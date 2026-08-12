# MultiModalFlatField._reduce_data

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/inputs.py`](vllm/multimodal/inputs.py) (lines 569–649)
- **Symbol:** `MultiModalFlatField._reduce_data`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0005`

## Description
Concatenates or pads variable-length NestedTensors into one flat tensor for fields whose batch dimension is represented by slices.

## Current approach
For same non-concat shapes it allocates torch.empty and calls torch.concat(out=...). For variable non-concat shapes it computes max_sizes in Python, creates torch.zeros, builds a list of slice objects per tensor, and assigns each tensor into the output one by one.

## Estimated impact explanation
This affects audio and video workloads with variable feature lengths. Fewer per-item copies or launches reduces multimodal prefill preparation latency, moving TTFT for those requests.

## Evolve rationale
The hot constructs are torch.zeros plus the per-tensor out[tuple(slices)] = tensor loop. For audio/video features with many variable-length items, padding/packing via fewer tensor ops or a specialized copy plan can reduce per-item indexing overhead and kernel launches while keeping the padded layout contract. Correctness oracle: tests/multimodal/test_inputs.py; results must match the existing concat/padded slice-assignment layout.

## Deep research proposals

### 1. Use torch.nested packed values+offsets and to_padded_tensor for variable-length _reduce_data
- **Finding:** `find-vllm_multimodal-0006` — *torch.nested — PyTorch 2.9 documentation*
- **Source URL:** <https://docs.pytorch.org/docs/2.9/nested.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/multimodal/inputs.py` `MultiModalFlatField._reduce_data` (lines 569-649), replace the variable-length branch (lines 607-646) — which allocates `torch.zeros(max_sizes)` and then executes a Python `for tensor in batch` loop performing `out[tuple(slices)] = tensor` per item — with a torch.nested-backed path.

Concretely, when the non-concat dimensions differ across `batch`:
1. Build a jagged nested tensor from `batch` via `torch.nested.nested_tensor(batch, layout=torch.jagged)` (or the values+offsets constructor `torch.nested.nested_tensor_from_jagged(values, offsets)` when the concat dim is 0 and inputs are already contiguous, letting us skip an intermediate copy for that common audio case).
2. Call `nt.to_padded_tensor(padding=0)` to materialize the same zero-padded dense layout the current code produces.
3. Apply the existing concat-dim offset logic: if `self.dim == 0` this yields exactly the current output; for `self.dim > 0`, transpose so the ragged dim aligns with jagged layout before/after padding, matching the current shape contract.
4. Keep the existing same-shape fast path (lines 596-605) and single-tensor fast path (lines 577-587) unchanged. Add a fallback to today's slice-assign loop for dtype/device combinations that torch.nested does not support, guarded by a capability check.

Correctness is verified against `tests/multimodal/test_inputs.py`; the padded layout (max across non-concat dims, sum across concat dim, zero fill) is preserved bit-for-bit.

**Proposal rationale.**

The candidate's stated hot constructs are `torch.zeros(max_sizes)` plus a Python-level per-tensor `out[tuple(slices)] = tensor` loop — one indexed copy kernel launch per batch item, plus per-item Python work to build `slices`. The finding points to torch.nested's packed values+offsets representation, whose `to_padded_tensor` is the exact PyTorch-native primitive for turning a ragged batch into a padded dense tensor in a single fused operation. This addresses the candidate's specific gap (per-item indexing overhead and kernel-launch count scaling with batch size) without changing the output contract downstream consumers depend on. For the target workload (multi-turn agentic with audio/video features of differing lengths), reducing per-item overhead in multimodal prefill preparation directly attacks median TTFT, matching the candidate's stated impact. The finding is not merely topically adjacent: `to_padded_tensor` is precisely the ragged-to-padded conversion this candidate performs by hand.

---

## Agent proposals

### 1. Fuse per-item slice-assign into a single torch._foreach_copy_ over narrow views, and elide zero-fill where padding is empty
- **Agent:** claude

**Detailed description.**

In `vllm/multimodal/inputs.py` `MultiModalFlatField._reduce_data` (lines 607-646, the variable-length branch), replace the Python-level `for tensor in batch: out[tuple(slices)] = tensor` loop with a two-part change that keeps the exact same padded-dense output layout.

1. Build a list of destination views up front using chained `torch.Tensor.narrow` calls, then execute all copies as one batched multi-tensor op via `torch._foreach_copy_(dst_views, batch)`. Concretely, for each item compute `view = out.narrow(dim, concat_offset, t.shape[dim])` and then, for every other axis `d` where `t.shape[d] < max_sizes[d]`, further narrow via `view = view.narrow(d, 0, t.shape[d])`. Collect all such views into `dst_views` and issue a single `torch._foreach_copy_(dst_views, batch)` (falling back to a manual loop if `_foreach_copy_` is unavailable for the given dtype/device — e.g., some CPU dtypes). `narrow` produces zero-copy strided views, so no extra allocation is introduced, and `_foreach_copy_` dispatches through a single multi-tensor kernel rather than N indexed-assignment dispatches, cutting Python-level overhead and PyTorch dispatcher cost that today scales linearly with batch size.

2. Elide the `torch.zeros` zero-fill when it is provably unnecessary. Before allocating, check whether any non-concat axis has heterogeneous sizes across `batch`: `needs_pad = any(max_sizes[d] != min(t.shape[d] for t in batch) for d in range(ndim) if d != dim)`. When `needs_pad is False` (i.e., every item already matches the max on every non-concat dim, so the padded region is empty — the common case for audio batches whose only variable axis is the time/concat dim), allocate with `torch.empty(max_sizes, ...)` instead of `torch.zeros`, skipping the full-buffer zeroing pass. When `needs_pad is True`, keep `torch.zeros` to preserve the zero-fill contract for the padded region.

Preserve the existing same-shape fast path (lines 596-605) and single-tensor fast path (lines 577-587). Correctness is verified against `tests/multimodal/test_inputs.py`; the padded layout (max across non-concat dims, sum across concat dim, zero fill in padded regions) is bit-for-bit identical.

**Novelty rationale.**

The existing deep_research_proposal (find-vllm_multimodal-0006) rewrites the branch around `torch.nested.nested_tensor` + `to_padded_tensor`, which changes the representation of `batch` into a jagged nested tensor and materializes the padded dense form via a nested-tensor primitive; it also requires care around ragged-axis alignment for `dim > 0` (transposes) and a capability-guarded fallback for dtypes torch.nested does not support. This proposal is orthogonal: it keeps the current explicit `torch.empty`/`torch.zeros` + per-slab layout unchanged (no nested tensors, no transposes, no ragged representation), and instead attacks the two remaining bottlenecks that the nested path does not address on their own: (a) fusing the N per-item Python-dispatched `out[tuple(slices)] = t` writes into a single `torch._foreach_copy_` batched multi-tensor kernel over pre-materialized `narrow` views, and (b) skipping the `torch.zeros` full-buffer clear when the padded region is provably empty (a common case that the nested path still pays for inside `to_padded_tensor`). The two ideas could even be composed later, but the mechanism here — foreach-copy over narrow views + conditional zero-fill elision — is not covered by the nested-tensor proposal.

---

### 2. Coalesce contiguous same-padding runs before copying into the padded output
- **Agent:** codex

**Detailed description.**

In `vllm/multimodal/inputs.py` `MultiModalFlatField._reduce_data`, add a middle path inside the variable non-concat branch that groups adjacent tensors whose non-concat dimensions are identical, then copies each run as one concatenated slab. After computing `max_sizes`, scan `batch` in order and form runs keyed by `tuple(t.shape[d] for d in range(ndim) if d != self.dim)`. For a run length greater than 1, allocate or reuse the destination view for the run's total concat length, call `torch.concat(run_tensors, dim=self.dim, out=dest_view)` when the view is contiguous-compatible, and advance the concat offset once for the whole run. For singleton runs, keep the existing slice assignment fallback. The output contract remains unchanged: the outer allocation is still zero-filled for padded regions, the concat-axis order is preserved exactly, and non-concat dimensions still occupy the leading slice of each padded slab. Add focused coverage in `tests/multimodal/test_inputs.py` for mixed runs such as shapes `[2, 3]`, `[4, 3]`, `[1, 5]`, `[2, 5]` with `dim=0`, verifying equality with the current per-item implementation.

**Novelty rationale.**

This is not the deep-research nested-tensor proposal: it keeps the existing dense padded output path and does not introduce `torch.nested`, jagged layouts, transposes, or `to_padded_tensor`. It is also distinct from Agent A's `_foreach_copy_` over per-item narrow views: this proposal reduces the number of logical copy operations by coalescing adjacent compatible items into larger concat slabs, using `torch.concat(..., out=dest_view)` for each run rather than issuing one copy per item through a foreach API. It targets workloads where multimodal items often arrive in shape-compatible bursts while still preserving the original item order.

---
