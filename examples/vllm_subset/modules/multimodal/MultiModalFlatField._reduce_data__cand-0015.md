# MultiModalFlatField._reduce_data

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/inputs.py`](vllm/multimodal/inputs.py) (lines 529–600)
- **Symbol:** `MultiModalFlatField._reduce_data`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0015`

## Description
Reduces flat multimodal field elements into a concatenated or padded tensor batch before model execution.

## Current approach
For tensor batches, the method checks shapes in Python, uses torch.concat for compatible shapes, and for variable-length non-concat dimensions computes max_sizes with Python loops, zero-initializes the full output, builds a slice list for every tensor, and assigns each tensor into the padded result.

## Estimated impact explanation
This runs when multimodal items are batched across requests, especially for variable-length audio/video features. Reducing Python overhead and avoidable zero-fill lowers media preprocessing and batching latency, moving media TTFT and batch construction overhead that can affect TPOT under load.

## Evolve rationale
The variable-length branch at lines 566-599 is a batching and memory-write target. Headroom includes vectorizing shape/offset computation, using torch.empty plus explicit padding fill instead of zeroing the entire output when much of it is overwritten, specializing common 2D/3D audio feature layouts, and reducing Python slice construction in the per-tensor copy loop. Oracle: tests/multimodal/test_utils.py, tests/multimodal/test_inputs.py, and the Ultravox variable-length coverage must preserve batched_tensors_equal results, output shapes, padding values, concat order, device placement, and pin_memory behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace slice-assign loop with per-tensor F.pad followed by a single torch.cat(out=...)
- **Agent:** claude

**Detailed description.**

In the variable-length branch of `MultiModalFlatField._reduce_data` (vllm/multimodal/inputs.py:566-599), replace the `torch.zeros` + Python slice-construction + slice-assign loop with a structurally different algorithm:

1. Compute `max_sizes` as today (or via a single `torch.tensor([t.shape for t in batch]).amax(dim=0)` / `.sum(dim=0)[dim]` pass to drop the two Python `max(...)`/`sum(...)` loops over `ndim`).
2. For each tensor `t` in `batch`, build a pad spec that is `(0, 0)` on the concat dim and `(0, max_sizes[d] - t.shape[d])` on every non-concat dim, then call `torch.nn.functional.pad(t, pad_spec)`. This is a single fused C++/CUDA op per tensor that only writes the boundary pad regions (never the interior).
3. Pre-allocate the final output with `torch.empty(max_sizes, dtype=batch[0].dtype, device=batch[0].device, pin_memory=pin_memory)` and call `torch.cat(padded_tiles, dim=self.dim, out=out)` to do the concat in one fused kernel.

Net effect: drop the full-output `torch.zeros` (no double-write of the regions today's slice-assign would overwrite anyway); drop the per-tensor `slices: list[slice]` construction (lines 589-596); replace N indexed-assign kernel launches with N `F.pad` launches + one `cat` launch — keeping more work on the device dispatcher side and reducing per-element Python overhead, which compounds across multimodal items in a batched multi-turn agentic workload.

Semantics preserved: padded regions still hold 0 (F.pad's default fill value), concat order matches today (iterate `batch` in order), and shape/dtype/device/pin_memory match the current branch. Validate against tests/multimodal/test_inputs.py and tests/multimodal/test_utils.py — in particular the Ultravox variable-length coverage referenced by the candidate — by asserting `batched_tensors_equal` between old and new outputs across the existing fixtures. Keep today's slice-assign path as a guarded fallback for any dtype/device combo where `F.pad` is unsupported.

**Novelty rationale.**

This candidate has no existing deep_research_proposals, so novelty is measured against the candidate's own evolve_rationale. That rationale enumerates generic directions ('vectorize shape/offset', 'torch.empty plus explicit padding fill', 'specialize 2D/3D layouts', 'reduce Python slice construction') but does not propose the specific algorithmic restructuring here: replacing the slice-assign loop with F.pad-per-tensor + a single `torch.cat(..., out=...)`. The 'torch.empty + explicit pad-fill' suggestion still keeps a slice-assign loop and just zeros the pad regions after the fact; this proposal removes the slice-assign loop and the explicit pad-fill step entirely, relying on F.pad's own boundary fill and cat's fused output kernel. It is also independent of layout specialization and complementary to (not subsumed by) shape-vectorization.

---

### 2. Assemble ragged flat batches directly on the target device
- **Agent:** codex

**Detailed description.**

Thread the effective `device` from `BaseMultiModalField.reduce_data()` into `MultiModalFlatField._reduce_data()` and add a guarded fast path in the variable-length branch at `vllm/multimodal/inputs.py:566-599`: when the caller requested a non-CPU device and the input tensors are CPU tensors, allocate the padded `out` tensor directly on the target device, zero it there, and copy each valid tensor block into its slice with `copy_(..., non_blocking=True)`. Keep the current CPU-staging path when `device is None`, `keep_on_cpu` applies, device types are unsupported, or a simple padding-waste threshold says one full staged H2D copy is likely cheaper. This avoids creating a full zero-padded CPU tensor and then transferring padding bytes to GPU for Ultravox-like ragged `[B, 80, T]` audio batches; only real media data crosses H2D while padding is produced on the final device. Validate by comparing against the existing CPU reference for ragged 2D/3D batches, checking output device/dtype/shape/zero padding/concat order, and covering the `group_and_batch_mm_kwargs(..., device=cuda, pin_memory=False)` path used by the v1 multimodal encoder runner.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes changing the intra-reduction algorithm to `F.pad` each tensor and `torch.cat(out=...)` on the current staging device, which still materializes a fully padded batch before any later `_nested_tensors_h2d` transfer. This proposal targets a different cost center: the post-reduction device transfer. It changes where the ragged batch is assembled so padded bytes are never copied from host to device, and it remains complementary to either the existing slice-copy loop or Agent A's padding/cat approach for same-device reductions.

---
