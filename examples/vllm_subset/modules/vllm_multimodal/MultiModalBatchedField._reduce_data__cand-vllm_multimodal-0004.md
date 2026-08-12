# MultiModalBatchedField._reduce_data

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/inputs.py`](vllm/multimodal/inputs.py) (lines 513–542)
- **Symbol:** `MultiModalBatchedField._reduce_data`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_multimodal-0004`

## Description
Reduces a list of per-item NestedTensors into a batched tensor via an unsqueeze fast path or torch.stack.

## Current approach
For one tensor it unsqueezes, then may call contiguous or pin_memory. For multiple same-shaped tensors it checks shapes in Python, allocates a fresh output tensor with torch.empty, and calls torch.stack(batch, out=out).

## Estimated impact explanation
Pixel values and embedding tensors pass through this path on multimodal prefills. Large host allocations and stack copies can dominate CPU prefill preparation, so reducing them can materially lower median TTFT.

## Evolve rationale
The concrete alloc/copy constructs are torch.empty(..., pin_memory=pin_memory) and torch.stack(..., out=out). A lifecycle-aware pinned output buffer pool, avoiding redundant contiguity/pinning work for already suitable tensors, and specializing common same-shape CPU batches can reduce allocation churn. Correctness oracle: tests/multimodal/test_inputs.py and tests/multimodal/test_utils.py; outputs must match torch.stack semantics byte-for-byte.

## Deep research proposals

### 1. Pool pinned output buffers for MultiModalBatchedField._reduce_data batched stacks
- **Finding:** `find-vllm_multimodal-0005` — *A guide on good usage of non_blocking and pin_memory() in PyTorch*
- **Source URL:** <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a lifecycle-aware pinned-buffer pool that MultiModalBatchedField._reduce_data (vllm/multimodal/inputs.py:513-542) uses to obtain the destination tensor for the multi-tensor same-shape path instead of calling torch.empty(..., pin_memory=True) on every prefill. The pool keys buffers by (dtype, shape, device) and lends a pinned tensor for the duration of the torch.stack(batch, out=out) call plus the subsequent H2D transfer; the buffer is returned once the copy has been synchronized against the transfer stream (per the tutorial's guidance to explicitly synchronize GPU-facing copies before releasing pinned memory back for reuse). For the single-tensor branch (lines 521-531), skip the redundant contiguous()/pin_memory() work when batch[0] is already contiguous and pinned; only allocate/copy when a property is actually missing. Pool acquisition happens inside _reduce_data so the pinning cost amortizes across requests; the downstream _nested_tensors_h2d path can then issue the copy non_blocking=True on a dedicated transfer stream, matching the tutorial's asynchronous-transfer pattern. Correctness is preserved by keeping torch.stack(..., out=out) semantics (byte-for-byte equivalent to torch.stack(batch)) and re-running tests/multimodal/test_inputs.py and tests/multimodal/test_utils.py.

**Proposal rationale.**

The candidate's hot path allocates a fresh pinned host tensor per multimodal prefill (torch.empty(..., pin_memory=pin_memory)) and, for the single-tensor case, may redundantly force contiguity/pinning on tensors that already satisfy those properties. The PyTorch tutorial specifically prescribes pooled pinned buffers with explicit synchronization and non-default streams as the remedy for per-request pinned-allocation churn, which is exactly the bottleneck named in the candidate's evolve_rationale. Because pixel-values and embedding tensors flow through this reducer during multi-turn agentic prefills, replacing per-call cudaHostAlloc-backed empties with a reused pinned pool addresses the concrete allocation/copy cost that drives TTFT here, while the tutorial's synchronization guidance provides the safety pattern needed to keep torch.stack output semantics intact.

---

## Agent proposals

### 1. Detect shared-storage contiguous slices to elide the torch.stack copy in _reduce_data
- **Agent:** claude

**Detailed description.**

In `MultiModalBatchedField._reduce_data` (vllm/multimodal/inputs.py:513-542), before falling into the same-shape allocate-and-stack branch (lines 532-540), add a zero-copy fast path that checks whether the `batch` list is composed of contiguous, equally-strided slices of a single underlying storage laid out as `(len(batch), *first_shape)`. Concretely: (a) verify all tensors share `untyped_storage().data_ptr()`, `dtype`, `device`, and `stride()`; (b) verify their `storage_offset()` values form an arithmetic progression whose step equals `first_shape.numel()` (i.e., item stride in elements) and whose first offset is aligned so a `torch.as_strided` view over the shared storage reproduces the stacked shape exactly. When this holds, return `batch[0].untyped_storage()`-backed `torch.as_strided((len(batch), *first_shape), (item_stride, *batch[0].stride()), storage_offset=batch[0].storage_offset())` (optionally `.contiguous()` only when `pin_memory` is requested and the view is not already contiguous). This eliminates both the `torch.empty(..., pin_memory=pin_memory)` host allocation and the N-tensor `torch.stack` copy in the common case where an HF processor produced one `(N, C, H, W)` tensor upstream that `MultiModalBatchedField.build_elems` sliced per-item (line 511 iterates `for item in data`). For batches where the check fails, fall through to today's stack path unchanged. Correctness is preserved because the resulting tensor is bit-identical to `torch.stack(batch)` by construction of the strided view. Validate against `tests/multimodal/test_inputs.py` and `tests/multimodal/test_utils.py`; add a targeted test that constructs elements as slices of a shared tensor and asserts equality with `torch.stack`.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_multimodal-0005) targets amortizing pinned-host allocation churn via a lifecycle-aware pinned-buffer pool and trimming redundant `contiguous()`/`pin_memory()` calls in the single-tensor branch. It still performs a `torch.stack` copy into a (now pooled) destination buffer for the multi-tensor branch. This proposal is orthogonal: it eliminates the copy itself for the frequent case where the per-item elements are already contiguous slices of a shared upstream tensor (typical for HF image processors that emit a single `(N, C, H, W)` tensor which `build_elems` splits at line 511). The pool proposal never inspects storage identity or stride patterns, so it cannot recognize this zero-copy opportunity; conversely, this fast path composes cleanly with the pool for batches that don't qualify.

---

### 2. Short-circuit stacking for explicit singleton repeated batches
- **Agent:** codex

**Detailed description.**

In `MultiModalBatchedField._reduce_data` (`vllm/multimodal/inputs.py:513-542`), add a narrow fast path before the multi-tensor same-shape `torch.empty`/`torch.stack(..., out=out)` branch that detects when every element in `batch` is the exact same tensor object or an alias to the same storage with identical `shape`, `stride`, `storage_offset`, `dtype`, and `device`. For immutable multimodal fields that are duplicated across batch items, return an expanded leading-batch view via `batch[0].unsqueeze(0).expand(len(batch), *batch[0].shape)` when `pin_memory` is false, falling back to the existing stack path when a pinned destination is required or when aliasing conditions do not hold. This preserves `torch.stack` values while avoiding both allocation and N repeated copies for repeated prompt-level multimodal tensors, which can occur in multi-turn agentic workflows that reuse the same image or embedding across requests/turns. Add a focused test in the existing multimodal input tests that passes repeated references to one tensor and asserts equality with `torch.stack(batch)`, expected shape, and no mutation assumptions beyond the current reducer contract.

**Novelty rationale.**

The deep research proposal focuses on pooling pinned output buffers and skipping redundant contiguity or pinning work, but it still materializes a stacked tensor for multi-element batches. Agent A's proposal handles consecutive slices from one larger upstream tensor by reconstructing a batched `as_strided` view over storage offsets. This proposal targets a different aliasing pattern: repeated identical tensor references or exact same-storage aliases with the same offset, where the batched result can be represented by a zero-stride expanded view rather than a contiguous slice layout. It is narrower than Agent A's arithmetic-progression slice case and would not be covered by that check because all storage offsets are equal, not separated by item stride.

---
