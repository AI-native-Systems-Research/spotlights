# compute_mm_prefix_range_tensor

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/utils.py`](vllm/v1/attention/backends/utils.py) (lines 49–75)
- **Symbol:** `compute_mm_prefix_range_tensor`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0020`

## Description
Converts TritonAttention multimodal prefix ranges from a per-request dict into a padded device tensor.

## Current approach
Builds Python lists for every request, computes max_ranges on CPU, pads each request's range list, then uploads the nested list with async_tensor_h2d and reshapes it.

## Estimated impact explanation
Multimodal agent turns can hit this every prefill or extend step on the Triton backend. Reducing Python list work and H2D traffic improves multimodal TTFT and per-step overhead.

## Evolve rationale
Hot constructs are list construction, per-request padding, and per-step H2D upload. Correctness oracle is exact tensor equality for randomized mm_prefix_range dictionaries and Triton multimodal attention output equality.

## Deep research proposals

### 1. Pinned staging + vectorized fill for mm_prefix_range tensor with single non-blocking H2D
- **Finding:** `find-vllm_v1_attention-0006` — *CUDA C++ Best Practices Guide*
- **Source URL:** <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Rewrite compute_mm_prefix_range_tensor in vllm/v1/attention/backends/utils.py (lines 49-75) to eliminate the nested Python list construction and per-request padding loop, and to issue a single pinned-memory asynchronous H2D transfer instead of routing the nested list through async_tensor_h2d.

Specifically: (1) Compute max_ranges by iterating over mm_prefix_range.values() once (mm_prefix_range is a dict, so most entries default to [(0,0)] and can be counted as 1 without materializing range_lists); short-circuit to None when the dict is empty or every non-default entry has zero-length ranges. (2) Allocate a persistent (or high-water-mark) int32 pinned-memory CPU staging buffer of shape (num_seqs, max_ranges, 2) using the same pinned-memory path as np_to_pinned_tensor / PIN_MEMORY already imported in this file; zero it in-place (torch.Tensor.zero_() on the used slice) rather than reconstructing Python lists of (0,0) tuples. (3) Fill the buffer directly with numpy: for each request that actually has a non-default range list, write its (start, end) pairs into rows[i, :len(r)] via a small vectorized assignment (or a numpy asarray on the request's own list, which is at most a few tuples). (4) Issue one .to(device, non_blocking=True) copy from the pinned buffer to device — a single contiguous H2D per call — and return the (num_seqs, max_ranges, 2) view.

As a follow-on that stays within the same file, expose the pinned buffer so that other Triton attention-metadata H2D constructs assembled in the same build step (e.g. indptr, last-page lengths, split metadata) can be packed into one contiguous staging region and uploaded with a single H2D, matching the CUDA best-practices guidance to batch small transfers.

**Proposal rationale.**

The finding explicitly recommends (a) minimizing host-device transfers, (b) batching many small transfers into one contiguous transfer, and (c) using pinned memory for asynchronous copies — quoting the CUDA C++ Best Practices Guide §10.1. The candidate's current_approach and evolve_rationale name exactly the costs this guidance targets: Python list construction and per-request padding on the CPU, plus a per-step H2D upload of a small nested list on the Triton multimodal attention path. In a multi-turn agentic multimodal workload (the stated caller objective, targeting median TTFT/TPOT), this function is invoked on every prefill/extend attention-metadata build, so removing per-request Python object churn and replacing async_tensor_h2d on a nested list with a single pinned non-blocking copy is a direct, transferable application of the finding to the candidate. It addresses a concrete gap (unnecessary Python work + small unpinned-style transfer path) without changing the correctness oracle (exact tensor equality of the padded (num_seqs, max_ranges, 2) int32 tensor is preserved).

---

## Agent proposals

### 1. Replace padded (num_seqs, max_ranges, 2) mm_prefix_range tensor with a jagged CSR layout (flat values + int32 offsets)
- **Agent:** claude

**Detailed description.**

Change compute_mm_prefix_range_tensor in vllm/v1/attention/backends/utils.py (lines 49-75) to stop materializing a padded (num_seqs, max_ranges, 2) tensor and instead return a jagged/CSR pair: a contiguous int32 values tensor of shape (total_ranges, 2) holding the concatenated (start, end) pairs in request order, plus an int32 offsets tensor of shape (num_seqs + 1,) so that request i's ranges live at values[offsets[i]:offsets[i+1]]. Requests whose only entry is the sentinel [(0,0)] contribute zero rows to values (offsets stay flat across them), and the empty-batch short-circuit still returns None. Build both arrays with a single numpy pass over mm_prefix_range: numpy.fromiter or a preallocated numpy array sized to total_ranges avoids nested Python list construction and the per-request padding loop entirely; then upload values and offsets in one contiguous H2D (either concatenated into a single staging buffer and sliced on device, or as two non_blocking copies from pinned memory).

Update the Triton multimodal-prefix attention kernel and its Python-side wrapper (the sole consumer of this tensor) to accept (values, offsets) and iterate the request's own slice `for k in range(offsets[i], offsets[i+1])`, dropping the padded-slot is_valid check on (0,0) rows. This removes wasted per-query kernel work proportional to (max_ranges - actual_ranges_for_this_request), which is significant in heterogeneous multimodal agentic batches where one request contributes many ranges (e.g. multi-image turns) and the rest contribute one or none — exactly the current_approach's worst case, since max_ranges is set by the batch maximum.

H2D bytes drop from 8 * num_seqs * max_ranges to 8 * total_ranges + 4 * (num_seqs + 1); on skewed batches this is a large reduction and shrinks the per-step contribution of this metadata to prefill/extend TTFT and to per-step TPOT overhead on the Triton multimodal path. Correctness oracle changes from exact equality of the padded tensor to (a) equality of the CSR (values, offsets) pair against a reference builder, and (b) unchanged Triton multimodal attention output on the same randomized mm_prefix_range dictionaries. Verification: extend the existing fill/build tests to compare against a padded reference reconstructed as `torch.zeros(num_seqs, max_ranges, 2, int32); rows[i, :offsets[i+1]-offsets[i]] = values[offsets[i]:offsets[i+1]]`, and rerun the Triton multimodal attention numerical-equality tests.

**Novelty rationale.**

The existing deep_research_proposal (find-vllm_v1_attention-0006) explicitly preserves the padded (num_seqs, max_ranges, 2) tensor shape and preserves 'exact tensor equality of the padded ... int32 tensor' as its correctness oracle — it optimizes only the CPU build path (pinned staging, vectorized fill, single non-blocking H2D, coalescing with sibling metadata). It does not change the representation and therefore cannot eliminate the padding overhead itself. This proposal is orthogonal: it changes the representation to CSR/jagged, which (a) removes the O(num_seqs * max_ranges - total_ranges) padded H2D bytes, (b) drops wasted per-query kernel work on (0,0) padded slots inside the Triton multimodal-prefix kernel, and (c) is strictly composable with the pinned-single-H2D approach — the two stack. It also requires a coordinated kernel-side change that the existing proposal does not touch.

---

### 2. Cache and update mm_prefix_range tensors only when request ranges change
- **Agent:** codex

**Detailed description.**

Add a small cache around `compute_mm_prefix_range_tensor` in `vllm/v1/attention/backends/utils.py`'s caller path so the padded device tensor is rebuilt only when the batch's `mm_prefix_range` content or shape changes. Compute a cheap signature from the ordered per-request ranges used for the current metadata build, for example `(num_seqs, per_request_lengths, flattened_start_end_pairs)` or a monotonic request metadata version if one already exists nearby, and store the resulting device tensor plus its signature on the Triton attention metadata builder/state. On decode/extend steps where multimodal prefix ranges are unchanged across turns or tokens, return the cached tensor directly instead of repeating Python range collection, padding, and H2D upload. Invalidate the cache when the request order changes, sequences are added/removed, any request's multimodal prefix ranges change, or `max_ranges` changes. Keep the existing tensor representation and semantics unchanged, and add tests that build metadata for the same `mm_prefix_range` across repeated steps and assert both tensor equality and that the upload/build helper is not reinvoked after a cache hit.

**Novelty rationale.**

The deep_research_proposal optimizes each rebuild by using pinned staging, vectorized fill, and a single non-blocking copy, but it still rebuilds and uploads the tensor on every call. Claude's proposal changes the representation to CSR to reduce padding bytes and kernel work, but likewise assumes a fresh representation is produced for each metadata build. This proposal targets temporal redundancy across repeated decode/extend metadata builds: avoid calling `compute_mm_prefix_range_tensor` at all when the multimodal prefix ranges are stable. It is orthogonal to both per-call build optimization and jagged layout changes.

---
