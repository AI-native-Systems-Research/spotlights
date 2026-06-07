# compute_sub_block_ptrs

[← v1.kv_offload](../v1.kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/gpu_worker.py`](vllm/v1/kv_offload/cpu/gpu_worker.py) (lines 39–86)
- **Symbol:** `compute_sub_block_ptrs`
- **Kind:** function
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0002`

## Description
Builds the flat int64 byte-pointer array for sub-block copies consumed by ops.swap_blocks_batch, and is called for both source and destination pointer lists during every transfer submission.

## Current approach
The block_size_factor == 1 path writes the output with one NumPy expression. The general path allocates a fresh sub_offsets array, casts block_ids to int64, builds a 2-D broadcasted all_ptrs array, flattens with ravel(), then slices [skip_count:skip_count+num_sub_blocks] into the caller-provided output buffer. SingleDirectionOffloadingHandler.transfer_async calls it twice per data_ref per KV-cache group.

## Estimated impact explanation
It reduces Python and NumPy overhead paid on every CPU<->GPU swap submission. The impact is medium because it only affects submission overhead, but frequent small offload/load operations in agentic workloads can make that overhead visible in TPOT tails.

## Evolve rationale
This is hot host-side transfer-prep code and allocates temporaries on every unaligned or multi-sub-block transfer. Headroom includes caching sub_offsets per factor on the handler, writing the requested window directly into output without all_ptrs/ravel, or replacing the NumPy expansion with a small native/vectorized routine. The correctness oracle is byte-exact equality with the current pointer formula across block_size_factor, skip_count, non-contiguous CPU strides, and the existing GPU-worker round-trip tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Hoist sub-block offset computation out of the per-data_ref loop in transfer_async
- **Agent:** claude

**Detailed description.**

Refactor `compute_sub_block_ptrs` (vllm/v1/kv_offload/cpu/gpu_worker.py:39-86) and its two call sites in `SingleDirectionOffloadingHandler.transfer_async` (lines ~262-279) so the expensive part of the expansion is computed once per group instead of once per data_ref. Concretely, split the function into (a) `build_sub_block_offsets(block_ids, block_size_factor, skip_count, num_sub_blocks, row_stride) -> np.ndarray[int64]` which returns the base-relative offset table `(block_ids[:,None].astype(int64) * row_stride + sub_offsets[None,:]).ravel()[skip_count : skip_count + num_sub_blocks]`, and (b) a tiny `output[:] = base_ptr + offsets` writer (a single np.add with `out=output`, no temporaries). In `transfer_async`, hoist the `build_sub_block_offsets` call above the `for data_ref in group_data_refs` loop: compute `src_offsets` and `dst_offsets` once per group using `self.src_tensors[group_data_refs[0].tensor_idx].stride(0)` / `self.dst_tensors[...].stride(0)`, asserting that all data_refs in the group share the same `row_stride` (true for symmetric KV-cache layers; if the assertion ever fails we fall back to the per-data_ref path). Then the inner loop reduces to two scalar-broadcast adds per data_ref: `np.add(self.src_tensors[t_idx].data_ptr(), src_offsets, out=all_src[op_idx:end_idx])` and the matching dst version. Preserve the `block_size_factor == 1` fast path inside `build_sub_block_offsets` (skip the broadcast/ravel and just return `block_ids[:num_sub_blocks].astype(int64) * row_stride`). Sub_offsets can be cached on the handler keyed by `(block_size_factor, sub_block_size)` since both are immutable handler attributes — this is cheap and complements the hoist. Net effect: for a group with N KV layers we go from 2N broadcast-multiply-ravel-slice operations to 2 broadcast-multiply-ravel-slice operations plus 2N pointer-aligned scalar-broadcast adds writing directly into the pre-allocated all_src/all_dst slices, eliminating the per-data_ref Python and NumPy temporary-allocation overhead that dominates today's submission cost. Correctness oracle: byte-exact equality with the existing implementation's output across the candidate's enumerated cases (varying block_size_factor, skip_count, non-contiguous CPU strides), plus the existing GPU-worker round-trip tests under `tests/v1/kv_offload/`.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals. The candidate's `evolve_rationale` enumerates three ideas: (1) cache `sub_offsets` per factor, (2) write the requested window directly into `output`, (3) replace the NumPy expansion with a native/vectorized routine. This proposal is materially distinct: it identifies that the dominant cost — the `block_ids.astype(int64)[:, None] * row_stride` broadcast plus ravel/slice — is invariant across the inner `for data_ref in group_data_refs` loop and hoists it to group scope, turning O(N_layers) expansions into O(1) per group. Caching `sub_offsets` (rationale item 1) only saves a tiny `np.arange(factor) * sub_block_size` vector; this proposal additionally caches the much larger expanded offset table within the call's group scope. Writing directly into `output` (rationale item 2) is a per-call micro-opt; this proposal restructures the call graph above `compute_sub_block_ptrs` so most data_refs skip the expansion entirely. A native routine (rationale item 3) speeds up each individual call but does nothing about the redundant work across data_refs. The hoist composes with all three rationale ideas rather than overlapping with any of them.

---

### 2. Compact contiguous copy spans before calling swap_blocks_batch
- **Agent:** codex

**Detailed description.**

After `compute_sub_block_ptrs` fills `all_src` and `all_dst` in `vllm/v1/kv_offload/cpu/gpu_worker.py`, add an in-place compaction pass over `all_src`, `all_dst`, and `all_sizes`: if the next entry starts exactly at `prev_src + prev_size` and `prev_dst + prev_size`, fold it into the previous entry by increasing `all_sizes[write_idx]`; otherwise keep it as a new span. Then pass `all_src[:num_spans]`, `all_dst[:num_spans]`, and `all_sizes[:num_spans]` to `ops.swap_blocks_batch` while leaving `num_transfer_bytes` based on the original logical bytes. This uses the existing raw-pointer-plus-size contract of `swap_blocks_batch` and is safe for non-contiguous CPU mmap strides because it only merges when both computed byte pointers prove adjacency. Add focused tests for contiguous GPU/CPU block runs, unaligned `skip_count`, non-contiguous CPU row strides, and random non-contiguous block IDs where the compacted length should remain unchanged.

**Novelty rationale.**

There are no listed deep_research_proposals. Agent A's proposal reduces redundant pointer expansion across KV tensors but still emits one copy operation per logical GPU page. This proposal changes the emitted copy list itself by merging adjacent byte ranges, reducing the number of `cuMemcpyBatchAsync` entries or fallback `cudaMemcpyAsync` calls. It is also distinct from the candidate rationale items: it does not cache sub-offsets, does not merely write the existing window more directly, and does not replace NumPy pointer generation with a native routine.

---
