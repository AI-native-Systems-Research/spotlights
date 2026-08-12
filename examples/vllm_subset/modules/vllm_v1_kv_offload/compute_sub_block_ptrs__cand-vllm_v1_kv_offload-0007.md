# compute_sub_block_ptrs

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/gpu_worker.py`](vllm/v1/kv_offload/cpu/gpu_worker.py) (lines 73–120)
- **Symbol:** `compute_sub_block_ptrs`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0007`

## Description
Expands block IDs into raw byte pointers for transfer descriptors, including sub-block fanout and partial-block skip handling.

## Current approach
Fast path handles blocks_per_chunk == 1. General path allocates sub_offsets, materializes a full (num_blocks, blocks_per_chunk) pointer matrix, flattens it, then slices by skip_count and output length.

## Estimated impact explanation
This fixed overhead is paid twice per group/data-ref pair inside transfer submission. It can noticeably reduce small-promotion CPU overhead, though the ceiling is bounded by the surrounding transfer scheduling and memory copy cost.

## Evolve rationale
The general path allocates and materializes intermediates on every call even when only a slice is needed. Alternatives include direct strided writes into output, cached sub-offsets, or a fused descriptor builder. Correctness oracle: numeric equality of pointer arrays across randomized block IDs, skip counts, and block fanouts.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Eliminate matrix materialization in compute_sub_block_ptrs via direct strided writes with cached invariants
- **Agent:** claude

**Detailed description.**

Rewrite the `blocks_per_chunk > 1` branch of `compute_sub_block_ptrs` (vllm/v1/kv_offload/cpu/gpu_worker.py:110-120) so it never allocates the full `(num_blocks, blocks_per_chunk)` `all_ptrs` matrix nor calls `.ravel()`. Instead:

1. Compute `first_partial = skip_count > 0` and `num_full_blocks = (num_sub_blocks - (blocks_per_chunk - skip_count)) // blocks_per_chunk` (with a small edge case when the entire output fits inside the first partial block). Only three regions can ever appear in `output`: (a) a leading partial sub-block slice `[skip_count : blocks_per_chunk]` from the first block, (b) zero or more full blocks that each contribute exactly `blocks_per_chunk` pointers, and (c) an optional trailing partial slice from the last block.

2. Write directly into `output` using strided numpy views. Materialize `block_ids[:effective_num_blocks].astype(np.uint64) * row_stride + base_ptr` into a temporary of size `effective_num_blocks` (one column), then for the full-blocks region use `output.reshape(num_full_blocks, blocks_per_chunk)[:] = block_bases[:, None] + sub_offsets[None, :]` (broadcasted assignment into the destination view, no intermediate flat array or ravel). Handle the leading/trailing partial regions with two small `+=`/scalar-broadcast writes.

3. Hoist the loop-invariants that today are recomputed on every one of the `2 * len(group_data_refs)` calls per group: cache `base_ptr = tensor.data_ptr()` and `row_stride = tensor.stride(0)` per `(src_tensors[t_idx], dst_tensors[t_idx])` at handler init (they never change once the KV cache tensors are allocated), and precompute a shared `sub_offsets` array per handler for each of `src_blocks_per_chunk` and `dst_blocks_per_chunk` (there are only ever two values). Pass those cached scalars/arrays into `compute_sub_block_ptrs` (or make it a method that reads them from `self`) so the function does zero Python-level `data_ptr()` / `stride()` / `np.arange` calls per invocation.

4. Micro-optimize the fast-path (`blocks_per_chunk == 1`): the current line `output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride` unconditionally allocates a new uint64 copy of `block_ids`. Pre-declare `block_ids` as uint64 upstream (or pass a pre-cast view via a small cache keyed on `id(block_ids)`) so the astype becomes a no-op — the two calls per data-ref then each save one allocation.

Correctness oracle: keep the current implementation behind a `_reference_compute_sub_block_ptrs` helper and add a randomized pytest that varies `num_blocks ∈ [1, 128]`, `blocks_per_chunk ∈ {1, 2, 4, 8, 16}`, `skip_count ∈ [0, blocks_per_chunk)`, `num_sub_blocks` over the full valid range, and non-contiguous tensors with `row_stride > blocks_per_chunk * block_page_size`. Assert bit-exact `np.array_equal` between the new and reference outputs.

Expected impact: for the multi-turn agentic workload TTFT/TPOT budget, this removes two numpy allocations and a `ravel()` per `(group, data_ref)` pair on the critical submission path, and turns per-call `tensor.stride(0)` / `.data_ptr()` / `np.arange` into O(1) attribute reads. Small-promotion transfer submission is called twice per group (H2D + D2H staging) and once per data-ref inside each, so the constant-factor savings compound across the many small transfers that dominate agentic KV re-attach.

**Novelty rationale.**

The candidate lists zero existing deep_research_proposals, so there is nothing to overlap. Beyond the candidate's own evolve_rationale bullet (which lists 'direct strided writes', 'cached sub-offsets', 'fused descriptor builder' as adjacent ideas), this proposal is concrete about how the three-region (leading-partial / full-blocks / trailing-partial) direct write into `output` avoids the flat intermediate, AND additionally identifies loop-invariant hoisting of `data_ptr()`/`stride(0)`/`sub_offsets` out of the per-`data_ref` loop in `transfer_async` (lines 332-349), plus the redundant uint64 astype on the fast path — optimizations orthogonal to the matrix-materialization fix and not enumerated in the candidate's rationale.

---

### 2. Reuse per-group relative pointer offsets across data refs
- **Agent:** codex

**Detailed description.**

Refactor `compute_sub_block_ptrs` so the block-id expansion can be computed as relative byte offsets independent of `tensor.data_ptr()`, then reused within the `transfer_async` group loop for every `data_ref` that has the same `blocks_per_chunk`, `row_stride`, `block_page_size`, `skip_count`, and `group_size`. In the current caller, `group_src`/`group_dst` and skip counts are fixed for the whole group, but `compute_sub_block_ptrs` is invoked once per `data_ref`, repeating the same block-id cast and sub-block fanout work for each tensor. Add a helper such as `_compute_sub_block_offsets(block_ids, blocks_per_chunk, row_stride, block_page_size, skip_count, output)` that writes offsets relative to base pointer. For each group and direction, compute that offset vector once into a scratch array when the participating tensors are shape/stride-compatible, then fill each descriptor slice with `base_ptr + offsets`; fall back to the existing per-tensor path when strides or page sizes differ. This keeps correctness for non-contiguous tensors while removing repeated expansion work in the common case where many KV data refs share the same layout.

**Novelty rationale.**

There are no deep_research_proposals to overlap with. Agent A proposes eliminating the full pointer matrix, caching tensor invariants/sub-offsets, and avoiding redundant `astype` allocations inside each call. This proposal is different: it targets cross-call reuse in the `transfer_async` group loop by computing the relative offsets once per group and applying different tensor base pointers for each `data_ref`, reducing the number of `compute_sub_block_ptrs` expansions rather than only making each expansion cheaper.

---
