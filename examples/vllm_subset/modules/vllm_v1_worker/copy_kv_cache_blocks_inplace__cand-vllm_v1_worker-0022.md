# copy_kv_cache_blocks_inplace

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/utils.py`](vllm/v1/worker/utils.py) (lines 565–601)
- **Symbol:** `copy_kv_cache_blocks_inplace`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0022`

## Description
Copies logical KV cache blocks in-place across backing storage tensors for prefix-cache promotion/copy-on-write.

## Current approach
Builds a unique storage list, transfers copy indices to GPU, then performs one advanced-index assignment per storage tensor on a uint8 view.

## Estimated impact explanation
Multi-turn agents reuse and promote prefixes frequently; fusing copies across storages reduces repeated launches and improves TPOT when many KV storage tensors are present.

## Evolve rationale
The per-storage advanced-index launch loop is owned code. Prefix-cache block-copy tests and the byte-divisibility assertion validate correctness for block-level copies.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Fuse per-storage KV block copies into a single CUDA-graph-captured copy kernel
- **Agent:** claude

**Detailed description.**

In `vllm/v1/worker/utils.py:copy_kv_cache_blocks_inplace` (lines 565-601), replace the Python-level `for tensor in storage_tensors` loop that issues one advanced-index assignment (`blocks[dst_indices] = blocks[src_indices]`) per unique backing storage with a single fused launch that copies all storages at once. Concretely: (1) pack the deduplicated storage base pointers, per-storage byte-strides-per-block, and shared (src_idx, dst_idx) pairs into small persistent device buffers (reused across calls via a small ring/cache keyed on `id(kv_caches)`), and (2) launch one custom CUDA kernel (or a Triton kernel) whose grid is `(num_copies, num_storages, ceildiv(page_bytes, BLOCK))` and whose body performs vectorized 16B (uint4) loads/stores from `src` to `dst` within each storage's block-major range. This collapses `S` advanced-index kernels + `S` index-broadcast prologues into a single launch, eliminates the repeated `torch.empty(0).set_(untyped_storage())` + `.view(num_blocks, -1)` overhead per call, and lets the copy be captured inside the persistent-batch CUDA graph rather than firing eager kernels on the hot path. For the multi-turn agentic workload, prefix promotions/COW happen every step across many attention + Mamba storages, so folding S launches into 1 measurably shrinks per-step host-side gap and shaves TPOT; the host-side dedup + h2d transfer is preserved so correctness (page-size divisibility, storage-alias handling) matches the existing implementation. Fall back to the current path when the fused kernel is unavailable (e.g., CPU/XPU) via a `current_platform.is_cuda_alike()` guard.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate (the list is empty), so a fused single-launch multi-storage copy kernel proposal is trivially not covered. It also goes beyond the candidate's own `evolve_rationale` (which only notes that fusing copies across storages reduces launches) by specifying the exact packing/kernel shape, CUDA-graph capture integration, and the persistent device-buffer reuse strategy needed to actually realize the TPOT win.

---

### 2. Add overlap-safe ordering for in-place KV block copies
- **Agent:** codex

**Detailed description.**

Change `vllm/v1/worker/utils.py:copy_kv_cache_blocks_inplace` to preserve correctness when a single call contains overlapping source and destination block IDs, such as chained promotions where one copy writes a block that another copy still needs to read. Before issuing the assignment, detect dependencies among `(src_block_number, dst_block_number)` pairs within the same storage: if `dst` appears in the source set, split the operation into dependency-safe phases using a temporary byte buffer only for the affected source blocks, while keeping the existing fast advanced-index path for the common non-overlapping case. Add focused tests that construct copy pairs like `0 -> 1, 1 -> 2` and `1 -> 2, 0 -> 1` across aliased storage tensors and verify the final byte contents are as if all sources were read from the pre-copy state. This reduces risk for prefix-cache copy-on-write paths without changing the hot non-overlap behavior that matters for median TTFT/TPOT.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal is a performance-oriented fusion of per-storage copies into one CUDA/Triton launch with persistent device buffers and CUDA graph capture; it assumes the same copy semantics as the current advanced-index assignment. This proposal is instead about strengthening in-place copy semantics for overlapping block mappings and adding targeted correctness coverage, so it is not a duplicate of the fused-kernel idea.

---
