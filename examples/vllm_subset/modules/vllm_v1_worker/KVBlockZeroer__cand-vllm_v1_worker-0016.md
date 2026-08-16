# KVBlockZeroer

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/utils.py`](vllm/v1/worker/utils.py) (lines 47–241)
- **Symbol:** `KVBlockZeroer`
- **Kind:** config_block
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0016`

## Description
Builds segment metadata and launch parameters for the KV block zeroing Triton kernel, then launches zeroing for newly allocated blocks.

## Current approach
Derives a single blk_size and MAX_CHUNKS from all segment page sizes, launches n_blocks x n_segs x max_chunks programs, and relies on an early return when a segment has fewer chunks.

## Estimated impact explanation
Fresh KV blocks are allocated during new turns and prefills; segment-aware tiling reduces wasted programs and improves TTFT/tail latency when page sizes vary across KV cache segments.

## Evolve rationale
The policy-defining constructs are blk_size, max_chunks, and the grid calculation. tests/v1/worker/test_kv_block_zeroer.py and tests/v1/worker/test_dsv4_packed_zeroer_geometry.py validate zeroing and packed geometry invariants.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace rectangular zero-kernel grid with a precomputed flat work-item table sized to exact per-segment chunk counts
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/utils.py (KVBlockZeroer at lines 47-241), the current design launches `n_blocks * n_segs * MAX_CHUNKS` Triton programs where `MAX_CHUNKS = max_page_size_el // blk_size` and `blk_size` is the GCD-derived largest_power_of_2_divisor across all `seg_page_sizes` (capped at 1024). This has two compounding inefficiencies when segment page sizes are imbalanced (common in MLA + DSA indexer setups, packed KV views, or mixed KV cache groups): (1) `blk_size` collapses to the smallest segment's page-size divisor, inflating chunk counts across all segments, and (2) every segment pays the `MAX_CHUNKS` toll of the largest segment, with the `if chunk_index >= page_size_el // BLOCK_SIZE: return` guard becoming pure launch overhead for the smaller segments.

Proposed change:

1. In `KVBlockZeroer.__init__` (around lines 207-219), after collecting `seg_page_sizes` and `seg_block_strides`, choose `blk_size` per-segment: compute `seg_blk_size[i] = min(largest_power_of_2_divisor(seg_page_sizes[i]), 1024)` and `seg_chunks[i] = seg_page_sizes[i] // seg_blk_size[i]`. Then still pick one kernel-wide `BLOCK_SIZE` (Triton `constexpr` needs to be fixed), but pick it as the *max* power-of-2 that divides every `seg_page_size` (same as today) OR — better — accept multiple kernels bucketed by their `blk_size`. As a simpler alternative that keeps one kernel: keep the existing GCD-derived `blk_size`, but precompute a flat work-item table.

2. Precompute a work-item table `work_items` of length `total_chunks = sum(seg_page_sizes[i] // blk_size for all i)`. Each entry stores `seg_id` (int32); the chunk offset within the segment is implicit as position minus the segment's start offset, or store `chunk_offset_el` (int32) directly to avoid a second lookup. Also build a companion `seg_start_offsets` if needed. Upload these as int32 tensors alongside the existing `seg_addrs`, `seg_block_strides`, `seg_page_sizes`.

3. Rewrite `_zero_kv_blocks_kernel` to a 2-D program space `(block_index, work_index)` with grid `(n_blocks * total_chunks,)`. Each program loads `seg_id = work_items_seg[work_index]` and `chunk_offset_el = work_items_off[work_index]`, then loads `seg_addr` / `block_stride_el` for that segment and stores `BLOCK_SIZE` zeros — with no early-return branch. Remove the `chunk_index >= page_size_el // BLOCK_SIZE` guard entirely; the flat table encodes exactly the valid `(seg, chunk)` pairs.

4. In `zero_block_ids` (lines 221-245), replace the grid computation `grid = (n_blocks * n_segs * max_chunks,)` with `grid = (n_blocks * total_chunks,)` where `total_chunks` was precomputed and stored in `self._meta`. `N_SEGS` and `MAX_CHUNKS` become unused constexprs and can be dropped.

The kernel does the same total useful work (`n_blocks * total_chunks` stores of `BLOCK_SIZE` zeros) but launches exactly that many programs instead of `n_blocks * n_segs * max_chunks`. When segments are balanced this is a no-op change. When they are imbalanced (e.g., one segment is 4x smaller than the max), the wall time savings scale with the fraction of no-op programs eliminated — which the current early-return incurs as both scheduler and TLB overhead on each zero call during prefill/new-turn allocation.

Validation:

- Extend `tests/v1/worker/test_kv_block_zeroer.py` with a case where segments have mixed page sizes (e.g., 512 and 4096 elements) and assert byte-for-byte that the freshly zeroed regions match the reference. The existing packed-geometry invariant test `tests/v1/worker/test_dsv4_packed_zeroer_geometry.py` should still pass unchanged.
- Micro-benchmark: for a representative multi-turn agentic workload's per-step new-block count (typically 8-64 blocks), time `zero_block_ids` before/after using CUDA events on a fixture with imbalanced segments (MLA + DSA-style config). Verify the ratio of programs launched is `sum(page_sizes) / (n_segs * max_page_size)` and that runtime drops proportionally for small n_blocks (launch-overhead-bound regime).
- End-to-end: measure median TTFT with `vllm bench serve` on a multi-turn agentic trace; the zeroer runs on new-block allocation on every turn, so the effect should be visible on TTFT particularly for short prompts where the zeroing latency is a meaningful fraction of prefill.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so this proposal cannot overlap prior work. The idea itself — precomputing a per-segment flat work-item table so the launch grid matches exact useful work count rather than `n_segs * max_chunks` — targets the specific `evolve_rationale` elements the candidate flagged (blk_size, max_chunks, grid calculation) and eliminates the early-return branch entirely, which is different from other plausible directions like autotuning `BLOCK_SIZE`, splitting into per-segment kernel launches, or fusing zeroing into the allocator.

---

### 2. Skip KV zeroing for CUDA allocations that are already guaranteed clean
- **Agent:** codex

**Detailed description.**

Add a fast path in `KVBlockZeroer.zero_block_ids` that avoids launching `_zero_kv_blocks_kernel` when the worker can prove the target KV cache storage is freshly created and has not been previously used by any request. Today `KVBlockZeroer` is structured as an unconditional sanitizer for newly allocated block ids, which is correct for reused blocks but pays the full zero-kernel cost even during cold-start/prefix phases where blocks come from never-touched KV cache pages. Track a per-cache-block cleanliness/ever-used bit alongside the block allocator state, initialize all blocks as clean immediately after KV cache allocation, mark blocks dirty once they have been handed to model execution, and have `zero_block_ids` compact/filter the incoming `block_ids` so the Triton kernel is launched only for dirty reused blocks. If every requested block is still clean, return without launching the kernel. Keep the existing segment metadata and kernel unchanged for the remaining dirty ids, so correctness for reused blocks and packed segment geometry is preserved. Validation should add a unit test around `tests/v1/worker/test_kv_block_zeroer.py` or the allocator-facing worker tests that verifies first-use block ids do not invoke zeroing while recycled block ids are still zeroed byte-for-byte, then benchmark multi-turn short-prompt traffic where cold or newly expanded cache allocation contributes to TTFT.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal optimizes the shape of the Triton launch by replacing the rectangular per-segment grid with an exact flat work-item table, but it still zeroes every block passed to `zero_block_ids`. This proposal is orthogonal: reduce launches and stores by filtering out blocks that provably do not require zeroing at all, while leaving the candidate's current chunk/grid policy untouched for blocks that do require sanitization.

---
