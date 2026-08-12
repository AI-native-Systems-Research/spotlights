# _compute_slot_mapping_kernel

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/block_table.py`](vllm/v1/worker/block_table.py) (lines 380–442)
- **Symbol:** `_compute_slot_mapping_kernel`
- **Kind:** kernel
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0010`

## Description
Legacy Triton kernel that maps scheduled token positions to KV-cache slot IDs and pads unused CUDA graph slots.

## Current approach
Launches one program per request plus one padder program, with a fixed BLOCK_SIZE=1024 vector per request tile regardless of actual decode length.

## Estimated impact explanation
Slot mapping runs every step per cache group; decode-heavy agent workloads often schedule one token per request, so the current tiling wastes lanes and affects steady TPOT.

## Evolve rationale
The fixed per-request 1024-lane tiling is the optimization target. A request-by-token-tile grid or adaptive block size can preserve slot-mapping and PAD_ID invariants validated by attention backend and slot-mapping tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Reshape slot-mapping kernel to a 2D token-tile grid with binary-search req lookup
- **Agent:** claude

**Detailed description.**

Restructure `_compute_slot_mapping_kernel` (vllm/v1/worker/block_table.py:379-442) from a `(num_reqs+1,)` request-per-program grid with fixed `BLOCK_SIZE=1024` into a `(cdiv(max_num_tokens, TILE),)` token-tile grid where each program handles exactly `TILE` (e.g., 64 or 128) contiguous global token positions. Steps:

1. In `BlockTable.compute_slot_mapping` (line 195), replace the launch `(num_reqs + 1,)` with `(triton.cdiv(self.max_num_batched_tokens, TILE),)` so the grid size is bounded by tokens rather than requests, and pad+compute are fused into one program per tile.
2. Inside the kernel, each program computes `token_offsets = pid * TILE + tl.arange(0, TILE)` and a validity mask `token_offsets < num_tokens`. Padding of the tail `[num_tokens, max_num_tokens)` is done by simply storing `PAD_ID` under the inverted mask in the same kernel — eliminating the dedicated padder program that currently serially strides at `BLOCK_SIZE=1024` per iteration.
3. To resolve which request each token belongs to, perform a per-lane branchless binary search over `query_start_loc_ptr[0..num_reqs]` (num_reqs+1 int32 entries, easily fits in L1 / can be `tl.load`-cached). Each lane derives its `req_idx` and thus `row_offset = req_idx * block_table_stride`. This is O(log num_reqs) integer ops per lane, cheap versus the memory work.
4. All existing arithmetic (virtual_block_indices, CP_KV_CACHE_INTERLEAVE_SIZE masking, `is_local`, `PAD_ID` semantics) is preserved verbatim — only the mapping from `pid` to `(req_idx, token_within_req)` changes.
5. Choose `TILE` as a `tl.constexpr` autotuned across {32, 64, 128, 256}; for the agentic decode case (1 token/req, hundreds of reqs), a small TILE (32/64) fully utilizes lanes across requests instead of wasting 1023/1024 lanes per program.

This converts wasted-lane-per-program into wasted-lane-per-tail-of-batch (one partial tile at most), collapses the +1 padder program into the main grid, and turns per-request serial iteration into massively parallel token-parallel execution — directly attacking the steady-state TPOT cost that this kernel adds on every scheduler step for every KV cache group. Correctness is validated by the existing attention-backend and slot-mapping tests, which assert PAD_ID for out-of-range slots and correct KV addresses for in-range slots.

**Novelty rationale.**

The candidate has no existing deep_research_proposals, so any concrete change is novel by construction. The specific proposal — restructuring the grid from request-parallel to token-tile-parallel with an in-kernel binary search over query_start_loc and fused tail padding — is a distinct optimization axis from what evolve_rationale merely hints at (adaptive BLOCK_SIZE or naive request-by-token retiling). Rather than adapting the block size per-request or introducing multi-dimensional request×token programs (which multiply grid dimensions and complicate the padder), it flattens execution to a single 1D token-parallel grid so decode-heavy agentic batches (one token per req, many reqs) achieve near-100% lane utilization instead of ~0.1%, and eliminates the dedicated padder program entirely.

---

### 2. Add a one-token decode fast path without req lookup
- **Agent:** codex

**Detailed description.**

Add a specialized Triton path for the common agentic decode case where every active request schedules exactly one token. Extend `BlockTable.compute_slot_mapping` in `vllm/v1/worker/block_table.py` to accept or derive an `is_decode_one_token_per_req` flag from the caller's `num_scheduled_tokens`, and when true launch a new `_compute_decode_slot_mapping_kernel` instead of `_compute_slot_mapping_kernel`. The decode kernel should use a compact request-tile grid, e.g. `BLOCK_REQS=128`, with `req_offsets = pid * BLOCK_REQS + tl.arange(0, BLOCK_REQS)`. For valid `req_offsets < num_reqs`, the global token offset is identical to the request index, so it can load `positions_ptr + req_offsets`, compute the same `virtual_block_indices`, `is_local`, CP interleave handling, `block_indices`, `block_numbers`, and `slot_ids` as the existing kernel, then store to `slot_mapping_ptr + req_offsets`. For lanes beyond `num_reqs` but below the padded CUDA graph token extent, store `PAD_ID`, so decode mapping and graph-tail padding are handled by the same tiled launch. Keep the existing `_compute_slot_mapping_kernel` as the fallback for prefill, chunked prefill, speculative batches, and any mixed request lengths. Validate with a focused slot-mapping test that covers `num_scheduled_tokens == 1` for many requests, CP interleave/local-rank masking, and padding from `num_reqs` to the padded token capacity.

**Novelty rationale.**

There are no listed deep_research_proposals. This is not the same as Claude's generic token-tile grid with per-lane binary search over `query_start_loc`: it deliberately targets only the dominant one-token decode shape, avoids binary search and `query_start_loc` loads entirely, preserves the current request-to-row relationship directly, and leaves the legacy general kernel in place for non-decode or mixed-length batches.

---
