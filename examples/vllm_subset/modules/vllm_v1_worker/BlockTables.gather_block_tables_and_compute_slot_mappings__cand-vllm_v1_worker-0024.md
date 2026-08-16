# BlockTables.gather_block_tables and compute_slot_mappings

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/block_table.py`](vllm/v1/worker/gpu/block_table.py) (lines 136–323)
- **Symbol:** `BlockTables.gather_block_tables and compute_slot_mappings`
- **Kind:** region
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0024`

## Description
New GPU runner block-table gather and slot-mapping preparation for attention metadata and KV cache writes.

## Current approach
Launches a gather kernel over num_kv_cache_groups x num_reqs_padded, then launches a slot-mapping kernel over num_groups x (num_reqs + 1), both with fixed 1024-element tiles and a dedicated padding program.

## Estimated impact explanation
Block-table gather and slot mapping run every step in the new runner; decode-heavy agent batches with many one-token requests waste tile lanes, so adaptive tiling/fusion reduces median TPOT.

## Evolve rationale
The fixed TRITON_BLOCK_SIZE=1024 launch policy and per-request grid are concrete kernel-scheduling targets. tests/v1/worker/test_gpu_block_table.py and attention backend tests validate block-table gather, staged writes, padding, and slot-mapping invariants.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Add a decode-fused path for compute_slot_mappings and gather_block_tables that maps one request per lane instead of one request per program
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu/block_table.py:136-323, add a specialised fast path used when the batch is decode-dominant (detectable at launch time as num_tokens == num_reqs, or equivalently when query_start_loc[i+1] - query_start_loc[i] == 1 for all i, which is the common shape for multi-turn agentic decode steps). In that case, replace the current `(num_groups, num_reqs + 1)` launch of `_compute_slot_mappings_kernel` — where each program computes a single request's slot mapping using a 1024-lane Triton tile and wastes ~1023/1024 lanes — with a new `_compute_slot_mappings_decode_kernel` launched as `(num_groups, cdiv(num_reqs, TILE))`. Each program processes TILE requests: one lane per request loads `req_state_idx = idx_mapping[batch_off + lane]`, `pos = positions[batch_off + lane]` (since `query_start_loc[i] == i` in the decode case, positions are contiguous and directly indexable), computes `block_indices = pos // block_size`, gathers `block_numbers = block_table[req_state_idx, block_indices]` with a single coalesced 32-bit load per lane, and stores `slot_id = block_numbers * block_size + pos % block_size` at `slot_mappings[group, batch_off + lane]`. Tail padding (the current `batch_idx == num_programs(1) - 1` branch) becomes a single tl.store with `mask = (batch_off + lane) < num_reqs` inside the same kernel, eliminating the dedicated padding program. Add a symmetric `_gather_block_tables_decode_kernel` that packs multiple requests' block-table rows into one program by tiling in two dimensions `(TILE_REQ, TILE_BLK)` with TILE_REQ*TILE_BLK == 1024, so short block-tables (typical for agent turns that have not yet grown a long KV) coalesce many requests into a single warp instead of leaving lanes idle; the padding-row zeroing branch also fuses into the same launch via mask. Keep the existing kernels as the fallback for mixed prefill/decode batches so semantics (chunked prefill, CP, PAD_SLOT_ID handling) are unchanged. Wire in a per-step feature flag (env or engine config) so tests/v1/worker/test_gpu_block_table.py can exercise both paths, and extend that test with a decode-only batch of many single-token requests to lock in equivalence. Expected: for a decode step of R requests, `compute_slot_mappings` goes from R×1024-lane launches doing 1 useful lane each to cdiv(R,1024) launches with full lane utilisation — a ~1024x drop in wasted SIMT work and one launch instead of R+1, directly reducing per-step overhead in the two hottest metadata paths of the new GPU runner.

**Novelty rationale.**

The candidate has no existing deep_research_proposals, so any concrete idea is novel. The specific mechanism proposed — switching the launch geometry from one-program-per-request with a 1024-lane tile to one-lane-per-request with a packed grid, gated by a runtime decode-shape check on query_start_loc — is not implied by the candidate description (which only names the fixed 1024 tile and per-request grid as tuning targets, not a fused decode geometry). It is also distinct from generic tile-size autotuning or kernel fusion: it changes the mapping between logical work items and Triton programs, which is what actually recovers the wasted lanes in decode-heavy agentic batches.

---

### 2. Bound slot-mapping padding to the returned padded token count
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu/block_table.py:178-210`, stop passing `slot_mappings.shape[1]` as the padding limit to `_compute_slot_mappings_kernel`; pass the method argument `num_tokens_padded` instead, and use that value for both the padding-program loop bound and the returned slice. The current kernel pads from `actual_num_tokens` through the full persistent `max_num_batched_tokens` capacity every step, even though callers receive only `slot_mappings[:, :num_tokens_padded]`. For decode-heavy agent batches, this can turn the dedicated padding program into a large fixed-cost memset over stale capacity that is not visible to attention metadata. Add a CUDA test in `tests/v1/worker/test_gpu_block_table.py` that seeds `slot_mappings` with non-`PAD_SLOT_ID` values, calls `compute_slot_mappings(..., num_tokens_padded=N)` where `N` is much smaller than `max_num_batched_tokens`, and asserts that `[actual_num_tokens:N]` is padded while valid token slots remain equivalent to the existing reference behavior. For CUDA graph safety, keep padding through the graph-padded token count, not only the actual token count.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposed a decode-only remapping/fusion strategy that changes Triton launch geometry and packs requests into lanes for both slot mapping and block-table gather. This proposal is different: it preserves the existing per-request slot-mapping kernel and focuses specifically on eliminating over-padding to the persistent tensor capacity by using the already supplied `num_tokens_padded` bound. It targets the dedicated padding program's unnecessary memory traffic rather than decode lane utilization or gather fusion.

---
