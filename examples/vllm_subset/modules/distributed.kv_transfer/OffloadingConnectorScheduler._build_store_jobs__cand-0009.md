# OffloadingConnectorScheduler._build_store_jobs

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`](vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) (lines 595–763)
- **Symbol:** `OffloadingConnectorScheduler._build_store_jobs`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0009`

## Description
Per-scheduler-step construction of offload store jobs from scheduled request data, KV groups, offload keys, and GPU block IDs.

## Current approach
The method iterates in Python over scheduled requests, groups, offloaded blocks, and GPU blocks; materializes flattened block lists and slices; checks pending-job fences; calls `manager.prepare_store`; builds source block lists, group sizes, block indices, sliding-window/non-sliding-window tracking lists, and `TransferJob` records.

## Estimated impact explanation
The method runs every scheduler iteration when offloading is enabled. Lower Python overhead reduces scheduler-step latency, improving TTFT for prefill bursts and median TPOT when stores are built concurrently with ongoing decode.

## Evolve rationale
This is a hot Python batching and mask-building path with a fixed metadata output contract. It can be evolved by precomputing masks, avoiding repeated slices, batching group work, or moving the inner block scan to a native/vectorized helper. Correctness oracle: for a fixed `SchedulerOutput` and mocked manager, produced `OffloadingConnectorMetadata.store_jobs`, `jobs_to_flush`, `TransferJobStatus`, and `_block_id_to_pending_jobs` must match current semantics; offloading scheduler tests include store-job fencing and metadata invariants.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Collapse the two-pass key/block scan into a single precomputed records list to halve Python overhead per scheduler step
- **Agent:** claude

**Detailed description.**

In `OffloadingConnectorScheduler._build_store_jobs` (vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:595-763), the method currently performs two near-identical scans per request: the first (lines 633-660) iterates `kv_group_configs`/`group_states`, slices `offload_keys[start:num_blocks]` and `block_ids[start*bsf+bsf-1:num_blocks*bsf:bsf]` to assemble `new_offload_keys` for `manager.prepare_store(...)`; the second (lines 685-721) re-iterates the same groups, re-slices `offload_keys[start_block_idx:num_blocks]`, and walks `block_ids[gpu_block_idx + i]` for every `i in range(block_size_factor)` to build `src_block_ids`, `group_sizes`, `block_indices`, and the sliding-window/non-sliding-window lists. Replace this with a single forward pass that records, per candidate offloaded block, a lightweight tuple `(group_idx, offload_key, is_sliding_window, gpu_block_idx_base, last_block_id)` into a per-request `pending_records` list while simultaneously appending to `new_offload_keys` only when `last_block_id != 0` (matching today's filter). Track per-group running counts (`num_group_blocks_so_far`, `start_gpu_block_idx_so_far`) inline so the second-pass bookkeeping is already accumulated. After `manager.prepare_store(...)` returns `keys_to_store`, emit `src_block_ids`/sliding-window lists by iterating `pending_records` exactly once, indexing `block_ids` via the precomputed `gpu_block_idx_base + i` (avoiding the inner `range(block_size_factor)` Python loop by unrolling for the common `block_size_factor in (1, 2, 4)` cases or using a precomputed slice `block_ids[gpu_block_idx_base:gpu_block_idx_base + block_size_factor]`). Additionally, hoist `block_size_factor`, `offloaded_block_size`, and `is_sliding_window` per group into local variables outside the inner block loop, and replace `keys_to_store = set(store_output.keys_to_store)` with a frozenset only when `len(store_output.keys_to_store) >= some_threshold` else use the list directly via membership in a pre-built `dict` keyed by id (since `OffloadKey` hashing dominates membership checks). The fence check on lines 614-625 is also redundantly re-flattens `new_block_id_groups`; precompute `new_blocks_flat` once and reuse for both `isdisjoint` and the subsequent generator. Net result: one forward scan over groups/blocks per request instead of two, eliminated re-slicing of `group_state.offload_keys`, and removal of the per-block `range(block_size_factor)` Python loop. Correctness is preserved by construction: the `pending_records` ordering matches today's group-then-offset ordering, and the `keys_to_store` filter is applied identically. Validate with the existing offloading scheduler tests (store-job fencing, metadata invariants, sliding-window vs non-sliding-window block_id assignment) and microbenchmark the scheduler step under a multi-turn agentic prefill burst (many small new_block_id_groups per step) to confirm the reduction in scheduler-step latency translates to lower median TTFT.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate (the list is empty), so by definition this proposal does not overlap any prior proposal. Beyond that, this proposal is structurally specific: it identifies that the two passes share the same `(group_idx, offloaded_block_idx)` enumeration and proposes a concrete `pending_records` intermediate (with named fields and exact filter semantics) to fuse them, rather than the broader evolve_rationale hints (masks, vectorization, batching) that describe directional ideas without a concrete fused-pass algorithm or the inner `range(block_size_factor)` unrolling tactic.

---

### 2. Add a per-request offload-block watermark to skip empty store scans
- **Agent:** codex

**Detailed description.**

In `OffloadingConnectorScheduler._build_store_jobs` (vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:595-763), add a cached `next_store_token_threshold` on `RequestOffloadState` equal to the minimum `(group_state.next_stored_block_idx + 1) * group_config.offloaded_block_size` across KV groups. After handling preemption, block-id updates, and the pending-store fence for `new_block_id_groups`, compute `num_offloadable_tokens` and immediately `continue` when it is below this threshold, before calling `req_status.update_offload_keys()` and before allocating/scanning `new_offload_keys`. Refresh the threshold in the same places that currently mutate `next_stored_block_idx`: `RequestOffloadState.advance_stored_idx`, the successful store-job path after setting each group to `num_blocks`, and `update_state_after_alloc` when loaded prefixes are marked as already stored. This preserves correctness because the threshold is exactly the negation of “any group has `num_blocks > next_stored_block_idx`”; skipped iterations are cases where the existing code would only discover that no complete offloaded block is available and leave state unchanged. Keep the new-block fence before the fast skip so pending stores are still flushed when the allocator reuses a block. Validate with existing offloading scheduler tests plus a focused test that several decode steps below the threshold produce no `prepare_store` call, then the boundary step produces identical `store_jobs`, `jobs_to_flush`, and pending-job index state to the current implementation.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A’s proposal optimizes the non-empty store path by fusing the two group/block scans and reusing per-block records once candidate offload blocks exist. This proposal targets a different hot case: scheduler iterations where no full offloaded block can be stored yet, especially decode steps between offload-block boundaries. It introduces a request-level watermark and moves key generation/scanning behind that guard, rather than changing the two-pass construction algorithm.

---
