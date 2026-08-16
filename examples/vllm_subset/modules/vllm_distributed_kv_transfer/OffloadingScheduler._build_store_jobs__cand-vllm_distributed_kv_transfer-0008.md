# OffloadingScheduler._build_store_jobs

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`](vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) (lines 1248–1441)
- **Symbol:** `OffloadingScheduler._build_store_jobs`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_distributed_kv_transfer-0008`

## Description
Builds store transfer jobs each scheduler step by scanning scheduled and finished requests, computing offloadable tokens, filtering reachable chunks, calling manager.prepare_store, and packing GPU source block metadata.

## Current approach
Nested Python loops over requests, KV groups, chunks, and blocks perform list slicing, reachability checks, set membership tests, event recording, and GPULoadStoreSpec packing. prepare_store is called per request rather than batched across the scheduler step.

## Estimated impact explanation
Adds CPU time to every scheduler step and therefore TPOT; with many concurrent agentic sessions, store bookkeeping can also delay scheduling of TTFT-critical loads.

## Evolve rationale
Store-job construction cost scales with active_requests * groups * chunks_per_request. Vectorized reachability masks, grouped prepare_store calls, or alignment-profile batching are contained changes preserving produced store_jobs, fences, and completion accounting. Oracle: offloading scheduler and worker unit tests verify produced store_jobs, fences, completion accounting, and subsequent load byte-equivalence.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Cross-request batched prepare_store with single-pass reachability + block-list build
- **Agent:** claude

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` `_build_store_jobs` (lines 1248-1441), the method currently walks the (requests × KV groups × chunks × blocks_per_chunk) space TWICE per scheduler step: once to compute `new_offload_keys` filtered by `is_store_reachable_swa_chunk` and non-zero `block_id`, and again after `prepare_store` returns to reconstruct `src_block_ids`, `group_sizes`, `block_indices`, `fenced_block_ids`, and `deferred_fence_block_ids`. It also calls `manager.prepare_store` once per request, which for multi-turn agentic workloads with many concurrent sessions serializes allocator work and inflates scheduler-step CPU time (directly hurting TPOT and delaying TTFT-critical loads scheduled in the same step).

Concrete change: restructure `_build_store_jobs` so that (1) the inner per-request loop builds, in ONE pass, a lightweight per-group `PendingStoreDraft` dataclass carrying `offload_keys: list[OffloadKey]`, `chunk_indices: list[int]` (the absolute `chunk_idx` for each surviving key), and the flattened `src_block_ids_by_chunk: list[list[int]]` — already filtered by `block_id != 0` and `is_store_reachable_swa_chunk`. Populate `is_sliding_window` once per group_config at scheduler init and cache it on `KVGroupConfig` (avoiding the redundant test inside the hot loop). (2) Aggregate the surviving keys from every request in the step into a single flat `all_keys` list plus a parallel `owner_index` array pointing back to `(req_status, draft_index)`. Call `self.manager.prepare_store(all_keys, batched_contexts)` ONCE with a new `prepare_store_batched(keys_by_req, contexts_by_req)` entry point (adding a thin adapter on `OffloadingManager` that fans internally to the existing allocator, giving the allocator visibility into the full step's demand for better packing and rejecting-oldest-first policy). (3) Consume the returned `keys_to_store` as a `set` and, using the pre-materialised `src_block_ids_by_chunk`/`chunk_indices` from the drafts, emit `TransferJob`s with a single `list.extend`/`itertools.chain` per group — no second scan of `group_state.offload_keys` or `group_state.block_ids`. `record_store`/`record_partial_store` are still called in that final pass since they need the actual accepted keys, but from the pre-computed `chunk_indices` list rather than by re-zipping.

Additionally cache the slice `group_state.offload_keys[start_chunk_idx:num_chunks]` and the strided `block_ids[start*bpc + bpc-1 : num*bpc : bpc]` computations in local vars so they are computed once rather than in both scan passes, and replace the `for i in range(blocks_per_chunk): block_id = block_ids[gpu_block_idx + i]` inner loop with a slice `block_ids[gpu_block_idx : gpu_block_idx + blocks_per_chunk]` plus a comprehension — measurably faster in CPython for the typical `blocks_per_chunk≤8` case. Preserve produced `store_jobs`, fences (both immediate `fenced_block_ids` and `deferred_fence_block_ids`), `_current_batch_jobs_to_flush` registration on finished requests, `advance_stored_idx` calls, and `_block_id_to_pending_jobs` updates — covered by the existing offloading scheduler/worker unit tests referenced in `evolve_rationale`.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so nothing overlaps by construction. The proposal is also distinct from the generic "vectorize with numpy" direction hinted at in `evolve_rationale`: it (a) eliminates the double scan by materialising per-chunk draft state on the first pass, (b) introduces a cross-request batched `prepare_store` entry point on the manager (allocator sees the full step's demand rather than one request at a time), and (c) targets specific CPython micro-inefficiencies (per-block index loop, redundant slice recomputation, per-iteration `is_sliding_window` re-derivation) that are contained edits preserving byte-identical stored KV and the fence/flush accounting invariants.

---

### 2. Defer active-request stores on load-heavy scheduler steps
- **Agent:** codex

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` `_build_store_jobs`, add a load-priority fast path that skips normal store-job construction for non-finished requests when the current scheduler step already has TTFT-critical work, e.g. `self._current_batch_load_jobs` is non-empty or `self._current_batch_jobs_to_flush` is non-empty. Keep processing `scheduler_output.finished_req_ids` in the same step so finished-request offloads and cleanup signaling are preserved, but for still-running requests leave `group_state.next_stored_chunk_idx` unchanged and do not call `manager.prepare_store`; those chunks will be reconsidered on the next step without pending loads. This can be guarded by an offloading config flag or a small helper such as `_should_defer_active_stores()` to keep policy isolated. Add scheduler tests covering: active scheduled request plus pending load does not call `prepare_store` and does not advance stored cursors; finished request still emits store jobs under the same condition; and a later no-load step resumes normal store construction for the deferred active request.

**Novelty rationale.**

There are no deep_research_proposals listed for this candidate. Agent A's proposal optimizes the existing eager store-building path by batching `prepare_store`, eliminating duplicate scans, and tightening inner-loop CPU costs. This proposal is a distinct scheduling-policy change: avoid entering the expensive active-request store path at all on load-heavy steps to protect TTFT, while preserving finished-request handling and eventual store behavior.

---
