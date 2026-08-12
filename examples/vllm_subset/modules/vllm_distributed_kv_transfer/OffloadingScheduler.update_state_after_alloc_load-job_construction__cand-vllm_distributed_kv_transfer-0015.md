# OffloadingScheduler.update_state_after_alloc load-job construction

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`](vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) (lines 993–1101)
- **Symbol:** `OffloadingScheduler.update_state_after_alloc load-job construction`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_distributed_kv_transfer-0015`

## Description
After external-token allocation, builds offloading load jobs by scanning groups and allocated blocks, collecting keys_to_load and destination block IDs, and registering TransferJob/TransferJobStatus entries.

## Current approach
Python loops over KV groups and group blocks repeatedly slice block lists, extend key and block arrays, update allocated-block sets, and call manager.prepare_load once for the request. Load job construction is per request rather than batched across the scheduler step.

## Estimated impact explanation
Every offloaded hit that will be loaded asynchronously passes through this before it can resume; reducing this path moves TTFT for multi-turn requests with cache hits.

## Evolve rationale
For prefix hits, this runs between scheduler allocation and worker load submission, so its CPU overhead is on the TTFT path. Batch construction across requests, vectorized block/null detection, or precomputed group chunk boundaries can reduce work while preserving the TransferJob contract. Oracle: tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py asserts load_jobs, transfer_jobs, chunk accounting, and partial-tail behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Precompute per-group null-safe flag and fuse group_blocks walks into one pass
- **Agent:** claude

**Detailed description.**

In OffloadingScheduler.__init__, derive a static boolean per KV group — `null_block_free[i]` — that is True when the group config has no sliding-window (`sliding_window_size_in_chunks is None`) and its underlying KV cache spec is `FullAttentionSpec` (i.e. neither `SlidingWindowSpec`, `ChunkedLocalAttentionSpec`, nor `MambaSpec`). Store it on `self` alongside `config.kv_group_configs`.

Rewrite the per-group body of `update_state_after_alloc` at scheduler.py:1012-1076 so that a single walk over `group_blocks[:num_gpu_blocks]` produces every output the current three passes produce:

1. Drop the standalone `self._current_batch_allocated_block_ids.update(block.block_id for block in group_blocks if block.block_id != 0)` generator (line 1017-1019) and instead update that set inline from the same loop, over `group_blocks[:num_gpu_blocks]`. The tail `group_blocks[num_gpu_blocks:]` (already-computed suffix beyond `num_cached_tokens`) is unused by this method and does not need to enter the allocated-block set for load bookkeeping — verify against the oracle test in tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py and, if that tail must remain accounted for, walk it via one extra generator; either way we save the null/block_hash scan below.
2. When `null_block_free[i]` is True, unconditionally set `num_locally_computed_gpu_blocks = num_gpu_blocks` and skip the `for i, block in enumerate(group_blocks[:num_gpu_blocks])` scan (lines 1029-1032) entirely. Full-attention groups cannot produce `is_null` blocks, and for allocated GPU blocks in this code path `block_hash is None` implies the block is at the pending boundary — which for full-attention is exactly `num_gpu_blocks`. This preserves the invariant checked at 1034-1037.
3. When `null_block_free[i]` is False (sliding window / mamba / chunked local), fall back to the existing scan but fuse it with the `dst_block_ids.extend(...)` slice at 1063-1068 so both are produced from one enumeration.

Additionally, replace `set(keys_to_load)` at line 1095 with a single set built once and shared by reference: keep `keys_to_load` as the ordered list used to construct `src_spec` (order matters — `prepare_load` in cpu/manager.py preserves key/block correspondence), but immediately after `prepare_load` returns, build `keys_set = set(keys_to_load)` once and pass the *same* object as both `TransferJobStatus.keys` and to `self._chunks_being_loaded.update(keys_set)` at 1099-1100. `set.update(other_set)` takes a fast internal path versus iterating a list, and reusing the same object improves cache locality for the later `difference_update(job_status.keys)` at 1570.

Oracle: run `.venv/bin/python -m pytest tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py -v`, which asserts load_jobs, transfer_jobs, chunk accounting, and partial-tail behavior — the changes above preserve every observable field of the constructed `TransferJob`, `GPULoadStoreSpec`, and `TransferJobStatus`, so the test should pass unchanged. For TTFT impact, benchmark with `vllm bench serve` on the multi-turn agentic profile and compare cache-hit-heavy percentiles.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals. The candidate's evolve_rationale enumerates three angles — batching across requests, vectorized block/null detection, and precomputed group chunk boundaries — all of which target *how* the inner work is done. This proposal is orthogonal: it removes the null-block scan *entirely* for full-attention groups by lifting a per-KVCacheSpec structural property to a scheduler-init-time flag (not a vectorization of the scan, and not a chunk-boundary precomputation), fuses the three separate walks over `group_blocks` into one to cut Python bytecode dispatch, and eliminates the `set(keys_to_load)` allocation by sharing a single set reference across `TransferJobStatus.keys`, `_chunks_being_loaded.update`, and the eventual `difference_update` on load completion — a lifetime-of-the-job data-structure reuse that no listed angle covers.

---

### 2. Pass allocation deltas into load-job construction
- **Agent:** codex

**Detailed description.**

Change the scheduler-to-connector call path so `OffloadingScheduler.update_state_after_alloc` receives the `new_blocks` returned by `KVCacheManager.allocate_slots` (or just the per-group new physical block IDs) in addition to the full `KVCacheBlocks`. Use that allocation delta to update `self._current_batch_allocated_block_ids` instead of scanning every `group_blocks` entry at scheduler.py:1017-1019. The flush logic only needs physical block IDs allocated in the current scheduler step to detect collisions with pending store jobs; it does not need all blocks already owned by the request. Keep `blocks` for destination layout construction, but make allocated-ID tracking O(number of newly allocated blocks) rather than O(total cached blocks in the request). Add/adjust unit coverage in `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` for the existing reallocated-block flush behavior, including a prefix-hit request with a long already-cached prefix and a small load tail, to prove pending jobs are still flushed when a newly allocated block ID collides.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes fusing the existing walks over `group_blocks` and optionally limiting the allocated-block scan to `group_blocks[:num_gpu_blocks]`; this proposal removes allocated-block discovery from the full block-list scan entirely by carrying the allocation delta that the core scheduler already has at allocation time. It changes the data handed into this method so collision tracking scales with fresh allocations, not with request prefix length, and is therefore distinct from A's loop-fusion/null-scan/set-reuse optimizations.

---
