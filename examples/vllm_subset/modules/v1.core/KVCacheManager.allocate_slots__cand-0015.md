# KVCacheManager.allocate_slots

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/kv_cache_manager.py`](vllm/v1/core/kv_cache_manager.py) (lines 225–416)
- **Symbol:** `KVCacheManager.allocate_slots`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0015`

## Description
Central KV-slot allocation method that validates requested work, computes local/external computed-token ranges, optionally applies full-sequence admission gating, removes skipped blocks, checks required free blocks, attaches computed blocks, allocates new blocks, and caches finalized blocks.

## Current approach
Runs multiple coordinator passes per call: optional full_sequence_must_fit get_num_blocks_to_allocate, remove_skipped_blocks, a second get_num_blocks_to_allocate, allocate_new_computed_blocks, allocate_new_blocks, and cache_blocks. Free-space checks are conservative and all policy is evaluated synchronously in Python.

## Estimated impact explanation
This affects TTFT during admission and TPOT when decode/lookahead allocation occurs. It is medium because much cost is delegated to coordinator and block-pool methods, but reducing repeated Python allocation-policy work can lower scheduler step latency.

## Evolve rationale
The concrete optimization unit is allocate_slots' staged allocation policy and repeated coordinator calls. Both RUNNING and WAITING scheduling paths call it, often inside retry/preemption loops. Headroom includes combining capacity prediction passes, memoizing per-request block requirements within a step, specializing common no-external/no-encoder/no-full-fit cases, and making admission caps less conservative without breaking sliding-window safety. Correctness oracles include tests/v1/core/test_prefix_caching.py, tests/v1/core/test_single_type_kv_cache_manager.py, and tests/v1/core/test_scheduler.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Short-circuit cache_blocks when no new block boundary is crossed
- **Agent:** claude

**Detailed description.**

At the tail of `KVCacheManager.allocate_slots` (vllm/v1/core/kv_cache_manager.py:410-414), `self.coordinator.cache_blocks(request, num_tokens_to_cache)` is invoked unconditionally on every call when caching is enabled and `delay_cache_blocks` is False. For long-decode multi-turn agentic workloads, allocate_slots is invoked once per output token, but `num_tokens_to_cache // block_size` only advances every `block_size` tokens; the other ~`(block_size-1)/block_size` of calls fan out into the coordinator and per-group `single_type_kv_cache_manager.cache_blocks` only to be early-returned by the `num_cached_blocks >= num_full_blocks` check at single_type_kv_cache_manager.py:289. That dispatch (one Python call per kv-cache group on hybrid models, plus dict lookups) is pure overhead on the TPOT critical path.

Proposed change, scoped to KVCacheManager.allocate_slots:

1. Add a `self._last_cached_full_blocks: dict[str, int]` map on KVCacheManager, populated lazily.
2. Expose `block_size` once at __init__ (the manager already constructs the coordinator and knows the kv_cache_config).
3. In allocate_slots, replace the unconditional cache_blocks call with:
   - `num_full_blocks_to_cache = num_tokens_to_cache // self.block_size`
   - `prev = self._last_cached_full_blocks.get(request.request_id, -1)`
   - call `self.coordinator.cache_blocks(...)` only when `num_full_blocks_to_cache > prev`, then store `prev = num_full_blocks_to_cache`.
4. Clear the entry inside `KVCacheManager.free()` (vllm/v1/core/kv_cache_manager.py:418) so request_id reuse stays correct.
5. Also reset/back-off the entry on preemption paths that call `coordinator.free` for a request still in flight, to avoid stale max values after a reset.

For block_size=16 and a typical decode tail this skips ~94% of cache_blocks dispatches with no behavioral change: the slow path runs identically the moment a full block becomes cacheable, and underlying correctness is still defended by the existing in-manager check. Validation: `tests/v1/core/test_prefix_caching.py`, `tests/v1/core/test_single_type_kv_cache_manager.py`, and `tests/v1/core/test_scheduler.py` (named in evolve_rationale) all exercise both boundary-crossing and non-crossing decode steps and should pass unchanged; add a microbench around scheduler.step for a long-decode request to confirm reduced Python time.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's evolve_rationale enumerates four headroom items (combining capacity prediction passes, memoizing per-request block requirements within a step, specializing common no-external/no-encoder/no-full-fit cases, and loosening admission caps) — all of which target the *allocation* side of allocate_slots (get_num_blocks_to_allocate, allocate_new_*, full_sequence_must_fit). None of them address the unconditional `coordinator.cache_blocks` dispatch at the tail of the method, which is the dominant per-call Python cost on the long-decode TPOT path where allocation work is essentially zero. The proposal therefore targets a distinct overhead surface (caching bookkeeping) that the listed rationale does not cover.

---

### 2. Skip allocate_new_blocks on zero-allocation decode steps
- **Agent:** codex

**Detailed description.**

In `KVCacheManager.allocate_slots` (`vllm/v1/core/kv_cache_manager.py:366-400`), use the already-computed `num_blocks_to_allocate` result to avoid the final `self.coordinator.allocate_new_blocks(...)` pass when it is known to be a no-op. For the common RUNNING decode path with no new prefix-cache hits and no external computed tokens, `num_blocks_to_allocate == 0` means every per-group `allocate_new_blocks` call will only recompute required block counts, look up `req_to_blocks`, and return `[]`. Add a narrow fast path after the capacity check and after preserving the existing `allocate_new_computed_blocks` side effects: if `num_blocks_to_allocate == 0` and `new_computed_block_list` is the manager's empty block tuple and `num_external_computed_tokens == 0`, set the result to `self.empty_kv_cache_blocks` instead of dispatching through the coordinator. Keep the normal path unchanged for prefix hits, external KV, encoder allocations, and actual block-boundary growth. This should skip roughly `block_size - 1` out of every `block_size` `allocate_new_blocks` coordinator passes during long decode tails, reducing TPOT Python overhead without changing allocation state. Validate with the existing scheduler and prefix-cache tests plus a small regression test that monkeypatches or spies on `coordinator.allocate_new_blocks` for an in-block decode step and confirms no call is made while the returned block IDs remain `None`.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A targets the tail `cache_blocks` dispatch when no new full cache block can be committed; this proposal targets a different coordinator pass, `allocate_new_blocks`, and uses the capacity predictor's zero-allocation result to skip a no-op allocation commit pass. It does not change caching behavior or duplicate Agent A's `_last_cached_full_blocks` idea.

---
