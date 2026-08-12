# OffloadingScheduler._lookup_complete_chunks

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`](vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) (lines 697–868)
- **Symbol:** `OffloadingScheduler._lookup_complete_chunks`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_distributed_kv_transfer-0007`

## Description
Core offloading prefix-hit lookup iterates KV groups, performs full-attention prefix and sliding-window suffix backend lookups, and may re-run groups when constraints tighten max_hit_size_tokens.

## Current approach
A Python convergence while loop repeatedly calls _maximal_prefix_lookup or _sliding_window_lookup one group at a time. Sliding-window and EAGLE constraints can force another pass over groups and another series of backend probes; eagle_verified tracks per-iteration state.

## Estimated impact explanation
Scheduler lookup latency is on the TTFT path for every offloaded prefix hit; long shared prefixes in multi-turn agentic prompts repeatedly exercise it.

## Evolve rationale
Every cache-eligible request calls this before scheduling can allocate external tokens. Batched backend lookup, memoized lookup slices across retries, or a one-pass group constraint solver would reduce backend calls while preserving the returned num_hit_tokens contract. Oracle: tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py asserts exact hit tokens, deferral behavior, SWA/EAGLE clipping, and load correctness.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Per-scheduler-step OffloadKey lookup cache shared across requests
- **Agent:** claude

**Detailed description.**

In OffloadingScheduler._lookup_complete_chunks (scheduler.py:697-868) and its helpers _maximal_prefix_lookup / _sliding_window_lookup (scheduler.py:599-664), every call to self.manager.lookup(key, req_context) goes to the backend one key at a time, and the outer convergence while-loop plus SWA/EAGLE retries can probe the same keys multiple times within a single request. In multi-turn agentic workloads many concurrent requests share long system-prompt / tool-schema prefixes, so the same OffloadKey is looked up over and over within the same scheduling step across sibling requests. Introduce a short-lived cache dict[OffloadKey, LookupResult] created at the start of Scheduler.get_num_new_matched_tokens (the public entry that drives _lookup) and passed down to _lookup_complete_chunks / _maximal_prefix_lookup / _sliding_window_lookup. Wrap manager.lookup(key, req_context) in a helper that consults the cache first; on miss, call the backend and store the result. Only cache the terminal HIT and MISS outcomes freely; HIT_PENDING and RETRY should be cached only for the remainder of that request's convergence loop (or invalidated the moment defer_lookup causes lookup_groups to be reset) so async manager state advances between scheduler steps still reach the backend. Events tracking (self._events_tracker.record_lookup) must fire on the first observation only, so gate it on a 'seen' set inside the cache to preserve the exact hit-token contract that tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py asserts. This turns O(convergence_iters * num_concurrent_prefix_shared_requests * num_groups * num_chunks) lookups into O(unique_keys) per step, which directly cuts TTFT on the shared-prefix path.

**Novelty rationale.**

The candidate lists no deep_research_proposals. The candidate's own evolve_rationale mentions 'batched backend lookup, memoized lookup slices across retries, or a one-pass group constraint solver' — all scoped to a single request's internal iteration. This proposal is different: it deduplicates lookups across sibling requests within the same scheduler step (a cross-request cache in Scheduler.get_num_new_matched_tokens), which is the dominant source of redundant probes for the stated multi-turn agentic workload with shared system prompts, and it also naturally covers the intra-request retry-memoization case as a special case.

---

### 2. Probe most-restrictive KV groups first in prefix-hit convergence
- **Agent:** codex

**Detailed description.**

Change OffloadingScheduler._lookup_complete_chunks in vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:697-868 so each convergence pass orders lookup_groups by the maximum chunk span they can currently validate, rather than always using self._lookup_groups order with full-attention groups first. Before calling _maximal_prefix_lookup or _sliding_window_lookup, compute each group's candidate end from max_hit_size_tokens, len(offload_keys), tokens_per_chunk, num_computed_tokens, sliding_window_size_in_chunks, and the EAGLE extra-chunk requirement. Probe the group with the smallest candidate span first, apply its max_hit_size_tokens tightening, then skip or shorten later group probes that would have scanned beyond the new boundary. Keep the existing retry/defer semantics and eagle_verified reset rules; this is only an ordering/bounding change, not a cache. Add or extend scheduler unit tests where a restrictive SWA/EAGLE group caps the hit before a full-attention group, asserting identical returned hit tokens and fewer manager.lookup calls via the existing fake manager/oracle style in tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposed a per-scheduler-step OffloadKey lookup cache that deduplicates identical backend calls across requests and retries. This proposal does not cache lookup results; it reduces unnecessary probes by changing the intra-request group ordering and tightening bounds before expensive prefix scans, so it remains useful even when every key is unique and no cache hit is possible.

---
