# PriorityRequestQueue.remove_request / remove_requests

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/sched/request_queue.py`](vllm/v1/core/sched/request_queue.py) (lines 175–184)
- **Symbol:** `PriorityRequestQueue.remove_request / remove_requests`
- **Kind:** method
- **Estimated impact:** low
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0012`

## Description
Removes one or more requests from the heap-backed priority queue by linear search or full list rebuild followed by heapify.

## Current approach
remove_request uses list.remove then heapq.heapify, and remove_requests rebuilds the heap list excluding removed requests then heapifies. There is no lazy deletion or indexed heap state.

## Estimated impact explanation
This affects TPOT only for PRIORITY policy and large removal batches, so the signal is workload-dependent. The asymptotic improvement is real but not on the common FCFS decode path.

## Evolve rationale
The code construct is the removal path for PRIORITY scheduling. It is exercised by finish/abort handling and stopped preempted requests. At high agentic concurrency, O(n) heap rebuilds can dominate completion batches. Headroom includes tombstone-based lazy deletion with skip-on-pop, an indexed heap, or a heap plus request-id map while preserving pop/peek ordering. Correctness oracles include tests/v1/core/test_scheduler.py and tests/v1/core/test_priority_scheduler_random.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Lazy tombstone deletion with skip-on-pop for PriorityRequestQueue
- **Agent:** claude

**Detailed description.**

Replace the O(n) `list.remove + heapq.heapify` paths in `PriorityRequestQueue.remove_request` and `remove_requests` (vllm/v1/core/sched/request_queue.py:175-184) with O(1) tombstone marking, deferring actual eviction until pop/peek. Concretely:

1. Add a `self._tombstones: set[str] = set()` (keyed by `Request.request_id`) and a `self._live_count: int = 0` counter alongside `self._heap`.
2. `add_request` increments `_live_count`; if the request_id is in `_tombstones` (rare resurrection case after abort+resubmit), discard the tombstone.
3. `remove_request(req)` becomes `self._tombstones.add(req.request_id); self._live_count -= 1` — O(1), no heapify.
4. `remove_requests(reqs)` updates the tombstone set in bulk and decrements `_live_count` accordingly — O(k) instead of O(n).
5. `pop_request` loops `heapq.heappop` while the popped request's id is in `_tombstones`, discarding tombstoned ids from the set as they are skipped (so the set never grows unboundedly).
6. `peek_request` performs the same skip loop, popping tombstoned heads off the heap so `self._heap[0]` is always a live request.
7. `__len__` returns `_live_count`; `__bool__` returns `_live_count > 0`; `__iter__` filters tombstoned ids.
8. Add an opportunistic compaction trigger: when `len(_tombstones) > max(32, len(_heap) // 2)`, do a single rebuild + heapify to bound worst-case memory, preserving the current behavior as a fallback.

This preserves the `(priority, arrival_time)` ordering already implemented via `Request.__lt__`, requires no changes to `Request`, and keeps the public `RequestQueue` ABC intact. Correctness is covered by `tests/v1/core/test_scheduler.py` and `tests/v1/core/test_priority_scheduler_random.py`; add a targeted test that interleaves `add_request`, `remove_request(s)`, and `pop_request` on a heap of >1k entries to exercise both the skip-on-pop path and the compaction threshold.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete design is novel relative to that list. The candidate's evolve_rationale only enumerates direction names ("tombstone-based lazy deletion", "indexed heap", "heap plus id map") without specifying the API surface, the resurrection-safe id-keyed tombstone semantics, the `_live_count` separation from `len(_heap)`, the peek-time eviction needed to keep `peek_request` correct, or the bounded-memory compaction threshold — all of which are required for a correct, drop-in implementation and are spelled out here.

---

### 2. Use an indexed heap for eager O(log n) priority removals
- **Agent:** codex

**Detailed description.**

Replace `PriorityRequestQueue`'s linear removal paths in `vllm/v1/core/sched/request_queue.py:175-184` with an eager indexed heap. Add `self._heap_index: dict[str, int]` keyed by `Request.request_id`, update it in `add_request` and `pop_request`, and implement small local heap helpers that swap entries while maintaining the index map. `remove_request(request)` can then look up the request id, swap the target with the last heap item, pop it, and restore the heap with one sift-up or sift-down pass, giving O(log n) removal without leaving stale entries in the heap. For `remove_requests(requests)`, collect request ids and choose between repeated indexed deletes for small batches and the existing rebuild+heapify approach for large batches, rebuilding `_heap_index` after heapify. Add focused tests around interleaved add/remove/pop/peek operations and a randomized invariant check that every request id maps to the correct heap position after removals.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposed lazy tombstones with live counters, skip-on-pop/peek behavior, and compaction. This proposal is intentionally different: it keeps the heap physically free of removed requests immediately, adds a request-id-to-index map, and restores heap order through eager indexed deletion, avoiding tombstone state and skip paths entirely.

---
