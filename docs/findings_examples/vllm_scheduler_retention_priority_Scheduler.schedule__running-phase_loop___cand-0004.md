# Scheduler.schedule (running-phase loop)

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 387–522)
- **Symbol:** `Scheduler.schedule (running-phase loop)`
- **Kind:** loop
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0004`

## Description
Running-request scheduling loop that budgets new tokens, handles encoder and Mamba constraints, allocates KV slots, and preempts requests when allocation fails.

## Current approach
Linearly scans self.running. On allocation failure, PRIORITY mode selects a victim with max(self.running, key=...), removes it via list.remove, and may also remove it from scheduled_running_reqs and pop entries from num_scheduled_tokens, req_to_new_blocks, scheduled_spec_decode_tokens, and scheduled_encoder_inputs, repeating inside the preemption retry loop.

## Estimated impact explanation
This is the primary per-step scheduling loop. Faster victim selection and rollback directly reduce scheduler latency and tail latency for memory-pressured priority workloads with many concurrent running requests.

## Evolve rationale
The preemption path can cascade under memory pressure, so max()+list.remove()+scheduled_running_reqs.remove can become O(P*n) per scheduler step. Maintaining indexed priority state for running requests plus O(1) scheduled-running membership would preserve priority semantics while avoiding repeated full-list scans and rollback searches. tests/v1/core/test_scheduler.py and tests/v1/core/test_priority_scheduler_random.py assert FCFS/priority ordering, preemption behavior, token-budget invariants, and scheduled-token accounting.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace linear victim search with a lazy max-heap and use a set for scheduled_running_reqs membership
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/sched/scheduler.py around lines 479-487, the PRIORITY preemption path repeatedly calls `max(self.running, key=lambda r: (r.priority, r.arrival_time))` followed by `self.running.remove(...)` and `scheduled_running_reqs.remove(...)` — each O(n), so cascading preemption under memory pressure is O(P*n) per scheduler step.

Concrete change, scoped to this loop and the small bookkeeping around it:

1. Maintain a long-lived `self._running_priority_heap: list[tuple[int, float, str]]` populated as `(-priority, -arrival_time, request_id)` (negated so heapq's min-heap yields the lowest-priority/oldest victim) alongside `self.running`. Push on every site that appends to `self.running` (scheduler add_request / waiting->running transitions in this file) and treat it as lazily deleted: do not remove on finish/preempt. Reset alongside `self.running` wherever it is re-initialized.

2. In the preemption branch, replace `max(...)` with a small helper that pops from the heap until the head's request_id is still present in `self.requests` and currently in `self.running` (use a `running_ids: set[str]` mirror updated where `self.running` is mutated). The first valid head is the victim — O(log n) amortized vs O(n).

3. Replace `scheduled_running_reqs` (currently a list) with either a parallel `set[str] scheduled_running_ids` for the `preempted_req in scheduled_running_reqs` test and the subsequent `.remove`, or change `scheduled_running_reqs` itself to an order-preserving dict keyed by request_id (`dict[str, Request]`) since dict iteration order is insertion-order in Python 3.7+. The second option avoids carrying two structures and turns both the membership check and removal into O(1). All downstream consumers that iterate `scheduled_running_reqs` would iterate `.values()`.

4. Keep FCFS semantics intact: only PRIORITY mode uses the heap; the FCFS branch (`self.running.pop()`) is unchanged.

Net effect: cascading preemption becomes O(P log n) for victim selection and O(P) for rollback bookkeeping, eliminating two of the three O(n) operations per preemption iteration. Existing invariants in tests/v1/core/test_scheduler.py and tests/v1/core/test_priority_scheduler_random.py (FCFS/priority order, preemption behavior, token-budget accounting) remain because (a) heap ordering exactly matches the previous `max(...)` key and (b) the rollback set/dict preserves the same membership semantics.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any concrete proposal is novel by construction. Beyond that, this proposal commits to specific data-structure choices (lazy max-heap with a running_ids set; converting scheduled_running_reqs to a dict keyed by request_id) and identifies the exact bookkeeping mutation sites that must push to the heap, which goes beyond the high-level 'indexed priority state' phrase in the candidate's evolve_rationale.

---

### 2. Fix cursor adjustment when preempting an earlier skipped running request
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/sched/scheduler.py` inside `Scheduler.schedule`, the PRIORITY preemption branch only does `req_index -= 1` when the victim is in `scheduled_running_reqs`. That is not the same as “the removed victim was before the current cursor”: an earlier running request can be skipped without being scheduled this step, for example when `num_new_tokens == 0` because of encoder budget, async/max-token checks, or Mamba alignment. If that skipped earlier request becomes the priority victim, removing it shifts the current request left while `req_index` stays unchanged; after the current request succeeds and increments the cursor, the next running request is skipped until a later scheduler step. Make victim removal index-aware: select `(preempted_idx, preempted_req)` from `enumerate(self.running)`, remove with `pop(preempted_idx)`, and decrement `req_index` whenever `preempted_idx < req_index`. Keep the scheduled rollback conditional separate, keyed only on whether the victim was actually scheduled in this step. Add a targeted priority-scheduler regression test with an earlier skipped running request, a current request that triggers preemption, and a later runnable request that must still be visited in the same call.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A’s proposal addresses data structures for faster victim selection and scheduled-running membership; it does not cover the cursor-correctness bug caused by using `scheduled_running_reqs` membership as a proxy for the victim’s position in `self.running`. This proposal is about preserving loop traversal semantics under mutation, and remains necessary even if victim selection is later optimized with a heap.

---
