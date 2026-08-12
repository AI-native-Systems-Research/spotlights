# _PUSH_WRITER_POLL_INTERVAL_MS

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py`](vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py) (lines 70–74)
- **Symbol:** `_PUSH_WRITER_POLL_INTERVAL_MS`
- **Kind:** config_block
- **Estimated impact:** medium
- **Id:** `cand-vllm_distributed_kv_transfer-0004`

## Description
Fixed 1.0 ms active-state poll cadence for the nixl-push-writer thread while unmatched push state exists.

## Current approach
A static sleep interval is used regardless of queue depth, recent notification rate, or number of pending PUSH_REG/finished-block matches. The writer is event-woken from idle but self-polls at this cadence while active.

## Estimated impact explanation
In push mode, D-side turn-2 TTFT can include this polling delay before registration and WRITE completion are matched; the bound is small but paid repeatedly under multi-turn reuse.

## Evolve rationale
This constant defines the push-mode latency/CPU tradeoff. Adaptive backoff, notification-readiness wakeups, or cadence selection based on pending state are local policy changes preserving writer behavior. Oracle: push-mode NIXL unit and integration tests cover PUSH_REG matching, done_recving/done_sending, and request-free ordering invariants.

## Deep research proposals

### 1. Replace fixed 1 ms active-state poll with NIXL notification-driven wakeups for the push writer
- **Finding:** `find-vllm_distributed_kv_transfer-0003` — *Enhancing Distributed Inference Performance with the NVIDIA Inference Transfer Library*
- **Source URL:** <https://developer.nvidia.com/blog/?p=113426>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Change the wake model for the `nixl-push-writer` loop in `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py` (around `_PUSH_WRITER_POLL_INTERVAL_MS` at lines 70-74 and its sole use in `_push_writer_loop` at lines 210-272) so that, while `_push_finished_blocks` is non-empty and the writer is otherwise waiting on a D-side PUSH_REG notification, it no longer relies on a fixed 1.0 ms self-poll for progress. Concretely: (1) drop into the same event-based wait as the idle branch (block on `_push_writer_wake`) but bound it with a small timeout as a safety fallback, and (2) drive the wakeup from NIXL's own notification-arrival signal — either by using a non-blocking `get_new_notifs` in a tight drain followed by a bounded `wait()` when it returns empty, or, if the NIXL Python binding exposes a notification-ready fd/condition (as suggested by the blog's emphasis on target-side notifications as the intended completion mechanism), waiting on that condition directly. Preserve the existing eviction path (`_evict_finished_inbox`) so unmatched entries can't pin the writer indefinitely, and keep the safety-net timeout larger than 1 ms (e.g., a few ms) to trade a small amount of worst-case latency for far less CPU spinning when unmatched state persists. Bench with the existing push-mode NIXL unit/integration tests (PUSH_REG matching, `done_recving`/`done_sending`, request-free ordering) and measure D-side turn-2 TTFT under the multi-turn agentic workload to confirm the change reduces the polling-induced latency floor.

**Proposal rationale.**

The candidate's fixed 1 ms cadence is only paid while the writer is waiting for a D-side PUSH_REG NIXL notification to match a P-side finished-blocks entry — i.e., a notification-arrival event. The NVIDIA NIXL blog explicitly frames NIXL as a fully non-blocking API in which target-side notifications, not busy polling, are the intended progress mechanism ("NIXL is designed to have a fully non-blocking API", plus the target-side-notification model in "What is NIXL / Setting up the agents"). That directly targets the gap this constant papers over: on multi-turn agentic workloads, D-side turn-2 TTFT is repeatedly gated by a 1 ms wait for a NIXL notification to become visible to `get_new_notifs`, even though NIXL itself already knows when new notifs have arrived. Replacing (or bounding much more loosely) the self-poll with a notification-driven wake collapses that per-match latency floor without changing writer semantics or ordering invariants, and it aligns the module with NIXL's documented usage pattern — exactly the kind of transferable, local policy change the candidate's evolve_rationale calls out.

---

### 2. Drain-to-empty per poll iteration to amortize the 1 ms cadence cost
- **Finding:** `find-vllm_distributed_kv_transfer-0007` — *10.39M Storage I/O Per Second From One Thread*
- **Source URL:** <https://spdk.io/news/2019/05/06/nvme/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Rework the active-state loop body around _PUSH_WRITER_POLL_INTERVAL_MS in vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py (lines 70-74 and the associated writer loop) so that each wake harvests every ready NIXL completion/notification in a bounded inner loop rather than one batch per sleep. Concretely: on each iteration, repeatedly call get_new_notifs and drain _reg_send_inbox / _finished_blocks_inbox / _deferred_push_inbox / _evict_finished_inbox until they all report empty (or a per-tick work cap is hit) before sleeping for _PUSH_WRITER_POLL_INTERVAL_MS. Match D registrations against P finished blocks and issue WRITE transfers in the same drained pass, so a single poll can satisfy many pending PUSH_REG matches. Because the per-iteration Python/FFI/telemetry overhead is amortized across many completions, the 1 ms constant can be tightened (or replaced with a shorter floor plus an idle-transition into the event-wake path) without a proportional CPU increase.

**Proposal rationale.**

The candidate's constraint is the per-poll fixed overhead that forces a 1 ms floor on active-state latency; SPDK's specific technique in the cited article is exactly this: reap many completions per poll and batch doorbell writes so a single syscall/FFI edge covers a large batch. Push mode's writer already services multiple queues plus NIXL notifs per tick, so a drain-to-empty shape maps directly onto the finding. It addresses the candidate's evolve_rationale (cadence selection based on pending state) by making each tick do proportionally more work when the queues are deep, cutting the D-side turn-2 TTFT contribution from PUSH_REG-match latency in the multi-turn agentic workload without changing the writer's ordering invariants (get_finished/done_recving/done_sending, request-free ordering).

---

## Agent proposals

### 1. Deadline-aware timed wait using per-request lease expirations instead of fixed 1 ms cadence
- **Agent:** claude

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py`, replace the fixed `_PUSH_WRITER_POLL_INTERVAL_MS = 1.0` (lines 70-74) and its sole use in `_push_writer_loop` (the `self._push_writer_stop.wait(timeout=sleep_s)` branch at lines 268-269) with a **deadline-aware timeout**. Concretely: track a companion dict `_push_finished_blocks_arrival: dict[ReqId, float]` populated at the moment an unmatched entry is inserted at line 241 (`self._push_finished_blocks[rid] = blocks`). On each writer loop iteration when `_push_finished_blocks` is non-empty, compute `now = time.perf_counter()` and derive `deadline = min(self._reqs_to_send[rid] for rid in self._push_finished_blocks if rid in self._reqs_to_send)` — that is, the earliest lease expiration among unmatched entries; if none of them are lease-tracked, fall back to `min(arrival[rid]) + MAX_UNMATCHED_MS` as a safety cap. Sleep on `self._push_writer_stop.wait(timeout=max(MIN_SLEEP_S, min(deadline - now, MAX_SAFETY_S)))` where `MIN_SLEEP_S` is a small floor (e.g. 0.25 ms) and `MAX_SAFETY_S` is a few tens of ms. Additionally, maintain an EWMA of observed match latencies (time from `_push_finished_blocks[rid] = blocks` at line 241 to the matching `_pop_matching_finished_blocks` hit in `_handle_push_reg_notif`); when the EWMA is stable, use it as the *initial* wait window for freshly-unmatched entries before falling through to the deadline-bounded wait. Because leases and matches use real per-request timescales — typically many ms to seconds — this collapses the writer's active-state wake count from ~1000 wakes/s (the current 1 ms cadence) to a handful of wakes across the entire lifetime of a batch of unmatched entries, while never sleeping past a lease deadline that the main thread would then need to evict. Push-mode unit and integration tests (`PUSH_REG` matching, `done_recving`/`done_sending`, request-free ordering, lease-expiry eviction via `_evict_finished_inbox`) exercise the correctness envelope; measure D-side turn-2 TTFT under multi-turn agentic workloads to confirm the polling floor is removed without regressing lease-expiry eviction latency.

**Novelty rationale.**

Existing proposal find-0003 replaces the fixed poll with NIXL notification-driven wakeups (bounded fallback still time-based, using an arbitrary safety timeout of a 'few ms'). Existing proposal find-0007 keeps polling but drains all queues per iteration to amortize per-tick cost. Neither uses the per-request lease-expiration data already maintained in `_reqs_to_send` to compute a *right-sized* wait window: #0003 introduces a new, still-hardcoded safety timeout, and #0007 doesn't change wake timing at all. This proposal derives the wait window from real per-entry deadlines the module already tracks, plus an EWMA of *observed* match latency, so the effective cadence adapts to the actual distribution of PUSH_REG arrival times and per-request lease budgets — a policy dimension neither existing proposal addresses. It is complementary rather than overlapping: it composes cleanly with #0003 (a notification-driven wake preempts the deadline-bounded wait) and with #0007 (each wake still drains to empty).

---

### 2. Add a bounded fast-path poll window before backing off
- **Agent:** codex

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py`, replace the single `_PUSH_WRITER_POLL_INTERVAL_MS = 1.0` active-state cadence with a two-phase policy keyed to when `_push_finished_blocks` first becomes non-empty. Track an `active_poll_started_at` / `fast_poll_until` timestamp in `_push_writer_loop`: when the writer inserts the first unmatched finished block, run a bounded fast-path window for a very short duration, for example 100-250 us, where the loop immediately re-drains inboxes and `nixl_wrapper.get_new_notifs()` using `time.sleep(0)` or a sub-100 us wait instead of sleeping 1 ms. If the PUSH_REG still has not arrived by the end of that budget, fall back to a slower sleep, e.g. the existing 1 ms or a capped exponential backoff, until `_push_finished_blocks` becomes empty again. Reset the fast-path window each time the active set transitions from empty to non-empty, not on every loop, so long-lived unmatched state cannot burn CPU indefinitely. This specifically targets the common median-TTFT case where P-side finished-block metadata and D-side PUSH_REG are racing and the missing counterpart arrives just after the current loop iteration; those matches no longer pay the full 1 ms floor, while stale or lease-expired entries still settle into low-CPU polling and are cleaned up through the existing `_evict_finished_inbox` path.

**Novelty rationale.**

The existing notification-driven proposal replaces polling with a NIXL readiness wakeup, and the drain-to-empty proposal amortizes work done per wake without changing the initial latency policy. Agent A's proposal computes longer deadline-aware waits from request leases and observed match latency. This proposal is different: it keeps the current local polling model but adds a bounded, age-based fast-path only on the empty-to-active transition to catch near-immediate PUSH_REG arrivals before the 1 ms sleep. It is focused on reducing median latency for closely racing metadata/notification arrivals, not on notification integration, batch draining, or lease-derived sleep deadlines.

---
