# DPCoordinator.run polling/publish loop

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/coordinator.py`](vllm/v1/engine/coordinator.py) (lines 257–283)
- **Symbol:** `DPCoordinator.run polling/publish loop`
- **Kind:** loop
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0014`

## Description
DP coordinator publish loop that chooses when to broadcast engine load snapshots to frontends.

## Current approach
Computes wait_for as stats_update_interval_ms when stats changed or 5000 ms otherwise, optionally enforces a 50 ms minimum wait under wave coordination, then publishes either last_step_counts or current engine counts only on poll timeout.

## Estimated impact explanation
Snapshot publication latency feeds directly into DP routing decisions; improving cadence can reduce stale-load misrouting and median TTFT for bursty multi-turn DP serving.

## Evolve rationale
The 5000 ms fallback and 50 ms lockstep minimum are policy constants that define frontend snapshot freshness and wave-coordination latency. Candidate changes include adaptive publish cadence, immediate publish on significant load changes, lower or conditional lockstep minimums, or coalescing by observed rank arrival skew. Correctness oracle: DP coordinator/client tests in tests/v1/engine can assert count propagation, wave state updates, FIRST_REQ wakeups, and load-balancer snapshots remain eventually consistent.

## Deep research proposals

### 1. Adaptive DP coordinator publish cadence via EWMA divergence of engine load
- **Finding:** `find-vllm_v1_engine-0015` — *GitHub - Netflix/concurrency-limits*
- **Source URL:** <https://github.com/Netflix/concurrency-limits>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the fixed cadence constants in DPCoordinator.run's publish loop (vllm/v1/engine/coordinator.py:257-283) with an EWMA-divergence-based adaptive cadence inspired by Netflix concurrency-limits' Gradient2. Maintain two exponential moving averages over per-engine request-count vectors (or their aggregate/imbalance metric): a short-window EWMA reflecting recent load and a long-window EWMA reflecting baseline. Compute a divergence signal (e.g., L1 or max norm of short_ewma - long_ewma, normalized by long_ewma). Use this signal to modulate wait_for: when divergence exceeds a threshold (queueing trend rising or rank imbalance growing), shrink wait_for toward stats_update_interval_ms or trigger an immediate publish; when divergence is small and stats_changed is False, allow wait_for to grow beyond the current 5000 ms floor up to a bounded ceiling. Under wave coordination, keep the 50 ms minimum only while divergence indicates unsettled step boundaries; otherwise permit a lower minimum. Update the EWMAs each time _get_engine_counts is consulted or when engine stats arrive on output_back, so the signal tracks observed rank arrival skew without adding new IPC. Preserve the existing publish-on-timeout structure and last_step_counts handling so wave state, FIRST_REQ wakeups, and eventual consistency guarantees exercised by tests/v1/engine remain intact.

**Proposal rationale.**

The candidate's current cadence is governed by static constants (5000 ms fallback, stats_update_interval_ms, 50 ms lockstep floor) that do not react to how quickly engine load is actually changing. The finding's core technique — tracking divergence between a short and long EWMA to detect queueing trends — directly supplies an observed-behavior signal the coordinator currently lacks. Applying it here lets the publish loop broadcast fresher snapshots exactly when DP routing decisions are most sensitive to staleness (bursty multi-turn agentic traffic with shifting rank load) while backing off when load is stable, addressing the evolve_rationale's adaptive-cadence and immediate-publish-on-significant-change directions and plausibly reducing stale-load misrouting that inflates median TTFT.

---

## Agent proposals

### 1. Event-driven publish on routing-decision threshold crossings in DPCoordinator.run
- **Agent:** claude

**Detailed description.**

Modify the publish loop in vllm/v1/engine/coordinator.py:257-283 to trigger an immediate front-end publish whenever an engine's stats update (received on output_back, around lines 384-417) causes a routing-relevant threshold crossing relative to the last-published snapshot — rather than waiting for the next poll timeout. Maintain a small per-engine cached copy of the last values published to the front-end (num_waiting_reqs, num_running_reqs, kv_cache_usage). After the existing stats update at lines 414-417, compare the new values to that cache and flag a 'priority publish' when any of a few discrete conditions hold: (a) an engine's num_waiting_reqs transitions across zero (0→>0 or >0→0), which flips it between 'idle' and 'queueing' from the router's perspective; (b) kv_cache_usage crosses a saturation threshold (e.g., configurable, defaulting to ~0.9) in either direction; (c) the identity of the argmin-loaded engine (using the same score the DP load balancer uses on the client side) changes. If any condition fires, publish immediately after finishing draining currently-pending events in this loop iteration, subject to a small min-interval floor (e.g., 5-10 ms) tracked via last_publish_time to prevent flooding under bursty stats arrival. Preserve the existing 5000 ms/stats_update_interval_ms timeout-driven publish, the 50 ms lockstep floor, and the last_step_counts handling so wave-coordinated snapshots and eventual consistency are unchanged. Under wave coordination, only allow the priority publish when last_step_counts is None (i.e., we already have a coherent step-boundary snapshot to publish), otherwise defer until the step boundary is reached, so lockstep semantics are not broken. Correctness oracle: existing DP coordinator/client tests in tests/v1/engine assert count propagation, wave-state updates, FIRST_REQ wakeups, and load-balancer snapshot consistency — all continue to pass because timeout-driven publishes and wave messages are untouched; the added path only fires extra publishes on routing-relevant deltas.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_engine-0015) modulates the *cadence* of the existing timeout-driven publish via EWMA divergence — it changes wait_for and the lockstep minimum but keeps the 'publish only on poll timeout' control structure. This proposal is orthogonal: it changes *what triggers a publish*, adding an event-driven path that fires the moment an engine's stats cross a specific routing-decision threshold (queue empty↔nonempty, KV saturation, argmin-load flip). These are the exact moments when the DP client's routing choice would change, so a fresher snapshot has direct impact on request placement — a signal EWMA divergence cannot capture because divergence is a scalar aggregate that averages over exactly the discrete transitions that matter for routing. The two mechanisms compose (an EWMA-adaptive cadence would still benefit from immediate publishes on threshold crossings), so this is not a subset of the existing proposal.

---

### 2. Publish zero-wait snapshots immediately after FIRST_REQ wakeups
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/coordinator.py:257-283`, add a narrow fast path for the transition that matters most to multi-turn TTFT: when the coordinator receives a `FIRST_REQ` wakeup from a frontend for a previously idle or zero-waiting engine, publish a fresh load snapshot as soon as the wakeup has been processed and the current event drain completes. This should use `_get_engine_counts()` and the existing frontend publish mechanism, but gate it with a very small `last_publish_time` floor to avoid duplicate broadcasts when many frontends wake the same coordinator at once. Under wave coordination, keep the existing `last_step_counts` semantics by publishing `last_step_counts` if available and otherwise deferring until coherent counts are available, matching the current timeout path. Add or extend `tests/v1/engine` coverage to assert that a FIRST_REQ wakeup causes frontend-visible load counts to refresh without waiting for the 5000 ms idle fallback, while the existing timeout-based eventual consistency path remains intact.

**Novelty rationale.**

The deep-research proposal adapts the timeout cadence based on EWMA load divergence, and Claude's proposal publishes on stats-update threshold crossings from `output_back`. This proposal is triggered by the frontend demand-side signal already represented by `FIRST_REQ`, before backend stats necessarily change enough to cross a routing threshold. It specifically removes the idle-fallback delay for the first request of a new burst, which is a distinct source of stale routing in multi-turn agentic workloads and composes with both EWMA cadence and stats-threshold publishing.

---
