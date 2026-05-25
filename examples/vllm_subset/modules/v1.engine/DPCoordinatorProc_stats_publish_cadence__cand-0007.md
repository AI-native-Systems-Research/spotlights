# DPCoordinatorProc stats publish cadence

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/coordinator.py`](vllm/v1/engine/coordinator.py) (lines 151–285)
- **Symbol:** `DPCoordinatorProc stats publish cadence`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0007`

## Description
Defines and applies the DP coordinator's stats-publication cadence for frontend load balancing: 100 ms default stats interval, 5 s unchanged-stats interval, and 50 ms initial settling timeout before first publication.

## Current approach
Static timing constants drive poll-timeout selection: min_stats_update_interval_ms defaults to 100, unchanged stats wait 5000 ms, and first-step publication waits 50 ms. Publications happen on poll timeout or immediate wave-state change, not on magnitude of load change.

## Estimated impact explanation
These timings bound how stale DPLBAsyncMPClient's lb_engines view can be. Fresher or more selective publications improve DP routing accuracy and reduce median TTFT variance for bursty agent turns, at the cost of coordinator and socket traffic.

## Evolve rationale
The concrete policy is the timeout computation at lines 263-272 plus the default interval at lines 152-164. It can be evolved with threshold-triggered publish, EWMA/hysteresis on counts, faster publish after queue-depth shocks, or backoff during steady state. Correctness oracle: coordinator/frontend tests in tests/v1/engine/test_engine_core_client.py plus a focused coordinator replay must preserve decoded (counts, wave, engines_running) message semantics and ensure frontends eventually receive the latest counts.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Trigger immediate publish on least-loaded-engine flip, not just elapsed cadence
- **Agent:** claude

**Detailed description.**

In `vllm/v1/engine/coordinator.py` at lines 263-285 (the poll/timeout/publish loop) and 385-409 (the per-engine stats update path), augment the existing `stats_changed` gate with a routing-relevance gate: keep a cached `last_published_min_idx` (the engine with the lowest queue depth at the last publication, breaking ties by index) and a cached `last_published_counts`. Whenever a stats update arrives from `output_back` and updates `self.engines[eng_index].request_counts`, recompute the current `argmin` (waiting+running). If it differs from `last_published_min_idx`, OR if the gap between the previous min and the new min in `last_published_counts` exceeds a small threshold (e.g., >=2 requests), short-circuit the next iteration's poll timeout to 0 so a publish happens on the very next loop turn, regardless of `stats_update_interval_ms`. Concretely, replace the `wait_for = self.stats_update_interval_ms if stats_changed else 5000` line with `wait_for = 0 if routing_changed else (self.stats_update_interval_ms if stats_changed else 5000)`, and reset `routing_changed` immediately after publish at line 283. The `min_stats_update_interval_ms=100` default and 5000ms steady-state interval stay intact for unchanged-ordering ticks, so socket traffic is unchanged in steady state. Validation: extend `tests/v1/engine/test_engine_core_client.py` with a coordinator replay that injects a sequence of stats where engine 0 starts least-loaded, then a burst makes engine 1 least-loaded, and asserts a publication is emitted within <=10ms of the flip rather than ~100ms later. Encoded message shape `(counts, wave, engines_running)` is preserved.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's own evolve_rationale lists generic ideas (threshold-triggered publish, EWMA/hysteresis, queue-depth shocks, steady-state backoff) — all of which trigger on *magnitude* of count change. This proposal triggers on the *argmin flip*, i.e., the precise routing-relevant invariant that DPLBAsyncMPClient consumes when picking an engine. EWMA/hysteresis would actively *suppress* these flips (smoothing harms routing accuracy), and a magnitude threshold misses small flips that nonetheless change which engine the frontend picks. The proposal is also strictly additive to the existing cadence (does not raise the steady-state publish rate) and cheap (O(engine_count) per stats update), making it different in mechanism and goal from the brainstorm hints in evolve_rationale.

---

### 2. End the 50 ms settling wait once all engines report the step
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/coordinator.py` around the timeout calculation at lines 263-272 and the scheduler-stats update path around lines 383-409, replace the unconditional `min_timeout = 50 if last_step_counts is None else 0` settling delay with a per-step completeness gate. Track the latest `(current_wave, step_counter)` key and a small set/bitset of engine indexes that have reported stats for that key. When every current engine has reported the latest key, set the settling timeout to `0`; otherwise keep the existing 50 ms fallback so partial or out-of-order stats still coalesce. The normal `stats_update_interval_ms` cadence remains the outer rate limit, but once the 100 ms cadence is already due the coordinator can publish immediately if it has a complete same-step snapshot instead of waiting for an extra quiet 50 ms. Reset/resize the tracking state on step/wave advancement and on elastic engine-count changes. Validation: add a focused coordinator replay in `tests/v1/engine/test_engine_core_client.py` that injects stats from all engines for the same step after the cadence is due and asserts a publication is emitted without the 50 ms delay, plus a partial-report case that still waits/falls back. The published tuple shape `(counts, wave, engines_running)` stays unchanged.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. This is not Claude's least-loaded-engine flip trigger: it does not look at the routing argmin or count magnitude, and it does not add a new routing-relevance threshold. It specifically removes avoidable latency from the coordinator's existing 50 ms stats-settling policy by using the already-carried scheduler step/wave metadata to know when settling is complete.

---
