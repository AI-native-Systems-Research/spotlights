# DPEngineCoreProc._has_global_unfinished_reqs

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/core.py`](vllm/v1/engine/core.py) (lines 1846–1863)
- **Symbol:** `DPEngineCoreProc._has_global_unfinished_reqs`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0006`

## Description
Controls how often DP ranks all-reduce unfinished-request state to decide whether the global wave is still running or can pause.

## Current approach
Increments step_counter each call and skips ParallelConfig.sync_dp_state on 31 of every 32 steps via a hard-coded modulo, returning True without consulting actual rank state.

## Estimated impact explanation
When DP ranks drain unevenly during short agent turns, late wave-completion detection wastes dummy work and delays the next wave. Smarter sync cadence can improve next-turn TTFT, with impact bounded by DP size and all-reduce cost.

## Evolve rationale
The optimization unit is the modulo-32 sync gate at lines 1847-1852. Adaptive probing, such as syncing more aggressively when the local rank is empty or after pause requests and backing off when all ranks are clearly busy, is a self-contained change. Correctness oracle: DP integration tests and pause-mode tests must eventually reach the same pause_consensus/current_wave state and produce identical request outputs.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Pipeline DP unfinished-state all-reduce with async_op overlap
- **Agent:** claude

**Detailed description.**

Replace the modulo-32 skip in DPEngineCoreProc._has_global_unfinished_reqs (vllm/v1/engine/core.py:1846-1863) with a pipelined async all-reduce, so the sync runs every step at near-zero marginal cost instead of once per 32 steps blocking. Concretely: (1) keep a handle self._dp_sync_pending = (work_handle, tensor, prev_local_unfinished, prev_pending_pause). On each call, if a handle is in flight and not yet ready, return the previously-known has_unfinished result (treat unfinished as True conservatively, never reporting false until a fresh result lands) and immediately issue the next all_reduce with async_op=True on a fresh tensor populated from the current local_unfinished and pending_pause. (2) When the previous handle is ready, .wait() it (which is non-blocking once complete), decode (has_unfinished_global, pause_consensus) using the same logic as ParallelConfig.sync_dp_state (vllm/config/parallel.py:667-691), and then kick off the next async all-reduce. (3) Force a synchronous sync (wait the in-flight handle, then issue and wait one more cycle) only on real state transitions: local just became empty, pending_pause flipped True, or a max-staleness budget (e.g. 64 steps) elapsed. This requires factoring sync_dp_state into a launch/finalize pair (e.g. begin_sync_dp_state returning (handle, tensor) and finalize_sync_dp_state(handle, tensor, dp_size)) so the engine loop can interleave the collective with the next forward step. The pause_consensus/wave logic at lines 1820-1844 still observes the same eventual values, just one step later in the steady state and immediately on local transitions.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's own evolve_rationale only suggests adaptive *cadence* (sync more or less often based on local emptiness/pause/busy heuristics) but keeps each sync synchronous and blocking. This proposal is orthogonal: it changes the sync from a blocking CPU all-reduce into a one-step-pipelined async collective overlapped with the next engine iteration, eliminating the latency cost rather than amortizing it. It uses torch.distributed's async_op=True path (not currently exercised by sync_dp_state) and adds an event-driven force-sync only on local state transitions, which is a different mechanism from rate adaptation.

---

### 2. Drive coordinated DP wave completion from coordinator state
- **Agent:** codex

**Detailed description.**

For coordinator-backed DP, replace the modulo-gated finish all-reduce in `DPEngineCoreProc._has_global_unfinished_reqs` (`vllm/v1/engine/core.py:1846-1863`) with an event-driven coordinator path. Engines already publish per-wave scheduler counts via `_maybe_publish_request_counts`; extend that state message to also carry `pending_pause` and make it publish when either counts or pause state changes. In `DPCoordinatorProc`, track the latest `(current_wave, step_counter, waiting, running, pending_pause)` per engine, ignore stale wave/step updates using the existing ordering checks, and broadcast a new DP-state control message when either all engines report zero waiting/running requests for the current wave or all engines report `pending_pause=True`. Each `DPEngineCoreProc` records that coordinator decision in `_handle_client_request`; `_has_global_unfinished_reqs` then increments `step_counter` and returns `False` immediately for a matching coordinator-complete wave, applying the existing `pause_consensus` side effects when the coordinator reports pause consensus. Keep the current `ParallelConfig.sync_dp_state` path only for deployments without coordinator support, and gate the new path as an all-ranks capability so no rank enters the old collective while peers are waiting for coordinator messages.

**Novelty rationale.**

There are no deep-research proposals for this candidate. Agent A proposes keeping the same DP all-reduce semantics but launching it asynchronously and overlapping it with engine work. This proposal does not pipeline or retune the collective; it removes the steady-state finish collective in coordinator-backed deployments by reusing the existing coordinator aggregation channel as the source of global unfinished/pause state. It also differs from the candidate rationale's adaptive sync-cadence idea because completion is driven by explicit per-rank state updates, not by changing how often `_has_global_unfinished_reqs` probes with `sync_dp_state`.

---
