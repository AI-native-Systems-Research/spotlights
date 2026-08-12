# DPEngineCoreProc._has_global_unfinished_reqs

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/core.py`](vllm/v1/engine/core.py) (lines 2165–2182)
- **Symbol:** `DPEngineCoreProc._has_global_unfinished_reqs`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0002`

## Description
DP wave-completion detector that periodically synchronizes all ranks to decide whether the busy loop may pause.

## Current approach
Increments step_counter and performs ParallelConfig.sync_dp_state only when step_counter % 32 == 0; all other steps assume global unfinished work remains.

## Estimated impact explanation
The interval trades all-reduce overhead against wave-end latency; improving it can reduce median TPOT/TTFT gaps for short, bursty agentic waves that currently wait up to 31 extra steps to observe global idle.

## Evolve rationale
The hard-coded 32-step finish-sync interval is a policy constant on the DP scheduling path. Alternatives include adaptive cadence based on local queue transitions, shorter cadence near idle, longer cadence under saturation, or event-driven sync after no rank schedules useful work. Correctness oracle: DP pause/wave tests in tests/v1/engine and tests/v1/engine/test_async_llm.py should still show pause consensus, wave_complete emission, and request liveness.

## Deep research proposals

### 1. Replace fixed 32-step DP finish-sync cadence with EWMA-divergence adaptive trigger
- **Finding:** `find-vllm_v1_engine-0015` — *GitHub - Netflix/concurrency-limits*
- **Source URL:** <https://github.com/Netflix/concurrency-limits>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In DPEngineCoreProc._has_global_unfinished_reqs (vllm/v1/engine/core.py:2165-2182), replace the hard-coded `step_counter % 32 == 0` gate with an adaptive trigger inspired by Netflix concurrency-limits' Gradient2 algorithm. Maintain two exponential moving averages over per-step local queue/idle signals derived from local_unfinished and local scheduler activity: a short-window EWMA (fast, e.g., alpha ~ 1/8) and a long-window EWMA (slow, e.g., alpha ~ 1/64). Trigger the ParallelConfig.sync_dp_state all-reduce when the divergence between the two EWMAs indicates a queueing trend change consistent with an approaching wave boundary — specifically, when the short-window average of local_unfinished drops below the long-window average by more than a small threshold (rank is trending idle), and also on a safety-cap interval (e.g., every 64 steps) to bound worst-case latency. Under saturation (both EWMAs high, divergence small), sync less often than the current 32 steps; near idle (short EWMA collapsing), sync sooner. Keep the existing pause_consensus handling (lines 2177-2180) unchanged. Add the EWMA state (short_avg, long_avg, last_sync_step) as instance attributes initialized alongside step_counter; reset them on wave_complete alongside step_counter (line 2161).

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out that the 32-step interval is a policy constant trading all-reduce overhead against wave-end latency, and suggests adaptive cadence based on local queue transitions and shorter cadence near idle as alternatives. The Netflix concurrency-limits Gradient2 technique provides a concrete, well-tested mechanism — dual-EWMA divergence — for exactly this kind of trend detection from noisy per-step signals without requiring cross-rank coordination for the trigger itself (the all-reduce still happens, just at smarter moments). This directly targets the caller-context objective of reducing median TTFT/TPOT for multi-turn agentic workloads, where short bursty waves currently pay up to 31 extra steps of wave-end latency; a divergence-driven trigger would fire the sync as soon as a rank's local idle trend becomes detectable, cutting that tail. The finding contributes a specific transferable signal-processing idea (short/long EWMA divergence as a queueing trend detector) rather than merely restating 'be adaptive.'

---

## Agent proposals

### 1. Event-driven DP finish-sync triggered by local wave-boundary transitions
- **Agent:** claude

**Detailed description.**

In DPEngineCoreProc._has_global_unfinished_reqs (vllm/v1/engine/core.py:2165-2182), replace the fixed `step_counter % 32 == 0` cadence with an edge-triggered mechanism that fires the ParallelConfig.sync_dp_state all-reduce on locally observable state transitions rather than on a step count. Track two pieces of instance state alongside step_counter: (a) prev_local_unfinished (bool, last-step value of local_unfinished) and (b) steps_since_last_sync (int, monotonically increasing). Fire the sync immediately on the falling edge — i.e., when prev_local_unfinished was True and the current local_unfinished is False — because that is the exact moment this rank first believes it may be at a wave boundary and any per-step-count delay is pure wave-end latency. Also fire on the rising edge (False→True) so that a rank that just received new work quickly rejoins pause consensus rather than being seen as idle for up to 31 extra steps. Keep a hard safety cap (e.g., `steps_since_last_sync >= 64`) as a backstop against pathological oscillation and to bound worst-case drift when local_unfinished stays constant for a long time under saturation. Reset prev_local_unfinished and steps_since_last_sync on wave_complete alongside step_counter (line 2161). Add a small hysteresis: after firing on a falling edge, suppress a subsequent rising-edge fire for K steps (e.g., K=4) to avoid a burst of all-reduces when local_unfinished flickers within a single scheduling tick. Leave the pause_consensus handling at lines 2177-2180 unchanged; only the *when-to-sync* predicate changes.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_engine-0015) proposes a *statistical* trigger — dual-EWMA divergence over noisy per-step signals — that must accumulate enough smoothed evidence before firing, and it still sync-caps at 64 steps as a safety net. This proposal is categorically different: it is *edge-triggered on a boolean state transition* (local_unfinished True→False and False→True), so on the falling edge the sync fires on the very next step after the rank goes idle — zero smoothing lag — which is exactly the tail the caller-context objective (median TTFT/TPOT on bursty multi-turn agentic waves) is trying to cut. It also adds a novel rising-edge trigger and short suppression window (hysteresis) that the EWMA proposal does not describe, addressing the symmetric case where a rank re-acquires work and must rejoin pause consensus quickly. The mechanism (edge detection + hysteresis on a boolean) does not overlap the mechanism (dual-EWMA divergence on a continuous signal) — they are independently implementable and could even coexist, but this one is simpler, has no tuning of alpha/threshold, and reacts one step faster on the critical falling-edge case.

---

### 2. Add a wall-clock latency budget for DP finish-sync
- **Agent:** codex

**Detailed description.**

In `DPEngineCoreProc._has_global_unfinished_reqs` (`vllm/v1/engine/core.py:2165-2182`), keep the existing step-based backstop but add a time-based sync trigger using `time.monotonic_ns()`. Track `last_finish_sync_ns` as instance state initialized with the other engine-loop counters and reset it when `wave_complete` resets `step_counter`. The sync predicate should fire when either the existing step interval is reached or the elapsed wall-clock time since the previous finish-sync exceeds a small configurable budget while this rank is locally idle, for example `not local_unfinished and now_ns - last_finish_sync_ns >= dp_finish_sync_idle_budget_ns`. After `ParallelConfig.sync_dp_state`, update `last_finish_sync_ns`. This bounds wave-end detection by real elapsed time instead of by an arbitrary number of engine iterations, which matters because 32 iterations can be cheap during empty/dummy batches but much more expensive when recent batches had long decode or scheduling work. Add tests that mock or inject the monotonic clock to verify that an idle rank syncs before 32 steps once the latency budget expires, while saturated ranks still use the lower-overhead step backstop.

**Novelty rationale.**

The deep_research_proposal replaces the fixed cadence with dual-EWMA trend detection over local queue signals, and Agent A proposes boolean edge detection with hysteresis. This proposal is neither statistical nor edge-triggered: it introduces an elapsed-time service-level budget for finish-sync. It addresses a different failure mode of the current policy, namely that a fixed number of steps does not correspond to a fixed latency when per-step duration varies. It could coexist with either EWMA or edge detection, but the actionable change here is specifically to bound idle wave-completion latency in wall-clock time while preserving the current step-count backstop for overhead control.

---
