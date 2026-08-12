# DPEngineCoreProc._should_throttle_prefills

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/core.py`](vllm/v1/engine/core.py) (lines 2086–2093)
- **Symbol:** `DPEngineCoreProc._should_throttle_prefills`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0010`

## Description
DP prefill admission throttle that decides whether new prefills may be scheduled on the current step.

## Current approach
Returns true whenever prefill_schedule_interval > 1 and step_counter is not divisible by that fixed interval; the cadence is deterministic and identical across DP ranks.

## Estimated impact explanation
Prefill admission timing drives TTFT for new turns and interacts with decode throughput; better cadence can lower median TTFT under bursty agentic arrivals without overloading DP ranks.

## Evolve rationale
This is an owned scheduling heuristic that gates prefill entry in DP mode. Alternatives include adaptive intervals from rank imbalance, queue depth, KV pressure, per-rank staggered offsets, or separate burst/steady-state policies. Correctness oracle: DP scheduling tests should ensure prefills are eventually admitted, requests finish, and configured interval semantics remain testable where required.

## Deep research proposals

### 1. Urgency-based bypass of DP prefill throttle for waiting requests
- **Finding:** `find-vllm_v1_engine-0001` — *Taming Request Imbalance: SLO-Aware Scheduling for Disaggregated LLM Inference*
- **Source URL:** <https://papers.cool/arxiv/2605.02329>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify `DPEngineCoreProc._should_throttle_prefills` at `vllm/v1/engine/core.py:2086-2093` to bypass the fixed-cadence throttle when at least one waiting request has high urgency, keeping the cadence gate as the default. Concretely, when `prefill_schedule_interval > 1` and `step_counter % prefill_schedule_interval != 0`, additionally consult the scheduler's waiting queue for an urgency signal — for example the maximum queued wait time or, equivalently, the minimum remaining TTFT slack against a configured target — and return False (admit prefills this step) when that urgency exceeds a threshold. Preserve the fresh-wave fast path (step_counter == 0 still admits). To keep DP-rank determinism, derive urgency from state that is either lockstep-synchronized (e.g. per-rank counts already published via `_maybe_publish_request_counts`) or reduced across ranks — otherwise fall back to the cadence-only decision. Wire two new config knobs (urgency threshold and, optionally, target TTFT) with defaults that reproduce today's behavior. Correctness is preserved because the override only *loosens* throttling: prefills are still eventually admitted on cadence, and interval semantics remain testable by disabling the urgency threshold in tests.

**Proposal rationale.**

The candidate's current heuristic is a fixed, load-oblivious cadence; the finding's urgency-based prefill selection directly targets the same decision point and is explicitly motivated by reducing median TTFT in long-tail multi-turn workloads — matching the caller's objective and the workload hint. It contributes a concrete, transferable mechanism (an urgency signal computed from queue wait / TTFT slack) that plugs into `_should_throttle_prefills` without discarding the DP-balancing cadence, addressing the gap where a short new-turn prefill can otherwise wait a full interval behind long prefills purely because of step-counter alignment.

---

### 2. Replace fixed-cadence prefill throttle with load-aware gate on prefill/decode token pressure
- **Finding:** `find-vllm_v1_engine-0006` — *Scheduler*
- **Source URL:** <https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/03-scheduler.md>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `DPEngineCoreProc._should_throttle_prefills` (vllm/v1/engine/core.py:2086-2093), replace the deterministic `step_counter % prefill_schedule_interval` gate with a load-aware admission decision that treats prefill (input-token) and decode (output-token) load as separate dimensions, mirroring the shape-aware scheduling principle from sglang-jax. Concretely: keep the current cadence as an upper bound / fallback, but allow prefills through earlier when local decode-token load is high relative to prefill-token load (so decodes are not starved by a queued prefill), and hold prefills back when local prefill-token load is already large even if the cadence step aligns. Signals available on the rank include `scheduler.get_request_counts()` (already used by `_maybe_publish_request_counts` at core.py:2069-2084) for waiting/running counts, `scheduler.get_kv_cache_usage()` for KV pressure, and the scheduler's running batch to estimate scheduled prefill vs decode tokens for the next step. Add a cheap-overhead fast path (analogous to sglang-jax's LPM->FCFS fallback at queue length >128) so that when the waiting queue is short the throttle short-circuits to `False` and skips any load computation, keeping scheduler overhead flat under bursty arrivals. The `prefill_schedule_interval` remains as a bound/safety net so configured-interval semantics stay testable.

**Proposal rationale.**

The candidate's current approach is explicitly a fixed, DP-rank-identical modulo cadence, which is exactly the kind of shape-blind heuristic the finding argues against: it can hold prefills back on a rank that is decode-heavy (hurting TTFT for newly arriving turns in a multi-turn agentic workload) or admit prefills on a rank that is already prefill-saturated (hurting TPOT for in-flight decodes). The finding contributes two transferable ideas that map onto this gate: (1) treat prefill-token and decode-token load as separate scheduling dimensions when making admission decisions, and (2) guard scheduler overhead with a queue-length fast-path so the adaptive logic does not itself become the bottleneck when the waiting queue is small. Both directly address the evolve_rationale's suggestion of `adaptive intervals from rank imbalance, queue depth, KV pressure` and align with the caller's TTFT/TPOT objective under bursty agentic arrivals. The proposal keeps existing correctness oracles satisfied (prefills are still eventually admitted; `prefill_schedule_interval` remains a bound).

---

### 3. Replace fixed prefill cadence with progress-informed adaptive throttle
- **Finding:** `find-vllm_v1_engine-0009` — *Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs*
- **Source URL:** <https://arxiv.gg/abs/2506.03296>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Rework `DPEngineCoreProc._should_throttle_prefills` (vllm/v1/engine/core.py:2086-2093) so that admission is gated by observed progress signals rather than the deterministic `step_counter % prefill_schedule_interval` cadence. Concretely: (1) sample lightweight per-step signals already available on the engine core — recent decode step latency, per-rank scheduler queue depth from `scheduler.get_request_counts()`, KV-cache usage from `scheduler.get_kv_cache_usage()`, and the current rank's imbalance versus other DP ranks (already published via `_maybe_publish_request_counts`); (2) maintain short EMAs of decode step time and prefill step time so the throttle can predict whether admitting a prefill this step would overshoot a target step-time budget; (3) admit a prefill when predicted step time stays under budget AND this rank is not the most-loaded rank (imbalance guard), otherwise defer. Preserve the existing configured `prefill_schedule_interval` as an upper-bound ceiling so operators retain a deterministic worst-case cadence. Keep the DP-lockstep invariant by driving the decision from `step_counter`-stamped statistics that are consistent across ranks (or by taking a conservative rank-local decision that only relaxes, never tightens, the global ceiling).

**Proposal rationale.**

The finding argues for replacing fixed timing (fixed sleeps / fixed cadences) with profiling-informed dispatch that predicts subtask execution times and reacts to observed progress. The candidate is exactly a fixed-cadence heuristic: it throttles prefills on a rigid modulo schedule that is blind to actual decode-step latency, queue depth, KV pressure, or DP rank imbalance. Under the caller's bursty multi-turn agentic workload with a median-TTFT/TPOT objective, a fixed interval is either too conservative (delayed TTFT for new turns arriving just after a prefill step) or too aggressive (prefill collides with a heavy decode batch and inflates TPOT). Applying the finding's predict-and-adapt idea addresses this gap directly by letting the throttle react to observed step-time and imbalance signals the engine already computes, without inventing new instrumentation or breaking the deterministic ceiling that DP scheduling tests rely on.

---

### 4. Adaptive DP prefill throttle via short/long EWMA divergence (Gradient2-style)
- **Finding:** `find-vllm_v1_engine-0015` — *GitHub - Netflix/concurrency-limits*
- **Source URL:** <https://github.com/Netflix/concurrency-limits>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the fixed-interval gate in DPEngineCoreProc._should_throttle_prefills (vllm/v1/engine/core.py:2086-2093) with a Gradient2-style adaptive admission signal borrowed from Netflix/concurrency-limits. Maintain two EWMAs — a short-window and a long-window — over a per-rank pressure metric (e.g., decode step latency, waiting-queue depth, or KV-cache usage published in SchedulerStats near lines 2069-2084). Compute the gradient short/long as a queueing trend indicator: when short << long (latency/pressure improving), permit prefills every step; when short >> long (queueing building), throttle prefills for more steps than the configured cadence would; when they agree, fall back to the current prefill_schedule_interval cadence. Because step_counter is lockstep across DP ranks and stats are already published to peers via _maybe_publish_request_counts, aggregate the divergence across ranks (e.g., max short/long ratio) so the admission decision remains deterministic and identical across DP workers, preserving the invariant that today’s fixed-interval check relies on. Keep prefill_schedule_interval as a floor/ceiling on cadence so configured interval semantics remain testable, and admit prefills unconditionally on step_counter == 0 to preserve fresh-wave behavior.

**Proposal rationale.**

The candidate explicitly calls out adaptive intervals from rank imbalance and KV pressure as alternatives to the deterministic modulo gate, and its stated impact is TTFT under bursty agentic arrivals — exactly the regime where a fixed cadence either admits prefills into a congested step or needlessly delays them during quiet periods. The finding’s Gradient2 technique (divergence between short- and long-window EWMAs to detect queueing trend) provides a concrete, transferable control law that consumes signals vLLM already gathers (SchedulerStats, kv_cache_usage, step_counter published across DP ranks) and produces a bounded admission decision. It addresses the specific gap of the current heuristic — no feedback from observed load — while remaining deterministic across DP ranks when computed over shared, lockstep-published stats, which is the correctness constraint the current implementation preserves.

---

## Agent proposals

### 1. Stagger DP prefill throttle phase offset per rank to interleave prefill admission across ranks
- **Agent:** claude

**Detailed description.**

In `DPEngineCoreProc._should_throttle_prefills` at `vllm/v1/engine/core.py:2086-2093`, add a deterministic per-rank phase offset so DP ranks admit prefills on staggered steps instead of on the same modulo-aligned steps. Replace the current test with `offset = (self.dp_rank * self.prefill_schedule_interval) // self.dp_size` and return `self.prefill_schedule_interval > 1 and (self.step_counter + offset) % self.prefill_schedule_interval != 0`. With `prefill_schedule_interval=4, dp_size=4`, rank 0 admits prefills on steps {0,4,8,...}, rank 1 on {3,7,11,...}, rank 2 on {2,6,10,...}, and rank 3 on {1,5,9,...}. Because `dp_rank` and `dp_size` are already available on `DPEngineCoreProc` (set in `_init_data_parallel`), no new signals, EMAs, cross-rank reductions, or per-request state are introduced — each rank's decision remains a pure function of `(step_counter, dp_rank, dp_size, prefill_schedule_interval)` and stays deterministic and lockstep-computable. To preserve fresh-wave behavior, keep an early-fresh-wave fast path that admits prefills unconditionally while `step_counter < dp_size` (so rank 0's step 0, rank 1's step 1, etc. still admit immediately after idle). The `prefill_schedule_interval` per-rank cadence is unchanged: every rank still runs exactly one admission slot per interval, so aggregate prefill volume, DP balancing, and the correctness oracle (prefills eventually admitted, interval semantics testable by setting `dp_size=1` or `prefill_schedule_interval=1`) all remain intact. The benefit is that at any global step >= dp_size, at least one DP rank is in prefill-admission mode, so the worst-case wait before *some* rank can pick up a newly arriving request drops from up to `prefill_schedule_interval - 1` steps to `ceil(prefill_schedule_interval / dp_size)` steps — directly reducing median TTFT under bursty multi-turn agentic arrivals, and smoothing aggregate decode-token throughput because prefill bursts no longer align across ranks. This composes cleanly with (and is orthogonal to) any of the adaptive/urgency schemes in the existing proposals: they can adjust the phase-shifted cadence rather than a synchronized one.

**Novelty rationale.**

All four existing deep_research_proposals share the invariant that the throttle decision is identical across DP ranks and vary the decision only along the time dimension: find-0001 adds a per-request waiting-time urgency bypass, find-0006 adds a load-aware gate over prefill/decode token pressure with a queue-length fast path, find-0009 adds progress-informed EMAs on step latency plus a rank-imbalance guard, and find-0015 adds Gradient2 short/long-EWMA divergence over pressure signals. My proposal varies the decision along the *rank* dimension instead — introducing a per-rank phase offset in the cadence itself so that different DP ranks admit prefills on different steps by construction. It requires no adaptive signals, no EMAs, no cross-rank reductions, and no per-request urgency computation; it is a purely structural change to the base modulo gate. None of the four proposals mention rank staggering, phase offsets, or breaking the 'lockstep-identical across ranks' invariant, and each of them can be applied on top of this staggered base rather than the synchronized one — so this contributes a distinct, composable primitive rather than an alternative signal.

---

### 2. Skip DP prefill throttle when no decodes are running
- **Agent:** codex

**Detailed description.**

Change `DPEngineCoreProc._should_throttle_prefills` in `vllm/v1/engine/core.py:2086-2093` so the fixed interval only suppresses prefills while the local scheduler already has running decode work to protect. Before applying the modulo check, consult the scheduler's current request counts or equivalent local running-state signal; if `num_running == 0` and the waiting queue is non-empty, return `False` to admit a prefill immediately, regardless of `step_counter % prefill_schedule_interval`. Keep the existing `prefill_schedule_interval <= 1` and `step_counter == 0` behavior unchanged, and keep the modulo cadence once there is active decode work. This targets the common multi-turn agentic gap where a rank becomes idle between bursts but a newly arrived turn can still wait for the next deterministic cadence slot, producing avoidable TTFT with no TPOT benefit because there are no running decodes to shield. Add a focused DP scheduling test that advances `step_counter` to a throttled phase, drains running requests, enqueues a new waiting request, and verifies `_should_throttle_prefills` does not block admission; also keep an existing interval test with running requests to preserve configured cadence semantics.

**Novelty rationale.**

The deep_research proposals all add feedback or adaptive signals: request urgency, prefill/decode token pressure, progress/latency EMAs, or short/long EWMA divergence. This proposal is a narrower idle-rank fast path based on the absence of running decode work, not on wait-time urgency, KV pressure, token load, or latency trends. Agent A's proposal changes the modulo phase across DP ranks; this keeps the cadence synchronized and unchanged during active decode work, and only bypasses it when throttling cannot improve TPOT because there are no decodes running on that rank.

---
