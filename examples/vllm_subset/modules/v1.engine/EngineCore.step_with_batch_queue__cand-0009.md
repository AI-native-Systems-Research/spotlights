# EngineCore.step_with_batch_queue

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/core.py`](vllm/v1/engine/core.py) (lines 443–559)
- **Symbol:** `EngineCore.step_with_batch_queue`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0009`

## Description
Pipeline-parallel engine step policy that schedules new batches into a batch_queue, decides when to return without draining outputs, blocks on the oldest future when needed, updates the scheduler from model output, and handles deferred structured-output sampling.

## Current approach
Hard-coded fill-before-drain policy at lines 497-507: if a model-executing batch was enqueued, the queue is not full, and the oldest future is not done, return None to schedule more work. A note at lines 535-537 says current deferred handling favors TTFT over TPOT/throughput.

## Estimated impact explanation
For pipeline-parallel serving this policy trades pipeline bubbles against output latency. Better fill/drain decisions can reduce median TPOT while protecting TTFT for short agent turns, but it is inactive when batch_queue_size is one.

## Evolve rationale
The optimization unit is the scheduling/draining decision at lines 497-507 and the deferred-sampling placement at lines 535-557. Adaptive policies can choose fill vs drain based on queue depth, future age, prompt/decode mix, structured-output pressure, or request latency class. Correctness oracle: tests/v1/engine/test_engine_core.py and tests/distributed/test_pipeline_parallel.py must produce identical token streams, scheduler state transitions, and finish reasons for PP and non-PP comparisons.

## Deep research proposals

### 1. Phase-aware fill-vs-drain policy in step_with_batch_queue using prefill/decode mix
- **Finding:** `find-0004` — *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*
- **Source URL:** <https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the unconditional fill-before-drain branch in vllm/v1/engine/core.py:497-507 (and the deferred-sampling placement at lines 535-557) with a phase-aware policy that consults the prefill/decode composition of (a) the just-scheduled batch and (b) the oldest in-flight future before deciding to return None vs. block on the oldest future. Concretely: classify each scheduled batch as prefill-dominant (any new prompt tokens beyond decode-step-1) or decode-dominant using fields already present on SchedulerOutput / the scheduled requests; when the oldest in-flight future is decode-dominant and a new prefill-dominant batch was just enqueued, prefer to drain (block on the oldest future) so a long prefill cannot stall ongoing decodes and inflate median TPOT; when the oldest future is prefill-dominant and the just-scheduled batch is decode-dominant or empty, retain the current fill-through behavior to keep the pipeline busy and protect TTFT. Also use this classification to gate the deferred structured-output path at lines 535-557: if there is a queued prefill that would race against an outstanding decode-future on structured-output sampling, drain the decode-future first so its TPOT is not deferred. Keep batch_queue_size==1 paths unchanged. Validate via tests/v1/engine/test_engine_core.py and tests/distributed/test_pipeline_parallel.py for identical token streams and finish reasons under non-PP and PP runs, and add a microbenchmark with a multi-turn agentic mix to confirm median TPOT improvement at equal or better TTFT.

**Proposal rationale.**

The candidate's evolve_rationale explicitly lists "prompt/decode mix" as a signal an adaptive policy could use, and the in-code note at lines 535-537 already flags that the current policy favors TTFT over TPOT. DistServe's central observation is that prefill and decode have qualitatively different latency profiles and SLOs, which is why it provisions them on separate GPU pools coordinated by TTFT/TPOT targets. We cannot replicate physical disaggregation inside a single EngineCore, but the transferable idea — let the scheduler's choice depend on whether the work in flight is prefill or decode and on which SLO is at risk — maps directly onto the fill-vs-drain branch and the deferred-sampling placement that are this candidate's optimization unit. This addresses the candidate's stated gap (the hard-coded policy is TTFT-biased and ignores workload phase) and the caller objective of lowering median TPOT in a multi-turn agentic workload, where decode-phase tokens dominate steady-state latency.

---

### 2. Run scheduler one batch ahead in step_with_batch_queue to overlap CPU scheduling with GPU execution
- **Finding:** `find-0010` — *SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs*
- **Source URL:** <https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Restructure EngineCore.step_with_batch_queue (vllm/v1/engine/core.py:443-559) so that the next batch's CPU-side preparation — scheduler.schedule(), grammar bitmask computation for non-deferred structured output, and any per-request metadata building — is performed before the blocking future.result() at line 521, while the GPU is still executing the oldest in-flight batch. Concretely: (1) When the queue has at least one in-flight future and there is room (len(batch_queue) < batch_queue_size), always call self.scheduler.schedule() and dispatch self.model_executor.execute_model(..., non_block=True) before deciding whether to block on batch_queue[-1]; this generalizes the current fill-before-drain branch at lines 497-507 from "only when the oldest future is not done" into a deliberate run-ahead. (2) Move the deferred-sampling work currently performed at lines 535-557 (after future.result()) into a pre-block path so that get_grammar_bitmask and sample_tokens dispatch happen while the prior batch is still executing, eliminating the TTFT-favoring CPU bubble noted in the comment at lines 535-537. (3) For pure PP=1 (batch_queue_size == 1), enable a single-slot variant where the next scheduler.schedule() call is launched immediately after dispatch and its result cached so the next step_with_batch_queue invocation can skip the schedule() cost on the critical path. Preserve correctness by keeping update_from_output ordering unchanged and gating the run-ahead behind a boolean (e.g., VLLM_SCHEDULE_AHEAD) so the existing TTFT-favoring policy remains the default until the new path is validated against tests/v1/engine/test_engine_core.py and tests/distributed/test_pipeline_parallel.py.

**Proposal rationale.**

The candidate explicitly trades pipeline bubbles against output latency at lines 497-507 and acknowledges at lines 535-537 that the current deferred-sampling placement favors TTFT over TPOT/throughput — exactly the CPU-side bubble that SGLang's zero-overhead scheduler addresses by running one batch ahead. The caller objective (reduce median TPOT for multi-turn agentic workloads) is the regime where this CPU/GPU overlap pays off most: short decode steps make scheduler/grammar/metadata CPU time a larger fraction of step latency, and the multi-turn pattern means structured-output and prefix-aware scheduling are common. The finding's contribution is concrete and transferable — it pinpoints which CPU work to overlap (schedule() + metadata prep) and where the bubble appears (between batches) — which maps directly onto the schedule/dispatch/block sequence in step_with_batch_queue rather than being merely topically adjacent.

---

## Agent proposals

### 1. Per-request TPOT-deadline-driven drain in step_with_batch_queue
- **Agent:** claude

**Detailed description.**

Augment the fill-vs-drain branch at vllm/v1/engine/core.py:497-507 with a per-request deadline check that runs before the `return None, True` fast-path. Maintain on the scheduler a cheap `last_token_emit_time[req_id]` (already implicitly available via metrics/output timestamps) and a `target_tpot[req_id]` derived from `SamplingParams` (or a global default from VllmConfig). Inside `step_with_batch_queue`, after appending the new future to `batch_queue` (line 499) but before the early-return at line 507, scan the requests carried by `batch_queue[-1][1]` (the oldest in-flight `SchedulerOutput`) and compute, in O(num_decode_requests_in_oldest_future), `min_slack = min(target_tpot[r] - (now - last_token_emit_time[r]))` for the decode-phase requests in that batch. If `min_slack` is below a threshold tied to the EWMA-tracked cost of the next `scheduler.schedule()` + `execute_model` dispatch (a value the engine can keep in a single float updated each step), suppress the early return and fall through to the blocking `future.result()` at line 521 — i.e., drain rather than fill. Mirror this gate in the deferred structured-output path at lines 535-557: if any deferred-decode request is already past its slack budget, prefer to drain the existing future before computing the deferred grammar bitmask + `sample_tokens` for the new prefill, since the bitmask path adds CPU latency to a request that is already TPOT-violating. Keep the policy gated behind a flag (e.g., VLLM_TPOT_DEADLINE_DRAIN) and a default `target_tpot` of +inf so behavior is unchanged unless the user sets per-request or global TPOT targets. Validate via tests/v1/engine/test_engine_core.py and tests/distributed/test_pipeline_parallel.py for unchanged token streams and finish reasons under the default config, and add a microbenchmark that mixes long-prefill turns into a steady decode stream and asserts that p50/p99 TPOT for the decode stream is bounded by the configured target while TTFT for new prefills regresses by less than a small bound.

**Novelty rationale.**

Neither existing proposal uses per-request temporal deadline pressure as the drain signal. find-0004 decides fill-vs-drain from *workload composition* (prefill-dominant vs decode-dominant batches) — a structural property of what is in flight, not how long any individual request has been waiting. find-0010 *always* overlaps CPU scheduler/grammar work with GPU execution and never opts to drain on TPOT grounds; it eliminates a CPU bubble but leaves the policy decision unchanged when a long prefill is enqueued behind aging decodes whose clients are perceiving stalls. This proposal adds the orthogonal axis the candidate's evolve_rationale calls out — `future age` and `request latency class` — and enforces a per-request SLO directly at the fill-vs-drain branch, so that even when the workload-mix heuristic of find-0004 says "fill" and the run-ahead heuristic of find-0010 is active, an aging streaming client still gets its next token before the engine adopts another prefill, which is precisely the failure mode the caller's multi-turn agentic median-TPOT objective is sensitive to.

---

### 2. Drain ready queue head before scheduling a replacement batch
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/core.py:443-559`, add a non-blocking ready-drain fast path before the current `self.scheduler.has_requests()` scheduling block. If `batch_queue` is non-empty and the oldest queued future `batch_queue[-1][0].done()` is already complete, pop it, call `future.result()` without blocking, run `_process_aborts_queue()` and `self.scheduler.update_from_output(...)`, and retain those `engine_core_outputs` for this invocation. Then, if `len(batch_queue) < self.batch_queue_size` and `self.scheduler.has_requests()`, run the existing schedule/execute/sample path once to refill the queue using the updated scheduler state. When this path has ready outputs, skip the `return None, True` fill-through fast path so tokens and finish events are returned immediately; when no oldest future is ready, preserve the existing fill-before-drain behavior. Factor the shared pop/result/update logic into a small helper to avoid duplicating the error handling around `future.result()`. Validate with `tests/v1/engine/test_engine_core.py` and `tests/distributed/test_pipeline_parallel.py`, and add a targeted concurrent-batches test where the oldest future is already done while new work is schedulable, asserting that the ready output is returned in that call and the queue is still refilled afterward.

**Novelty rationale.**

This is not the phase-aware prefill/decode policy from find-0004, because it does not classify workload phase or choose drain based on prompt/decode composition. It is not the schedule-ahead proposal from find-0010; it does the opposite only when draining is already non-blocking, first refreshing scheduler state and client-visible outputs before launching replacement work. It also differs from Claude's TPOT-deadline proposal, which drains based on per-request slack and may block; this proposal uses the already-completed status of the FIFO queue head to remove an avoidable CPU-side delay and stale-scheduler-state scheduling decision.

---
