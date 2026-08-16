# EngineCore.step

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/core.py`](vllm/v1/engine/core.py) (lines 580–610)
- **Symbol:** `EngineCore.step`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0015`

## Description
Default engine-core scheduler/model step that schedules work, overlaps grammar-mask construction with model execution, samples tokens, processes aborts, and updates the scheduler from model output.

## Current approach
Calls scheduler.schedule, launches execute_model(non_block=True), immediately computes get_grammar_bitmask, then blocks on future.result(); if the model runner did not sample internally, it calls sample_tokens, processes aborts, and calls scheduler.update_from_output.

## Estimated impact explanation
Model execution dominates many steps, but this method is paid once per decode iteration; reducing host-side blocking or wasted per-step work can lower median TPOT and TTFT for short multi-turn agentic requests.

## Evolve rationale
This is the non-pipeline scheduling boundary for every active engine step and owns the ordering between CPU scheduler work, GPU model execution, grammar-mask preparation, sampling, abort handling, and scheduler update. Candidate changes include more adaptive sampling placement, avoiding unnecessary grammar-mask work on empty/pooled steps, earlier abort handling, or tighter overlap with model execution. Correctness oracle: tests/v1/engine/test_engine_core.py, tests/v1/engine/test_abort_final_step.py, structured-output tests, and vllm bench should preserve completion, abort semantics, token parity, and scheduler stats while measuring TTFT/TPOT.

## Deep research proposals

### 1. Overlap abort processing with GPU execute in EngineCore.step using profiling-informed CPU placement
- **Finding:** `find-vllm_v1_engine-0009` — *Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs*
- **Source URL:** <https://arxiv.gg/abs/2506.03296>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/engine/core.py (EngineCore.step, lines 580-610), the CPU-side sequence is: schedule -> launch execute_model(non_block=True) -> compute grammar bitmask -> block on future.result() -> _process_aborts_queue -> update_from_output. Only the grammar-bitmask computation currently overlaps with GPU execution; abort-queue processing runs strictly after future.result() even though it is pure CPU work that does not depend on model output. Apply the finding's profiling-informed dispatch idea by (1) measuring, per step, the CPU cost of get_grammar_bitmask and _process_aborts_queue and the observed GPU-execute wall time, and (2) when there is predicted slack between the launched execute_model future and the CPU work already scheduled before future.result(), moving _process_aborts_queue (or a cheap prefix of it, e.g. draining self._aborts_queue snapshot without touching scheduler state that update_from_output will read) up to run before the blocking future.result() call. Concretely: after computing grammar_output, if the running EMA of CPU(abort_queue) < remaining predicted GPU time, invoke _process_aborts_queue inside the same overlap window; otherwise keep the current post-block placement. The decision is a lightweight EMA compare (no per-step profiling overhead beyond timestamps), matching the finding's guidance to 'limit scheduling overhead' while dispatching to 'maximize overlap.' Semantics remain unchanged because aborts observed at time T are still applied before update_from_output on the same iteration; the only change is when along the CPU timeline the drain happens. Guard behind a config flag and validate against tests/v1/engine/test_engine_core.py, tests/v1/engine/test_abort_final_step.py, and structured-output tests to preserve abort ordering and scheduler-stat parity.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'earlier abort handling' and 'tighter overlap with model execution' as target improvements, and the estimated_impact notes that reducing host-side blocking can lower median TPOT — the exact regime the finding addresses. Today's step() overlaps only grammar-mask CPU work with GPU execute; abort processing sits on the critical path after future.result(). The finding's core transferable idea is to predict CPU/GPU subtask times and dispatch CPU work into the GPU-idle window rather than serialising it. Applied here, that means moving abort drain into the pre-block overlap window when it fits, which directly shortens the per-step CPU tail for multi-turn agentic workloads (where aborts are common as tool-use turns are cancelled). The change is scoped, uses observed-progress signals (EMA timings) rather than fixed heuristics, and preserves the same iteration-level ordering of aborts relative to update_from_output.

---

## Agent proposals

### 1. Speculatively pre-schedule the next step's CPU work during the current GPU execute window
- **Agent:** claude

**Detailed description.**

In vllm/v1/engine/core.py EngineCore.step (lines 580-610), the CPU thread currently blocks on future.result() with only grammar-bitmask construction filling the GPU-execute window. When multi-turn agentic decode dominates (small batches, short GPU steps), the CPU has meaningful idle time that could be used to pre-compute the *next* step's scheduling decision speculatively. Concretely: after launching execute_model(non_block=True) and computing get_grammar_bitmask, invoke a new scheduler.speculative_prepare(scheduler_output) that (a) snapshots the current running/waiting queues, (b) applies the projected token-slot deltas that update_from_output would apply if the in-flight model_output produces exactly one token per running request with no stops/aborts, and (c) runs the token-budget and block-allocator dry-run of scheduler.schedule() against that projected state, caching the resulting *candidate* SchedulerOutput and grammar prefetch. When future.result() returns, the fast path validates the projection (no aborts occurred, no requests hit stop-tokens, no preemption triggered, no new arrivals since the snapshot) and, if valid, reuses the cached scheduler_output and grammar-bitmask on the *next* call to step(), skipping the schedule() + get_grammar_bitmask() cost entirely on that iteration. On mismatch, discard and fall back to the normal path. The projection is cheap because in steady-state decode most iterations advance every running request by exactly one token; validation is O(#running + #aborts + #new_arrivals). Cache is invalidated on any add_request/abort/preemption between steps. Gate behind an engine_config flag (e.g. speculative_schedule_prefetch), and validate via tests/v1/engine/test_engine_core.py, tests/v1/engine/test_abort_final_step.py, and structured-output tests to confirm scheduler-stat parity, token parity, and abort ordering under both projection-hit and projection-miss regimes; measure TPOT/TTFT deltas via vllm bench serve with multi-turn agentic traces.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_engine-0009) targets *intra-step* CPU/GPU overlap by moving _process_aborts_queue into the current step's GPU-idle window using EMA-based dispatch. This proposal is orthogonal: it introduces *inter-step* speculative pre-scheduling that reuses the current step's GPU-execute window to prepare the *next* step's scheduler_output and grammar bitmask, so that step N+1 skips schedule() and get_grammar_bitmask() entirely on the projection-hit path. The mechanism (state projection + validate-or-discard), the work being moved (schedule() and get_grammar_bitmask, not abort drain), and the target savings (eliminating scheduling latency on the next iteration's critical path, not shortening this iteration's post-block tail) are all distinct from the abort-drain overlap proposal.

---

### 2. Run deferred sampling asynchronously and overlap its wait with post-forward CPU work
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/core.py` `EngineCore.step` lines 580-610, split the `model_output is None` path so the engine launches `self.model_executor.sample_tokens(grammar_output, non_block=True)` instead of calling the blocking form immediately. When `execute_model(..., non_block=True)` returns `None`, start the sampling future, then run CPU work that must occur before `scheduler.update_from_output`, primarily `_process_aborts_queue()` and any cheap iteration-detail/stat preparation that does not inspect sampled tokens, before blocking on `sample_future.result()`. Keep the current path unchanged when `execute_model` returns a `ModelRunnerOutput` directly. This mirrors the executor API already used by `step_with_batch_queue` and targets standard GPU-model-runner steps where forward produces logits and sampling is a second worker call. Validation should cover `tests/v1/engine/test_engine_core.py`, `tests/v1/engine/test_abort_final_step.py`, structured-output tests, and a `vllm bench` run comparing TPOT for short decode-heavy agentic traces with and without the new async-sampling path.

**Novelty rationale.**

The deep-research proposal moves abort processing into the `execute_model` GPU-forward window before `future.result()`. This proposal instead targets the separate deferred-sampling window after `execute_model` has completed and only when `model_output is None`, using the existing `sample_tokens(..., non_block=True)` executor path. Agent A's proposal speculatively prepares the next scheduler step and reuses cached schedule/grammar results; this proposal does not predict or cache next-step scheduler state and does not alter scheduling decisions. It is a distinct intra-step change focused on reducing the blocking tail of external sampling.

---
