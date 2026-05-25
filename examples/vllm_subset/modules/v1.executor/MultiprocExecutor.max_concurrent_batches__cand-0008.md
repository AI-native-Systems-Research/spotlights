# MultiprocExecutor.max_concurrent_batches

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/multiproc_executor.py`](vllm/v1/executor/multiproc_executor.py) (lines 474–478)
- **Symbol:** `MultiprocExecutor.max_concurrent_batches`
- **Kind:** config_block
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0008`

## Description
MQ executor batch-pipeline depth heuristic. It returns 2 for async scheduling without pipeline parallelism and otherwise returns pipeline_parallel_size; RayExecutorV2 inherits this policy.

## Current approach
A cached property reads pp_size and returns the hard-coded expression 2 if pp_size <= 1 and async_scheduling else pp_size. EngineCore uses this value as batch_queue_size, controlling how many scheduler batches may be in flight.

## Estimated impact explanation
Pipeline depth directly trades overlap against queueing and KV pressure. For multi-turn agentic workloads, improving this heuristic can reduce median TPOT without regressing TTFT under bursty decode traffic.

## Evolve rationale
The policy-defining constant is the literal 2 at line 478. It can be evolved into a workload-aware or memory-aware depth using batch size, KV-cache slack, observed decode latency, or queue occupancy, with hysteresis to avoid oscillation. Correctness oracle: functional generation outputs should remain unchanged; scheduler batch-queue tests plus serving benchmarks should validate ordering, memory bounds, TTFT, and TPOT.

## Deep research proposals

### 1. Condition batch-queue depth on chunked-prefill uniformity
- **Finding:** `find-0005` — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- **Source URL:** <https://www.usenix.org/conference/osdi24/presentation/agrawal>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Change MultiprocExecutor.max_concurrent_batches at vllm/v1/executor/multiproc_executor.py:474-478 from the literal `2 if pp_size <= 1 and async_scheduling else pp_size` to a small policy that takes chunked-prefill state into account. When async scheduling is enabled and pp_size <= 1, inspect the scheduler config: if chunked prefill is active (long_prefill_token_threshold / max_num_batched_tokens makes per-iteration shapes near-uniform), allow a deeper queue (e.g. 3) so the executor can keep the pipeline filled with mixed chunk+decode batches, matching Sarathi-Serve's stall-free batching ideal. When chunked prefill is disabled or a long unsplit prefill is in flight, fall back to 2 to avoid letting a heavy prefill batch sit ahead of decodes in the queue and inflate TPOT. Keep the value as a cached_property but compute it from already-available VllmConfig fields (scheduler_config.chunked_prefill_enabled, max_num_batched_tokens, long_prefill_token_threshold) so RayExecutorV2 inherits the same logic. No new tunables, no runtime feedback loop in this step — just a config-derived constant that aligns executor depth with the workload uniformity that chunked prefill provides.

**Proposal rationale.**

The candidate's hard-coded depth of 2 was chosen to bound queueing under arbitrary batch shapes, where a queued prefill behind a decode batch can spike TPOT. Sarathi-Serve's central insight is that chunked prefill makes per-iteration work near-uniform, removing exactly the head-of-line risk that motivated depth=2. That makes a slightly deeper queue safe and beneficial precisely when chunked prefill is on — improving pipeline fill and median TPOT under the multi-turn agentic workload — while preserving the conservative depth=2 when chunked prefill is off. The change is local, reuses existing config, and is gated by an already-meaningful flag, so it is a concrete and transferable application of the finding to this specific knob.

---

### 2. Replace fixed batch-queue depth with backpressure-driven adaptive cap
- **Finding:** `find-0011` — *Pattern: Using ray.wait to limit the number of pending tasks*
- **Source URL:** <https://docs.ray.io/en/master/ray-core/patterns/limit-pending-tasks.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py at MultiprocExecutor.max_concurrent_batches (lines 474-478), replace the hard-coded literal 2 (and the static pipeline_parallel_size fallback) with an adaptive cap derived from observed in-flight batch completion, modeled on Ray's pending-task backpressure pattern. Concretely: track the number of submitted-but-not-yet-completed scheduler batches (the analog of ray.wait's pending set) and expose a depth that grows when completions consistently outpace submissions and shrinks when in-flight batches accumulate beyond a threshold. The property would return a cap computed from: (a) a baseline floor (2 for async scheduling, pp_size otherwise, preserving current behavior on cold start), (b) a ceiling tied to KV-cache slack and pp_size, and (c) a smoothed observed completion-rate signal with hysteresis to avoid oscillation. Because RayExecutorV2 inherits this property, the same backpressure logic transfers directly to the Ray executor, where ray.wait-based pacing is the canonical implementation. EngineCore consumes this value as batch_queue_size, so increasing it under low pressure widens pipeline overlap and lowers TPOT, while the cap prevents stale pending work from inflating TTFT under bursty multi-turn traffic.

**Proposal rationale.**

The candidate's current_approach is a static literal that does not react to workload state; the evolve_rationale explicitly calls out queue occupancy and observed decode latency as signals for a workload-aware depth. The Ray docs pattern addresses exactly this gap: it prescribes capping in-flight work at the useful concurrency window using observed completion as the backpressure signal, which is transferable to MultiprocExecutor's batch pipeline (and inherited by RayExecutorV2 where ray.wait is the native primitive). For the stated multi-turn agentic objective (reduce median TTFT and TPOT), a backpressure-bounded cap is plausibly better than a fixed 2 because it lets pp=1 async scheduling exploit additional overlap when decode latency is low while clamping depth back when pending batches build up, protecting tail TTFT.

---

## Agent proposals

### 1. Scale batch-queue depth with CUDA-graph capture state
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py at MultiprocExecutor.max_concurrent_batches (lines 474-478), replace the literal `2 if pp_size <= 1 and async_scheduling else pp_size` with a config-derived value that also looks at whether CUDA graphs will be captured for the decode path. Concretely: keep pp_size as the dominant signal when pp>1, but when pp_size <= 1 and async_scheduling is on, return 3 if CUDA graphs are enabled (i.e. not enforce_eager and the compilation/cudagraph config will capture decode shapes) and 2 otherwise. The rationale is structural, not workload-adaptive: when CUDA graphs are captured, each decode step has near-constant, low GPU latency, so the CPU-side scheduler+sampler+output-handling work has a smaller window to hide behind a single in-flight batch — a queue of 3 lets the executor keep the GPU saturated without enlarging the head-of-line risk that motivated depth=2 (because graph-captured decode steps are precisely the regime where per-step time variance is minimal). When CUDA graphs are off (eager mode, or explicitly disabled), per-step latency is higher and more variable, so retaining depth=2 preserves today's TPOT bound. The check uses already-available VllmConfig fields (compilation_config / model_config.enforce_eager) and stays a cached_property, so RayExecutorV2 inherits the same logic. This is a single config-derived constant — no runtime feedback, no new tunables, no interaction with prefill shape — and it specifically targets the multi-turn agentic objective where decode steps dominate and CUDA graphs are typically active.

**Novelty rationale.**

Neither existing proposal uses CUDA-graph capture state as the gating signal. find-0005 keys depth off chunked-prefill configuration (a prefill-uniformity argument), and find-0011 introduces a runtime backpressure loop driven by observed completion rate. This proposal is orthogonal: it argues depth should rise when per-step *decode* latency is small and uniform because graphs are captured — a compile-time-known property of the engine's compilation_config, not of scheduler config and not of runtime queue occupancy. It also stays static (no feedback loop, no hysteresis), unlike find-0011, and is independent of whether chunked prefill is on, unlike find-0005, so the two existing proposals could be adopted alongside it without conflict.

---

### 2. Normalize queue depth for speculative decode lookahead
- **Agent:** codex

**Detailed description.**

In `vllm/v1/executor/multiproc_executor.py` at `MultiprocExecutor.max_concurrent_batches`, keep the existing `pp_size` requirement for pipeline parallelism, but make the `pp_size <= 1 && async_scheduling` path account for `self.speculative_config`. Treat the queue limit as an approximate outstanding-output-token budget rather than a raw scheduler-batch count: when speculative decoding is enabled with a large lookahead, e.g. `num_speculative_tokens >= 4` or `max_num_new_slots_for_drafting > 0` with a large reserved-slot fraction, return `1` or otherwise clamp to the current `2`; when speculation is absent or shallow, keep the current `2`. This prevents two queued scheduler batches from representing many unobserved decode tokens per request, which can inflate TTFT, streaming latency, abort responsiveness, and KV/slot pressure in multi-turn agentic traffic. Add focused tests for async scheduling with speculative config to assert the default non-spec path is unchanged and the high-lookahead path uses the reduced depth; validate with existing async spec-decode correctness tests plus serving benchmarks that report TTFT/TPOT and cancellation latency.

**Novelty rationale.**

This is not covered by the chunked-prefill proposal, which keys on prefill shape uniformity, nor by the adaptive backpressure proposal, which uses runtime completion/KV-slack signals. It is also different from Claude's CUDA-graph proposal: CUDA graphs reason about per-step decode latency, while this proposal reasons about how many future output tokens a single queued scheduler batch can embody under speculative decoding. The signal is speculative lookahead/reserved-slot configuration, and the action is to reduce or clamp queue depth for high-lookahead speculation rather than increase it for graph-captured or chunked-prefill regimes.

---
