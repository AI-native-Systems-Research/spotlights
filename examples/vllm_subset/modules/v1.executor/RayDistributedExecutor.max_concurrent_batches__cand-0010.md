# RayDistributedExecutor.max_concurrent_batches

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/ray_executor.py`](vllm/v1/executor/ray_executor.py) (lines 99–105)
- **Symbol:** `RayDistributedExecutor.max_concurrent_batches`
- **Kind:** config_block
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0010`

## Description
Legacy Ray executor batch-pipeline depth heuristic. It chooses the EngineCore batch queue size for Ray compiled-DAG execution.

## Current approach
The property returns 2 for async scheduling when pipeline_parallel_size <= 1, otherwise returns pipeline_parallel_size. This mirrors the multiprocess heuristic but applies to RayDistributedExecutor rather than MQ-based RayExecutorV2.

## Estimated impact explanation
This value controls how much Ray work the engine pipelines. Better tuning can reduce decode bubbles or avoid excess queueing, directly moving median TPOT for Ray async workloads and protecting TTFT during bursty multi-turn traffic.

## Evolve rationale
The policy-defining constant is the literal 2 at line 105. A candidate evolution can make Ray compiled-DAG concurrency depend on observed ray.get latency, queue occupancy, batch size, connector mode, or KV-cache headroom, with a functional oracle that outputs and ordering remain unchanged. Correctness oracle: Ray executor tests and serving benchmarks should validate identical generation outputs while measuring TTFT/TPOT and memory pressure across async and PP settings.

## Deep research proposals

### 1. Replace static max_concurrent_batches with ray.wait-based backpressure
- **Finding:** `find-0011` — *Pattern: Using ray.wait to limit the number of pending tasks*
- **Source URL:** <https://docs.ray.io/en/master/ray-core/patterns/limit-pending-tasks.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

At vllm/v1/executor/ray_executor.py:99-105, replace the static heuristic (return 2 for async scheduling when pipeline_parallel_size <= 1, else pipeline_parallel_size) with a dynamic backpressure policy modeled on Ray's documented limit-pending-tasks pattern. Rather than fixing the EngineCore batch queue depth at construction time, treat max_concurrent_batches as an upper bound and gate new compiled-DAG submissions on ray.wait over the set of in-flight ObjectRefs, only releasing a new batch once the count of unresolved refs falls below a target window. The window itself can be initialized to the existing values (2 for async PP=1; pipeline_parallel_size otherwise) and adapted from observed signals already available to the executor: ray.get latency on prior batches, current batch size, and pipeline-parallel depth. The property's contract (an integer cap consumed by EngineCore) is preserved; the change is in how that cap is honored at submission time inside the Ray compiled-DAG driver path so that bursty multi-turn traffic does not accumulate stale pending work behind the queue. Outputs and submission ordering are unchanged, satisfying the candidate's functional oracle.

**Proposal rationale.**

The candidate's current_approach pins concurrency to a literal 2, which under multi-turn agentic bursts can either under-pipeline (idle decode bubbles raising TPOT) or over-queue (stale pending tasks raising TTFT for newly arriving turns). The finding directly addresses this gap: Ray's limit-pending-tasks pattern prescribes ray.wait-based backpressure precisely to cap in-flight work at the useful concurrency window without fixed pre-allocation. Applying it to RayDistributedExecutor's compiled-DAG submission path turns the static heuristic into an adaptive cap aligned with the caller's stated objective of reducing median TTFT and TPOT under multi-turn agentic workloads, while remaining a minimal, transferable change at the policy-defining line.

---

## Agent proposals

### 1. Scale max_concurrent_batches by KV-cache headroom and connector mode
- **Agent:** claude

**Detailed description.**

At vllm/v1/executor/ray_executor.py:99-105, replace the static `return 2` (and the `pipeline_parallel_size` branch) with a depth chosen from two signals the existing proposal does not use: (1) free KV-cache headroom reported by the worker's KVCacheManager / GPU block pool, and (2) whether a KV connector (offload, P/D disagg) is configured on the executor. Concretely: compute a base depth equal to today's heuristic, then multiply by a headroom factor `f_kv = clamp(free_blocks / total_blocks, 0.5, 1.0)` so that when the cache is near saturation the executor advertises a smaller cap to EngineCore (reducing preemption thrash that inflates TPOT under multi-turn bursts where prior turns retain prefix blocks), and add a connector bonus `+1` when `vllm_config.kv_transfer_config` is set with a non-trivial connector role, because compiled-DAG submissions overlap with asynchronous KV transfers and therefore tolerate (and benefit from) one extra in-flight batch. The property still returns a single integer consumed by EngineCore, preserving the contract; correctness oracle is unchanged because only the integer cap varies, not output computation or submission ordering. Cache the chosen value at executor construction (or refresh on a coarse cadence via a worker RPC) to keep the property cheap; fall back to today's literal when the signals are unavailable so behavior is identical on configs without a KV connector and at fresh cache.

**Novelty rationale.**

The existing deep_research_proposal (find-0011) adapts the cap via ray.wait-based backpressure driven by ray.get latency, batch size, and pipeline-parallel depth. It explicitly does not consult KV-cache headroom or kv_transfer_config / connector mode — both of which the candidate's evolve_rationale calls out as legitimate signals. This proposal targets exactly those untapped signals and is orthogonal: it changes the integer the property returns, while find-0011 changes how that integer is honored at submission time. The two could even compose.

---

### 2. Make the Ray async-depth branch reachable and tested
- **Agent:** codex

**Detailed description.**

At `vllm/v1/executor/ray_executor.py:99-105`, first make the `return 2` async-scheduling branch actually reachable for `RayDistributedExecutor`. The class currently inherits `Executor.supports_async_scheduling()` as false, while `UniProcExecutor` and `MultiprocExecutor` override it, so vLLM config validation can disable async scheduling before this property is used. Add a `supports_async_scheduling()` classmethod for the Ray compiled-DAG executor after covering the non-blocking `execute_model` / `sample_tokens` path with a Ray regression test: construct Ray backend configs with `async_scheduling=True`, assert `max_concurrent_batches == 2` for PP=1 and `== pipeline_parallel_size` for PP>1, and run a small deterministic generation comparison against non-async. This turns the existing concurrency heuristic into an exercised behavior before layering on more adaptive policies.

**Novelty rationale.**

The deep-research proposal assumes the cap is already live and changes submission backpressure with `ray.wait`; Claude's proposal also assumes the cap is live and changes the returned integer using KV-cache and connector signals. Neither addresses the executor capability gate that can prevent `scheduler_config.async_scheduling` from being enabled for this class at all. This proposal is about branch reachability and test coverage for `RayDistributedExecutor.max_concurrent_batches`, not another latency signal, KV signal, or backpressure mechanism.

---
