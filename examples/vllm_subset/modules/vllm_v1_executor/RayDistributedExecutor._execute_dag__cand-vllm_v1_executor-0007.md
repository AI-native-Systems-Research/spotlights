# RayDistributedExecutor._execute_dag

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/ray_executor.py`](vllm/v1/executor/ray_executor.py) (lines 434–468)
- **Symbol:** `RayDistributedExecutor._execute_dag`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_executor-0007`

## Description
Per-step Ray compiled-DAG execution boundary: submits scheduler output to the compiled DAG, blocks or returns a FutureWrapper, detaches Ray SHM views, and aggregates multi-worker outputs when a KV connector is present.

## Current approach
The method lazily compiles the DAG, calls self.forward_dag.execute, then either blocks on refs[0].get() and detaches one output, returns FutureWrapper(refs[0]), or ray.gets all refs for connector mode, detaches outputs sequentially, and calls kv_output_aggregator.aggregate.

## Estimated impact explanation
This method is the driver-side per-token wait point for the Ray compiled-DAG backend. Any overlap or reduction in get-detach-aggregate wall time directly reduces median TPOT, especially in multi-node or connector-enabled Ray deployments.

## Evolve rationale
The specific constructs are forward_dag.execute at line 444, refs[0].get/ray.get at lines 450 and 462, and the sequential detach loop at lines 463-464. Headroom includes overlapping detach work with Ray ref readiness in connector mode, moving single-output detach into the future path consistently, and pipelining Python-side output post-processing without changing DAG execution semantics. Correctness oracle: for a fixed SchedulerOutput and GrammarOutput, the returned ModelRunnerOutput or KV aggregation result is byte/value-equivalent to the current synchronous path, output order into KVOutputAggregator is preserved, and existing Ray PP/TP integration tests still pass.

## Deep research proposals

### 1. Use ray.wait to overlap detach and KV aggregation with worker readiness in connector mode
- **Finding:** `find-vllm_v1_executor-0005` — *ray.wait*
- **Source URL:** <https://docs.ray.io/en/latest/ray-core/api/doc/ray.wait.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In RayDistributedExecutor._execute_dag (vllm/v1/executor/ray_executor.py:434-468), replace the blocking `outputs = ray.get(refs)` followed by the sequential `for output in outputs: detach_zero_copy_from_model_runner_output(output)` loop on lines 462-464 with a readiness-driven loop built on `ray.wait`. Concretely: maintain an index map `ref -> rank`, then repeatedly call `ready, refs = ray.wait(refs, num_returns=1)` (or a larger `num_returns` when many refs are outstanding); for each newly-ready ref, immediately `ray.get` it and run `detach_zero_copy_from_model_runner_output` on the result, placing the detached output into a pre-sized list at its original rank. When the loop drains, call `self.kv_output_aggregator.aggregate(ordered_outputs)` exactly as today. This preserves the rank ordering that KVOutputAggregator relies on and leaves the single-worker path (lines 449-456) and the non-blocking FutureWrapper path (line 468) untouched. The FutureWrapper path in the connector branch could later adopt the same readiness-driven drain internally, but that is out of scope for this proposal.

**Proposal rationale.**

The candidate's connector-mode blocking path is a strict pipeline: wait for the slowest worker via `ray.get(refs)`, then serially detach each output, then aggregate. Because Ray compiled-DAG workers do not necessarily finish in rank order, the wall-clock cost of the detach loop is added on top of the slowest worker's readiness. `ray.wait` returns the subset of refs already ready, which is exactly the primitive needed to overlap the Python-side zero-copy detach (and any aggregator pre-processing) with the tail workers still in flight — directly reducing the per-step driver-side wait time called out in the candidate's `estimated_impact_explanation`. It changes only scheduling of already-computed work, so byte/value equivalence of the aggregated ModelRunnerOutput and the ordering guarantees into KVOutputAggregator are preserved, matching the candidate's correctness oracle.

---

### 2. Set explicit _max_inflight_executions on the compiled DAG to bound nonblocking step queueing
- **Finding:** `find-vllm_v1_executor-0010` — *ray.dag.input_node.InputNode.experimental_compile*
- **Source URL:** <https://docs.ray.io/en/latest/ray-core/compiled-graph/doc/ray.dag.input_node.InputNode.experimental_compile.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/ray_executor.py, extend _compiled_ray_dag (invoked lazily from _execute_dag at line 442) so that the final forward_dag.experimental_compile(...) call at line 617 additionally passes _max_inflight_executions (and, where supported, a matching buffered-results bound) sized to the async scheduler's max concurrent steps rather than relying on Ray Compiled Graph defaults. Thread the limit through as a config-derived value (e.g. from the parallel/scheduler config or a new envs.VLLM_RAY_CDAG_MAX_INFLIGHT knob, with a conservative default that reflects the number of in-flight FutureWrapper results the driver can hold across execute_model/sample_tokens/_execute_dag). No changes are needed inside _execute_dag itself at lines 444, 450, 456, 462 or the detach loop 463-464; the correctness oracle is preserved because DAG semantics, output order into KVOutputAggregator, and blocking-vs-FutureWrapper return paths are unchanged. In the connector path (lines 460-465) and the PP FutureWrapper path (line 456), the bound now imposes explicit backpressure so that when several nonblocking model futures accumulate in a multi-turn agentic loop, the driver stalls at .execute rather than at an implicit Ray-internal queue.

**Proposal rationale.**

The finding points at a concrete parameter of experimental_compile (_max_inflight_executions) documented as 'The maximum number of in-flight executions', and this candidate's compiled-DAG is created via forward_dag.experimental_compile at line 617 and driven step-by-step from _execute_dag. The candidate specifically supports nonblocking execution (FutureWrapper at line 456 and 468), which in multi-turn agentic workloads can allow multiple model steps to pile up behind refs[0].get / ray.get. Setting an explicit inflight bound aligned with the async scheduler gives _execute_dag a well-defined backpressure boundary at line 444's forward_dag.execute call, which is on the driver's per-token critical path and thus directly relevant to median TPOT under the stated workload. It is a small, transferable configuration change grounded in the cited Ray Compiled Graph API rather than a refactor of DAG semantics.

---

## Agent proposals

### 1. Detach only the output_rank output in connector-mode blocking path
- **Agent:** claude

**Detailed description.**

In RayDistributedExecutor._execute_dag (vllm/v1/executor/ray_executor.py:460-465), replace the unconditional per-rank detach loop `for output in outputs: detach_zero_copy_from_model_runner_output(output)` with a single call `detach_zero_copy_from_model_runner_output(outputs[output_rank])`, where `output_rank` matches the value KVOutputAggregator.aggregate uses to select the returned output (currently the default 0 — see vllm/distributed/kv_transfer/kv_connector/utils.py:161). Rationale grounded in the aggregator: `aggregate` only iterates the non-output-rank ModelRunnerOutputs to read small `kv_connector_output` metadata (finished_sending/recving req_id sets, kv_connector_stats, kv_cache_events, invalid_block_ids) and then discards them when the local `outputs` list goes out of scope at the end of the call. It never retains their `logprobs` or `routed_experts` — those heavy NumPy fields are only read from `outputs[output_rank]`. The detach helper exists (per its own docstring in vllm/v1/executor/ray_utils.py:198) specifically because retaining SHM-backed NumPy views across scheduler iterations can stall the Ray channel. Non-output-rank outputs are not retained across iterations, so their read-only SHM views are released within the same `_execute_dag` call and do not need the `arr.copy()` in `_copy_if_readonly`. Only the returned `ModelRunnerOutput` (from `output_rank`) can outlive the call and therefore requires detach. The change is one-line: keep `outputs = ray.get(refs)` at line 462, replace lines 463-464 with a single detach on `outputs[output_rank]` (reading `output_rank` from `self.kv_output_aggregator` or defaulting to 0 to match `aggregate`'s default), then call `self.kv_output_aggregator.aggregate(outputs)` unchanged at line 465. Byte/value equivalence: aggregate reads only metadata (already Python-owned sets/dicts) from non-output-rank outputs and only returns/mutates `outputs[output_rank]`, so the aggregated ModelRunnerOutput and KVOutputAggregator state are identical. Ordering into KVOutputAggregator is preserved (list order unchanged). The single-worker path (lines 449-452) and the two FutureWrapper paths (lines 456, 468) are untouched; the FutureWrapper connector path's `result()` in ray_utils.py:270-272 can adopt the same reduction in a follow-up.

**Novelty rationale.**

Proposal 1 (finding_id=find-vllm_v1_executor-0005) restructures scheduling with `ray.wait` to overlap detach with tail-worker readiness, but still calls `detach_zero_copy_from_model_runner_output` on every ready ref. Proposal 2 (finding_id=find-vllm_v1_executor-0010) sets `_max_inflight_executions` on the compiled DAG and explicitly states no changes are needed inside `_execute_dag` at lines 462-464. Neither proposal observes that KVOutputAggregator.aggregate only retains `outputs[output_rank]` past the call, so the detach on the other N-1 outputs is dead work — regardless of whether they arrive together (current code) or one-by-one (Proposal 1). This proposal eliminates O(N) driver-side NumPy `.copy()` cost per token and reduces it to O(1), independent of ref-arrival scheduling and independent of DAG queue depth. It is compositional with both listed proposals: Proposal 1's `ray.wait` loop can drop the per-ref detach for non-output ranks; Proposal 2's inflight cap is unaffected.

---
