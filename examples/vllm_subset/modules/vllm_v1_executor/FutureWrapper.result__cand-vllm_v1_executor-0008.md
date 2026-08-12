# FutureWrapper.result

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/ray_utils.py`](vllm/v1/executor/ray_utils.py) (lines 250–272)
- **Symbol:** `FutureWrapper.result`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_executor-0008`

## Description
Ray executor future resolution path for non-blocking compiled-DAG execution; resolves Ray refs, detaches zero-copy buffers, and optionally aggregates all worker outputs.

## Current approach
result() calls ray.get(self.ref_or_refs, timeout=timeout). Without an aggregator it detaches the single output and returns it. With an aggregator it iterates outputs sequentially, detaches each output in Python, and calls self.aggregator.aggregate(outputs, output_rank=0).

## Estimated impact explanation
This is directly on the async Ray decode completion path. It can move median TPOT under connector or multi-output cases, with medium scope because non-connector single-output runs perform only one detach.

## Evolve rationale
The key construct is ray.get followed by the sequential for output in outputs detach loop. In async Ray scheduling, this is where the engine turns in-flight DAG refs back into model outputs. Headroom includes processing refs as they become ready, parallelizing or batching detach for multiple outputs while preserving list order, and reducing duplicate detachment work when aggregation consumes only a subset. Correctness oracle: aggregate receives outputs in the same rank order as ray.get(refs), single-output futures return the same object values, and existing Ray async scheduling tests observe the same result/exception behavior.

## Deep research proposals

### 1. Use ray.wait to overlap detach with in-flight worker outputs in FutureWrapper.result
- **Finding:** `find-vllm_v1_executor-0005` — *ray.wait*
- **Source URL:** <https://docs.ray.io/en/latest/ray-core/api/doc/ray.wait.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/ray_utils.py at FutureWrapper.result (lines 250-272), replace the current ray.get(self.ref_or_refs, timeout=timeout) followed by the sequential `for output in outputs: detach_tensors_for_zero_copy(output)` loop with a readiness-driven pattern based on ray.wait when self.aggregator is not None and self.ref_or_refs is a list. Concretely: keep a mapping from ref -> rank index; in a loop call ready, pending = ray.wait(pending, num_returns=1, timeout=remaining_timeout); for each ready ref call ray.get on it, run detach_tensors_for_zero_copy immediately on that single output, and store it in a pre-sized results list at its rank index. Continue until pending is empty, honoring the overall timeout budget. Then pass the rank-ordered results list to self.aggregator.aggregate(outputs, output_rank=0) exactly as today. The single-ref (non-aggregator) path is unchanged. This preserves ordered return semantics (results are placed back by rank), preserves exception behavior (ray.get on each ready ref still raises), and preserves timeout semantics by subtracting elapsed time from the remaining budget.

**Proposal rationale.**

The candidate explicitly identifies the sequential detach loop after a blocking ray.get as headroom: with multiple workers, all detach work waits for the slowest ref before any zero-copy detachment begins. ray.wait is the standard Ray primitive for readiness-driven consumption of multiple futures and directly addresses this gap by letting per-output detach start as soon as each worker's ref is ready, overlapping detach cost with the tail of in-flight workers. Because results are re-placed by rank before aggregation, the aggregator still sees outputs in ray.get(refs) order, satisfying the stated correctness oracle. This is most impactful on the multi-worker aggregation path (connector/multi-output cases) that the candidate calls out as the medium-TPOT lever, and it maps cleanly onto a multi-turn agentic workload where every decode step pays this cost.

---

## Agent proposals

### 1. Skip zero-copy detach on non-output-rank worker outputs before aggregation
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/ray_utils.py at FutureWrapper.result (lines 250-272), when self.aggregator is not None, only run detach_zero_copy_from_model_runner_output on outputs[output_rank] (i.e. output_rank=0), not on every element of the outputs list. Rationale grounded in the code path: KVOutputAggregator.aggregate (vllm/distributed/kv_transfer/kv_connector/utils.py, class KVOutputAggregator around lines 54-172) returns outputs[output_rank] as the ModelRunnerOutput and only reads kv_connector_output-related fields (finished_sending/finished_recving, kv_connector_stats, kv_connector_worker_meta, kv_cache_events, invalid_block_ids, expected_finished_count) from the non-output-rank workers. Meanwhile detach_zero_copy_from_model_runner_output (same file, lines 198-247) only copies output.logprobs and output.routed_experts numpy arrays — fields that are discarded for all outputs except outputs[output_rank]. Copying those SHM-backed numpy arrays on ranks whose whole ModelRunnerOutput is about to be thrown away is pure waste: it burns CPU and memory bandwidth per decode step and keeps SHM channel entries live longer than needed. Concrete change: replace the current `for output in outputs: detach_zero_copy_from_model_runner_output(output)` loop with a single `detach_zero_copy_from_model_runner_output(outputs[0])` (output_rank is hard-coded to 0 in the current aggregate call), then hand the outputs list to self.aggregator.aggregate(outputs, output_rank=0) unchanged. To preserve the docstring's stated purpose of not retaining Ray SHM references across scheduler iterations for the non-rank-0 outputs, drop those objects immediately after aggregation returns by clearing the local list (e.g. `outputs.clear()` before returning the aggregated result) so any read-only numpy views from non-rank-0 workers go out of scope right away and the compiled-DAG SHM channel is not stalled by lingering references. Correctness oracle: the single-ref (non-aggregator) path is unchanged; the returned aggregated ModelRunnerOutput comes from outputs[0] whose logprobs/routed_experts are still detached exactly as before; the kv_connector_output aggregation reads only the small Python-native fields on other ranks, which detach never touches. Existing Ray async scheduling tests observe identical result/exception behavior because ray.get semantics and the aggregator call signature are unchanged.

**Novelty rationale.**

The existing deep_research_proposal (find-vllm_v1_executor-0005) focuses on parallelizing/overlapping the detach work with in-flight worker refs using ray.wait — it still runs detach on every output, just earlier and concurrently. This proposal is orthogonal and complementary: it eliminates the detach work for non-output-rank outputs entirely by observing that KVOutputAggregator.aggregate reads only kv_connector_output fields (never logprobs or routed_experts) from ranks != output_rank, so detaching their numpy arrays is dead work. Removing work is strictly cheaper than parallelizing it, and the two ideas can be combined but neither subsumes the other.

---
