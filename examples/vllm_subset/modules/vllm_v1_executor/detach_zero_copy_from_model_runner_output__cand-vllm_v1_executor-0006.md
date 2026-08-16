# detach_zero_copy_from_model_runner_output

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/ray_utils.py`](vllm/v1/executor/ray_utils.py) (lines 198–247)
- **Symbol:** `detach_zero_copy_from_model_runner_output`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_executor-0006`

## Description
In-place detachment of read-only numpy arrays inside ModelRunnerOutput so Ray compiled-DAG shared-memory buffers are not retained across scheduler iterations.

## Current approach
The nested _copy_if_readonly checks isinstance(arr, np.ndarray) and arr.flags.writeable, then calls arr.copy() for read-only arrays. logprobs and routed_experts fields are unpacked, copied field-by-field, and their enclosing tuple/namedtuple is rebuilt only if at least one child array changed.

## Estimated impact explanation
The copies are on the Ray output critical path and scale with emitted logprob/routing metadata. Optimizing them moves median TPOT for Ray backends when agentic workloads request logprobs or use expert routing, but has little effect when those fields are absent.

## Evolve rationale
The concrete optimization unit is the repeated _copy_if_readonly(arr).copy() sequence over logprobs and routed_experts. This runs after each Ray compiled-DAG output with logprobs or routed expert metadata. Headroom includes avoiding copies for fields proven not to escape the current scheduler iteration, pooling destination arrays for stable shapes, and combining detach with downstream aggregation so the output is copied at most once. Correctness oracle: returned ModelRunnerOutput no longer aliases read-only Ray SHM arrays, field types and values are unchanged, and existing Ray executor logprob/routed-expert tests produce identical outputs without RAY_CGRAPH_get_timeout stalls.

## Deep research proposals

### 1. Skip detach copies via opt-in, per-step routed-experts and logprobs payloads
- **Finding:** `find-vllm_v1_executor-0004` — *[RFC] Redesign enable_return_routed_experts to avoid blocking EngineCore event loop*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/38079>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/ray_utils.py:198-247, restructure detach_zero_copy_from_model_runner_output so it does not unconditionally scan and copy every read-only numpy array on the Ray SHM output critical path. Following the RFC's pattern of routing optional per-request metadata as deltas that are populated only for opted-in requests and only on the steps that need it, teach the executor output path to know which of {logprobs, routed_experts} are actually populated for this step. When a field is None or empty for every request in the batch, skip the tuple/namedtuple unpack, the per-child _copy_if_readonly checks, and the tuple rebuild entirely. When a field is populated for a subset of requests, restrict the read-only copy to the sub-arrays that will actually be consumed by the scheduler this iteration (e.g. the slice belonging to opted-in requests) rather than copying the full concatenated array. Keep the isinstance(np.ndarray)+writeable guard as the correctness oracle, and preserve the existing behavior that returned ModelRunnerOutput no longer aliases Ray SHM buffers and that field types/values are unchanged.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'avoiding copies for fields proven not to escape the current scheduler iteration' as headroom. The finding's RFC on enable_return_routed_experts prescribes exactly that shape for the routed_experts payload feeding this function: opt-in per request, present only on the steps that need it, and kept off the critical path. Aligning the detach step with that per-request/per-step opt-in signal is a concrete transferable idea: it turns detach_zero_copy_from_model_runner_output from an always-on O(fields) scan-and-copy into a no-op on steps where logprobs/routed_experts are absent for all requests, which is the common case for the multi-turn agentic workload in the caller context and thus directly reduces median TPOT for the Ray backend.

---

### 2. Overlap zero-copy detach with outstanding worker outputs via ray.wait
- **Finding:** `find-vllm_v1_executor-0005` — *ray.wait*
- **Source URL:** <https://docs.ray.io/en/latest/ray-core/api/doc/ray.wait.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/ray_utils.py, restructure the multi-worker path in FutureWrapper.result() (lines 264-272) that currently calls `ray.get(self.ref_or_refs, timeout=timeout)` and then sequentially iterates outputs calling `detach_zero_copy_from_model_runner_output(output)` before `self.aggregator.aggregate(outputs, output_rank=0)`. Replace the blocking `ray.get` + sequential detach with a `ray.wait`-driven loop: repeatedly call `ray.wait(pending_refs, num_returns=1, timeout=remaining_timeout)` to fetch whichever worker refs are ready, immediately `ray.get` each ready ref, run `detach_zero_copy_from_model_runner_output` on that output (so its read-only logprobs/routed_experts arrays are copied out of Ray SHM as soon as they're materialized), and stash the detached output in a rank-indexed slot (`results[original_index] = output`). After all refs have completed, pass the rank-ordered list to `self.aggregator.aggregate(..., output_rank=0)`, preserving current ordering semantics. The single-ref (no-aggregator) branch is unchanged. Keep detach itself in-place and identical; only its scheduling changes.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names 'combining detach with downstream aggregation so the output is copied at most once' and reducing detach latency on the Ray output critical path as headroom. Today, detach only starts after the slowest worker returns, so per-output detach work (which scales with logprobs/routed_experts size) is fully serialized behind the tail worker. `ray.wait` returns the subset of refs that are already ready, which lets us begin detaching earlier-arriving workers' outputs while the tail worker is still producing. This directly addresses median TPOT on multi-turn agentic workloads with logprobs or expert routing (the estimated_impact scenario), because on each iteration the wall-clock cost of detach is hidden inside the natural per-worker skew rather than added after `ray.get` returns. The correctness oracle is preserved: outputs are placed back in rank order before aggregation, and the in-place detach function itself is unchanged, so field types/values and existing Ray executor logprob/routed-expert tests remain identical.

---

## Agent proposals

### 1. Copy SHM arrays into a rank/field-keyed reusable pool instead of allocating fresh numpy buffers
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/ray_utils.py:198-247, replace the `arr.copy()` call inside `_copy_if_readonly` with a copy into a pre-allocated destination array held in a small per-caller pool keyed by (worker_rank, field_name, dtype). Attach the pool to `FutureWrapper` (or module-level, keyed by `id(self)` of the executor) so it survives across scheduler iterations. On each detach, for each read-only source `arr`, look up the pool slot for (rank, field, dtype); if the slot's `shape == arr.shape`, do `np.copyto(dst, arr)` and return `dst`; otherwise reallocate the slot with `np.empty_like(arr)` and copy. This keeps the correctness oracle (returned ModelRunnerOutput no longer aliases Ray SHM, field types/values unchanged) while eliminating the per-step allocator churn from `arr.copy()`, which is the dominant cost when shapes are stable across iterations (the normal case for a fixed batch of ongoing agentic turns emitting logprobs). Keep the isinstance/writeable guard unchanged; the tuple/namedtuple rebuild still fires only when a child was replaced. For safety, cap the pool at O(num_fields * num_ranks) slots and drop it when the FutureWrapper (or executor) is torn down so the buffers don't outlive the workload.

**Novelty rationale.**

Neither existing proposal addresses allocator pressure. Finding 0004 focuses on *skipping* copies when fields are unpopulated/opted-out (a control-flow change), and finding 0005 focuses on *scheduling* detach earlier via `ray.wait` overlap (a latency-hiding change). Both still call `arr.copy()`, which allocates a fresh buffer every iteration and, for stable-shape logprobs/routed_experts, incurs O(bytes) malloc+memcpy plus later GC pressure on the hot path. Pooling destination buffers keyed by (rank, field, dtype) turns the copy into a pure `memcpy` into a warm allocation, which is orthogonal to and composable with both prior proposals (skip-when-empty still short-circuits; overlap-via-`ray.wait` still hides latency, and each per-rank detach now hits a smaller, cache-warm destination).

---

### 2. Detach only the aggregated rank’s surviving metadata
- **Agent:** codex

**Detailed description.**

In `vllm/v1/executor/ray_utils.py`, change the multi-worker `FutureWrapper.result()` path so `logprobs` and `routed_experts` are detached only on the `ModelRunnerOutput` that survives `KVOutputAggregator.aggregate(outputs, output_rank=0)`. Today every worker output is passed through `detach_zero_copy_from_model_runner_output()` before aggregation, but `KVOutputAggregator` ultimately returns `outputs[output_rank]` and only aggregates `kv_connector_output` from the other ranks. Add a narrow variant or parameter to `detach_zero_copy_from_model_runner_output` that detaches the scheduler-visible numpy metadata fields, then call it for the selected output after aggregation; leave non-selected worker outputs undetached because their `logprobs` and `routed_experts` are discarded when the local `outputs` list goes out of scope. If needed for safety, explicitly clear `logprobs`/`routed_experts` on non-output ranks before aggregation or immediately after extracting their `kv_connector_output`, so those Ray SHM references cannot accidentally be retained by future aggregator changes. The correctness oracle is unchanged for the returned output: it must not alias read-only Ray SHM arrays, and its field types/values remain identical. Add or extend a Ray executor aggregation test with multiple worker outputs containing read-only numpy `logprobs`/`routed_experts`, asserting only the returned rank’s arrays are copied and non-returned rank metadata is not preserved.

**Novelty rationale.**

This is distinct from deep_research proposal 0004, which skips or narrows copies based on whether requests opted into metadata for a step; this proposal skips copies based on worker-rank liveness after aggregation. It is distinct from proposal 0005, which overlaps detach work with `ray.wait` but still detaches every worker output. It is also distinct from Claude’s pooling proposal, which keeps copying every read-only array but reuses destination buffers. Here the optimization removes whole copies for non-output ranks whose `logprobs` and `routed_experts` are not part of the returned scheduler output.

---
