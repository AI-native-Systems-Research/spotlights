# RayWorkerWrapper.execute_model_ray

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/ray_utils.py`](vllm/v1/executor/ray_utils.py) (lines 125–177)
- **Symbol:** `RayWorkerWrapper.execute_model_ray`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_executor-0013`

## Description
Worker-side Ray compiled-DAG node body: sets the CUDA device if needed, unpacks scheduler/intermediate inputs, runs model_runner.execute_model, prepares PP intermediate handoff, and materializes final async outputs before returning through Ray.

## Current approach
Every DAG invocation calls setup_device_if_necessary, branches on tuple length, calls worker.model_runner.execute_model, checks _is_intermediate_tensors, may loop over scheduled_new_reqs to clear mm_features for PP transfer, calls AsyncModelRunnerOutput.get_output for final execute_model output, checks _is_last_rank, and may call sample_tokens plus another get_output when execute_model returned None on the last PP rank.

## Estimated impact explanation
This is the per-worker body of the Ray steady-state execution graph. Removing host/device sync or Python overhead here affects every token step across all Ray workers, so it can reduce median TPOT in multi-turn workloads that use the Ray backend.

## Evolve rationale
The concrete hot constructs are setup_device_if_necessary at line 135, the mm_features clearing loop at lines 159-160, and the AsyncModelRunnerOutput.get_output calls at lines 163-176. This method runs once per Ray compiled-DAG worker per scheduler step. Headroom includes caching PP rank booleans, reducing per-step Python branching in the DAG node, and replacing blocking get_output materialization with a staged serializable payload or overlapped D2H copy before Ray channel return. Correctness oracle: the DAG node returns the same ModelRunnerOutput or the same (SchedulerOutput, GrammarOutput, IntermediateTensors) tuple, non-last PP ranks still suppress final outputs, mm_features are stripped only where required, and existing Ray compiled-DAG PP tests remain value-equivalent.

## Deep research proposals

### 1. Restructure PP handoff return path in execute_model_ray to enable Ray CG GPU communication overlap
- **Finding:** `find-vllm_v1_executor-0002` — *Experimental: Overlapping communication and computation*
- **Source URL:** <https://docs.ray.io/en/master/ray-core/compiled-graph/overlap.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/ray_utils.py::RayWorkerWrapper.execute_model_ray (lines 125-177), adapt the worker-side DAG node body so that the intermediate-tensor return path is friendly to Ray Compiled Graph's `_overlap_gpu_communication=True` mode (set at the corresponding `dag.experimental_compile()` call site). Concretely: (1) keep the PP intermediate return branch at line 161 producing GPU-resident IntermediateTensors on the same stream used by execute_model, and avoid inserting any host-visible sync (no `.cpu()`, no `.item()`, no `torch.cuda.synchronize()`) between the model_runner.execute_model call at line 145 and the tuple return; (2) move the mm_features clearing loop at lines 159-160 to run before (or in parallel with) execute_model when supports_mm_inputs and is_first_rank so it doesn't extend the critical path between compute completion and the Ray channel send; (3) on the last PP rank, defer AsyncModelRunnerOutput.get_output() calls at lines 163-164 and 175-176 as late as possible (or replace with a staged serializable payload) so that non-last ranks' intermediate handoff can be scheduled as an overlappable GPU op by the DAG scheduler rather than being blocked by Python-visible work in this same function. Pair this with enabling `_overlap_gpu_communication=True` on the corresponding `experimental_compile()` invocation.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names 'overlapped D2H copy before Ray channel return' and reducing per-step blocking work as headroom; the finding provides a concrete Ray mechanism (`_overlap_gpu_communication=True`) that lets the DAG scheduler hide inter-stage GPU transfers under downstream compute. That mechanism only pays off if the worker-side DAG node (this method) does not interpose host-visible blocking work between the compute that produced the tensor and the Ray channel send. The proposal addresses exactly that gap: it restructures execute_model_ray's return path so the PP intermediate handoff at line 161 and the final-rank materialization at lines 163-176 do not serialize the transfer, making the overlap flag effective for the multi-turn agentic workload where every PP step's TPOT compounds.

---

### 2. Overlap AsyncModelRunnerOutput D2H materialization with next Ray DAG step via pinned buffers on a side stream
- **Finding:** `find-vllm_v1_executor-0003` — *CUDA C++ Best Practices Guide*
- **Source URL:** <https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-best-practices-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/ray_utils.py, RayWorkerWrapper.execute_model_ray at lines 125-177 currently blocks on AsyncModelRunnerOutput.get_output() at lines 164 and 176 before returning through the Ray compiled-DAG channel. Restructure the tail of the method so that D2H copies of the async payload (sampled tokens, logprobs, routed-expert metadata, and other token metadata that AsyncModelRunnerOutput would otherwise materialize synchronously) are issued into pinned host buffers on a dedicated non-default CUDA stream at output-production time, and the current worker step returns as soon as those copies are enqueued. The subsequent scheduler tick (the next entry into execute_model_ray on this worker) consumes the previously staged pinned-host payload by waiting only on that specific copy event before it is serialized onto the Ray channel, so the D2H transfer for step N overlaps model_runner.execute_model compute for step N+1. Concretely: (1) add a per-worker slot that holds the last step's pinned-host payload plus a CUDA event recorded on the side copy stream; (2) at line 163-164 and 175-176, replace the blocking get_output() with an enqueue-on-copy-stream path that returns a lightweight handle backed by pinned memory; (3) at the top of the next execute_model_ray call (after setup_device_if_necessary at line 135), drain the previous slot by synchronizing on its copy event (cheap if the compute step took longer than the copy) and package the pinned-host tensors into the ModelRunnerOutput that is returned through Ray. Preserve the existing branches for _is_intermediate_tensors, the mm_features stripping loop at lines 159-160, and the non-last-rank suppression at lines 165-169 so that non-last PP ranks still return (scheduler_output, grammar_output, None) and only the last-rank final-output path is staged. Correctness oracle from evolve_rationale is retained: the DAG node still yields the same ModelRunnerOutput or the same PP tuple, just delayed by one step's copy latency rather than by a synchronous get_output().

**Proposal rationale.**

The candidate's identified headroom explicitly names replacing blocking get_output materialization with a staged serializable payload or overlapped D2H copy before Ray channel return. The CUDA C++ Best Practices Guide section 9.1.2 prescribes exactly the missing ingredients for that headroom: pinned host memory plus a non-default stream are the two preconditions the guide gives for asynchronous D2H transfers that overlap with subsequent kernel work. AsyncModelRunnerOutput.get_output() today is a natural synchronization point that hides behind future/result resolution, which is precisely the anti-pattern the guide contrasts against its asynchronous version. Because execute_model_ray runs once per worker per scheduler step and get_output is on the last-rank final-output path, moving the copy off the critical path directly targets median TPOT in the multi-turn agentic workload flagged by the caller, without touching PP intermediate handoff or the mm_features stripping invariant.

---

### 3. Defer AsyncModelRunnerOutput materialization in Ray DAG node to overlap D2H with channel return
- **Finding:** `find-vllm_v1_executor-0004` — *[RFC] Redesign enable_return_routed_experts to avoid blocking EngineCore event loop*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/38079>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/ray_utils.py::RayWorkerWrapper.execute_model_ray (lines 125-177), replace the blocking AsyncModelRunnerOutput.get_output() calls at lines 164 and 176 with a staged pattern that keeps GPU→CPU synchronization off the critical Ray compiled-DAG return path, in the spirit of the RFC's 'No GPU→CPU sync on the critical path' data flow. Concretely: (1) split the AsyncModelRunnerOutput handling into a lightweight 'schedule H2D-ready + Ray-serializable payload' step that records the CUDA event and metadata cheaply, from a 'wait + materialize' step that currently blocks on the copy; (2) issue the async D2H copy for the sample-token/final output before or overlapped with any remaining Python bookkeeping in the DAG node (device setup, PP-rank branching, mm_features stripping), so the CUDA copy runs while Python work completes; (3) only synchronize (call the equivalent of get_output()) at the last possible moment before Ray requires a serializable object, so PP intermediate tuples (which are returned at line 161 without materialization) and non-last-rank paths (line 169) are unaffected. Also apply the RFC's 'copied only for opted-in requests and only on the steps that need it' idea to any per-request auxiliary payload built in this method: guard the mm_features clearing loop at lines 159-160 and any future optional per-request output metadata behind a per-request opt-in flag so non-opted-in requests skip the Python-level work entirely. The DAG node still returns the same ModelRunnerOutput or (SchedulerOutput, GrammarOutput, IntermediateTensors) tuple, non-last PP ranks still suppress final outputs, and existing Ray compiled-DAG PP tests remain value-equivalent.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'replacing blocking get_output materialization with a staged serializable payload or overlapped D2H copy before Ray channel return' as headroom, which is the same pattern the RFC advocates: keep feature-specific payload construction and GPU→CPU sync off the critical path, and only pay for optional per-request work when requested. The RFC's supporting quote 'No GPU→CPU sync on the critical path' directly targets the AsyncModelRunnerOutput.get_output() calls at lines 164 and 176, which are the concrete blocking sync sites in this per-worker per-step method. Because this DAG node runs once per Ray worker per scheduler step, overlapping the D2H materialization with the surrounding Python branching plus gating optional per-request payloads (like mm_features handling and future auxiliary metadata) behind an opt-in flag can plausibly reduce median TPOT in the multi-turn agentic workload, matching the caller objective. The change is transferable rather than merely topically adjacent because it addresses a specific, identified stall (blocking get_output before Ray return) with a specific mechanism from the finding (deltas + opt-in + off-critical-path construction).

---

## Agent proposals

### 1. Specialize execute_model_ray into role-bound closures at DAG compile time to eliminate per-step Python branching
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/ray_utils.py::RayWorkerWrapper.execute_model_ray (lines 125-177), replace the single generic method body with a role-specialized closure that is bound once at Ray Compiled Graph construction time and installed as the DAG node's callable. Concretely: (1) Add a helper on RayWorkerWrapper (called from the same site that today invokes execute_model_ray as the DAG bind target) that, after the worker is fully initialized, snapshots the invariants this method re-derives on every invocation: `is_last_rank = get_pp_group().is_last_rank`, `is_first_rank = get_pp_group().is_first_rank`, `supports_mm_inputs = self.worker.model_runner.supports_mm_inputs`, `needs_device_setup = not current_platform.is_tpu()`, and `device = self.worker.device`. (2) Using those constants, generate one of a small set of specialized closures and assign it as the DAG-invoked entrypoint: e.g. `_execute_model_ray_first_rank_mm`, `_execute_model_ray_middle_rank`, `_execute_model_ray_last_rank`, `_execute_model_ray_single_rank`. Each closure hard-codes: whether to expect a 2-tuple or 3-tuple input (middle/last-rank expect the 3-tuple, first/single expect the 2-tuple), whether to run the mm_features stripping loop at lines 159-160 (only the first-rank-with-mm closure includes it and inlines `req.mm_features = []`), whether to branch on `_is_intermediate_tensors` at all (non-last ranks skip the last-rank materialization branch entirely; the last-rank closure skips the intermediate-return branch), and whether to call `sample_tokens`/second `get_output` at lines 170-176 (only the last-rank closure). (3) Replace the runtime `setup_device_if_necessary()` guard at line 135 with a one-shot device-set performed inside the closure factory before installation, plus a Ray-CG-thread-affinity check that no-ops after the first call using a bare boolean local closed over the closure (avoiding the `self.compiled_dag_cuda_device_set` attribute read/write on every step). (4) Keep `AsyncModelRunnerOutput.get_output()` at its current sites within the last-rank closure so this change composes cleanly with the existing D2H-overlap proposals rather than duplicating them. The correctness oracle from evolve_rationale is preserved verbatim: each specialized closure returns the same ModelRunnerOutput or `(SchedulerOutput, GrammarOutput, IntermediateTensors)` / `(SchedulerOutput, GrammarOutput, None)` tuple its role would have returned under the generic method, mm_features are stripped only in the first-rank-with-mm variant, and non-last ranks still suppress final outputs. Fallback: retain the current generic `execute_model_ray` as the default entrypoint used when the compiled-DAG construction site has not opted in, so non-Ray-CG execution paths and non-Ray executors are untouched.

**Novelty rationale.**

None of the three listed deep_research_proposals address per-step Python dispatch overhead in the DAG node body. find-vllm_v1_executor-0002 targets `_overlap_gpu_communication=True` and PP GPU-transfer overlap; find-vllm_v1_executor-0003 pipelines D2H copies via pinned buffers on a side stream across steps; find-vllm_v1_executor-0004 stages AsyncModelRunnerOutput materialization and gates optional per-request payloads. All three keep the generic method body and its per-step branching intact — they only reshape where GPU→CPU work happens. This proposal instead attacks the Python-side cost that the candidate's evolve_rationale explicitly names as headroom ("caching PP rank booleans, reducing per-step Python branching in the DAG node"): the repeated `setup_device_if_necessary` guard read, `get_pp_group()` calls in `is_first_rank`/`_is_last_rank`, `supports_mm_inputs` attribute lookup, `isinstance` dispatch in `_is_intermediate_tensors`, and tuple-length branching, all of which have compile-time-constant answers per worker. Compile-time specialization into role-bound closures is a distinct mechanism (static dispatch specialization) from D2H staging or `_overlap_gpu_communication`, and it stacks with all three existing proposals rather than overlapping them.

---

### 2. Return a PP-transfer view of SchedulerOutput instead of mutating mm_features per step
- **Agent:** codex

**Detailed description.**

In `vllm/v1/executor/ray_utils.py::RayWorkerWrapper.execute_model_ray` lines 157-161, replace the in-place loop that clears `req.mm_features` on every first-rank multimodal PP handoff with a lightweight PP-transfer projection of `SchedulerOutput`. Concretely, when `output` is `IntermediateTensors` and the worker is the first PP rank with multimodal support, build or request a `SchedulerOutput` transfer view whose `scheduled_new_reqs` entries exclude the heavy `mm_features` field from Ray serialization, while preserving the original `scheduler_output` object for local accounting and correctness-sensitive consumers. The return at line 161 would send `(scheduler_output_for_pp_transfer, grammar_output, output)` instead of mutating `scheduler_output.scheduled_new_reqs` in place. This can be implemented as a small helper invoked only in this branch, ideally using existing dataclass/model-copy facilities or a purpose-built serialization method rather than hand-editing each request object. The correctness oracle is that downstream PP stages still receive no multimodal feature payload, all other scheduler fields remain identical, and the first-rank local object is no longer destructively modified after `execute_model` has consumed it.

**Novelty rationale.**

The deep-research proposals focus on Ray GPU communication overlap and AsyncModelRunnerOutput D2H staging; the only mentions of `mm_features` are moving the existing clearing loop earlier or gating optional work. Agent A proposes role-specialized closures but still keeps the first-rank multimodal variant doing the same per-step `req.mm_features = []` loop. This proposal is different: it removes the destructive per-request mutation from `execute_model_ray` by returning a serialization/transfer view that omits `mm_features`, reducing Python object churn and avoiding side effects while preserving the PP payload contract.

---
