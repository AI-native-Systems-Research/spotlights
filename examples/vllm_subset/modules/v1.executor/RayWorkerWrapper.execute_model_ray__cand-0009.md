# RayWorkerWrapper.execute_model_ray

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/ray_utils.py`](vllm/v1/executor/ray_utils.py) (lines 122–174)
- **Symbol:** `RayWorkerWrapper.execute_model_ray`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0009`

## Description
Worker-side Ray compiled-DAG node invoked on every Ray model step. It sets device context if needed, unpacks scheduler/grammar/intermediate tensors, runs model execution, forwards intermediate tensors for non-last PP stages, and samples tokens on the last stage when needed.

## Current approach
Each invocation calls setup_device_if_necessary(), branches on tuple length, calls model_runner.execute_model(), checks _is_intermediate_tensors(), may inspect supports_mm_inputs and get_pp_group().is_first_rank before clearing mm_features, handles AsyncModelRunnerOutput.get_output(), calls _is_last_rank() via get_pp_group(), and may call sample_tokens().

## Estimated impact explanation
This runs once per Ray worker per decode step. Removing repeated Python branching and distributed-group lookups should reduce Ray worker-side host overhead, improving median TPOT most noticeably for small batches and multi-stage Ray deployments.

## Evolve rationale
The hot constructs are the branch ladder and repeated group/model-mode checks at lines 132-173. Evolution can generate specialized call paths per PP stage and model mode, precompute is_first_rank/is_last_rank/supports_mm_inputs, split intermediate-stage and last-stage handling, and reduce repeated tuple/type checks while preserving Ray compiled-DAG contracts. Correctness oracle: Ray PP and compiled-DAG tests must preserve intermediate tensor routing, multimodal feature stripping, output serializability, and final token outputs compared with the current path.

## Deep research proposals

### 1. Specialize execute_model_ray per PP/model mode using compiled-DAG static topology
- **Finding:** `find-0002` — *Ray Compiled Graph (beta)*
- **Source URL:** <https://docs.ray.io/en/latest/ray-core/compiled-graph/ray-compiled-graph.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Leverage the fact that Ray Compiled Graph fixes the actor topology and call sites at DAG-build time to replace the runtime branch ladder in vllm/v1/executor/ray_utils.py:122-174 (RayWorkerWrapper.execute_model_ray) with statically bound worker callables. Concretely: at the time the compiled DAG is constructed, query get_pp_group() once on each worker to materialize is_first_rank, is_last_rank, and supports_mm_inputs, then bind one of a small set of specialized handler closures (e.g. first_rank_no_mm, first_rank_with_mm, middle_rank, last_rank_sampling, last_rank_async) onto the actor. The DAG node would invoke that pre-bound callable directly, so per-step work skips setup_device_if_necessary re-checks, the tuple-length branch, _is_intermediate_tensors() probing, supports_mm_inputs lookup, get_pp_group() rank checks, and the AsyncModelRunnerOutput vs sync output branch. Intermediate-stage workers would have a path that only forwards IntermediateTensors; the last-stage worker would have a path that always invokes sample_tokens(). The Ray compiled-DAG contract (input/output tensor shapes and serializability) is preserved because we are only collapsing Python control flow that is already constant for the lifetime of the DAG.

**Proposal rationale.**

Ray Compiled Graph's value proposition (“<50us system overhead for workloads that repeatedly execute the same task graph”) only materializes if the per-invocation Python work on each worker is also minimized; otherwise the worker-side branch ladder dominates. The finding's transferable idea — preallocate resources and prebuild paths so the same fanout graph runs with sub-ms overhead — maps directly to specializing this worker entry point per node identity in the static graph. Because the compiled DAG guarantees the PP rank, model mode, and output mode of each actor are fixed for the DAG's lifetime, the repeated checks at lines 132–173 are pure overhead that the compiled-graph model lets us eliminate, addressing the candidate's stated bottleneck (Python branching and distributed-group lookups on every decode step) and improving median TPOT for the multi-turn agentic workload, especially at small batch sizes where Ray host overhead is most visible.

---

## Agent proposals

### 1. Move PP IntermediateTensors hop onto Ray Compiled DAG NCCL TorchTensor channels
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/ray_utils.py:122-174 (RayWorkerWrapper.execute_model_ray), the non-last-rank return path packs IntermediateTensors into a Python tuple together with SchedulerOutput/GrammarOutput (`return scheduler_output, grammar_output, output` at line 158) and lets Ray Compiled Graph send the whole tuple over its default object channel. That path pickles every torch.Tensor inside IntermediateTensors and round-trips them through the Ray object store / shared memory, even though the receiving PP-stage worker is colocated on the same node or reachable by NCCL. Concretely: split the worker DAG output into two channels — (a) a lightweight Python channel carrying only `(scheduler_output, grammar_output)` plus a fixed-shape descriptor (sequence-id list, batch metadata) that the next stage needs for bookkeeping, and (b) a `TorchTensorType(transport="nccl")` channel (or one such channel per named tensor in IntermediateTensors, e.g. hidden_states / residual) that is registered when the compiled DAG is built. On the producing worker, replace lines 145-158 with a path that writes each IntermediateTensors field into a pre-registered NCCL output buffer (no tuple packing, no pickling) and emits only the Python metadata over channel (a). On the consuming worker, the input deserialization path is symmetric: the IntermediateTensors object is reconstructed from references to the NCCL-received buffers without going through the `len(execute_model_input) == 3` tuple branch. This is concrete because Ray Compiled Graph already exposes `with_type_hint(TorchTensorType(transport="nccl"))`, and because IntermediateTensors has a small, fixed set of tensor fields whose shapes are determined by the compiled-DAG batch shape (so the output buffer can be reused step-to-step). Correctness is preserved by validating against the existing Ray PP compiled-DAG tests: intermediate tensor values, multimodal feature stripping (still done on the lightweight metadata channel), and final token output must match the current synchronous tuple path. The mm_features stripping at lines 156-157 stays correct because scheduler_output now travels on channel (a) where it is mutated as before.

**Novelty rationale.**

The existing deep_research_proposal (find-0002) explicitly limits itself to collapsing worker-side Python control flow by prebinding specialized handler closures at DAG-build time, and explicitly states its change preserves the Ray Compiled DAG contract because it only touches Python branching. It does not change the *transport* of the IntermediateTensors hop, nor the *serialization* path between PP stages — those remain on Ray's default object channel. This proposal targets a different bottleneck (per-step pickle + object-store round-trip of hidden-state tensors) by switching the PP boundary onto NCCL TorchTensor channels and splitting the DAG output into a metadata channel and a tensor channel. It is complementary to find-0002 (the prebound handlers can write into the new NCCL buffers) rather than overlapping, and it addresses a cost — tensor serialization across PP stages — that find-0002 leaves on the table.

---

### 2. Bypass PP hops for GrammarOutput bitmasks
- **Agent:** codex

**Detailed description.**

Change the Ray compiled-DAG contract around vllm/v1/executor/ray_utils.py:122-174 so GrammarOutput is not carried inside the stage-to-stage payload. Today every non-last PP stage returns `(scheduler_output, grammar_output, output)` and the next stage unpacks that tuple, even though repository usage shows `grammar_output` is only consumed by `model_runner.sample_tokens()` on the last PP stage. Instead, split the DAG input at build time: pass only the scheduler/intermediate payload down the PP chain, and bind `grammar_output` as a direct side input only to final-PP worker calls (including the PP=1 case). Then update `RayWorkerWrapper.execute_model_ray` to accept that optional final-stage grammar argument and have intermediate stages return `(scheduler_output, intermediate_tensors)` or `(scheduler_output, None)`. This avoids serializing and forwarding `GrammarOutput.grammar_bitmask` through every PP boundary for structured-output agentic workloads while preserving final sampling behavior. Validate with Ray PP tests covering structured outputs, `grammar_output is None`, no-scheduled-token steps, PP size 1 and >1, and async output serializability.

**Novelty rationale.**

The deep_research_proposal specializes the worker-side control flow and precomputes static PP/model-mode facts, but it keeps the same logical payload flowing through the DAG; it does not remove GrammarOutput from intermediate PP returns or add a direct final-stage side input. Agent A targets IntermediateTensors transport by splitting tensor data onto NCCL channels, and its metadata channel still carries `(scheduler_output, grammar_output)` between PP stages. This proposal targets a different object and cost: repeated Python/Ray serialization of structured-output bitmasks through stages that never consume them.

---
