# RayDistributedExecutor._compiled_ray_dag transport policy

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/ray_executor.py`](vllm/v1/executor/ray_executor.py) (lines 570–637)
- **Symbol:** `RayDistributedExecutor._compiled_ray_dag transport policy`
- **Kind:** config_block
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0011`

## Description
Ray compiled-DAG communication policy for pipeline-parallel execution. It selects the tensor transport, optional wrapped PP communicator, and Ray compiled-DAG GPU communication overlap flag.

## Current approach
The method reads VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE, validates it against auto/nccl/shm, applies with_tensor_transport() only between non-last PP stages when the channel is not shm, optionally registers RayPPCommunicator when VLLM_USE_RAY_WRAPPED_PP_COMM is enabled, and passes VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM to experimental_compile().

## Estimated impact explanation
This does not affect uniprocess or non-PP runs, but for Ray PP deployments it controls inter-stage tensor movement and communication overlap on every decode step. Better selection should reduce pipeline bubbles and median TPOT, with possible TTFT gains from avoiding a poor first compiled-DAG transport choice.

## Evolve rationale
The policy-defining constructs are the channel_type branch at lines 570-575, the PP-stage transport application at lines 597-608, the communicator selection at lines 612-623, and the _overlap_gpu_communication compile argument at lines 634-637. Evolution can make transport and overlap selection topology-aware, interconnect-aware, or PP-size-aware instead of fixed by environment defaults. Correctness oracle: Ray PP and compiled-DAG tests plus e2e generation parity must preserve intermediate tensor routing, output values, and failure behavior while benchmarks compare TTFT and TPOT across shm, nccl, wrapped-PP, and overlap settings.

## Deep research proposals

### 1. Make Ray compiled-DAG transport selection topology-aware via placement-group inspection
- **Finding:** `find-0008` — *Placement Groups*
- **Source URL:** <https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/ray_executor.py around lines 570-637, replace the env-only channel_type/overlap selection with a policy that inspects the Ray placement group and per-worker bundle locations of the PP stage actors before deciding the transport. Concretely: (1) at the channel_type branch (570-575), when VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE is unset or 'auto', query the placement group / node ids of adjacent PP stage workers and pick 'shm' for stage pairs co-located on the same node (PACK-style locality) and 'nccl' for stage pairs that span nodes (SPREAD), instead of a single global channel. (2) At the PP-stage transport application (597-608), apply with_tensor_transport per producer→consumer edge using the per-edge decision so intra-node edges use shared memory and inter-node edges use NCCL. (3) At the communicator selection (612-623), only register RayPPCommunicator when at least one PP edge actually crosses nodes (where a custom wrapped communicator could matter); skip it for fully co-located PP. (4) At experimental_compile() (634-637), enable _overlap_gpu_communication only when there is at least one cross-node NCCL edge, since overlap has limited value when all transports are shm. The placement-group/node info can be obtained from ray.util.get_current_placement_group() and ray.get(actor.get_node_id.remote()) on the existing PP workers without changing how workers are launched.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out making transport and overlap selection topology-aware and PP-size-aware, but today the code is uniformly driven by env vars regardless of where PP workers actually landed. The Ray Placement Groups doc supplies the missing primitive: PACK vs SPREAD bundle placement defines exactly the locality information needed to choose shm vs nccl per edge and to decide whether overlap or a wrapped PP communicator can pay off. Using placement-group/node-id introspection at decision time turns a static, possibly-mismatched policy into a per-edge decision that should reduce inter-stage latency for co-located PP pairs (shm beats nccl on a single node) while still using nccl plus overlap for cross-node edges, directly addressing the median TPOT objective on multi-turn agentic Ray PP deployments.

---

## Agent proposals

### 1. Add a decode fast-path that bypasses Ray compiled DAG for co-located PP with small payloads
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/ray_executor.py around lines 570-637, introduce a second execution path alongside the compiled Ray DAG that is used specifically for decode iterations when (a) all PP stage workers landed on the same node (PACK locality) and (b) per-step inter-stage activations are small — i.e. hidden_size * num_running_seqs * 1 (one new token) * dtype_bytes is below a threshold (e.g. <= 256 KiB by default, tunable via a new VLLM_RAY_DECODE_FASTPATH_BYTES env). For those iterations, route the (hidden_states, residual) tensor between adjacent PP stages through a pre-allocated pinned-CPU shared-memory ring buffer (created once at executor init, sized for the worst-case decode payload) rather than through Ray's compiled DAG channel, while still using the compiled DAG for prefill iterations where tensors are large and the DAG's static scheduling pays off. Wire this in by: (1) at the channel_type branch (570-575), in addition to selecting a channel for the compiled DAG, allocate the shared ring on each PP-adjacent worker pair when locality permits; (2) leave the with_tensor_transport / RayPPCommunicator / _overlap_gpu_communication code paths (597-637) unchanged for the prefill DAG; (3) add a small Python-level dispatcher in execute_model() that picks fastpath vs compiled-DAG at step boundaries based on whether the scheduler output is pure-decode and the payload size estimate. Correctness is preserved because the fastpath only carries the same hidden_states/residual that the DAG would, and falls back to the DAG on any size-threshold miss or mixed prefill/decode batch. Validate with the existing Ray PP tests (forcing both paths via env), e2e generation parity vs main, and TTFT/TPOT benchmarks on a multi-turn agentic trace.

**Novelty rationale.**

find-0008 keeps the compiled Ray DAG as the only execution path and only varies transport (shm vs nccl), wrapped-PP, and overlap on a per-edge topology basis. It does not consider that for a decode-heavy multi-turn agentic workload the dominant per-step cost on co-located PP is the compiled-DAG channel/scheduling overhead itself — even with shm — because the per-token activation is tiny. This proposal adds an orthogonal axis (bypass the DAG entirely for small-payload decode steps via a pre-allocated pinned-CPU ring), driven by per-step payload size and batch composition rather than just static placement-group topology, and remains complementary to the topology-aware DAG configuration in find-0008.

---

### 2. Side-input PP control metadata outside the tensor transport edge
- **Agent:** codex

**Detailed description.**

In `vllm/v1/executor/ray_executor.py` around the `_compiled_ray_dag` construction at lines 577-608, change the PP DAG shape so the inter-stage edge carries only the `IntermediateTensors`, while `SchedulerOutput` and `GrammarOutput` are supplied to every PP stage directly from the `InputNode` as side inputs. Concretely, have `_execute_dag()` pass both a full first-stage control tuple and a compact downstream control tuple into the compiled DAG; the downstream tuple should preserve scheduling fields needed by later PP stages but strip first-stage-only multimodal payloads such as `mm_features` before Ray serializes it. Then update the worker Ray entrypoint to accept `(scheduler_output, grammar_output, intermediate_tensors)` as separate bind arguments for PP ranks > 0, and return only `IntermediateTensors` from non-last PP stages. Apply `with_tensor_transport()` only to that intermediate tensor output before binding the next stage. This lets Ray broadcast small control metadata to downstream PP workers in parallel with stage-0 compute instead of serializing it through each producer-to-consumer PP edge, reducing first-token pipeline bubbles for media turns and avoiding repeated non-tensor object payloads on every decode step. Validate with Ray PP text and multimodal generation parity, plus a regression that stage > 0 receives stripped multimodal fields while final outputs match the current tuple-chaining DAG.

**Novelty rationale.**

The deep_research_proposal changes which transport, communicator, and overlap flag are selected for each PP edge based on placement topology; it still keeps the current data dependency where `SchedulerOutput` and `GrammarOutput` are returned by one PP stage and forwarded to the next with the tensors. Agent A proposes bypassing the compiled DAG for small co-located decode payloads via a new shared-memory ring. This proposal does neither: it keeps the Ray compiled DAG and the existing transport choices, but changes the DAG dataflow so non-tensor control metadata is side-input once and only activations use the configured tensor transport edge.

---
