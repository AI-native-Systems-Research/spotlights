# RayDistributedExecutor._compiled_ray_dag

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/ray_executor.py`](vllm/v1/executor/ray_executor.py) (lines 527–620)
- **Symbol:** `RayDistributedExecutor._compiled_ray_dag`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_executor-0012`

## Description
Compile-time construction of the Ray compiled DAG used for steady-state model execution, including PP/TP topology, tensor transport policy, optional vLLM PP communicator registration, and Ray overlap flag.

## Current approach
The method validates Ray/cgraph support, sets RAY_CGRAPH_get_timeout default, builds a MultiOutputNode by chaining execute_model_ray.bind across pp_tp_workers, applies with_tensor_transport for non-shm intermediate PP edges, optionally registers RayPPCommunicator, then calls experimental_compile with _overlap_gpu_communication from VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM.

## Estimated impact explanation
Ray compiled-DAG topology and transport choices directly affect per-step communication and synchronization in multi-node PP/TP serving. Better choices can reduce both median TTFT after graph creation and steady-state median TPOT for Ray-backed agentic workloads.

## Evolve rationale
The optimization unit is the owned DAG topology and transport-selection logic at lines 560-620. It defines the steady-state Ray execution schedule. Headroom includes topology-aware per-edge transport selection, adaptive use of RayPPCommunicator versus Ray NCCL, PP-stage edge specialization, and overlap-flag tuning based on cluster topology and tensor sizes. Correctness oracle: for a fixed SchedulerOutput, the compiled DAG returns the same final ModelRunnerOutput and intermediate PP handoff semantics as the current DAG; existing Ray PP/TP integration tests and connector tests must remain value-equivalent.

## Deep research proposals

### 1. Enable Ray CGraph GPU comm/compute overlap by default for PP DAGs and tune per topology
- **Finding:** `find-vllm_v1_executor-0002` — *Experimental: Overlapping communication and computation*
- **Source URL:** <https://docs.ray.io/en/master/ray-core/compiled-graph/overlap.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `RayDistributedExecutor._compiled_ray_dag` (vllm/v1/executor/ray_executor.py:527-620), change how `_overlap_gpu_communication` is chosen for the `experimental_compile` call at lines 617-620 so that overlap of inter-PP-stage transfers with downstream TP compute is enabled whenever the topology actually benefits from it, rather than being globally gated off by default via `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` (default `0` in vllm/envs.py). Concretely: (1) when `len(self.pp_tp_workers) > 1` (real PP present) and the selected `channel_type` is `"auto"` or `"nccl"` (i.e., the non-shm transports that already produce GPU-backed edges via `with_tensor_transport` at lines 580-591), default `_overlap_gpu_communication` to `True` unless the user has explicitly set `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=0`; (2) when `len(self.pp_tp_workers) == 1` (PP=1) or `channel_type == "shm"` (no GPU inter-stage edge to overlap), keep overlap disabled to avoid the overhead Ray's docs flag for cases where there is no communication to hide; (3) log the effective overlap decision alongside the existing `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` info logs at lines 548-551 so operators can see the topology-derived choice; (4) do not change the DAG topology or transport-selection logic itself — only the flag passed to `experimental_compile`. The `with_tensor_transport(transport=transport)` calls at lines 588-591 already mark the exact edges (non-last PP boundary, non-shm) that Ray's overlap machinery targets, so no new edge annotation is needed.

**Proposal rationale.**

The caller objective is reducing median TTFT and TPOT on multi-turn agentic workloads, where Ray-backed PP steady-state steps are dominated by inter-stage IntermediateTensors handoffs interleaved with per-stage TP compute. The finding cites Ray Compiled Graph's own documented mechanism — `_overlap_gpu_communication=True` on `dag.experimental_compile()` — for scheduling those transfers as overlappable GPU ops instead of Python-visible blocking work. The candidate already threads this exact flag at line 619 but sources it from `envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM`, which defaults to `False` in vllm/envs.py, so today most Ray PP deployments run without the overlap Ray's docs recommend. The gap is a defaults/topology-awareness one, not a missing API: the DAG at lines 560-593 already produces the multi-stage GPU-edge structure that overlap targets, and lines 580-591 already restrict non-shm transport to the exact inter-PP edges that benefit. Making overlap the default when (PP>1 and channel_type != shm), while keeping the env override, is a concrete transferable application of the finding to this specific candidate and preserves the correctness oracle (same final ModelRunnerOutput and PP handoff semantics for a fixed SchedulerOutput) because only the scheduling of already-existing edges changes.

---

### 2. Topology-aware PP/TP rank placement and per-edge transport selection in Ray compiled DAG
- **Finding:** `find-vllm_v1_executor-0007` — *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning*
- **Source URL:** <https://www.alphaxiv.org/abs/2201.12023>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend RayDistributedExecutor._compiled_ray_dag (vllm/v1/executor/ray_executor.py:527-620) to make DAG construction topology-aware, following Alpa's device-mesh insight. Two concrete changes: (1) Before the InputNode/MultiOutputNode construction (lines 560-593), inspect the Ray placement group / node IDs of self.pp_tp_workers to detect intra-node vs. cross-node boundaries and prefer arrangements that keep TP groups colocated on high-bandwidth intra-node links (NVLink/PCIe) while placing PP stage boundaries on lower-bandwidth inter-node links; if the current pp_tp_workers layout violates this (e.g., a TP group split across nodes), log a warning and, where feasible, reorder tp_group membership per PP rank so the SPMD collective within TP happens on the fastest available fabric. (2) Replace the single-valued VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE fan-out at lines 580-591 with a per-edge transport policy: for each PP edge between pp_rank and pp_rank+1, pick 'shm' when the producer/consumer workers share a node, and 'nccl' (or the wrapped RayPPCommunicator path already toggled by VLLM_USE_RAY_WRAPPED_PP_COMM at line 595) when they cross nodes. This preserves the DAG's MultiOutputNode topology and per-edge with_tensor_transport shape so the correctness oracle (identical ModelRunnerOutput and PP handoff semantics for a fixed SchedulerOutput) is unchanged, while adapting transport to the observed mesh. Also expose the derived topology (per-edge transport, intra/inter-node classification) via a debug log adjacent to the existing channel-type log at lines 544-551 so operators can validate placement.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names topology-aware per-edge transport selection and adaptive RayPPCommunicator-vs-Ray-NCCL choice as headroom, and today's implementation applies a single VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE uniformly to every non-last PP edge regardless of whether the edge crosses a node boundary. Alpa's device-mesh formulation, partition the cluster so high-bandwidth links carry the communication-heavy tensor-parallel groups and lower-bandwidth links carry pipeline stages, maps directly onto the pp_tp_workers structure this method already owns: TP groups should live intra-node, PP edges should be classified per-edge and use shm intra-node and NCCL (or the wrapped PP communicator) inter-node. For the stated multi-turn agentic workload, reducing cross-node collective and handoff latency on the steady-state Ray DAG is a plausible lever for both median TTFT after graph creation and median TPOT, which are exactly the objectives listed in the caller context.

---

### 3. Topology- and message-size-aware per-PP-edge transport selection in the Ray compiled DAG
- **Finding:** `find-vllm_v1_executor-0008` — *Massively Scale Your Deep Learning Training with NCCL 2.4*
- **Source URL:** <https://developer.nvidia.com/blog/?p=13452>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `RayDistributedExecutor._compiled_ray_dag` (vllm/v1/executor/ray_executor.py:527-620), replace the current uniform `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` choice with a per-PP-edge policy that mirrors NCCL's hierarchical algorithm selection. Concretely: (1) When building the DAG in the `for pp_rank, tp_group in enumerate(self.pp_tp_workers)` loop, inspect the Ray placement / node IDs of workers in `self.pp_tp_workers[pp_rank]` and `self.pp_tp_workers[pp_rank+1]` (derivable from bundle indices already used in worker placement) to classify each PP handoff edge as intra-node vs inter-node. (2) Estimate the per-edge IntermediateTensors payload size from `self.model_config` / `self.parallel_config` (hidden_size, dtype, expected max num_tokens per step). (3) For intra-node edges, prefer `shm` (skip `with_tensor_transport`); for inter-node edges with large payloads, apply `with_tensor_transport(transport="nccl")` and, when `VLLM_USE_RAY_WRAPPED_PP_COMM` is set, keep the vLLM-wrapped `RayPPCommunicator` path so PP handoffs reuse the existing NCCL `_PP` GroupCoordinator (hierarchical intra/inter-node ring). For inter-node edges with small handoff payloads (e.g., single-token decode steps in an agentic workload), fall back to Ray's default channel to avoid NCCL-per-step launch overhead. (4) Log the resolved per-edge transport map alongside the existing `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` / `VLLM_USE_RAY_WRAPPED_PP_COMM` info lines. Preserve the existing `"auto"|"nccl"|"shm"` env override semantics: when the env is set to a non-`auto` value, honor it uniformly (current behavior); the new policy only activates in `auto` mode. Keep `_overlap_gpu_communication` unchanged, but gate it off when the resolved policy is all-`shm` (where GPU-comm overlap has no effect).

**Proposal rationale.**

The candidate's `evolve_rationale` explicitly calls out headroom in "topology-aware per-edge transport selection" and "adaptive use of RayPPCommunicator versus Ray NCCL", which is precisely what NCCL 2.4's hierarchical-ring/tree idea generalizes: choose the communication path based on the intra-node/inter-node hierarchy and the message size. Today the DAG builder applies a single transport uniformly to every non-last PP edge, ignoring whether that edge crosses a NIC or stays on NVLink, and ignoring per-step handoff size. For a multi-turn agentic workload dominated by short decode steps, an inter-node NCCL handoff per step is often worse than shm-adjacent alternatives, while for large prefill batches an NCCL ring is better than default shm marshaling. Applying the finding's hierarchy-aware selection at DAG compile time can reduce both post-compile TTFT (fewer redundant NCCL setups on intra-node edges) and steady-state TPOT (better-matched transport per PP hop) without changing the DAG's I/O contract, which the correctness oracle requires.

---

### 4. Topology-aware per-PP-edge transport selection in _compiled_ray_dag
- **Finding:** `find-vllm_v1_executor-0009` — *[Bug]: RayExecutorV2 multi-node DP hangs on shm_broadcast — cross-node ranks can't share single-host shared memory*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/43420>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify RayDistributedExecutor._compiled_ray_dag (vllm/v1/executor/ray_executor.py, lines 560-620) so that transport for each intermediate PP edge is chosen per-edge based on the co-location of the producing and consuming workers rather than a single global VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE. During DAG construction, for each PP stage transition between tp_group[pp_rank] and tp_group[pp_rank+1], resolve the Ray node id of the producer worker and the consumer worker (using data already available on RayWorkerWrapper / placement group bundles), then: (a) if producer and consumer are co-located on the same node, keep the default shared-memory transport (skip with_tensor_transport, matching today's 'shm' fast path); (b) if they are on different nodes, invoke with_tensor_transport with a cross-node-safe transport — either 'nccl' when CUDA IPC / NCCL is available across those hosts, or Ray's default RPC channel when it is not. Treat channel_type='auto' as this hybrid mode; keep 'shm' and 'nccl' as explicit overrides that force a single transport everywhere for debugging. Log the resolved (pp_rank -> transport) mapping once at compile time. Preserve the existing RayPPCommunicator registration path (VLLM_USE_RAY_WRAPPED_PP_COMM) and _overlap_gpu_communication behavior unchanged. The oracle stated in evolve_rationale (same final ModelRunnerOutput and PP handoff semantics for a fixed SchedulerOutput) is preserved because only the transport implementation of intermediate PP edges changes, not their values or ordering.

**Proposal rationale.**

The finding documents a concrete failure mode of assuming a single-host shared-memory transport across a multi-node Ray deployment and proposes a hybrid same-node-shm / cross-node-non-shm design. The candidate today applies one transport uniformly to all non-last PP edges (lines 580-591), which is exactly the class of decision the finding critiques: it is optimal only when PP topology happens to be all-intra-node (shm wins) or all-inter-node (nccl/RPC wins), and pessimal in mixed layouts common to multi-node PP/TP serving. Making the transport a per-edge function of worker co-location closes that gap without introducing new IPC primitives — it reuses Ray's existing with_tensor_transport hook already invoked here — and directly targets the workload's median TPOT, since PP handoffs sit on the steady-state critical path of every decode step in a multi-turn agentic run.

---

### 5. Set explicit _max_inflight_executions on Ray compiled DAG to bound async concurrency
- **Finding:** `find-vllm_v1_executor-0010` — *ray.dag.input_node.InputNode.experimental_compile*
- **Source URL:** <https://docs.ray.io/en/latest/ray-core/compiled-graph/doc/ray.dag.input_node.InputNode.experimental_compile.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend RayDistributedExecutor._compiled_ray_dag (vllm/v1/executor/ray_executor.py:527-620) so the final experimental_compile call passes an explicit _max_inflight_executions argument alongside the existing enable_asyncio and _overlap_gpu_communication kwargs. Concretely: (a) introduce a knob (e.g. a new VLLM_RAY_COMPILED_DAG_MAX_INFLIGHT env var, defaulting to a value derived from the async scheduler's concurrency budget for multi-turn agentic workloads) plumbed through the executor; (b) log its value next to the existing RAY_CGRAPH_get_timeout / channel_type / overlap logs so operators can see the effective backpressure setting; (c) pass it into forward_dag.experimental_compile(...) at line 617-620. This exposes the Ray Compiled Graph capacity as a first-class, tunable parameter of the compiled DAG owned by this method, rather than relying on Ray's implicit default.

**Proposal rationale.**

The candidate owns the exact experimental_compile call whose parameters the finding documents, and today only enable_asyncio and _overlap_gpu_communication are set — inflight capacity is left implicit. For the stated caller objective (reduce median TTFT / TPOT on multi-turn agentic workloads under enable_asyncio=True), several nonblocking execute_model futures can accumulate across PP stages; without an explicit _max_inflight_executions the DAG either over-buffers (raising queueing latency and tail TPOT) or stalls on Ray's default cap in ways that are hard to reason about. The finding contributes a concrete, transferable knob — _max_inflight_executions — that maps directly onto this candidate's compile site and gives a principled backpressure lever tied to scheduler concurrency, addressing a real gap in the current DAG construction path.

---

## Agent proposals

### 1. Eagerly compile and warm up the Ray CGraph in _init_executor to hide compile latency from first-request TTFT
- **Agent:** claude

**Detailed description.**

Move invocation of RayDistributedExecutor._compiled_ray_dag (vllm/v1/executor/ray_executor.py:527-620) off the first-request critical path so its cost does not land on the first request's TTFT. Concretely: (1) At the end of _init_executor (around vllm/v1/executor/ray_executor.py:97, after _init_workers_ray and connector/sampler setup complete), add a step that calls self.forward_dag = self._compiled_ray_dag(enable_asyncio=False) synchronously so the MultiOutputNode construction, per-edge with_tensor_transport annotation (lines 580-591), optional RayPPCommunicator registration (lines 595-611), and the experimental_compile call at lines 617-620 — which allocates Ray channels, sets up cross-actor buffers, and (when channel_type != 'shm') establishes NCCL/CUDA-IPC state — all happen during executor construction rather than lazily inside _execute_dag at lines 441-442. (2) Immediately after eager compile, issue one no-op warmup execution through self.forward_dag.execute(...) using a minimal, model-safe SchedulerOutput (e.g. the empty/finished-only scheduler output already understood by the workers) and ray.get the returned refs; this forces first-time channel materialization, cudagraph capture on workers, and NCCL comm creation for the PP handoff edges so the first real request pays only steady-state cost. (3) Gate the whole thing behind an env knob (e.g. VLLM_RAY_EAGER_COMPILE_DAG, default enabled when the executor is used in an online-serving code path and disabled for offline/batch use where amortized cost is fine), and log the measured compile+warmup wall time next to the existing RAY_CGRAPH_get_timeout / channel_type / overlap info logs at lines 540-551. (4) When enable_asyncio must switch (the eager compile uses enable_asyncio=False today per line 442, but async paths may want enable_asyncio=True), fall back to lazy recompile only in the async-first case, and log that the eager DAG is being discarded so operators can see the extra cost. Do not change the DAG topology, transport selection, or the arguments passed to experimental_compile — only when the method is invoked and one added warmup pass.

**Novelty rationale.**

None of the five existing deep_research_proposals address when _compiled_ray_dag is invoked; all five modify what the compile does (overlap flag defaulting, per-edge transport selection by topology/message-size/colocation, _max_inflight_executions backpressure). Today the DAG is built lazily inside _execute_dag on the first request (vllm/v1/executor/ray_executor.py:441-442), so DAG construction, channel allocation, and cross-node NCCL/CUDA-IPC setup all count against the first request's TTFT. The proposed change is orthogonal: it shifts compile+first-touch cost into executor construction (a startup budget the caller already pays) and adds a warmup step to force NCCL comm creation and channel materialization before user traffic arrives. This directly targets the caller's median TTFT objective for multi-turn agentic workloads — where a new session frequently corresponds to a first request against a freshly warmed executor — in a way that composes with, rather than duplicates, the transport/overlap/backpressure proposals already on the candidate.

---

### 2. Stop forwarding scheduler metadata through PP tensor handoff edges
- **Agent:** codex

**Detailed description.**

Refactor the DAG built in `RayDistributedExecutor._compiled_ray_dag` (`vllm/v1/executor/ray_executor.py:560-620`) so inter-PP edges carry only `IntermediateTensors`, while the original `(SchedulerOutput, GrammarOutput)` input is provided directly to every PP stage as a side input from the `InputNode`. Today `RayWorkerWrapper.execute_model_ray` returns `(scheduler_output, grammar_output, intermediate_tensors)` for non-final PP stages, so every PP boundary serializes and transports the same scheduler and grammar metadata along with the activation tensors. Add a Ray-worker entrypoint that accepts `(input_data, intermediate_tensors)` for `pp_rank > 0` and reconstructs the existing local call to `model_runner.execute_model(scheduler_output, intermediate_tensors)`; for non-final stages it should return only the next `IntermediateTensors`, and for the final stage it should return the normal `ModelRunnerOutput`. In `_compiled_ray_dag`, initialize the first stage from `input_data`, then bind later stages with both the original `input_data` and the previous stage's intermediate output, applying `with_tensor_transport(...)` only to the intermediate-tensor edge. Preserve the current output contract of `_execute_dag` and connector aggregation; this changes the DAG payload shape on internal PP edges, not the model execution semantics.

**Novelty rationale.**

The existing deep_research_proposals focus on overlap defaults, topology/message-size transport selection, RayPPCommunicator choice, and `_max_inflight_executions`. Agent A focuses on when the DAG is compiled and warmed. This proposal is different: it reduces the data carried by each internal PP edge by changing the DAG's internal payload factoring, so repeated scheduler/grammar metadata is not serialized through every PP handoff. It composes with all transport and eager-compile ideas because it changes what each PP edge transports, not which transport is chosen or when compilation occurs.

---
