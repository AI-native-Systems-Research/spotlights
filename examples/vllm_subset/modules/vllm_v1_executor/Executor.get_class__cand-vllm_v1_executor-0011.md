# Executor.get_class

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/abstract.py`](vllm/v1/executor/abstract.py) (lines 50–92)
- **Symbol:** `Executor.get_class`
- **Kind:** plugin_seam
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_executor-0011`

## Description
Runtime executor implementation selection surface keyed by parallel_config.distributed_executor_backend, with an additional env-controlled Ray v2 branch and qualified-name extension path.

## Current approach
The interface symbol is Executor in vllm/v1/executor/abstract.py. Reference implementations include MultiprocExecutor in vllm/v1/executor/multiproc_executor.py, RayDistributedExecutor in vllm/v1/executor/ray_executor.py, RayExecutorV2 in vllm/v1/executor/ray_executor_v2.py, and UniProcExecutor in vllm/v1/executor/uniproc_executor.py. The selector is parallel_config.distributed_executor_backend with recognized values ray, mp, uni, external_launcher, an Executor subclass, or a qualified-name string resolved by resolve_obj_by_qualname; VLLM_USE_RAY_V2_EXECUTOR_BACKEND selects RayExecutorV2 for the ray backend.

## Estimated impact explanation
The selector itself is not hot, but it is the lowest-blast-radius path to evaluate alternative control-plane and data-plane executor designs. Such variants can move median TTFT/TPOT substantially; this site is medium because it enables the optimization rather than performing it directly.

## Evolve rationale
The concrete registration site is the get_class branch table that binds config values to Executor subclasses, plus the qualified-name branch for externally supplied Executor subclasses. This is a legitimate seam for adding a new executor topology, such as an RDMA-oriented or batched-broadcast executor, while keeping the Executor.collective_rpc/execute_model/sample_tokens contract fixed. Correctness oracle: the new executor must satisfy the Executor ABC, return the same ModelRunnerOutput as UniProcExecutor or MultiprocExecutor for fixed SchedulerOutput inputs, and pass existing executor integration tests for health, KV cache initialization, LoRA control calls, and async scheduling support declarations.

## Deep research proposals

### 1. Add Ray Compiled Graph executor with overlapped GPU communication for pipeline-parallel handoffs
- **Finding:** `find-vllm_v1_executor-0002` — *Experimental: Overlapping communication and computation*
- **Source URL:** <https://docs.ray.io/en/master/ray-core/compiled-graph/overlap.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the Executor.get_class branch table in vllm/v1/executor/abstract.py (lines 50-92) to register a new executor topology (e.g., "ray_cgraph" or a variant of the existing ray_v2 branch) that binds to a new RayCompiledGraphExecutor subclass of Executor. The new executor builds a Ray Compiled Graph over the worker actors used for pipeline-parallel stages, declaring inter-stage tensor handoffs as DAG edges with NCCL/GPU transports, and calls dag.experimental_compile(_overlap_gpu_communication=True) so that inter-stage sends are scheduled as overlappable GPU operations rather than Python-visible blocking calls. The public contract stays fixed: the new class implements Executor.collective_rpc/execute_model/sample_tokens and returns the same ModelRunnerOutput for a given SchedulerOutput as UniProcExecutor/MultiprocExecutor. Selection remains driven by parallel_config.distributed_executor_backend, mirroring the pattern already used for the VLLM_USE_RAY_V2_EXECUTOR_BACKEND env-controlled branch (either as a new distinct backend value or as an env-controlled sub-mode of the ray backend). No changes are required outside the get_class seam and the new executor module; the existing executor integration tests (health, KV cache initialization, LoRA control calls, async scheduling support) serve as the correctness oracle.

**Proposal rationale.**

The candidate is explicitly the plugin seam intended to enable alternative executor topologies while holding the Executor ABC fixed, and its impact rationale calls out control-plane/data-plane variants that shift TTFT/TPOT. The finding contributes a concrete, transferable mechanism (Ray Compiled Graph with _overlap_gpu_communication=True) that directly targets the pipeline-stage tensor handoff pattern already present in Ray-backed executors: on multi-turn agentic workloads with pipeline parallelism, hiding inter-stage NCCL transfers under downstream compute reduces per-token stage latency and thus median TPOT, and reduces the first-token critical path for TTFT. Because Ray Compiled Graph is a Ray-native construct, it slots naturally alongside the existing RayDistributedExecutor and RayExecutorV2 registrations in get_class without perturbing the mp/uni/external_launcher branches or the qualified-name extension path, matching the candidate's "lowest-blast-radius path" framing.

---

### 2. Register a poll-based multiproc executor backend that drains all rank response queues in one pass
- **Finding:** `find-vllm_v1_executor-0006` — *zmq_poller(3)*
- **Source URL:** <https://zeromq.github.io/libzmq/zmq_poller.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the Executor.get_class dispatch table in vllm/v1/executor/abstract.py (lines 50-92) with a new distributed_executor_backend value (e.g. "mp_poll") that resolves to a new PolledMultiprocExecutor sibling of MultiprocExecutor. The new executor keeps the Executor ABC contract (collective_rpc, execute_model, sample_tokens, check_health, KV/LoRA control calls) and reuses MultiprocExecutor's rpc_broadcast_mq / per-rank worker_response_mq layout, but replaces the sequential per-rank collection loop in MultiprocExecutor.collective_rpc.get_response (multiproc_executor.py:405-421, which today does `for mq in response_mqs: mq.dequeue(...)`) with a zmq_poller-based multi-queue drain: register each response queue's underlying zmq socket with a single zmq_poller, call zmq_poller_wait_all with the RPC deadline, drain every socket the poller reports as ready in the returned events array, repeat until every rank has produced a response, then assemble results back into rank order before returning. The output_rank == unique-reply short path is preserved (single-queue case degenerates to zmq_poller_wait with one socket). Error propagation (ResponseStatus.SUCCESS vs. RuntimeError) and TimeoutError semantics on deadline expiry are preserved. Since MessageQueue currently exposes a blocking dequeue, this proposal wires poll access at the zmq layer that already backs MessageQueue (or adds a narrow poll_and_dequeue helper on MessageQueue used only by this executor) so the change is scoped to the new executor plus a small MessageQueue accessor. The registration site is the branch table in Executor.get_class; the qualified-name path can also be used to prototype externally before promoting to a named backend.

**Proposal rationale.**

The Executor.get_class seam's stated purpose in the candidate description is exactly this: enable alternative executor topologies (e.g. batched-broadcast) behind a fixed collective_rpc/execute_model/sample_tokens contract. The finding contributes a concrete, transferable primitive (zmq_poller_wait_all returns the count of ready events in one array, per the cited lines 95-101) that maps onto a real inefficiency in the current reference implementation: MultiprocExecutor.collective_rpc.get_response serializes per-rank dequeues, so worker N's response cannot be observed until worker 0..N-1's are drained even when their sockets became ready simultaneously. For multi-turn agentic workloads where every decode step incurs an execute_model + sample_tokens fan-in, cutting that per-step serialized wait shortens both median TTFT (prefill collective) and, more meaningfully, median TPOT (per-token decode collective). The change is guarded by a distinct backend value so existing users are unaffected, and correctness is checkable against the candidate's stated oracle: the new executor must return the same ModelRunnerOutput as UniProcExecutor/MultiprocExecutor for fixed SchedulerOutput inputs and pass the existing executor integration tests for health, KV cache init, LoRA control, and async scheduling.

---

### 3. Add a topology-aware device-mesh executor backend to Executor.get_class
- **Finding:** `find-vllm_v1_executor-0007` — *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning*
- **Source URL:** <https://www.alphaxiv.org/abs/2201.12023>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the executor selection seam at vllm/v1/executor/abstract.py:47-92 (Executor.get_class) with a new distributed_executor_backend value (e.g. "mesh" or "topology_aware") that resolves to a new Executor subclass implementing Alpa-style device-mesh rank placement. The new class would subclass the existing MultiprocExecutor/RayDistributedExecutor infrastructure (reusing collective_rpc/execute_model/sample_tokens) but override _init_executor to: (1) probe intra-node interconnect bandwidth (NVLink/NVSwitch vs PCIe) and inter-node NIC topology, (2) partition the world into device meshes so that tensor-parallel groups (parallel_config.tensor_parallel_size) are colocated within high-bandwidth intra-node meshes, and (3) place pipeline-parallel stages (parallel_config.pipeline_parallel_size) across lower-bandwidth cross-node boundaries. Concretely, this means computing a rank permutation before torch.distributed process group creation and passing it through to workers so TP all-reduces stay on NVLink while PP send/recv crosses NICs. The Executor ABC surface (collective_rpc, execute_model, sample_tokens, check_health, KV cache init, LoRA control, supports_async_scheduling) remains unchanged, so existing executor integration tests continue to serve as the correctness oracle; the qualified-name path already lets us prototype the class out-of-tree before promoting it to a named branch. Env var (e.g. VLLM_TOPOLOGY_AWARE_EXECUTOR) can gate the ray/mp variants similar to the existing VLLM_USE_RAY_V2_EXECUTOR_BACKEND branch.

**Proposal rationale.**

The candidate's evolve_rationale explicitly identifies get_class as the low-blast-radius seam for evaluating alternative control-plane/data-plane executor designs. Alpa's contribution (device-mesh partitioning with TP on high-bandwidth intra-mesh links and PP across low-bandwidth boundaries) is exactly the kind of executor-topology variant that this seam is designed to admit. For the caller's multi-turn agentic workload targeting median TTFT/TPOT, cross-node TP all-reduces during prefill (TTFT) and decode (TPOT) are a known latency source on multi-node deployments where naive rank assignment straddles NIC boundaries; keeping TP intra-node and pushing PP cross-node directly reduces the collective latency that shows up in both metrics. The change fits the seam's contract (Executor ABC, same ModelRunnerOutput for fixed SchedulerOutput) and adds one branch plus one subclass rather than modifying hot paths.

---

### 4. Register a hybrid intra-/inter-node executor variant via Executor.get_class
- **Finding:** `find-vllm_v1_executor-0009` — *[Bug]: RayExecutorV2 multi-node DP hangs on shm_broadcast — cross-node ranks can't share single-host shared memory*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/43420>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the get_class branch table in vllm/v1/executor/abstract.py (lines 50-92) to bind a new distributed_executor_backend value (e.g. "ray_hybrid") to a HybridRayExecutor Executor subclass. The new executor keeps the Executor ABC surface (collective_rpc/execute_model/sample_tokens) intact and reuses MultiprocExecutor's shared-memory MessageQueue for same-node worker fan-out while routing cross-node collectives through Ray RPC (or Ray collective groups), matching the composition already visible in RayDistributedExecutor and RayExecutorV2. Selection remains a pure config-driven branch in get_class: parallel_config.distributed_executor_backend == "ray_hybrid" (optionally gated by the existing VLLM_USE_RAY_V2_EXECUTOR_BACKEND-style env for opt-in), preserving the qualified-name extension path so external subclasses can supply their own hybrid transports. No changes to Executor's contract, ModelRunnerOutput shape, health/KV/LoRA control paths, or async-scheduling declarations are required; existing executor integration tests remain the correctness oracle.

**Proposal rationale.**

The finding identifies a concrete, transferable design — shm for intra-node, Ray RPC for inter-node — that directly addresses a topology assumption baked into today's Ray executors (single-host shm_broadcast). The candidate is precisely the plugin seam for evaluating such a topology without disturbing the Executor contract: a new branch in get_class is the lowest-blast-radius place to introduce and A/B a hybrid transport. For the caller's multi-turn agentic workload, avoiding cross-node shm fallbacks and using the right transport per hop is a plausible lever on median TTFT/TPOT (fewer stalls on control-plane fan-out, better tail behavior across nodes), which matches the candidate's medium-impact framing that this seam enables optimization rather than performing it.

---

## Agent proposals

### 1. Register an io_uring/eventfd-driven multiproc executor that batches worker wakeups via a single kernel syscall
- **Agent:** claude

**Detailed description.**

Extend the Executor.get_class branch table in vllm/v1/executor/abstract.py (lines 50-92) with a new distributed_executor_backend value (e.g. "mp_uring") that resolves to a new UringMultiprocExecutor sibling of MultiprocExecutor. The new executor preserves the Executor ABC (collective_rpc, execute_model, sample_tokens, check_health, KV/LoRA control, async scheduling declarations) and returns identical ModelRunnerOutput for a given SchedulerOutput. The mechanism: replace the per-rank blocking dequeue loop with an io_uring-backed readiness path. On _init_executor, create one io_uring instance in the driver process and register an eventfd per worker rank; each worker's MessageQueue is extended (or subclassed) so that after enqueuing a response it writes 1 to its registered eventfd. The driver's collective_rpc fan-in submits IORING_OP_POLL_ADD SQEs for every worker eventfd in a single io_uring_submit call, then blocks on io_uring_wait_cqes with the RPC deadline; each completion CQE identifies exactly which rank produced a response, and the driver drains only those workers' MessageQueue slots before re-arming the poll. On Linux kernels without io_uring (or on macOS dev boxes), the class falls back to epoll on the same eventfd set, preserving semantics. Unlike a zmq_poller-based drain, this path (a) avoids the zmq userspace layer entirely for the readiness signal since the payload channel is already shared-memory MessageQueue, (b) submits N poll registrations as one batched syscall via io_uring rather than N individual zmq socket registrations, and (c) supports IORING_SETUP_SQPOLL so the driver's fan-in can become fully syscall-free during steady-state decode. Error propagation (ResponseStatus.SUCCESS vs RuntimeError) and TimeoutError semantics on deadline expiry are preserved by mapping io_uring_wait_cqes timeout to the existing RPC deadline. The change is scoped to the new executor module plus one narrow eventfd accessor on MessageQueue; the get_class branch is a single elif. Selection can additionally be gated by an env var (e.g. VLLM_USE_URING_EXECUTOR) mirroring the VLLM_USE_RAY_V2_EXECUTOR_BACKEND pattern so users opt in.

**Novelty rationale.**

The listed proposals cover: (1) Ray Compiled Graph with overlapped GPU comm for pipeline-parallel handoffs — a Ray/DAG data-plane change; (2) a zmq_poller-based drain of MessageQueue response queues — a userspace zmq readiness change; (3) topology-aware device-mesh rank placement — a rank-assignment change; and (4) hybrid shm-intra-node + Ray-inter-node transport — a transport composition change. This proposal is orthogonal: it targets the same MultiprocExecutor per-rank serialized fan-in that #2 addresses, but the mechanism is fundamentally different — io_uring/eventfd-based kernel-level readiness with batched syscall submission and optional SQPOLL for syscall-free steady state, decoupled from zmq entirely. Neither the Ray-DAG proposal (data-plane GPU comm), the topology-aware proposal (placement, not readiness), nor the hybrid transport proposal (per-hop transport selection) contemplates a kernel-level readiness primitive or SQPOLL for the driver fan-in. The zmq_poller proposal remains bound to zmq's userspace poll and per-socket registration; this proposal replaces the readiness signal with eventfd and batches registrations through a single io_uring_submit, which is a distinct mechanism with distinct steady-state syscall behavior. For the caller's multi-turn agentic workload, removing per-decode-step driver-side syscalls on the fan-in path is a direct TPOT lever.

---

### 2. Register a fused multiproc execute-and-sample executor backend
- **Agent:** codex

**Detailed description.**

Extend `Executor.get_class` in `vllm/v1/executor/abstract.py` with a new opt-in backend value such as `"mp_fused_sample"` that resolves to a `FusedSampleMultiprocExecutor` sibling of `MultiprocExecutor`. The subclass would preserve the `Executor` ABC but override `execute_model` and `sample_tokens` for sampling workloads: `execute_model(scheduler_output, non_block=True)` caches the `SchedulerOutput` and returns `None` immediately when tokens are scheduled and sampling is required, while `sample_tokens(grammar_output, non_block=...)` sends one `collective_rpc` callable to workers that runs `worker.execute_model(scheduler_output)` followed by `worker.sample_tokens(grammar_output)` inside the same worker busy-loop dispatch, returning only from `output_rank` as today. For pooling/no-token paths it falls back to the parent `MultiprocExecutor.execute_model`. This removes one driver-to-worker broadcast and one worker-to-driver response per decode step for common non-structured or already-ready grammar paths, while keeping output shape, `ModelRunnerOutput` semantics, KV aggregation, health checks, LoRA calls, and async-scheduling declarations unchanged. The branch can be gated behind a distinct backend name or env flag so existing `mp` behavior remains the baseline.

**Novelty rationale.**

The existing proposals cover Ray compiled graph GPU handoffs, zmq poll fan-in, topology-aware rank placement, hybrid intra/inter-node transport, and Agent A's io_uring/eventfd fan-in readiness path. This proposal is different because it reduces the number of executor RPC rounds themselves in the multiproc decode loop, rather than making fan-in polling faster, changing transport selection, changing placement, or using Ray DAG execution. It specifically ports the already-compatible deferred execution shape visible in the Ray executor to a new multiproc backend by fusing `execute_model` and `sample_tokens` into one worker dispatch, which targets median TPOT by eliminating an entire per-token control-plane round trip rather than optimizing the readiness mechanism for that round trip.

---
