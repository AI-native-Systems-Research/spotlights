# RayExecutorV2._init_executor

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/ray_executor_v2.py`](vllm/v1/executor/ray_executor_v2.py) (lines 304–478)
- **Symbol:** `RayExecutorV2._init_executor`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_executor-0016`

## Description
RayExecutorV2 actor placement, MessageQueue topology, GPU mapping, worker initialization, and response-queue setup for the MQ-based Ray backend.

## Current approach
The method chooses bundle assignments from VLLM_RAY_BUNDLE_INDICES or get_bundles_sorted_by_node, counts driver-node-local workers to size the broadcast MessageQueue local readers, creates one RayWorkerProc actor per bundle, gathers physical GPU IDs with ray.get, initializes workers with local ranks and assigned_physical_gpu_ids, collects response MessageQueue handles, starts each actor's run loop, waits for all queues to become ready, and sets output_rank.

## Estimated impact explanation
The method is initialization code, but it fixes the MessageQueue locality and worker placement used for every RayExecutorV2 decode step. Optimizing it can reduce median TPOT in multi-node MQ-backed Ray deployments and reduce TTFT during startup; impact is medium because single-node or already-local placements see limited benefit.

## Evolve rationale
The concrete constructs are bundle assignment selection at lines 306-313, n_local MessageQueue sizing at lines 340-347, the actor creation loop at lines 364-405, the GPU discovery/init loops at lines 408-448, and response queue setup at lines 450-471. These policies define the control-plane transport locality used by the inherited MultiprocExecutor.collective_rpc hot path. Headroom includes topology-aware bundle ordering, better local-vs-remote MQ layout, batching Ray actor initialization barriers, and placement choices that reduce scheduler-output broadcast and response latency without changing worker semantics. Correctness oracle: each rank is initialized exactly once with the same VllmConfig-visible model-parallel role, MessageQueue readiness and response rank ordering are preserved, and RayExecutorV2 PP/TP integration tests return the same ModelRunnerOutput and failure callbacks.

## Deep research proposals

### 1. Precompile Ray actor graph for steady-state token loop in RayExecutorV2
- **Finding:** `find-vllm_v1_executor-0001` — *Ray Compiled Graphs: Optimized AI Workloads with Native GPU Communication*
- **Source URL:** <https://www.anyscale.com/blog/announcing-compiled-graphs>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend RayExecutorV2._init_executor (vllm/v1/executor/ray_executor_v2.py:304-478) to build a Ray Compiled Graph over the worker actors created during the actor loop (lines 364-405) once, immediately after workers are initialized and their response MessageQueues are wired up (lines 450-471). The compiled graph would capture the steady-state control-plane call used by the inherited MultiprocExecutor.collective_rpc hot path (execute_model / dequeue-scheduler-output on each worker + gather ModelRunnerOutput from output_rank) as a static DAG whose task submission overhead is ~50us instead of Ray's 1-2ms per submission. Concretely: (1) after ray.get on _init_worker (line 448) and MQ readiness (lines 463-467), construct a `ray.dag.InputNode` -> per-actor `worker.execute_model.bind(...)` -> `MultiOutputNode` graph, `experimental_compile()` it, and stash the handle on self; (2) route steady-state execute_model calls through this compiled DAG rather than issuing fresh ray.remote() invocations each decode step, while keeping the MessageQueue path for scheduler-output broadcast intact (or optionally moving broadcast onto the DAG's native NCCL/shared-memory transport). Correctness is preserved because rank identity, VllmConfig, local-rank/GPU assignments, and output_rank collection at lines 469-471 are unchanged; only the transport of the per-step call changes. Fall back to the current uncompiled path if `experimental_compile` is unavailable or PP/TP shapes preclude a static graph.

**Proposal rationale.**

The candidate's evolve_rationale explicitly flags 'batching Ray actor initialization barriers' and reducing 'scheduler-output broadcast and response latency' as headroom, and identifies the control-plane transport used every decode step as the target. Ray Compiled Graphs directly attack that gap: they amortize actor invocation, tensor transport, and communicator setup into a one-time compile so each subsequent step pays ~50us instead of ~1-2ms of Ray task submission overhead. For a multi-turn agentic workload driving many short decode steps, this compounds across every token and every turn, reducing median TPOT on multi-node Ray-backed deployments where control-plane RPC dominates - the exact regime the candidate targets. TTFT also benefits modestly because the compile happens once at init and the first decode already uses the fast path. The change is localized to _init_executor's post-worker-init tail plus a thin execute path swap, respecting the correctness oracle (same rank init, same MQ readiness ordering, same output_rank).

---

### 2. Topology-aware bundle-to-rank assignment: keep TP intra-node, PP across nodes
- **Finding:** `find-vllm_v1_executor-0007` — *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning*
- **Source URL:** <https://www.alphaxiv.org/abs/2201.12023>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In RayExecutorV2._init_executor (vllm/v1/executor/ray_executor_v2.py, lines 304-325), replace the current linear enumeration produced by get_bundles_sorted_by_node with a topology-aware rank layout that maps logical (tp, pp, pcp) coordinates onto bundles using the Alpa-style device-mesh view. Concretely: (1) group placement-group bundles by node_id (a proxy for a high-bandwidth mesh) using the same node information already gathered at lines 313-325; (2) for each node/mesh, pack contiguous tensor-parallel groups of size tp_size (and pcp within TP where applicable) so that ranks that participate in the same TP all-reduce/all-gather share intra-node NVLink/PCIe bandwidth; (3) stripe pipeline-parallel stages across nodes so that only the smaller PP point-to-point sends cross node boundaries. Implement as a small helper that consumes the (bundle_id_idx, node_id, node_ip) triples currently produced at line 313 and returns the same shape of list, sorted so that rank r's TP peers land on the same node whenever tp_size fits within a node. Preserve current behavior when VLLM_RAY_BUNDLE_INDICES is set (lines 306-311) to keep the user-pinned override authoritative. Downstream, the driver-local counter at line 341 (n_local for MessageQueue local readers) and the per-node local_rank/assigned_physical_gpu_ids computation at lines 415-448 already key off node_id, so they will naturally reflect the new placement without further changes; the actor creation loop at lines 364-405 and the response-queue collection at lines 450-471 remain unchanged. Add a small assertion that the final assignment still contains exactly world_size ranks with unique bundle_id_idx values, to preserve the correctness oracle (each rank initialized exactly once, MessageQueue readiness and response ordering unchanged).

**Proposal rationale.**

The Alpa paper's core placement idea—partition the cluster into device meshes with preferably high-bandwidth intra-mesh connections, and route communication-heavy parallelism (TP) inside a mesh while pipelining across mesh boundaries—maps directly onto the gap in this candidate. Today, bundle_to_node_id is only 'sorted_by_node' (line 313), which packs bundles by node but does not deliberately align TP groups with node boundaries or stripe PP across nodes; a suboptimal ordering can place TP peers across nodes and thereby route hot all-reduce traffic over slower inter-node links, inflating both TTFT (startup and prefill collectives) and TPOT (per-decode-step collective_rpc broadcast plus per-token TP collectives). Because the caller objective is median TTFT/TPOT on a multi-turn agentic workload—where every decode token pays the collective cost—reshaping the rank-to-bundle assignment so TP is intra-node and PP is inter-node targets exactly this bottleneck. The change is initialization-only, does not alter worker semantics or MessageQueue readiness ordering, and leaves the VLLM_RAY_BUNDLE_INDICES escape hatch intact, matching the candidate's stated correctness oracle.

---

### 3. Hybrid intra-node shm / inter-node Ray RPC broadcast for RayExecutorV2
- **Finding:** `find-vllm_v1_executor-0009` — *[Bug]: RayExecutorV2 multi-node DP hangs on shm_broadcast — cross-node ranks can't share single-host shared memory*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/43420>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Change RayExecutorV2._init_executor's broadcast transport setup (vllm/v1/executor/ray_executor_v2.py lines 338-348 and the actor spawn/init loops at 364-448) so the single-host shared-memory MessageQueue is only used for workers colocated on the driver node, and cross-node ranks receive scheduler outputs through a Ray-native fan-out. Concretely: (1) partition bundle_assignments by node_id, keeping n_local as it is used today but treating it as the shm reader count only; (2) still construct self.rpc_broadcast_mq as an intra-node MessageQueue sized to the driver-node local workers and pass its handle only to actors created for bundles where bundle['node_id'] == driver_node in the Step 5 loop (lines 364-405); (3) for bundles on remote nodes, pass a Ray-RPC-backed broadcast handle (e.g., a per-remote-node relay actor that owns a node-local MessageQueue, or a direct actor-method call fan-out) instead of the shm handle, so RayWorkerProc initialization does not attempt to attach to shared memory across hosts; (4) mirror this on the response side (Step 8, lines 450-461) by only creating MessageQueue.create_from_handle for same-node actors and routing remote-node responses through Ray object refs or a node-local response relay; (5) keep the Step 10 readiness barrier at lines 468-471 correct by only waiting on the MQs that actually exist for the driver node, and add per-remote-node relay readiness waits. Preserve the correctness oracle: each rank is still initialized exactly once with the same VllmConfig-visible parallel role, response ordering is preserved by rank, and RayExecutorV2 PP/TP integration tests still return the same ModelRunnerOutput and failure callbacks.

**Proposal rationale.**

The candidate's current approach sizes a single broadcast MessageQueue with n_local counted only on the driver node and connects on ray.util.get_node_ip_address() (lines 340-347), and it passes that one shm handle to every actor regardless of node (lines 388-394). The linked issue #43420 reports exactly this failure mode: cross-node ranks cannot share single-host shared memory and hang on shm_broadcast in multi-node DP. The finding's proposed fix — 'fall back to Ray RPC for inter-node collectives while keeping shm for intra-node' — is a concrete, transferable design that maps directly onto Steps 4, 5, 7, 8, and 10 of this method and addresses a real correctness/tail-latency gap in the RayExecutorV2 control plane used by the inherited MultiprocExecutor.collective_rpc hot path, which is on the median TPOT path for the caller's multi-turn agentic workload.

---

## Agent proposals

### 1. Pipeline actor bring-up: overlap GPU discovery, worker init, and run() startup to eliminate straggler barriers
- **Agent:** claude

**Detailed description.**

Restructure Steps 6-9 of RayExecutorV2._init_executor (vllm/v1/executor/ray_executor_v2.py:406-467) to remove the three sequential `ray.get(...)` fan-in barriers and instead pipeline actor bring-up per-worker, so slow-node discovery and heavy model loading on one node do not stall progress on the others. Concretely: (1) Add a lightweight actor method on RayWorkerProc (e.g. `initialize_and_get_mq`) that combines the current `get_node_and_physical_gpu_ids` + `initialize_worker` + `wait_for_init` flow into a single call that returns `(node_id, physical_gpu_ids, response_mq_handle, ready_status)`. (2) In the executor, fire per-actor `get_node_and_physical_gpu_ids.remote()` calls at line 408 as today but consume them with `ray.wait(..., num_returns=1)` in a loop instead of a single `ray.get([...])` barrier at lines 408-413. As each future resolves, record the (node_id, physical_gpu_ids) tuple and, once all workers on a given node have reported, immediately dispatch `initialize_worker.remote(local_rank, worker_env_vars, driver_env_vars, assigned_physical_gpu_ids=...)` for every worker on that node — do not wait for other nodes' discovery. (3) Similarly consume the `initialize_worker` futures with `ray.wait` and, as each returns, immediately call `handle.actor.wait_for_init.remote()` and `handle.run()` for that rank rather than waiting for the global barrier at line 448 followed by another barrier at line 452. (4) Collect response MQ handles as `wait_for_init` futures resolve into the rank-indexed `self.response_mqs` list, preserving ordering by rank not by completion time. (5) Keep the final `self.rpc_broadcast_mq.wait_until_ready()` + per-mq `wait_until_ready()` fence at lines 468-471 as the single global gate — it now runs concurrently with the tail of the slowest node's init. The per-node mapping consistency check at lines 443-447 still holds because it only fires when there is a single node; multi-node paths already tolerate incremental per-node mapping population. Correctness is preserved: each rank is still initialized exactly once, response MQs are still indexed by rank, output_rank derivation is unchanged, `initialize_ray_cluster` and Steps 1-5 are unchanged, and the failure path is preserved by raising `RuntimeError(f"Worker {i} failed to initialize")` from inside the `ray.wait` consumer when any future returns a non-READY status or errors. Wall-clock startup drops from `sum(max_per_barrier)` to `max_per_worker_end_to_end`, and workers that finish init early start subscribing to `self.rpc_broadcast_mq` while slower peers are still loading weights, reducing the Step 10 barrier wait time.

**Novelty rationale.**

None of the three existing deep_research_proposals touches the sequential `ray.get` fan-in barriers or the strict `initialize_worker` -> `wait_for_init` -> `run()` ordering. Proposal #1 (Ray Compiled Graphs) targets steady-state execute_model transport, not init-time barriers. Proposal #2 (topology-aware bundle mapping) only reorders the assignment list produced at line 313, leaving the barrier structure at lines 408, 448, 451 untouched. Proposal #3 (hybrid shm/RPC broadcast) partitions the MessageQueue transport by node but still relies on the same three sequential ray.get barriers to bring workers online. This proposal is orthogonal: it keeps transport, mapping, and DAG unchanged and instead exploits the fact that GPU discovery is per-actor, model loading is per-node, and MQ subscription can begin before all peers finish loading — collapsing three straggler-bound barriers into overlapping per-worker pipelines. On multi-node clusters where one node's `initialize_worker` (heavy weight load) or `get_node_and_physical_gpu_ids` (cold ray.remote RTT) is slow, this directly cuts TTFT during startup without altering any correctness invariant flagged in the candidate's oracle.

---

### 2. Cache and reuse per-node physical GPU discovery across RayExecutorV2 restarts
- **Agent:** codex

**Detailed description.**

In `RayExecutorV2._init_executor` (`vllm/v1/executor/ray_executor_v2.py:408-448`), add a small executor-side cache keyed by the concrete Ray placement group bundle assignment tuple `(node_id, bundle_id_idx)` sequence and actor resource labels so repeat initialization on the same live placement can skip the full per-actor `get_node_and_physical_gpu_ids.remote()` discovery fan-out. On the first initialization, keep the existing discovery path and store the resulting `node_workers` map plus each rank's `physical_gpu_ids`. On subsequent `_init_executor` calls for the same placement topology, validate that every actor still reports the expected `node_id` cheaply (or invalidate on mismatch), then compute `local_rank` and `assigned_physical_gpu_ids` directly from the cached per-node rank ordering before calling `initialize_worker.remote(...)`. Preserve the single-node consistency assertion at lines 443-447 by comparing cached physical GPU ID ordering against any refreshed sample. This targets median TTFT for multi-turn agentic deployments that repeatedly tear down and recreate RayExecutorV2 over a stable placement group, avoiding a cold Ray RPC round trip per rank and repeated GPU ID probing while leaving worker semantics, rank ordering, response queues, and steady-state TPOT untouched.

**Novelty rationale.**

The existing proposals do not cache or reuse placement/GPU discovery metadata. Proposal #1 builds a steady-state Ray Compiled Graph after initialization; proposal #2 changes bundle-to-rank ordering; proposal #3 changes broadcast/response transport for cross-node workers; Claude's proposal overlaps initialization phases with `ray.wait` but still performs GPU discovery for every actor on every initialization. This proposal is orthogonal: it reduces repeated TTFT overhead by memoizing the deterministic placement-to-physical-GPU mapping when the same placement topology is reused, without changing barrier structure, topology policy, or MQ/RPC transport.

---
