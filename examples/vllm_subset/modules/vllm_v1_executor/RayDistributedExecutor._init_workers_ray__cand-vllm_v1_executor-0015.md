# RayDistributedExecutor._init_workers_ray

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/ray_executor.py`](vllm/v1/executor/ray_executor.py) (lines 143–379)
- **Symbol:** `RayDistributedExecutor._init_workers_ray`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_executor-0015`

## Description
RayDistributedExecutor worker placement and rank-topology construction, including bundle selection, actor creation, driver-first sorting, rank remapping, per-node GPU mapping, and pp_tp_workers layout for compiled-DAG construction.

## Current approach
The method chooses bundle_indices from VLLM_RAY_BUNDLE_INDICES or the first GPU bundles in the placement group, creates one RayWorkerWrapper actor per rank, fetches worker IPs, sorts workers by driver-node preference, worker count per IP, and IP, adjusts worker ranks through collective_rpc('adjust_rank'), discovers physical GPU IDs, initializes workers, and finally builds pp_tp_workers as sequential TP groups per PP stage.

## Estimated impact explanation
Placement and rank layout determine the communication pattern used by every Ray compiled-DAG token step. Better topology choices can reduce cross-node synchronization and tensor transport time, moving median TPOT for multi-node PP/TP agentic workloads; the effect is smaller on single-node runs.

## Evolve rationale
The concrete policy constructs are the bundle_indices selection at lines 168-188, sort_by_driver_then_worker_ip at lines 234-251, the rerank_mapping adjustment at lines 253-259, and the pp_tp_workers construction loop at lines 370-379. These choices define the physical placement and logical rank layout consumed by _compiled_ray_dag. Headroom includes topology-aware rank assignment that keeps TP groups intra-node, minimizes PP handoff distance, and accounts for heterogeneous node interconnects instead of sorting only by driver locality and IP counts. Correctness oracle: the same world_size workers are created, each rank receives the same logical model-parallel role set, SchedulerOutput inputs produce value-equivalent ModelRunnerOutput values, and existing Ray PP/TP plus placement-group tests pass with unchanged failure behavior.

## Deep research proposals

### 1. Topology-aware rank remapping: pack TP groups intra-node, span PP across nodes
- **Finding:** `find-vllm_v1_executor-0007` — *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning*
- **Source URL:** <https://www.alphaxiv.org/abs/2201.12023>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the driver-first/IP-count sort at lines 234-251 and the naive contiguous rerank at lines 253-259 with an Alpa-style device-mesh assignment before `collective_rpc('adjust_rank', ...)` is called. Concretely, after `worker_ips` and `ip_counts` are gathered (lines 217-232), group workers by node IP into per-node buckets, then assign adjusted ranks so that each contiguous block of `tensor_parallel_size` consecutive ranks lands entirely on a single node (a TP submesh on high-bandwidth intra-node links, e.g. NVLink/xGMI) and the `pipeline_parallel_size` such blocks are spread across distinct nodes (the PP dimension crossing lower-bandwidth inter-node links). Keep the current tiebreakers as secondary keys: prefer the driver node for PP stage 0 so the engine still colocates with the first stage, and use `ip_counts`/lexicographic IP order to deterministically break ties among equivalent nodes. When `world_size == tp * pp` does not evenly divide the bundle-to-node mapping (heterogeneous bundle counts per node), fall back to the existing sort so behavior is unchanged. Feed the resulting permutation into `rerank_mapping` exactly as today; the downstream `pp_tp_workers` construction at lines 370-379 then naturally yields TP groups that are intra-node and PP stages that are inter-node without changing its loop structure. Bundle selection at lines 168-188 is untouched; only the logical-rank permutation applied via `adjust_rank` changes. Correctness oracle from `evolve_rationale` is preserved: same world_size actors, same model-parallel role set per rank, and the existing Ray PP/TP placement-group tests continue to pass because the change is a pure permutation of ranks over the same set of actors.

**Proposal rationale.**

The candidate's current sort (lines 234-251) only optimizes for driver locality and per-IP worker counts; it has no notion of which ranks form a TP group versus a PP stage, so on multi-node deployments TP groups can straddle node boundaries and pay inter-node collective latency on every token step through the compiled Ray DAG. Alpa's core insight — partition the cluster into device meshes with preferably high-bandwidth intra-mesh connections and reserve slower links for pipeline boundaries — directly targets that gap: TP all-reduces are the bandwidth-heavy, per-token collective and belong on NVLink-class links, while PP handoffs are point-to-point activations that tolerate inter-node hops. Applying this policy to the rerank step is a low-risk, contained change (a permutation feeding an existing `adjust_rank` RPC) that plausibly reduces per-token collective time in the exact multi-turn agentic, multi-node PP/TP scenario named in `estimated_impact_explanation`, improving median TPOT without altering worker creation, GPU discovery, or DAG construction.

---

### 2. Hierarchical 2D rank assignment keeping TP intra-node and ordering PP stages for bandwidth-aware handoffs
- **Finding:** `find-vllm_v1_executor-0008` — *Massively Scale Your Deep Learning Training with NCCL 2.4*
- **Source URL:** <https://developer.nvidia.com/blog/?p=13452>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In RayDistributedExecutor._init_workers_ray (vllm/v1/executor/ray_executor.py:143-379), replace the current sort_by_driver_then_worker_ip key (lines 234-251) and the pp_tp_workers construction loop (lines 370-379) with a hierarchical, topology-aware rank assignment that mirrors NCCL's 2D (intra-node/inter-node) ring model. Concretely: (1) After collecting worker_ips (lines 217-225), group worker metadata by node IP, then form contiguous rank blocks of size tensor_parallel_size out of each node's workers so every TP group is fully intra-node whenever a node has >= tensor_parallel_size workers; only split a TP group across nodes when unavoidable, and log when this fallback triggers. (2) Order the resulting TP blocks across nodes to form pipeline_parallel_size PP stages such that adjacent PP stages sit on nodes that are 'close' in the available topology signal — start by preferring the driver node for PP stage 0 (preserving current driver-first behavior) and grouping remaining nodes by ip_counts and lexical IP to keep a deterministic, dense ordering, but structure the code so a richer distance/interconnect signal can slot in later. (3) Feed this ordering into rerank_mapping via collective_rpc('adjust_rank', ...) (lines 253-259) so the logical rank of each worker matches its physical position in the 2D layout. (4) Rewrite the pp_tp_workers builder (lines 370-379) so it consumes the same block structure directly instead of re-deriving rank = pp_rank * tp_size + tp_rank from a flat list, guaranteeing the compiled-DAG PP handoffs cross the fewest inter-node hops. Preserve VLLM_RAY_BUNDLE_INDICES behavior (lines 168-188) as an explicit user override that bypasses the new grouping. Correctness oracle from the candidate is unchanged: same world_size workers, same logical model-parallel role set per rank, and existing Ray PP/TP plus placement-group tests must pass.

**Proposal rationale.**

The finding describes NCCL's hierarchical 2D (intra-node/inter-node) rings as a way to align communication with the physical topology, choosing algorithms and rank layouts by hierarchy. The candidate's current placement policy sorts only by (driver_node, ip_counts, ip) and then slices flat ranks into pp_tp_workers, which incidentally clusters same-node workers but does not guarantee that a TP group of size tensor_parallel_size fits within one node, and does nothing to keep adjacent PP stages topologically close. Multi-turn agentic PP/TP workloads pay the resulting cross-node TP collectives and PP handoffs on every compiled-DAG token step, hitting the median TPOT. Explicitly building TP groups as intra-node blocks and ordering PP stages by node proximity — the direct 2D-ring analogue of the finding — targets the evolve_rationale's stated headroom (topology-aware rank assignment, intra-node TP, minimized PP handoff distance) at exactly the two code sites (sort key at lines 234-251, pp_tp_workers build at lines 370-379) the candidate flags, without changing the worker-count or role contract.

---

## Agent proposals

### 1. NUMA/PCIe-affinity-aware bundle selection to cut host-to-device prefill staging latency for TTFT
- **Agent:** claude

**Detailed description.**

Modify the bundle_indices selection block in RayDistributedExecutor._init_workers_ray at vllm/v1/executor/ray_executor.py:168-188 so that, when VLLM_RAY_BUNDLE_INDICES is not set, the executor does not simply pick the first world_size GPU bundles from the placement group. Instead, before creating actors, launch a short-lived Ray task on each candidate bundle that reads (a) the PCI bus id / NUMA node of the visible CUDA device (via pynvml.nvmlDeviceGetNumaNodeId or /sys/class/pci_bus/*/device/numa_node), (b) the NUMA node of the CPU cores the bundle's CPU reservation would land on (sched_getaffinity + /proc/self/status Cpus_allowed_list mapped via /sys/devices/system/node), and (c) the NUMA node of the primary NIC used for Ray object-store traffic (readlink /sys/class/net/<iface>/device -> PCI addr -> numa_node). Score each bundle by GPU-CPU NUMA match and GPU-NIC NUMA match, then pick the top world_size bundles greedily while still respecting the per-node grouping the downstream sort at lines 234-251 relies on (prefer bundles whose GPU/CPU NUMA nodes agree, breaking ties by matching NIC NUMA, then by driver-node preference). Cache the affinity map keyed by placement-group id in the executor so restarts inside the same PG reuse the probe results. Leave the sort/rerank/pp_tp_workers logic at lines 234-379 untouched - this proposal only changes which of the placement-group's GPU bundles become the actor set, not the logical rank permutation over that set. When probes fail or a heterogeneous PG makes the greedy pick infeasible, fall back to the current 'first N GPU bundles' behavior with a warning. The correctness oracle is preserved: same world_size actors are created, each rank receives the same logical model-parallel role set, and existing Ray PP/TP placement-group tests still pass because the change is a permutation over equivalent GPU bundles.

**Novelty rationale.**

Both existing deep_research_proposals (find-0007 and find-0008) explicitly leave bundle_indices selection (lines 168-188) untouched and instead modify only the logical-rank permutation via adjust_rank (lines 234-259) and the pp_tp_workers construction loop (lines 370-379). They target inter-GPU collective bandwidth for TP/PP by keeping TP groups intra-node - a device-side communication concern. This proposal is orthogonal: it changes which physical bundles are chosen in the first place, and optimizes host-side staging (GPU-CPU NUMA affinity and GPU-NIC affinity), which primarily affects TTFT via faster prompt tensor upload and Ray-object-store transfer during prefill, not TP all-reduce latency. Neither existing proposal mentions NUMA, PCIe topology, NIC affinity, or bundle re-selection; both are pure rank-permutation strategies over an already-chosen bundle set.

---

### 2. Order ranks within each TP group by local GPU interconnect topology
- **Agent:** codex

**Detailed description.**

Extend `RayDistributedExecutor._init_workers_ray` in `vllm/v1/executor/ray_executor.py:143-379` so the rerank step can use intra-node GPU topology, not just node IPs. Before calling `collective_rpc('adjust_rank', ...)`, collect each actor's visible physical GPU id/bus id with a lightweight RPC, group workers by node IP as today, and within each node order candidate workers by an NVLink/PCIe proximity score from NVML topology APIs such as `nvmlDeviceGetTopologyCommonAncestor` or CUDA bus ids. Use that ordering only inside the existing same-node buckets, then build `rerank_mapping` so each contiguous TP group has a deterministic local GPU order that favors high-bandwidth adjacency for NCCL ring/tree construction. If topology probing fails, GPU ids are unavailable, or the node has fewer workers than `tensor_parallel_size`, fall back to the current driver/IP/count ordering. The change preserves actor count, bundle selection, and logical PP/TP roles; it only changes the order of ranks assigned to already-created workers before the existing `adjust_rank` call.

**Novelty rationale.**

The deep-research proposals cover node-level topology: keep TP groups intra-node and arrange PP stages across nodes. They do not specify ordering ranks among GPUs within the same node by NVLink/PCIe topology. Agent A's proposal changes which Ray placement-group bundles are selected based on CPU/GPU/NIC NUMA affinity, but leaves rank ordering untouched. This proposal is a finer-grained intra-node rank-ordering change over the selected actors, targeting TP collective efficiency without duplicating node packing or bundle selection.

---
