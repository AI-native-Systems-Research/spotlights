# vllm/v1/executor

[← All modules](../index.md)

## Module
- **Path:** `vllm/v1/executor`
- **Description:** Distributes work across worker processes via Ray, multiprocessing, or single-process backends.
- **Depends on:** _(none)_
- **Main files:**
  - `vllm/v1/executor/abstract.py` — Executor interface
  - `vllm/v1/executor/multiproc_executor.py` — Multiprocessing executor
  - `vllm/v1/executor/ray_executor.py` — Ray executor
  - `vllm/v1/executor/uniproc_executor.py` — Single-process executor
- **Run status:** DEGRADED
- **Findings:** 12
- **Issues:** 1

## Candidates

| Candidate | Impact | Deep research proposals |
|---|---|---:|
| [`RayDistributedExecutor._compiled_ray_dag`](vllm_v1_executor/RayDistributedExecutor._compiled_ray_dag__cand-vllm_v1_executor-0012.md) | high | 5 |
| [`Executor.get_class`](vllm_v1_executor/Executor.get_class__cand-vllm_v1_executor-0011.md) | medium | 4 |
| [`MultiprocExecutor.collective_rpc`](vllm_v1_executor/MultiprocExecutor.collective_rpc__cand-vllm_v1_executor-0001.md) | high | 3 |
| [`RayWorkerWrapper.execute_model_ray`](vllm_v1_executor/RayWorkerWrapper.execute_model_ray__cand-vllm_v1_executor-0013.md) | high | 3 |
| [`RayExecutorV2._init_executor`](vllm_v1_executor/RayExecutorV2._init_executor__cand-vllm_v1_executor-0016.md) | medium | 3 |
| [`detach_zero_copy_from_model_runner_output`](vllm_v1_executor/detach_zero_copy_from_model_runner_output__cand-vllm_v1_executor-0006.md) | medium | 2 |
| [`RayDistributedExecutor._execute_dag`](vllm_v1_executor/RayDistributedExecutor._execute_dag__cand-vllm_v1_executor-0007.md) | high | 2 |
| [`RayDistributedExecutor._init_workers_ray`](vllm_v1_executor/RayDistributedExecutor._init_workers_ray__cand-vllm_v1_executor-0015.md) | high | 2 |
| [`FutureWrapper.result`](vllm_v1_executor/FutureWrapper.result__cand-vllm_v1_executor-0002.md) | medium | 1 |
| [`WorkerProc.worker_busy_loop`](vllm_v1_executor/WorkerProc.worker_busy_loop__cand-vllm_v1_executor-0003.md) | medium | 1 |
| [`WorkerProc.enqueue_output/handle_output/async_output_busy_loop`](vllm_v1_executor/WorkerProc.enqueue_output_handle_output_async_output_busy_loop__cand-vllm_v1_executor-0004.md) | high | 1 |
| [`FutureWrapper.result`](vllm_v1_executor/FutureWrapper.result__cand-vllm_v1_executor-0008.md) | medium | 1 |
| [`RayDistributedExecutor.collective_rpc`](vllm_v1_executor/RayDistributedExecutor.collective_rpc__cand-vllm_v1_executor-0009.md) | low | 0 |
| [`AsyncOutputFuture.result/UniProcExecutor.collective_rpc`](vllm_v1_executor/AsyncOutputFuture.result_UniProcExecutor.collective_rpc__cand-vllm_v1_executor-0014.md) | medium | 0 |

## Findings (full list)

1. **Ray Compiled Graphs: Optimized AI Workloads with Native GPU Communication**
   - Source type: blog
   - URL: <https://www.anyscale.com/blog/announcing-compiled-graphs>
   - Technique: Use a static, precompiled actor graph for the steady-state token loop so worker invocation, tensor transport, communicator setup, and deadlock-free scheduling are prepared once instead of rebuilt per step. This directly targets executor control-plane overhead and GPU handoff latency for autoregressive workloads.
   - Evidence: Quote: "Compiled Graphs provide minimal task submission overhead (~50 us) compared to Ray’s standard task submission overheads (1~2 ms)." Pointer: introduction, lines 16-17.
2. **Experimental: Overlapping communication and computation**
   - Source type: docs
   - URL: <https://docs.ray.io/en/master/ray-core/compiled-graph/overlap.html>
   - Technique: Enable and tune Ray Compiled Graph's GPU communication overlap, especially for pipeline-stage tensor handoffs where communication can be hidden under downstream compute. The transferable idea is to select DAG edges and transports so inter-stage transfers are scheduled as overlappable GPU operations rather than Python-visible blocking work.
   - Evidence: Quote: "specify `_overlap_gpu_communication=True` when calling `dag.experimental_compile()`." Pointer: Ray Compiled Graph overlap docs, enablement section.
3. **CUDA C++ Best Practices Guide**
   - Source type: docs
   - URL: <https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-best-practices-guide/index.html>
   - Technique: Stage async model outputs into pinned host buffers on a non-default stream and consume them later, rather than synchronizing at future/result resolution. This can reduce median TPOT by overlapping D2H copies for logprobs, routed experts, and token metadata with the next scheduler/worker step.
   - Evidence: Quote: "the asynchronous transfer version requires pinned host memory" and "different, non-default streams." Pointer: section 9.1.2, lines 352 and 361-363.
4. **[RFC] Redesign enable_return_routed_experts to avoid blocking EngineCore event loop**
   - Source type: issue
   - URL: <https://github.com/vllm-project/vllm/issues/38079>
   - Technique: Move optional per-request output metadata through the normal model-output channel as deltas, copied only for opted-in requests and only on the steps that need it. The same pattern can reduce executor output-path stalls by avoiding always-on shared-memory synchronization and by keeping feature-specific payload construction off the critical path.
   - Evidence: Quote: "No GPU→CPU sync on the critical path." Pointer: Data flow / Why this works, lines 234-258.
5. **ray.wait**
   - Source type: docs
   - URL: <https://docs.ray.io/en/latest/ray-core/api/doc/ray.wait.html>
   - Technique: Use readiness-driven result handling for multi-worker futures: wait for whichever refs are ready, detach or post-process them immediately, then place results back into rank order before aggregation. This can overlap zero-copy detach and KV aggregation with outstanding worker outputs without changing ordered return semantics.
   - Evidence: Quote: "Return a list of IDs that are ready and a list of IDs that are not." Pointer: Ray API docs, `ray.wait` description.
6. **zmq_poller(3)**
   - Source type: docs
   - URL: <https://zeromq.github.io/libzmq/zmq_poller.html>
   - Technique: Replace sequential per-rank response dequeues with a multi-queue poll/wait-all style response collector that drains all currently ready queues in one pass. This directly addresses executor response fan-in overhead while preserving rank-ordered assembly after the poll returns.
   - Evidence: Quote: "zmq_poller_wait_all returns the number of events signalled and returned in the events array." Pointer: return value, lines 95-101.
7. **Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning**
   - Source type: paper
   - URL: <https://www.alphaxiv.org/abs/2201.12023>
   - Technique: Adopt topology-aware rank placement using device meshes: keep communication-heavy tensor-parallel groups on high-bandwidth intra-node links and place pipeline stages across lower-bandwidth boundaries. This can improve TTFT/TPOT by reducing executor-induced cross-node handoff and collective latency in distributed backends.
   - Evidence: Quote: "partitioning the cluster into a number of device meshes, each of which contains devices with preferably high-bandwidth connections." Pointer: overview/contributions, lines 30-33.
8. **Massively Scale Your Deep Learning Training with NCCL 2.4**
   - Source type: blog
   - URL: <https://developer.nvidia.com/blog/?p=13452>
   - Technique: Choose communication algorithms and rank topologies based on message size and hierarchy, e.g. hierarchical intra-node/inter-node rings for bandwidth and tree variants for lower latency at scale. Executor DAG construction and placement can adapt this idea for pipeline handoffs and compiled-DAG transport choices.
   - Evidence: Quote: "The hierarchical ring is a 2D ring (intra-node/inter-node being the 2 dimensions)." Pointer: Effect on DL training, lines 76-83.
9. **[Bug]: RayExecutorV2 multi-node DP hangs on shm_broadcast — cross-node ranks can't share single-host shared memory**
   - Source type: issue
   - URL: <https://github.com/vllm-project/vllm/issues/43420>
   - Technique: Use a hybrid executor transport: shared-memory message queues for same-node workers and Ray RPC or Ray collectives for cross-node workers. This is an actionable design for preserving low local latency while avoiding cross-node shared-memory assumptions that can hurt reliability and tail latency.
   - Evidence: Quote: "fall back to Ray RPC for inter-node collectives while keeping shm for intra-node." Pointer: Fix path suggestions, lines 222-224.
10. **ray.dag.input_node.InputNode.experimental_compile**
   - Source type: docs
   - URL: <https://docs.ray.io/en/latest/ray-core/compiled-graph/doc/ray.dag.input_node.InputNode.experimental_compile.html>
   - Technique: Set explicit compiled-DAG inflight and buffered-result limits to match async scheduler concurrency, rather than relying on implicit capacity. This gives the executor a concrete backpressure knob for multi-turn agentic workloads where several nonblocking model futures can accumulate.
   - Evidence: Quote: "_max_inflight_executions – The maximum number of in-flight executions". Pointer: Ray Compiled Graph API parameters.
11. **Serialization**
   - Source type: docs
   - URL: <https://docs.ray.io/en/latest/ray-core/objects/serialization.html>
   - Technique: Exploit protocol-5 zero-copy buffers for large read-mostly array payloads, but pair them with explicit detachment or lifetime control when results must not block subsequent channel reuse. This informs output-copy decisions for logprob and routed-expert arrays crossing executor boundaries.
   - Evidence: Quote: "Numpy arrays in the object store are shared between workers on the same node (zero-copy deserialization)." Pointer: Ray serialization overview.
12. **Issue 17025: reduce multiprocessing.Queue contention**
   - Source type: issue
   - URL: <https://bugs.python.org/issue17025>
   - Technique: Move serialization/deserialization work outside contended queue locks and critical sections. The executor can adapt this by pre-encoding repeated callable/control payloads and keeping message-queue enqueue/dequeue critical sections focused on transport bookkeeping only.
   - Evidence: Quote: "serialize/unserialize the objects before/after holding the locks. This leads to reduced contention." Pointer: issue description.

## Issues

### module_deep_research
- **[warning, recoverable]** codex: Some Ray documentation pages returned HTTP 429 on direct fetch after search; retained findings only where search results or other fetches provided concrete, attributable source text.
