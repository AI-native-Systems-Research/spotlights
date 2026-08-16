# MultiprocExecutor.collective_rpc

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/multiproc_executor.py`](vllm/v1/executor/multiproc_executor.py) (lines 365–427)
- **Symbol:** `MultiprocExecutor.collective_rpc`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_executor-0001`

## Description
Per-step multiprocessing control-plane RPC for execute_model, sample_tokens, and related worker calls; broadcasts work, selects response queues, and returns a FutureWrapper that drains worker replies.

## Current approach
Each call computes a deadline, normalizes kwargs, allocates either a partial KVOutputAggregator.aggregate or identity lambda, serializes callable methods with cloudpickle, enqueues one tuple on rpc_broadcast_mq, builds a per-call get_response closure, and wraps it in FutureWrapper. Multi-rank responses are drained sequentially with a fresh remaining-time calculation before each MessageQueue.dequeue.

## Estimated impact explanation
This is directly on the per-token executor path for multiprocessing and RayExecutorV2. Reducing Python allocation, serialization, and response-drain overhead can lower median TPOT and per-step jitter in multi-turn agentic workloads with many short decode steps.

## Evolve rationale
The concrete optimization unit is the enqueue plus get_response closure in MultiprocExecutor.collective_rpc. It is invoked for each v1 multiprocessing execute_model/sample_tokens step and is also reused by RayExecutorV2. Headroom includes avoiding per-call closure/partial allocation on string-method hot paths, caching serialized callable payloads when a callable is reused, reducing deadline bookkeeping, and replacing sequential per-rank dequeues with a batched or multi-queue wait while preserving rank-ordered results. Correctness oracle: existing multiproc execute_model/sample_tokens and PP/TP integration tests return the same ModelRunnerOutput or rank-ordered list, with identical worker-failure and timeout propagation.

## Deep research proposals

### 1. Precompile a static Ray Compiled Graph for the steady-state execute_model/sample_tokens control plane
- **Finding:** `find-vllm_v1_executor-0001` — *Ray Compiled Graphs: Optimized AI Workloads with Native GPU Communication*
- **Source URL:** <https://www.anyscale.com/blog/announcing-compiled-graphs>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the per-step enqueue + get_response closure path in MultiprocExecutor.collective_rpc (vllm/v1/executor/multiproc_executor.py:365-427) for the hot execute_model and sample_tokens methods with a precompiled static actor DAG (Ray Compiled Graphs / aDAG) built once at executor init and reused every step. Concretely: (1) At MultiprocExecutor/RayExecutorV2 startup, compile one DAG per hot RPC method (execute_model, sample_tokens, and any other steady-state calls) whose nodes are worker.execute_method invocations wired with a fixed fanout to all worker ranks and a fixed aggregation node (KVOutputAggregator.aggregate or identity) as the sink. Cloudpickle-serialize the callable payload once at compile time instead of on each call. (2) In collective_rpc, detect the fast path (string method name in the precompiled set, no non-serializable kwargs, no per-call timeout override) and dispatch by calling dag.execute(args, kwargs) instead of building a per-call functools.partial, closure, and FutureWrapper. Return a thin FutureWrapper that wraps the compiled-graph future so timeout, cancellation, and worker-failure propagation semantics are preserved. (3) Fall back to today's rpc_broadcast_mq + get_response path for cold/rare methods, callables, and any call whose kwargs or timeout signal a non-steady-state invocation, so correctness is unchanged for control RPCs. (4) Replace the sequential per-rank MessageQueue.dequeue loop with the compiled DAG's built-in multi-rank gather for the fast path, preserving rank-ordered results by ordering DAG outputs by rank at compile time.

**Proposal rationale.**

The finding reports Ray Compiled Graphs cut task submission overhead from ~1-2 ms to ~50 us by moving invocation, transport, and scheduling setup out of the per-step path into a one-time compile. That is exactly the gap in MultiprocExecutor.collective_rpc: today each execute_model/sample_tokens step re-allocates a partial/closure, re-serializes the callable, re-normalizes kwargs, and drains ranks sequentially with fresh deadline math, all of which are fixed per-step Python and IPC costs the candidate's evolve_rationale explicitly calls out. Because RayExecutorV2 reuses this same path, a Ray-native compiled DAG is a natural fit, and the multiproc backend can adopt the same precompiled-dispatch pattern over its existing MessageQueue by caching the serialized payload and using a fixed multi-rank gather. The optimization is confined to steady-state string-method RPCs (execute_model, sample_tokens) with a cold-path fallback, which matches the candidate's correctness oracle: identical ModelRunnerOutput / rank-ordered results and identical worker-failure and timeout propagation. Given multi-turn agentic decode is dominated by many short steps, shaving per-step control-plane overhead should directly lower median TPOT and per-step jitter, aligning with the caller's TTFT/TPOT objective.

---

### 2. Batch response fan-in with a zmq_poller wait-all across per-rank MessageQueues
- **Finding:** `find-vllm_v1_executor-0006` — *zmq_poller(3)*
- **Source URL:** <https://zeromq.github.io/libzmq/zmq_poller.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `MultiprocExecutor.collective_rpc` at vllm/v1/executor/multiproc_executor.py:405-421, replace the sequential `for mq in response_mqs: mq.dequeue(timeout=...)` loop in `get_response` with a poller-based fan-in. Since each `MessageQueue` reader already owns a zmq notify socket + `zmq.Poller` (see `shm_broadcast.py:163-165, 206`), expose the underlying notify sockets (or an equivalent `poll_ready(timeout)` helper on `MessageQueue`) and, in `MultiprocExecutor`, hold a single `zmq.Poller` registered with all `self.response_mqs` notify sockets. In `get_response`, loop: compute the remaining deadline once, call `poller.poll(timeout_ms)` (analogous to `zmq_poller_wait_all`) to learn which subset of response queues have data, drain each ready queue non-blockingly via the existing `acquire_read`/`dequeue` path, mark those ranks as complete, and repeat until all expected ranks (or the single `output_rank`) have responded or the deadline elapses. Preserve the current API by assembling `responses` in rank order after the poll loop and returning `responses[0]` when `output_rank is not None`. Keep worker-failure handling (`ResponseStatus`) and TimeoutError wrapping identical. When `output_rank` is set (single-rank case), the fast path stays as a single blocking `dequeue` to avoid setting up a poller.

**Proposal rationale.**

The candidate's `evolve_rationale` explicitly calls out `replacing sequential per-rank dequeues with a batched or multi-queue wait while preserving rank-ordered results` as a headroom item. The finding's zmq_poller_wait_all primitive is exactly the mechanism that enables this: `MessageQueue` already uses `zmq.Poller` internally for its own notify+cancel sockets, so composing a poller across response queues is a natural extension of the existing infrastructure rather than a new transport. Under TP/PP with N ranks whose local runtimes vary, sequential dequeue forces the executor to block on `response_mqs[0]` even when ranks 1..N-1 are already ready, adding per-step tail latency that directly hits median TPOT in multi-turn agentic decode loops with many short steps. A poller-based drain converts N ordered blocking waits into one wait bounded by the slowest rank, cutting response-fan-in overhead on every execute_model/sample_tokens step while keeping rank-ordered assembly, timeout semantics, and worker-failure propagation intact — matching the candidate's correctness oracle.

---

### 3. Cache cloudpickle payloads for repeated callable RPCs in MultiprocExecutor.collective_rpc
- **Finding:** `find-vllm_v1_executor-0012` — *Issue 17025: reduce multiprocessing.Queue contention*
- **Source URL:** <https://bugs.python.org/issue17025>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py at MultiprocExecutor.collective_rpc (lines 365-427), the branch at lines 395-398 unconditionally calls cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL) whenever `method` is a callable rather than a string. When the same callable object is reused across steps (e.g., a bound worker method or module-level function repeatedly dispatched by higher-level control flow such as RayExecutorV2 or agentic control paths), this re-serializes an identical payload on every collective_rpc invocation before the enqueue on rpc_broadcast_mq. Adapt the finding by adding a small bounded cache keyed by `id(method)` (holding a weak reference to the callable and the memoized cloudpickle bytes) so the serialization step is executed once per distinct callable and reused on subsequent calls. The str-method fast path (execute_model, sample_tokens, execute_dummy_batch, take_draft_token_ids, sample_tokens) is unchanged, and the identity of the cached callable is preserved by the weak-reference key so a garbage-collected or replaced callable is re-encoded rather than aliased. The cached bytes are the same value cloudpickle would produce on each call today, so worker-side deserialization is byte-identical and the correctness oracle (existing multiproc execute_model/sample_tokens and PP/TP integration tests returning the same ModelRunnerOutput or rank-ordered list, and identical worker-failure/timeout propagation) is preserved.

**Proposal rationale.**

The finding's transferable idea is: keep transport critical sections focused on transport and pre-encode repeated payloads so the same object is not serialized twice. The candidate's evolve_rationale explicitly calls out `caching serialized callable payloads when a callable is reused` as headroom on this hot path, and the current implementation performs cloudpickle.dumps on every callable dispatch with no memoization. Under the stated multi-turn agentic workload with many short decode steps, even modest per-step savings on Python-level allocation and serialization for callable dispatches contribute to median TPOT and per-step jitter. The change is scoped to the callable branch, does not alter enqueue/dequeue semantics of the SHM MessageQueue, and does not change what workers receive on the wire.

---

## Agent proposals

### 1. Precompile per-hot-method RPC descriptors to eliminate per-step Python allocation on both leader and worker
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py, split MultiprocExecutor.collective_rpc (lines 365-427) into a slow generic path (today's code, unchanged) and a set of precompiled per-method 'hot RPC descriptor' fast paths bound at executor init for the finite set of steady-state string methods actually invoked per step: `execute_model`, `sample_tokens`, `execute_dummy_batch`, and `take_draft_token_ids`. Concretely: (1) At the end of `_init_executor`, once `self.output_rank`, `self.kv_output_aggregator`, and `self.response_mqs` are known, build a small immutable descriptor object per hot method that stores (a) the method name string, (b) the resolved `output_rank` (`None` for execute_model/sample_tokens because `kv_output_aggregator` is set, `self.output_rank` for the others), (c) the precomputed response_mqs slice (already just `(response_mqs[output_rank],)` for the aggregator-less cases, and the full sequence when aggregator is set), (d) a preallocated `functools.partial(kv_output_aggregator.aggregate, output_rank=self.output_rank or 0)` for execute_model/sample_tokens, or the identity function for the other two, and (e) the fixed `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` deadline base (or `None`). (2) Add a corresponding fast-path entry point `_collective_rpc_hot(descriptor, args)` that skips: the `isinstance(method, str)` branch, the `kwargs = kwargs or {}` reassignment, the aggregate/partial allocation, the `send_method` branch, the per-call closure creation, and the per-call lambda; it enqueues a preallocated 4-tuple where only the `args` slot is filled in (`(descriptor.method_name, args, EMPTY_KWARGS, descriptor.output_rank)` using a shared frozen `EMPTY_KWARGS = {}` singleton), then invokes an inlined `get_response` that reuses the descriptor's response_mqs slice and precomputed aggregate. (3) Rewrite the `execute_model`, `sample_tokens`, `execute_dummy_batch`, and `take_draft_token_ids` methods (lines 332-363) to call `_collective_rpc_hot(self._descriptor_<method>, args)` directly instead of going through `collective_rpc`. (4) Introduce a small reusable `FutureWrapper` pool (or a lighter-weight non-`concurrent.futures.Future` shim that provides just `.result()` and drains via `self.futures_queue`, avoiding the `Condition`+`Lock` construction that `Future.__init__` performs on every step) — since `_wait_for_response` currently uses `set_result`/`set_exception` cross-thread only from the caller's own thread in the drain loop, the full `Future` lock machinery is unnecessary on the hot path. (5) Symmetrically, in `WorkerProc.worker_busy_loop` (lines 1004-1033), cache the bound method resolution for the same finite set of hot string method names: on first encounter, populate `self._hot_method_cache: dict[str, Callable]` with `getattr(self.worker, name)` for each of the four names known to be dispatched every step, then in the loop replace `func = getattr(self.worker, method)` with `func = self._hot_method_cache.get(method) or getattr(self.worker, method)`. This eliminates a Python attribute-lookup + bound-method-object allocation per rank per step. (6) The slow generic `collective_rpc` remains for callable dispatches, unknown string methods, custom timeouts, or any invocation whose kwargs are non-empty — preserving worker-failure/timeout semantics unchanged and preserving the correctness oracle (identical ModelRunnerOutput or rank-ordered list from existing multiproc execute_model/sample_tokens and PP/TP tests). RayExecutorV2 inherits the same benefit if it dispatches through the same executor entry points.

**Novelty rationale.**

None of the three existing proposals touches the per-step Python allocation on the string-method fast path or the worker-side dispatch cost. find-0001 (Ray Compiled Graphs) is a Ray-only transport swap that explicitly says it leaves cold/rare paths on today's mq path and does not remove the leader-side `partial`/lambda/closure/`FutureWrapper` allocation for the multiproc-only steady state; it targets the whole DAG rather than the intra-Python control-flow allocations. find-0006 optimizes only the response fan-in (zmq poller wait-all across response_mqs) and does not touch enqueue-side or worker-side allocation. find-0012 caches cloudpickle bytes strictly for the callable branch (lines 395-398) — the string branch, which is the actual per-step hot path (execute_model, sample_tokens), has no serialization to cache and is untouched by that proposal. My proposal is additive and orthogonal: it removes per-step Python object construction (partial, lambda, closure, `Future.__init__` with its Condition/Lock) on the leader, eliminates the per-step `getattr` bound-method allocation on every worker rank, and preserves the SHM MessageQueue transport and rank-ordered response semantics; it composes cleanly with find-0006's poller fan-in and with find-0012's byte cache if either lands.

---
