# UniProcExecutor async RPC path

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/uniproc_executor.py`](vllm/v1/executor/uniproc_executor.py) (lines 40–100)
- **Symbol:** `UniProcExecutor async RPC path`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0005`

## Description
Single-process executor async execution path, including the one-thread async output executor, the max_concurrent_batches-derived enablement, and collective_rpc() future construction.

## Current approach
When max_concurrent_batches > 1, _init_executor creates a ThreadPoolExecutor(max_workers=1). collective_rpc() runs the worker method inline, submits AsyncModelRunnerOutput.get_output or a get_output_list closure to the thread pool, otherwise allocates a new Future and sets result or exception. Blocking single_value calls are already direct.

## Estimated impact explanation
Single-GPU serving is the common low-latency path. The affected overhead is paid on every non-blocking decode step, so reducing future/thread-pool churn and improving async depth should move median TPOT and short-request TTFT.

## Evolve rationale
The concrete constructs are ThreadPoolExecutor(max_workers=1) at lines 40-44, the hard-coded async depth policy at lines 63-65, and Future/closure allocation at lines 83-99. Evolution can use a lighter completed-future object for immediate results, avoid get_output_list closure allocation, specialize the always-single-value execute_model/sample_tokens path, and tune async queue depth beyond the fixed value 2. Correctness oracle: v1 engine/uniproc tests must preserve output values, exception timing in execute_model(non_block=True), and scheduler batch-queue behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace ThreadPoolExecutor + Future with a CUDA-event completion handle on the uniproc async-output path
- **Agent:** claude

**Detailed description.**

Today, lines 40-44 of vllm/v1/executor/uniproc_executor.py spin up a single-worker ThreadPoolExecutor whose only job is to call AsyncModelRunnerOutput.get_output() (a blocking D2H wait), and lines 83-99 wrap every non-blocking call in a freshly-allocated concurrent.futures.Future. Both exist solely to give the engine loop a `.done()`/`.result()`-shaped object whose readiness tracks the worker's D2H copy. Replace this with a small `_UniprocCompletionHandle` class that holds (a) the AsyncModelRunnerOutput (or already-materialized result/exception) and (b) the recording CUDA event already produced by the worker for its async D2H copy. Implement `done()` as `event.query()` (non-blocking, GIL-free), `result()` as `event.synchronize(); return self._async_output.get_output()`, and `exception()` symmetrically. In `_init_executor`, drop the `async_output_thread = ThreadPoolExecutor(...)` allocation entirely. In the `non_block` branch of `collective_rpc`, when the worker returns an `AsyncModelRunnerOutput`, return a `_UniprocCompletionHandle` directly; for already-materialized results, return a module-level `_COMPLETED_NONE`/typed singleton wrapper instead of constructing a new `Future` and calling `set_result`. Also collapse the dead `single_value=False` branch (which only fires through a `get_output_list` closure that is never hit on the hot path — execute_model/sample_tokens always pass single_value=True per vllm/v1/engine/core.py:414,474,490,556). Net effect on the multi-turn agentic decode hot path: per-step removal of ThreadPoolExecutor.submit (lock + wakeup of the worker thread + GIL handoff), Future allocation + set_result, run_method string dispatch, and the closure object allocation. The engine's `if output.done(): output.result()` early-failure check (uniproc_executor.py:112-114) still works because `event.query()` returns True the moment the kernels and D2H finish, with no thread hop. The correctness oracle (v1 engine/uniproc tests) is preserved: exception timing for `non_block=True` is unchanged because synchronous worker-side errors are surfaced eagerly via the materialized branch, and async D2H errors are raised at `.result()` time exactly as before. To address the hard-coded `max_concurrent_batches = 2` policy at lines 63-65, add an env-tunable `VLLM_UNIPROC_ASYNC_DEPTH` (default 2) so deeper pipelining can be measured for agentic decode without changing default behavior.

**Novelty rationale.**

The candidate has no existing deep_research_proposals, so there is nothing to overlap with. The evolve_rationale alludes generically to "a lighter completed-future object" and "tune async queue depth," but does not specify the concrete mechanism proposed here: piggy-backing on the worker's existing CUDA event so that `.done()` becomes a non-blocking `cudaEventQuery` and the ThreadPoolExecutor can be removed outright (not merely "lightened"), plus collapsing the dead single_value=False/get_output_list branch.

---

### 2. Bind the uniproc async-output thread to the worker device
- **Agent:** codex

**Detailed description.**

In `vllm/v1/executor/uniproc_executor.py`, make the `WorkerAsyncOutput` thread inherit the same device setup as the driver worker before it ever calls `AsyncModelRunnerOutput.get_output()`. The analogous multiproc async-output loop explicitly calls `current_platform.set_device(self.worker.device)` before synchronizing CUDA events; the uniproc `ThreadPoolExecutor(max_workers=1)` at lines 40-44 currently has no initializer, so the first `get_output()` call on that thread can hit CUDA runtime APIs with the default thread-local device, potentially creating work/context on device 0 or paying device setup on the request path. Move or wrap the executor construction so the async thread runs a small initializer after worker device initialization, e.g. `ThreadPoolExecutor(..., initializer=_init_async_output_thread, initargs=(self.driver_worker,))`, where the initializer checks `hasattr(worker, "device")` and calls `current_platform.set_device(worker.device)`. Also shut the executor down explicitly in `shutdown()` once pending worker output is no longer needed. This is a narrow lifecycle fix: it keeps the existing Future/ThreadPoolExecutor contract intact while removing avoidable first-output latency and non-default-GPU memory pressure from the uniproc async path.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes replacing the ThreadPoolExecutor/Future path with a CUDA-event completion handle and making async depth tunable; it does not cover the existing thread's CUDA device initialization or lifecycle. This proposal is specifically about making the current one-thread async-output executor device-correct and avoiding request-path CUDA context setup if that executor remains in place.

---
