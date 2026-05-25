# MultiprocExecutor.WorkerProc output emission path

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/multiproc_executor.py`](vllm/v1/executor/multiproc_executor.py) (lines 901–942)
- **Symbol:** `MultiprocExecutor.WorkerProc output emission path`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0004`

## Description
Worker response-emission path. It converts AsyncModelRunnerOutput to concrete output, wraps success/failure status, and either enqueues directly to the response MessageQueue or hands output to the async-output thread.

## Current approach
Synchronous mode calls enqueue_output() directly. Async scheduling uses queue.Queue.put() from handle_output(), a dedicated async_output_busy_loop thread with queue.Queue.get(), and enqueue_output() blocks on AsyncModelRunnerOutput.get_output() before response_mq.enqueue().

## Estimated impact explanation
This path gates when worker results become visible to the scheduler in async mode. Lower handoff overhead and better event-wait placement should reduce inter-step gaps, improving median TPOT for async multi-turn workloads.

## Evolve rationale
The hot constructs are queue.Queue put/get at lines 921-942 and the get_output synchronization at lines 906-914. An evolved implementation can use a lighter single-producer/single-consumer handoff, reduce lock/condition churn, and tune where CUDA-event completion is waited for while preserving response ordering and device-context setup. Correctness oracle: v1 async-scheduling tests and PP tests must see the same sequence of (ResponseStatus, output) messages and exception behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace queue.Queue + dedicated thread with CUDA stream-callback-driven response emission
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py:901-942, eliminate both the queue.Queue handoff (`async_output_queue.put`/`.get`) and the dedicated `async_output_busy_loop` thread for the success path. Instead, when `handle_output` receives an `AsyncModelRunnerOutput` (concretely `AsyncOutput`/`AsyncPoolingOutput` from vllm/v1/worker/gpu/async_utils.py:12,72 whose `get_output()` does `self.copy_event.synchronize()`), register a host-function callback on the copy_stream via `torch.cuda.Stream.add_callback` (a thin wrapper over cudaLaunchHostFunc) that runs as soon as the D2H copy event fires. The callback (a) builds the `(SUCCESS, model_runner_output)` tuple by performing the CPU-only `tolist()`/dict-construction work originally inside `get_output()` (the `synchronize()` becomes a no-op since the callback is scheduled at copy completion) and (b) pushes it onto `worker_response_mq` directly. To preserve strict FIFO across steps required by the v1 async-scheduling oracle, attach a monotonically increasing sequence number per step and use a tiny SPSC `collections.deque` + `threading.Event` reorder buffer so a small straggler cannot leapfrog an in-flight predecessor (deque append/popleft are atomic under the GIL; no Lock/Condition churn). Synchronous mode continues to call `enqueue_output()` directly. Keep a fallback path on the busy-loop thread for non-CUDA platforms or when `add_callback` is unavailable. Net effect on lines 906-914: the blocking `get_output()` no longer runs on a critical worker thread; on lines 921-942: the queue.Queue.put/get and the dedicated thread are removed for the hot path, shaving the condvar handoff (~10-30µs) and a context switch from every step. Device-context setup that `async_output_busy_loop` performed via `current_platform.set_device(self.worker.device)` is moved to a one-shot init inside the worker thread that records the callback (the callback itself does no CUDA API calls). Validation: the existing v1 async-scheduling and PP tests assert the exact `(ResponseStatus, output)` sequence and exception ordering — the sequencer makes ordering bit-identical; exceptions still flow through the synchronous `enqueue_output` path inside `worker_busy_loop`'s except branch, untouched.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate (the list is empty), so any concrete proposal is novel by construction. Beyond that, the candidate's evolve_rationale only gestures at 'lighter SPSC handoff' and 'tune where CUDA-event completion is waited for'; this proposal is specifically structured around moving the wait off the busy-loop thread entirely by piggybacking on the GPU's own copy-completion signal via cudaLaunchHostFunc/`Stream.add_callback`, which both removes the dedicated thread and replaces the queue.Queue rather than merely tuning either.

---

### 2. Use SimpleQueue for the async output handoff
- **Agent:** codex

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py, change the async scheduling handoff created around line 624 from `queue.Queue()` to `queue.SimpleQueue()`, with a matching type annotation such as `queue.SimpleQueue[Any]`. The hot path at lines 921-942 only uses unbounded blocking `put()` and `get()` between one producer (`worker_busy_loop`) and one consumer (`async_output_busy_loop`); it never uses `maxsize`, `task_done()`, `join()`, or timeout semantics. `SimpleQueue` preserves FIFO behavior for this use case while avoiding `queue.Queue`'s extra condition variables and unfinished-task bookkeeping. Keep `async_output_busy_loop()` and `enqueue_output()` otherwise unchanged so device-context setup, blocking `AsyncModelRunnerOutput.get_output()`, response ordering, and failure wrapping remain identical. Validate with the existing v1 async-scheduling and PP tests, plus a small microbenchmark that repeatedly calls `handle_output()` with already-materialized outputs to confirm lower handoff overhead without changing emitted `(ResponseStatus, output)` sequences.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. This is also not the same as Claude's proposal: Claude removes the queue and dedicated thread from the success path by using CUDA stream callbacks plus a sequencer. This proposal intentionally keeps the current thread, `get_output()` wait placement, and response enqueue path, and only swaps the generic synchronized queue implementation for Python's lighter unbounded FIFO queue in the existing design.

---
