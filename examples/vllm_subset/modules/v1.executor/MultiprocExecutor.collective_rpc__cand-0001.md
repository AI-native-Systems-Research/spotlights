# MultiprocExecutor.collective_rpc

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/multiproc_executor.py`](vllm/v1/executor/multiproc_executor.py) (lines 339–403)
- **Symbol:** `MultiprocExecutor.collective_rpc`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0001`

## Description
Driver-side MQ RPC path used by execute_model(), sample_tokens(), and control RPCs. It serializes the command, broadcasts it, selects response queues, builds a per-call response collector, and returns or drains a FutureWrapper.

## Current approach
Each call computes a deadline, normalizes kwargs, allocates either a partial() KV aggregator or an identity lambda, optionally cloudpickle.dumps() a callable method, enqueues (method, args, kwargs, output_rank), creates a get_response closure over response_mqs/deadline/method, allocates a FutureWrapper, and immediately drains it for blocking calls.

## Estimated impact explanation
The driver pays this Python overhead once or twice per scheduled decode step. Reducing it should improve median TPOT for small-batch and multi-turn agentic serving where host scheduling overhead is a visible part of each token step.

## Evolve rationale
The hot constructs are the per-call allocations and queue selection at lines 357-401. Evolution can pre-bind the identity aggregator, specialize the common single-output-rank execute_model/sample_tokens path, call the selected MessageQueue directly in the no-timeout case, and avoid FutureWrapper allocation for blocking calls when there are no older in-flight futures. Correctness oracle: existing multiprocess/distributed executor tests plus end-to-end LLM.generate parity against the current executor, including timeout and worker-failure propagation.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace FutureWrapper(concurrent.futures.Future) with a lock-free single-thread result holder for driver-thread RPCs
- **Agent:** claude

**Detailed description.**

FutureWrapper at vllm/v1/executor/multiproc_executor.py:69-99 inherits from concurrent.futures.Future, whose __init__ allocates a threading.Condition (RLock + bookkeeping), and whose set_result/result acquire that lock. In v1's engine-core driver, every collective_rpc (execute_model, sample_tokens, take_draft_token_ids, control RPCs) creates one of these per step, even though the producer and consumer are the same engine-core thread — both the synchronous `result()` drain at line 403 and the non_block=True consumers in vllm/v1/engine/core.py:414, 474, 490, 556 run on the driver thread. The locking and Condition allocation are pure overhead.

Proposal: introduce `_DriverFuture` (no inheritance, __slots__ = ('_done', '_result', '_exception', 'futures_queue', 'get_response', 'aggregate')) that implements only the surface actually used: done(), result(), set_result(), set_exception(), and _wait_for_response(). No threading.Condition, no internal lock, no callbacks list (or a plain list if any caller uses add_done_callback — verify by grep on the FutureWrapper return path). Use it in place of FutureWrapper at the construction site at lines 397-401. This complements (does not duplicate) the rationale's `avoid FutureWrapper allocation for blocking calls when there are no older in-flight futures` optimization: that fast-path only fires when futures_queue is empty AND non_block=False. The async-scheduling path (non_block=True, the call site that generates the steady-state TPOT win for multi-turn agentic via overlapped execute_model/sample_tokens) MUST keep a future-like object across the engine loop; this proposal makes that object cheap. Likewise, when the blocking caller has older in-flight futures it still needs a real Future-shaped object in the deque, and this proposal makes those cheap too.

Verification: existing distributed/multiproc executor tests, plus tests/v1/engine tests that exercise async_scheduling (which uses non_block=True), plus an end-to-end LLM.generate parity check; micro-bench by timing 10k synthetic execute_model RPCs against a noop worker to isolate driver-side savings.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's own evolve_rationale targets only (a) pre-binding the identity aggregator, (b) specializing the unique-output-rank path, (c) calling MessageQueue directly when timeout is None, and (d) skipping FutureWrapper construction when both non_block=False AND futures_queue is empty. None of those address the unavoidable cases — non_block=True (the async-scheduling path that drives TPOT in agentic decode) and blocking-with-pending-futures — where a Future-shaped object is still required. Replacing the concurrent.futures.Future base class with a lock-free thread-unsafe holder targets exactly those residual cases and is orthogonal to all four optimizations the rationale already enumerates.

---

### 2. Preserve kwargs=None through the RPC payload and call workers without **{}
- **Agent:** codex

**Detailed description.**

In `vllm/v1/executor/multiproc_executor.py:339-403`, stop normalizing `kwargs` to a freshly allocated empty dict for no-kwargs calls. Enqueue `kwargs` as `None` when the caller passed no keyword arguments, and update the worker-side `worker_busy_loop` in the same file to dispatch with `func(*args)` when `kwargs is None`, falling back to `func(*args, **kwargs)` only for real keyword calls. The hot `execute_model()` and `sample_tokens()` paths always pass positional args only, so this removes a per-RPC dict allocation, avoids serializing/deserializing an empty dict in `MessageQueue.enqueue/dequeue`, and avoids the slower empty-kwargs call form on every participating worker. Keep the public `collective_rpc(..., kwargs=None)` contract unchanged and add a focused test or microbenchmark that exercises both no-kwargs and kwargs RPCs so control calls such as `sleep(kwargs=...)` still work.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A targets the `FutureWrapper` object and its `concurrent.futures.Future` locking overhead. This proposal targets the RPC command payload and worker dispatch path instead: it eliminates empty-kwargs allocation, pickling, and `**{}` invocation for no-kwargs calls. It is also more specific than the candidate's listed optimizations, which mention identity aggregation, unique-output-rank response specialization, no-timeout dequeue shortcuts, and skipping or replacing future allocation, but do not cover preserving `kwargs=None` across the message boundary.

---
