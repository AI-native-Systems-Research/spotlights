# RequestOutputCollector.put/get_nowait/get

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/output_processor.py`](vllm/v1/engine/output_processor.py) (lines 62–96)
- **Symbol:** `RequestOutputCollector.put/get_nowait/get`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0017`

## Description
Per-request async output handoff buffer that stores or merges RequestOutputs before generate() consumes them.

## Current approach
Maintains a single pending output plus an asyncio.Event; put either stores the output, raises readiness, or merges RequestOutput deltas with RequestOutput.add when the producer is ahead, while get/get_nowait clear the slot and event.

## Estimated impact explanation
Every streamed token batch crosses this handoff; reducing event and merge overhead can lower frontend CPU cost and median TPOT for many concurrent short generations.

## Evolve rationale
This collector is the final handoff point for AsyncLLM streaming output and encodes the current backpressure and delta-aggregation policy. Candidate changes include lower-allocation merge paths, explicit bounded batching, specialized DELTA vs non-DELTA collectors, or avoiding redundant event churn when output is already ready. Correctness oracle: tests/v1/engine/test_async_llm.py and tests/v1/engine/test_output_processor.py should preserve streaming delivery, DELTA aggregation, exception propagation, and final-output behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace asyncio.Event with lazy consumer Future in RequestOutputCollector
- **Agent:** claude

**Detailed description.**

In vllm/v1/engine/output_processor.py at RequestOutputCollector (lines 45-96), remove the always-allocated `self.ready = asyncio.Event()` and replace it with a lazily-allocated `self._waiter: asyncio.Future | None = None`. Rework the three hot methods:

- `__init__`: drop the Event; set `self._waiter = None`.
- `put(output)`: same slot/merge logic as today, but instead of `self.ready.set()`, do `if (w := self._waiter) is not None and not w.done(): w.set_result(None); self._waiter = None`. When the producer is ahead of the consumer (common under load), no waiter exists and put() becomes a pure attribute assignment / RequestOutput.add() — no Event bookkeeping, no waiter-list walk, no scheduler churn.
- `get()`: keep the synchronous fast path (`if self.output is not None: ...`) so that when the producer has already deposited output there is zero await and zero Future allocation. Only on a genuine miss do we create `self._waiter = asyncio.get_running_loop().create_future()` and `await self._waiter`. After the future resolves, re-check `self.output` in a loop (spurious wakeups from cancellation are still handled). On consumption, simply set `self.output = None` — no `ready.clear()` needed because the future is single-shot and already consumed.
- `get_nowait()`: unchanged except drop the `self.ready.clear()` call.
- `close()` / `__del__`: additionally cancel `self._waiter` if pending so a hanging consumer coroutine is released with a CancelledError rather than deadlocking.

Why this helps the caller's median TPOT / TTFT goal: in multi-turn agentic workloads with many concurrent short generations, `put()` is called once per scheduler tick per active request (thousands of times per second aggregate). `asyncio.Event.set()` walks its `_waiters` deque and calls `loop.call_soon` on each; `asyncio.Event.clear()` on every get() writes to the internal flag. A single `Future` allocated only when the consumer actually has to block collapses the steady-state to a single attribute write in `put()` and a single attribute read in `get()`. Under a producer-ahead regime — which is exactly when TPOT matters — the Event and its `clear()` are pure overhead. This also eliminates the Event's per-put branch that redundantly re-sets an already-set flag.

Correctness/oracle: preserve the exact semantics tested by tests/v1/engine/test_async_llm.py and tests/v1/engine/test_output_processor.py — DELTA aggregation via `RequestOutput.add(..., aggregate=True)`, non-DELTA overwrite for PoolingRequestOutput, exception propagation on `get()`/`get_nowait()`, final-output delivery, and safe close during a pending await. The state machine is unchanged; only the wakeup primitive is swapped.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate (the list is empty). The candidate's own evolve_rationale mentions generic directions — 'lower-allocation merge paths, explicit bounded batching, specialized DELTA vs non-DELTA collectors, or avoiding redundant event churn' — but does not specify how. This proposal is a concrete, minimal implementation of the 'avoiding redundant event churn' direction: replacing asyncio.Event with a lazily-created single-shot Future so the steady-state producer-ahead path performs zero synchronization primitive work, while keeping the merge/DELTA-aggregation and exception-propagation semantics untouched. It is orthogonal to (and composable with) any future work on merge allocation or DELTA specialization.

---
