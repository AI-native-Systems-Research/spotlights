# FutureWrapper.result

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/multiproc_executor.py`](vllm/v1/executor/multiproc_executor.py) (lines 88–105)
- **Symbol:** `FutureWrapper.result`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_executor-0002`

## Description
FIFO synchronization point for non-blocking multiprocessing futures; resolving one future drains all earlier futures from the shared deque by calling _wait_for_response.

## Current approach
result(timeout=None) rejects timeouts, then repeatedly pops the oldest FutureWrapper from futures_queue and calls _wait_for_response until this future is done. _wait_for_response synchronously calls aggregate(get_response()) and stores either result or exception on the Future.

## Estimated impact explanation
The loop is on the async decode synchronization path. Improvements reduce median TPOT most when multiple steps are pipelined; scope is narrower than collective_rpc because it only appears when callers use non_block futures.

## Evolve rationale
The specific construct is the while-not-done drain loop over futures_queue. In async scheduling, several per-step FutureWrappers can be in flight, so this loop is where the engine catches up with worker output. Headroom includes batching response waits for adjacent futures, allowing get_response implementations to poll multiple queues once per drain, and avoiding redundant aggregate/get_response work when an earlier future has already been resolved. Correctness oracle: FIFO resolution remains intact, Future.set_result/set_exception semantics are unchanged, and existing async scheduling tests observe the same ordered sequence of ModelRunnerOutputs and exceptions.

## Deep research proposals

### 1. Poll all per-rank response queues in one pass inside FutureWrapper drains
- **Finding:** `find-vllm_v1_executor-0006` — *zmq_poller(3)*
- **Source URL:** <https://zeromq.github.io/libzmq/zmq_poller.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py lines 88-105, FutureWrapper.result drains earlier FutureWrappers FIFO by calling _wait_for_response, which invokes get_response (defined at lines 405-419 in collective_rpc). Today get_response iterates response_mqs sequentially, calling mq.dequeue(timeout=...) per rank; wall time is dominated by the slowest rank plus per-queue wakeup latency, and this cost is paid on every FutureWrapper in the drain loop. Change get_response (and, where beneficial, the outer drain loop in result()) to a readiness-poll-then-collect pattern modeled on zmq_poller_wait_all: register all currently outstanding response_mqs with a poller-style primitive, wait once for any subset to become ready (with a deadline derived from `deadline - time.monotonic()`), then dequeue only from the ready queues and repeat until all ranks for the current FutureWrapper have delivered. Rank-ordered assembly is preserved by indexing responses by their originating queue after the poll returns, and FIFO across FutureWrappers is preserved because the outer `while not self.done()` loop still pops and completes wrappers in queue order. Because vLLM uses MessageQueue (shm_broadcast) rather than raw zmq sockets, implementation options include (a) exposing a readiness fd/event on MessageQueue so a `select`/`multiprocessing.connection.wait` can multiplex across ranks, or (b) adding a module-level `MessageQueue.wait_any(mqs, timeout)` that returns the subset with data available. Correctness contract is maintained: aggregate() runs on the full rank set only after all its inputs are collected, Future.set_result/set_exception semantics are unchanged, and the FIFO order of ModelRunnerOutputs observed by callers is identical. Optionally, when draining several FutureWrappers in one result() call, the same poller can be reused across the batch of pending wrappers so a single kernel wait covers responses for multiple pipelined steps.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names 'allowing get_response implementations to poll multiple queues once per drain' as headroom, and the finding provides exactly that primitive: a wait-all-style poll that returns the set of ready queues in one call. On the async decode synchronization path the drain executes per-step with world_size per-rank dequeues each, so replacing sequential dequeues with a single readiness wait cuts response fan-in latency contributing to median TPOT under pipelined async scheduling. The technique preserves the correctness oracle (FIFO across FutureWrappers, rank-ordered aggregation, unchanged Future semantics) because ordering is imposed after the poll during assembly rather than by the wait itself.

---

## Agent proposals

### 1. Overlap aggregate() with next future's get_response() in the drain loop via a single-slot prefetcher
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py at FutureWrapper.result (lines 88-105), the drain loop processes pending FutureWrappers strictly serially: each iteration pops the oldest future and synchronously runs _wait_for_response, which calls get_response() (blocking on shm_broadcast dequeues across ranks) and then aggregate() (which, when kv_output_aggregator is active, performs a nontrivial per-rank reduction — see collective_rpc at lines 386-390 where aggregate is bound to kv_output_aggregator.aggregate). Under async scheduling with multiple in-flight steps, these two phases run back-to-back per future even though the aggregate for future N and the queue dequeue for future N+1 have no data dependency. Introduce a small, drain-scoped single-slot prefetcher: when result() begins draining and futures_queue holds >=2 items, submit the head future's get_response to a dedicated single-threaded executor (a lazily-created threading.Thread pool of size 1 stored on MultiprocExecutor, gated by an env like VLLM_MP_EXECUTOR_PREFETCH=1), then, while the calling thread runs aggregate() and set_result() for the current future, the background thread pulls responses for the next future from the response_mqs. On the next drain iteration, _wait_for_response consumes the already-materialized response list instead of re-entering mq.dequeue. Correctness contract: FIFO across FutureWrappers is preserved because the drain loop still pops in queue order and calls set_result/set_exception in that order; the prefetcher never runs aggregate() (that stays on the completing thread so exception attribution and CUDA-context assumptions are unchanged); if the prefetched call raises, the exception is stashed on the wrapper and surfaced only when its turn comes; on timeout=None (the current-only supported mode) there is no deadline race. This is complementary to a wait_any-style multi-queue poll — that reduces the per-rank fan-in inside one get_response call, whereas this proposal overlaps aggregate CPU cost of future N with the fan-in of future N+1, which is the dominant cost when kv_output_aggregator does per-rank work. On the async decode TPOT path with pipelined steps, this converts serial (dequeue + aggregate) x K into approximately max(dequeue, aggregate) x K plus a single trailing aggregate.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_executor-0006) targets get_response internals — replacing the sequential per-rank mq.dequeue loop with a poller-style wait_any across response_mqs to cut within-a-single-future fan-in latency. It explicitly preserves the outer while-not-done drain loop and does not overlap distinct FutureWrappers' work. This proposal instead pipelines across adjacent FutureWrappers by prefetching future N+1's get_response concurrently with future N's aggregate()/set_result(), attacking a different bottleneck (aggregate CPU cost overlapping IPC latency of the next step) that find-0006 does not address. The two changes are compatible and independently valuable: even if get_response is already wait_any-optimized, the aggregate cost of the current future remains serial with the dequeue of the next one, and this prefetcher removes that gap.

---

### 2. Move FIFO response draining to a dedicated completion thread
- **Agent:** codex

**Detailed description.**

Replace the caller-driven drain in `vllm/v1/executor/multiproc_executor.py::FutureWrapper.result` with an executor-owned FIFO completion loop that starts when the first non-blocking `FutureWrapper` is enqueued. The loop owns `futures_queue.pop()` and calls `_wait_for_response()` in order until the queue is empty, guarded by a condition/event that is notified from `FutureWrapper.__init__` after `appendleft`. `FutureWrapper.result(timeout=None)` then becomes a normal wait on the underlying `Future`, still rejecting unsupported non-`None` timeouts if desired. This keeps the same FIFO `set_result`/`set_exception` ordering, but removes response fan-in from the foreground path of whichever engine thread first calls `result()`: worker outputs are collected as soon as they are available, so by the time the scheduler needs a future, earlier pipelined futures may already be resolved. For multi-turn agentic workloads this can reduce median TPOT and tail stalls caused by bursty `result()` calls catching up several decode steps at once. The change should be gated behind a small executor option/env at first, and tests should cover ordered results, ordered exception surfacing, and shutdown behavior when pending futures exist.

**Novelty rationale.**

The deep research proposal optimizes the inside of one `get_response()` call by polling all rank queues together, while preserving the foreground `while not self.done()` drain. Agent A overlaps adjacent futures only during a caller-entered drain with a single-slot prefetcher. This proposal changes the ownership and timing of the FIFO drain itself: a persistent completion thread resolves futures before `result()` is called, so foreground result calls no longer perform catch-up work. It is compatible with both prior ideas but addresses a different latency source: deferred completion work accumulating until synchronization.

---
