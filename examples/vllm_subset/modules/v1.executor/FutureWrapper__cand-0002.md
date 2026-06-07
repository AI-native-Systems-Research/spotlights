# FutureWrapper

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/multiproc_executor.py`](vllm/v1/executor/multiproc_executor.py) (lines 69–99)
- **Symbol:** `FutureWrapper`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0002`

## Description
Ordering wrapper for in-flight multiprocess RPC responses. It appends itself to a shared deque and drains older futures before returning its own result.

## Current approach
__init__ stores get_response and aggregate callables, calls Future.__init__(), and appendlefts itself into a deque. result() repeatedly pops the deque tail until this future is done, and _wait_for_response() calls aggregate(get_response()) before set_result/set_exception under suppress(InvalidStateError).

## Estimated impact explanation
The object and deque costs are small but occur on every non-blocking token step. Lowering this overhead should improve median TPOT under async scheduling and PP, where executor-side ordering sits between consecutive scheduled batches.

## Evolve rationale
The deque drain protocol at lines 76-95 is the self-contained ordering mechanism behind async scheduling and pipeline-parallel overlap. It can be evolved into lighter ordered response slots, an identity-aggregate fast path, or a specialized single-in-flight path while preserving FIFO response observation. Correctness oracle: v1 engine batch-queue/async-scheduling tests and PP tests must observe identical output ordering, exception propagation, and blocking behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Skip FutureWrapper allocation entirely on the blocking RPC path
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py at the FutureWrapper construction site (lines 397-403), the blocking branch (`non_block=False`) constructs a FutureWrapper only to immediately call `.result()`, which performs a wasted appendleft+pop on the shared `futures_queue`, allocates a `concurrent.futures.Future` (with its RLock/Condition/callbacks list), and runs the same drain-then-set_result/set_exception path. Replace this with a synchronous fast path: when `non_block=False`, first drain the existing `futures_queue` in-order by popping each pending FutureWrapper and calling its `_wait_for_response()` (preserving FIFO observation of responses for any prior async-scheduled / PP in-flight work), then directly execute `aggregate(get_response())` and return the value, raising any exception directly. FutureWrapper construction is reserved exclusively for the `non_block=True` callers (async scheduling and PP overlap). The shared deque, FIFO ordering, exception propagation, and identical observable side effects (responses dequeued in MessageQueue order) are preserved because the blocking caller still drains pending wrappers before reading its own response. The change is local to the small region around lines 69-99 + 397-403 and removes one Future allocation, one deque mutation, one InvalidStateError-suppressing set_result, and the trivial `aggregate=identity` lambda call per blocking RPC. Validation: existing v1 engine batch-queue/async-scheduling and pipeline-parallel tests must observe identical token ordering and exception propagation; add a microbenchmark of `collective_rpc(..., non_block=False)` round-trip latency on a no-op worker method to confirm the reduction.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete idea is novel by definition. Beyond that, the candidate's `evolve_rationale` enumerates three directions — lighter ordered response slots, identity-aggregate fast path, and a specialized single-in-flight path — all of which still revolve around FutureWrapper's representation when at least one in-flight wrapper exists. This proposal is orthogonal: it removes FutureWrapper from the blocking path entirely (a path that is the norm for control-plane RPCs and any non-async callers) rather than making the wrapper cheaper, so it composes with — rather than overlaps — those listed directions.

---

### 2. Use a slot-based ordered response for async model RPCs
- **Agent:** codex

**Detailed description.**

In `vllm/v1/executor/multiproc_executor.py`, split the hot `execute_model(..., non_block=True)` / `sample_tokens(..., non_block=True)` path away from `concurrent.futures.Future`. Replace or supplement `FutureWrapper` at lines 69-99 with a small slot-based ordered response object that stores only `futures_queue`, `get_response`, `aggregate`, `_done`, `_result`, and `_exception`, and implements the two methods EngineCore actually uses: `done()` and `result()`. Its `result()` should keep the existing FIFO drain loop by popping the shared deque tail and calling `_wait_for_response()` until this object is done; `_wait_for_response()` should call `aggregate(get_response())` once and store either the value or exception directly. Keep the current `Future`-subclass wrapper only for generic `collective_rpc(..., non_block=True)` if preserving that public API contract is required. This removes `Future`'s per-token `Condition`/lock/callback/waiter allocation and the `set_result`/`set_exception` `InvalidStateError` path from async scheduling and pipeline-parallel overlap, while preserving ordered response observation, `done()` behavior, and exception propagation. Validate with v1 batch-queue async scheduling tests, PP tests, and a small benchmark that repeatedly enqueues no-op non-blocking `execute_model`/`sample_tokens`-shaped RPCs.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Claude's proposal removes `FutureWrapper` only from the blocking `non_block=False` path and still leaves async/PP token-step futures backed by `concurrent.futures.Future`; this proposal targets the non-blocking hot path itself by replacing the heavyweight Future internals with a minimal ordered response object while keeping the same deque ordering protocol.

---
