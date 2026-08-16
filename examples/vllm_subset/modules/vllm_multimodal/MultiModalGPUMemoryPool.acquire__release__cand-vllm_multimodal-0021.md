# MultiModalGPUMemoryPool.acquire/_release

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/gpu_ipc_memory.py`](vllm/multimodal/gpu_ipc_memory.py) (lines 54–110)
- **Symbol:** `MultiModalGPUMemoryPool.acquire/_release`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0021`

## Description
Implements the frontend GPU multimodal memory budget as a blocking byte-counting semaphore used by GPU video decode paths.

## Current approach
acquire validates nbytes, then waits on a threading.Condition while available bytes are insufficient. _release returns bytes and calls notify_all, waking every waiter. There is no fairness, request-size ordering, timeout/backoff policy, or batching of wakeups.

## Estimated impact explanation
This affects concurrent GPU video ingress. Better admission scheduling reduces TTFT queueing for video requests and avoids CPU wakeup storms that can interfere with token generation, improving median and tail latency under agentic bursts.

## Evolve rationale
The concrete synchronization policy is the while self._available < nbytes condition wait and notify_all in _release. A FIFO or size-aware wait queue, targeted notify, or bounded wait policy can reduce head-of-line blocking and thundering-herd wakeups while preserving the byte-budget invariant. Correctness oracle: tests/multimodal/test_gpu_ipc_memory.py; available_bytes accounting, double-release idempotence, blocking behavior, and oversize errors must remain correct.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace notify_all with FIFO wait queue and targeted per-waiter notify
- **Agent:** claude

**Detailed description.**

Refactor `MultiModalGPUMemoryPool.acquire`/`_release` in vllm/multimodal/gpu_ipc_memory.py:80-110 to use an explicit FIFO wait queue instead of the current `threading.Condition` + `notify_all` pattern. Concretely: (1) Replace the single `self._cond` wait with a per-waiter `threading.Event` (or dedicated Condition) that acquire() enqueues under an internal `_lock: threading.Lock`, along with the requested `nbytes` and the lease's monotonically increasing id, into a `_waiters: collections.deque[_Waiter]`. (2) In `acquire`, if `self._available >= nbytes` and the waiter is at the head of the queue (or the queue is empty), consume the bytes and skip the wait; otherwise append a `_Waiter` and block on its Event outside the lock. (3) In `_release`, after adding bytes back, walk `_waiters` from the head and, for each waiter whose `nbytes` fits in the current `_available`, decrement `_available`, mark that lease as outstanding, pop it, and `set()` its Event — stop as soon as the head waiter no longer fits (strict FIFO admission). This preserves head-of-line-safe fairness while eliminating the thundering herd where every waiter wakes on every release only to re-block. Keep the public API (`acquire`, `_release`, `available_bytes`, `MultiModalGPUMemoryLease`, double-release idempotence via `_outstanding`) and all existing invariants intact so `tests/multimodal/test_gpu_ipc_memory.py` continues to pass. Optionally add an internal `acquire_nowait()` fast-path used when the queue is empty and bytes are available, to avoid Event allocation for the uncontended common case (single-request bursts). Because acquisitions are already serialized by `_lock`, the FIFO order matches lease-id order, which also makes behavior deterministic and easier to reason about under agentic multi-turn bursts where many concurrent renderer threads compete for the same budget.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate (the list is empty), so any concrete refinement of the acquire/release synchronization policy is novel by construction. This proposal specifically operationalizes the candidate's `evolve_rationale` hint ("FIFO or size-aware wait queue, targeted notify") into an implementable design (per-waiter Event, strict FIFO head-of-queue admission, uncontended fast path) rather than restating the high-level idea, and it explicitly names the invariants that the existing test module must continue to enforce.

---

### 2. Add an adaptive small-request bypass to the memory admission policy
- **Agent:** codex

**Detailed description.**

Extend `MultiModalGPUMemoryPool.acquire`/`_release` in `vllm/multimodal/gpu_ipc_memory.py:54-110` with a size-aware admission rule that prevents one very large video decode request from blocking many smaller multimodal allocations when enough bytes are available for the smaller work. Concretely, keep the existing byte-budget invariant and lease accounting, but replace strict `while self._available < nbytes` admission with an internal waiter registry that can admit any waiter whose requested bytes fit when the current head waiter cannot fit, bounded by a starvation guard such as an enqueue timestamp or skip counter for the large waiter. `_release` would scan eligible waiters and wake only those that can be fully charged against `_available`, preferring older waiters normally but allowing bounded bypass for smaller requests. Add concurrency tests in `tests/multimodal/test_gpu_ipc_memory.py` that enqueue one request larger than the next release plus several small requests, then verify the small requests proceed without violating `available_bytes`, while the large waiter is eventually admitted after sufficient releases.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposed strict FIFO admission with targeted per-waiter notification, which intentionally stops when the head waiter does not fit. This proposal is different: it targets head-of-line blocking by allowing bounded, size-aware bypass of an oversized head waiter, while still avoiding unbounded starvation and preserving the same memory accounting invariants.

---
