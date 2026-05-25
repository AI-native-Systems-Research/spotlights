# SingleDirectionOffloadingHandler.get_finished

[← v1.kv_offload](../v1.kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/gpu_worker.py`](vllm/v1/kv_offload/cpu/gpu_worker.py) (lines 336–356)
- **Symbol:** `SingleDirectionOffloadingHandler.get_finished`
- **Kind:** method
- **Estimated impact:** low
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0004`

## Description
Polls completed transfers from the in-flight deque, emits TransferResult records, and recycles CUDA streams and timing events.

## Current approach
The method repeatedly queries only the deque head, computes elapsed time from the start/end timing events for each completed transfer, appends the result, returns the stream and events to local pools, and removes the job from _transfer_events. FIFO polling is currently consistent with transfer_async's stream chaining.

## Estimated impact explanation
This is not the primary data-movement cost because transfers are currently serialized, but it runs on the worker polling path. Reducing event timing and bookkeeping overhead can modestly shorten completion recognition for CPU->GPU loads, which affects TTFT for cache-warm requests waiting on restored blocks.

## Evolve rationale
The concrete headroom is in completion-poll overhead and timing collection: make transfer timing optional, skip elapsed_time when metrics do not need it, recycle event records with less per-completion work, or adapt the data structure if transfer_async later permits independent streams. Correctness oracles are that each submitted job_id is reported exactly once, transfer_size equals the scheduled byte count, wait(job_ids) still fences unfinished jobs, and existing GPU-worker tests keep passing.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Drop CUDA timing events from get_finished's hot path; use host-side timestamps for transfer_time
- **Agent:** claude

**Detailed description.**

In vllm/v1/kv_offload/cpu/gpu_worker.py around lines 336-356 (get_finished) together with transfer_async (lines ~295-331), replace the per-transfer `enable_timing=True` events plus the `start_event.elapsed_time(end_event)` call with cheaper bookkeeping:

1. Allocate the synchronization event with `torch.Event(enable_timing=False)` (or `torch.cuda.Event(blocking=False)`); CUDA timing-enabled events are markedly more expensive to record/query than non-timing events because they capture host-side wallclock data, and `elapsed_time()` itself is a non-trivial host call that runs once per completion in the polling loop.
2. Stop creating/recycling `start_event` entirely. The Transfer dataclass keeps only `end_event` for stream chaining and `wait()` fencing, halving event allocations and the size of `_event_pool`.
3. Capture `t_submit = time.monotonic_ns()` inside `transfer_async` right after `end_event.record(stream)` and store it on the `Transfer` record. In `get_finished`, compute `transfer_time = (time.monotonic_ns() - transfer.t_submit) * 1e-9` when the head's `end_event.query()` first returns True. This is a host-side upper bound on the device transfer (it includes queue wait), but the consumers of `TransferResult.transfer_time` are metrics/observability, not correctness oracles — so an approximate, monotonic timing is acceptable and matches what schedulers actually feel.
4. Gate even this lightweight timing behind a flag (e.g. reuse the existing stat-logging / Prometheus enablement signal already plumbed into the engine) so that when metrics are disabled `transfer_time` is left as 0.0 and the `monotonic_ns()` calls are skipped entirely.

Correctness invariants are preserved: each `job_id` is still popped exactly once from the deque, `transfer_size` is unchanged, `wait(job_ids)` still synchronizes on `end_event`, and `_transfer_events` bookkeeping is identical. The chained `stream.wait_event(last_event)` ordering in transfer_async does not require timing events. Existing GPU-worker tests should pass; add one assertion that `transfer_time >= 0` and (when timing disabled) `== 0.0`.

Expected effect on the workload: shaves a per-completion `elapsed_time()` call (typically tens of microseconds, sometimes more under contention) plus one event record/query/recycle off every CPU->GPU restore, which is exactly the path that gates TTFT for cache-warm multi-turn agentic requests waiting on restored blocks.

**Novelty rationale.**

The candidate has no listed deep_research_proposals, so any concrete change is novel by construction. The specific direction here — switching to non-timing CUDA events, eliminating the per-job `start_event`, and substituting host-side `monotonic_ns` timestamps gated on metrics enablement — is more targeted than the generic 'make transfer timing optional' phrasing in the candidate's evolve_rationale: it identifies that `enable_timing=True` itself (not just the elapsed_time call) imposes per-record cost, removes a whole event from the pool, and proposes a concrete substitute signal usable by metrics consumers rather than just dropping the field.

---

### 2. Add a FIFO tail-completion fast path to batch-drain finished transfers
- **Agent:** codex

**Detailed description.**

In `vllm/v1/kv_offload/cpu/gpu_worker.py` inside `SingleDirectionOffloadingHandler.get_finished`, exploit the existing FIFO stream chaining before the per-transfer pop loop. First query the head as today and return immediately if it is not ready. If the head is ready and the deque has several in-flight transfers, also query `_transfers[-1].end_event`; when the tail is ready, all earlier transfers are guaranteed complete, so drain the whole deque without calling `end_event.query()` for each intermediate transfer. Keep the current result construction, `elapsed_time`, event recycling, and `_transfer_events` deletion unchanged. Use a small threshold such as `len(_transfers) >= 4` before probing the tail so the common one-transfer case does not pay an extra query. Add a test that enqueues multiple transfers before polling, synchronizes or waits until the last event is complete, then asserts that `get_finished()` returns each `job_id` exactly once in FIFO order with unchanged `transfer_size`, and that `wait(job_ids)` remains a no-op for already reported jobs.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal removes timing-enabled events and changes how `transfer_time` is measured; this proposal leaves timing semantics alone and targets a different source of polling overhead: the O(number of completed transfers) CUDA event-query loop. It uses the current FIFO stream-chaining invariant to reduce completion detection work when a batch of queued transfers has already finished.

---
