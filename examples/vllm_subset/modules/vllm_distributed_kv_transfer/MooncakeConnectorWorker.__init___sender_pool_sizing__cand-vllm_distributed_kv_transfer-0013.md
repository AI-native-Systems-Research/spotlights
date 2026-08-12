# MooncakeConnectorWorker.__init__ sender pool sizing

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py) (lines 921–988)
- **Symbol:** `MooncakeConnectorWorker.__init__ sender pool sizing`
- **Kind:** config_block
- **Estimated impact:** high
- **Id:** `cand-vllm_distributed_kv_transfer-0013`

## Description
Configures Mooncake producer-side sender concurrency with num_sender_workers defaulting to 10 and num_sender_tasks fixed at twice that count, then constructs the sender ThreadPoolExecutor.

## Current approach
ThreadPoolExecutor(max_workers=num_sender_workers) and a 2x async task surplus are static heuristics from kv_connector_extra_config, independent of TP/PP world size, request concurrency, RDMA queue depth, or observed queueing latency.

## Estimated impact explanation
Producer-side send concurrency controls how quickly remote KV becomes available to decode nodes; many agentic sessions can queue turn-2 fetches concurrently.

## Evolve rationale
The pool size and task multiplier directly control how many send_kv_to_decode operations can be in flight from the producer. Adaptive sizing keyed to sender-queue latency, or backpressure-aware task admission, can be evaluated without changing the transfer contract. Oracle: Mooncake connector unit tests and vllm bench throughput/TTFT runs verify completed request sets, error propagation, and byte-correct transfer.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Adapt sender pool + async task budget from measured send latency and queue backlog
- **Agent:** claude

**Detailed description.**

In MooncakeConnectorWorker.__init__ (mooncake_connector.py:921-988), replace the static `num_sender_tasks = num_sender_workers * 2` heuristic and the fixed default of 10 workers with a lightweight adaptive controller that sizes the producer-side send concurrency from runtime signals. Concretely: (1) introduce a `SenderConcurrencyController` owned by the worker that maintains EWMA of per-send RDMA duration (already recorded around `self.sender_loop.run_in_executor(self._sender_executor, self._send_blocks, ...)` at ~line 1341 and via `xfer_stats.record_transfer` at line 1635) and of queue-wait time measured by timestamping items when they are `put` into `self.sender_worker_queue` (line 1159) and when they are `get`-ed inside `_sender_worker` (line 1174); (2) replace the raw `sender_worker_queue.get()` in `_sender_worker` with acquisition of an `asyncio.Semaphore` whose value is the controller's current in-flight budget, released after `send_kv_to_decode` completes, so async task fan-in can grow/shrink without recreating tasks; (3) periodically (every N completions or T ms) grow the budget when queue-wait < send-latency and observed RDMA completion latency is flat, and shrink when queue-wait grows faster than send-latency (signal of RDMA QP / NIC saturation); (4) derive the ThreadPoolExecutor `max_workers` from `min(cap, f(tp_size, pp_size))` using `self.tp_size`, `self.pp_size` already computed at lines 955-969, so per-rank RDMA fan-in is topology-aware rather than TP/PP-oblivious; (5) expose the current in-flight budget and EWMAs on `MooncakeKVConnectorStats` for observability without touching the wire protocol. The `kv_connector_extra_config` keys (`num_workers`) become soft caps rather than hard values, preserving user overrides. This does not change the transfer contract — `send_kv_to_decode`, bootstrap, and RDMA setup are untouched — so the existing Mooncake connector unit tests and `vllm bench` TTFT/throughput oracle from the candidate remain valid.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any concrete change is novel by construction. Beyond that, this proposal targets a specific, unlisted intervention — closed-loop control of both the async task budget (`num_sender_tasks`) and the executor width using two orthogonal producer-side signals (per-send RDMA latency EWMA and sender_worker_queue wait EWMA), combined with a TP/PP-aware cap derived from `self.tp_size`/`self.pp_size` that are already computed at init but currently unused for pool sizing. Existing knobs are purely static and TP/PP-oblivious, so an adaptive, topology-aware controller is not covered by any prior proposal on this candidate.

---

### 2. Split Mooncake sender executors by transfer size class
- **Agent:** codex

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` around the sender pool setup in `MooncakeConnectorWorker.__init__` (lines 921-988), replace the single producer-side `ThreadPoolExecutor(max_workers=num_sender_workers)` with two bounded executors and matching async task queues: a small-transfer lane for short or partial KV sends and a bulk-transfer lane for large block batches. Classify each `send_kv_to_decode` work item by the number of blocks or estimated bytes already available in the send request metadata before it is submitted to `run_in_executor`. Reserve at least one worker/task slot for the small lane and put a configurable cap on the bulk lane so one large prefill transfer cannot head-of-line block many small continuation transfers in multi-turn agentic traffic. Keep the existing total default budget (`num_sender_workers`, `num_sender_tasks`) as the aggregate cap, but split it with conservative defaults such as 25% small / 75% bulk and expose optional `kv_connector_extra_config` overrides for the threshold and split. This is a scheduling-only change: the Mooncake transfer contract, `_send_blocks`, and RDMA setup stay the same, while median TTFT/TPOT should improve when short follow-up turns are queued behind large KV sends.

**Novelty rationale.**

There are no deep_research_proposals listed. Agent A proposed adaptive resizing of the single sender budget using queue wait, RDMA latency, semaphores, and TP/PP-aware caps. This proposal is different: it keeps the aggregate concurrency budget static by default but changes the queueing discipline to size-aware, multi-lane scheduling to reduce head-of-line blocking between small interactive transfers and large bulk transfers. It does not rely on closed-loop latency feedback or dynamic pool resizing, so it is not covered by Agent A's adaptive controller.

---
