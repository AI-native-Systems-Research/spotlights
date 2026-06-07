# MoRIIOWriter._write_worker_loop / _process_deferred_tasks

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py`](vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py) (lines 115–154)
- **Symbol:** `MoRIIOWriter._write_worker_loop / _process_deferred_tasks`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0012`

## Description
MoRIIO producer-side write scheduler that waits for remote block allocation before executing queued RDMA write tasks.

## Current approach
A single daemon thread repeatedly rescans all deferred tasks, then does `Queue.get(timeout=0.01)` for new work. If a task's `transfer_id` is not in `done_remote_allocate_req_dict`, it is appended to `_deferred_tasks` and retried by full-list scan on subsequent loop iterations. There is no allocation-keyed wakeup, ready queue, batching of newly ready tasks, or backoff policy.

## Estimated impact explanation
Remote allocation readiness gates producer-side KV writes. Reducing polling delay and repeated scans can lower remote-prefill TTFT for MoRIIO-backed multi-turn requests and reduce background CPU overhead that can otherwise inflate TPOT.

## Evolve rationale
This is an owned polling and retry scheduler independent of the external MoRIIO transfer engine. It can be evolved into an event-driven ready queue keyed by `transfer_id`, batched execution of tasks released by the same remote allocation message, or adaptive sleep/backoff. Correctness oracle: with a fake `MoRIIOWrapper`, the same write tasks must call `_execute_write_task` only after matching remote allocation, complete the same transfer IDs, and emit the same notifications; `tests/v1/kv_connector/unit/test_moriio_connector.py` covers MoRIIO request and metadata behavior.

## Deep research proposals

### 1. Coalesce per-layer write tasks of the same transfer into a single batched RDMA call on allocation readiness
- **Finding:** `find-0004` — *[PD] optimize kv cache transfer directly using batch transfer*
- **Source URL:** <https://github.com/sgl-project/sglang/pull/9149>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor the producer-side scheduler at vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py:115-154 (and the immediately adjacent _execute_write_task/_do_layer_write at lines 188-295) so that, instead of executing each per-layer WriteTask individually via one moriio_wrapper.write_remote_data() call per layer, the scheduler accumulates all per-layer WriteTasks belonging to the same transfer_id and dispatches them as a single batch when the remote allocation becomes available.

Concrete shape of the change, scoped to this candidate region:
1. In _write_worker_loop / _process_deferred_tasks (lines 115-154), replace the current 'pop one task, check readiness, append to _deferred_tasks on miss, full rescan on next loop' pattern with a structure keyed by transfer_id: a dict transfer_id -> list[WriteTask] of pending tasks, plus a small ready-set populated when an allocation is observed in done_remote_allocate_req_dict. When a transfer_id transitions to ready, drain its full per-layer WriteTask list and hand it to a new batched executor (one wakeup, one drain) instead of executing tasks one-by-one across loop iterations.
2. Replace per-task _execute_write_task -> _do_layer_write dispatch with a batched path that, given the list of LayerTransferPlans for the same transfer_id (already each carrying sess_idx, transfer_local_offsets, transfer_remote_offsets, transfer_sizes, use_batch=True at lines 263-272), packs all layers' descriptors into a single batch transfer descriptor and issues one batch transfer call to moriio_wrapper rather than one write_remote_data() per layer (line 282) — i.e. compose the per-layer (sizes, local_off, remote_off, sess_idx) tuples into one batch call so the underlying transfer engine submits them as a single operation. If the wrapper currently exposes only a per-layer write_remote_data, route through whatever batch entry point it provides; otherwise gate the new path behind a capability check and fall back to the per-layer loop.
3. Preserve existing semantics: each task.event.synchronize() (line 211) must still gate that task's data before submission; finalization via _finalize_if_complete (line 230) must still fire once per task as today (or once per transfer_id with equivalent per-task notifications) so that downstream completion notifications and transfer_id bookkeeping in done_remote_allocate_req_dict and the connector's notification path are unchanged.

This reuses the already-present use_batch=True plan flag and existing _prepare_transfer_plan output without inventing new layer enumeration logic; the change is confined to (a) how _write_worker_loop groups and wakes tasks and (b) how _do_layer_write submits them.

**Proposal rationale.**

The finding's core technique — pack all layers' transfer parameters into one batch and call the batch-transfer interface directly instead of issuing one executor submission per layer — maps cleanly onto a real gap in this candidate. Today _execute_write_task (line 188) is invoked once per per-layer WriteTask, and _do_layer_write (line 274) issues one moriio_wrapper.write_remote_data() per layer (line 282), so the producer pays a per-layer launch/control cost on every transfer. Because all per-layer WriteTasks for one transfer_id become eligible at the same moment (when done_remote_allocate_req_dict gains that transfer_id), the scheduler is the natural place to gather them and emit a single batched call — exactly the structure the candidate's evolve_rationale calls out as 'batched execution of tasks released by the same remote allocation message'. This addresses both the polling-scheduler gap (no allocation-keyed wakeup, no batching) and the per-layer launch overhead the finding identifies, and is a plausible TTFT win for remote-prefill in the multi-turn agentic workload where many layers must transfer per request. The finding is concrete (quote: 'Pack all layers' transfer parameters to one single batch'), transferable (same producer-side per-layer-vs-batch tradeoff), and non-trivially distinct from the candidate's current approach (which has no cross-layer batching at the scheduler level).

---

### 2. Event-driven batched dispatch of remote-allocation-ready MoRIIO writes with I/O pipelining
- **Finding:** `find-0010` — *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*
- **Source URL:** <https://arxiv.org/abs/2510.09665>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the polling/full-rescan scheduler in vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py:115-154 (MoRIIOWriter._write_worker_loop / _process_deferred_tasks) with an event-driven, batched dispatch path inspired by LMCache's batched KV data movement and compute/I/O pipelining. Concretely: (1) Index deferred tasks in a dict keyed by transfer_id (transfer_id -> list[WriteTask]) instead of a flat list scanned every iteration. (2) Have the MoRIIOWrapper (or a thin wrapper around done_remote_allocate_req_dict) signal a threading.Event / Condition or push completed transfer_ids to a 'newly_ready' queue when a remote allocation message arrives, so the worker wakes only on actual readiness instead of a 10 ms blind timeout. (3) When woken, drain all transfer_ids that became ready in this tick and dispatch their WriteTasks as a single batch through _execute_write_task, allowing the underlying MoRIIO RDMA engine to coalesce posts (one batched write/notification submit per allocation event) rather than one task at a time. (4) Pipeline the wait-for-allocation with in-flight execution: a freshly arriving task on _write_task_q whose transfer_id is already in done_remote_allocate_req_dict goes straight to the batched dispatch path, while not-ready tasks register themselves under their transfer_id and return without scanning the rest of the deferred set. (5) Optional adaptive backoff: if no readiness events fire and the queue is empty, fall back to a longer sleep (e.g. exponential up to ~1 ms-10 ms) instead of the fixed 10 ms poll, and reset on activity. Preserve the existing correctness oracle: same _execute_write_task invocations, same completed transfer IDs, same notifications, and continue to satisfy tests/v1/kv_connector/unit/test_moriio_connector.py with a fake MoRIIOWrapper.

**Proposal rationale.**

The candidate's gap is twofold: (a) it polls and full-rescans _deferred_tasks every loop iteration with a fixed 10 ms timeout, adding latency and CPU overhead between a remote allocation arriving and the corresponding RDMA write being issued, and (b) it issues writes one task at a time even when a single allocation message releases multiple queued tasks. LMCache's reported gains come precisely from 'batched data movement operations' and 'compute and I/O pipelining' across KV tiers - the same two levers this scheduler is missing. Mapping those ideas onto an allocation-keyed ready index plus event wakeup and batched dispatch directly attacks the producer-side gating delay on remote-prefill TTFT in multi-turn agentic workloads (where the same remote allocation can release many block writes), and removes the steady-state rescan cost that can inflate TPOT under sustained load. The change is local to MoRIIOWriter and does not require modifying the external MoRIIO transfer engine, matching the candidate's evolve_rationale.

---

## Agent proposals

### 1. Decouple GPU-event readiness from allocation readiness via CUDA event callbacks to remove dispatcher head-of-line blocking
- **Agent:** claude

**Detailed description.**

In MoRIIOWriter._write_worker_loop / _process_deferred_tasks (vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py:115-154), introduce a two-axis readiness model so the single dispatcher thread never blocks on cudaEventSynchronize. Today readiness is a single check (transfer_id in done_remote_allocate_req_dict), and once the task is admitted to _execute_write_task it calls task.event.synchronize() at line 211 inline on the worker — meaning any layer whose producing GPU stream has not yet finished freezes dispatch of every other already-ready task. Replace this with: (a) keep a per-transfer_id pending dict (alloc_pending: transfer_id -> list[WriteTask]); (b) add a parallel gpu_pending dict (transfer_id -> set of WriteTasks whose CUDA event has not yet fired) and a gpu_ready set; (c) at task ingress, instead of calling event.synchronize() later, register a CUDA stream/event callback (torch.cuda.Stream.wait_event has no callback, but torch's AsyncEventHandler / cudaLaunchHostFunc / a tiny worker that polls via event.query() on a dedicated thread can be used) that, when the event completes, atomically moves the task from gpu_pending into gpu_ready and signals the dispatcher's condition variable; (d) the dispatcher dispatches a task only when its transfer_id is in done_remote_allocate_req_dict AND it is in gpu_ready, otherwise it returns immediately to wait on the condition variable; (e) _do_layer_write no longer needs event.synchronize() since GPU completion is now a precondition of dispatch. Preserve existing semantics: the same _execute_write_task body runs once per task, the same _finalize_if_complete fires, completions and notifications are unchanged, and tests/v1/kv_connector/unit/test_moriio_connector.py with a fake MoRIIOWrapper still passes (the fake never blocks on CUDA, so the gpu_ready path becomes a fast no-op for it). If torch does not expose a usable host callback, fall back to a tiny dedicated 'event poller' thread that does event.query() over outstanding events at sub-millisecond cadence — still strictly off the dispatcher's critical path.

**Novelty rationale.**

Both existing proposals (find-0004 batching, find-0010 event-driven allocation-keyed wakeup with adaptive backoff) treat readiness as a single dimension — remote allocation arrival — and leave the synchronous task.event.synchronize() at line 211 untouched on the dispatcher's critical path. Even with their changes, a slow GPU event on one ready task still serializes dispatch of every other allocation-ready task on the single worker thread, because the worker calls event.synchronize() inline before issuing the RDMA submit. This proposal targets a different gap: it makes GPU-event readiness an independent, asynchronously-signaled axis (CUDA event callback / dedicated event poller) so the dispatcher never blocks on cudaEventSynchronize, eliminating head-of-line blocking that batching and allocation-event wakeup alone do not address. It is composable with — not duplicative of — both prior findings (they would still help with batching and allocation wakeup), but it can stand alone as a TTFT-relevant change distinct from what they describe.

---

### 2. Expire allocation-orphaned MoRIIO write tasks from the deferred queue
- **Agent:** codex

**Detailed description.**

Add an explicit timeout/failure path to `MoRIIOWriter._write_worker_loop` / `_process_deferred_tasks` in `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py:115-154` so write tasks whose remote allocation never arrives cannot remain in `_deferred_tasks` forever. Use the existing `WriteTask.enqueue_time` field, and preferably the producer-side send deadline already carried through `MoRIIOConnectorMetadata.reqs_to_send`, to compute a per-request or per-transfer deadline. When `_process_deferred_tasks` sees a task that is still not remote-ready after that deadline, atomically drop all deferred tasks for the same `transfer_id`, log a warning with `request_id`/`transfer_id`/age/deferred count, and publish a failed-or-done signal through the wrapper so the producer can stop retaining the local KV blocks instead of rescanning the task every 10 ms indefinitely. Preserve the normal path exactly: tasks that become ready before the deadline still call `_execute_write_task`, completion notification remains success-only, and late allocation messages for an expired transfer are ignored with a warning. Add fake-wrapper unit coverage for three cases: not-ready but unexpired stays deferred, ready executes, expired missing-allocation task is pruned and never executes.

**Novelty rationale.**

The two deep-research proposals optimize successful readiness by adding allocation-keyed wakeups, ready queues, batching, RDMA coalescing, and optional sleep backoff; they do not add a lifecycle bound for transfers whose remote allocation is lost or whose decode-side request is aborted. Claude's proposal removes dispatcher head-of-line blocking on CUDA event synchronization after allocation readiness; it also assumes the transfer will eventually become eligible. This proposal targets a different failure mode in the same scheduler: orphaned deferred write tasks that permanently inflate scan cost and retain producer KV blocks, which can hurt median TPOT in long multi-turn workloads with cancellations or remote allocation failures.

---
