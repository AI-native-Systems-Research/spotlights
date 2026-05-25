# NixlConnectorWorker._pop_done_transfers

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`](vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py) (lines 1777–1822)
- **Symbol:** `NixlConnectorWorker._pop_done_transfers`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0003`

## Description
Worker-side polling loop that checks outstanding NIXL transfer handles and returns request IDs whose receive transfers have all completed. It is invoked from `get_finished` every engine step.

## Current approach
The method iterates over every request and every handle in Python, calls `self.nixl_wrapper.check_xfer_state(handle)` once per handle per step, records telemetry and releases each completed handle individually, and rewrites the in-progress handle list. There is no batched state query, adaptive polling cadence, prioritization, or step-budget-aware early exit.

## Estimated impact explanation
The loop runs every engine step on the worker hot path. Reducing per-handle polling overhead can lower median TPOT during steady decode when remote KV transfers are in flight, and can reduce TTFT jitter for requests waiting on transfer completion.

## Evolve rationale
This is an owned polling cadence and batching policy around NIXL handles. A batched C-side status query, adaptive cadence, or age/fairness-aware scan order can preserve the same completion semantics while reducing per-step Python and FFI overhead. Correctness oracle: with a deterministic fake wrapper, the same requests must eventually appear in `done_recving`, transfer telemetry counts must match completed handles, failed handles must mark the same invalid blocks, and `tests/v1/kv_connector/unit/test_nixl_connector.py` exercises these worker completion paths.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Defer telemetry+release off the hot path; only check_xfer_state inline
- **Agent:** claude

**Detailed description.**

In `_pop_done_transfers` (vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py:1777-1822), the hot path performs three FFI calls for every completed handle: `check_xfer_state`, `get_xfer_telemetry`, and `release_xfer_handle`. Only the first is required to decide whether a request can join `done_req_ids`. Refactor the loop so that on DONE the handle is appended to a worker-local `_pending_release: list[int]` (and, when telemetry sampling fires, to `_pending_telemetry: list[int]`), and the request is moved to `done_req_ids` immediately. After the per-request scan completes, drain `_pending_release` and `_pending_telemetry` in a single batched post-step pass — either inline at the end of `_pop_done_transfers` (cheap) or on a small background thread fed by a lock-free deque so the engine step does not pay for it at all. Pair this with a sampled-telemetry policy (e.g. 1-in-N or only when `xfer_stats.is_empty()` would otherwise be true at the next `get_kv_connector_stats` boundary) since `record_transfer` is statistical and consumers only see the aggregate from `clone_and_reset()` (worker.py:2315). Keep the existing failure path synchronous so `_handle_failed_transfer` still marks invalid blocks before `get_finished` returns. Net effect on the hot path: roughly 3× fewer FFI crossings per completed handle, and the per-step Python work shrinks to one `check_xfer_state` call plus list maintenance per in-flight handle.

**Novelty rationale.**

There are no existing deep_research_proposals for this candidate. The candidate's own evolve_rationale enumerates batched C-side state queries, adaptive polling cadence, and age/fairness-aware scan order — all of which target the *detection* side of the loop. This proposal is orthogonal: it leaves detection unchanged and instead removes the two FFI calls (`get_xfer_telemetry`, `release_xfer_handle`) that currently piggy-back on every DONE observation, by moving them to a batched/off-thread flush and adding telemetry sampling. None of the listed directions cover deferring release or sampling telemetry.

---

### 2. Flatten receive-transfer tracking with per-request remaining counts
- **Agent:** codex

**Detailed description.**

Replace `_recving_transfers: dict[req_id, list[handle]]` with a worker-local flat queue of `(req_id, handle)` entries plus `_recving_xfer_count_by_req`. When `_read_blocks` posts a handle, enqueue it and increment the request count. In `_pop_done_transfers` (worker.py:1777-1822), scan a snapshot of the queue, requeue only `PROC` handles, and decrement the request count on `DONE` or failure; add a request to `done_req_ids` only when its count reaches zero, then drop its counter. Keep telemetry, release, and failure handling synchronous for now so semantics remain unchanged. This removes the per-step `list(transfers.items())` copy, per-request `in_progress` list allocation, and dictionary rewrites while preserving the same one-state-check-per-live-handle behavior. Update shutdown to release handles from the flat queue, and extend `test_multi_xfer_one_engine`/failure tests to cover repeated handles for the same request and count cleanup.

**Novelty rationale.**

There are no listed deep_research_proposals. Agent A proposes moving DONE-side telemetry/release work off the hot path and sampling telemetry; this proposal leaves those calls inline and instead changes the in-memory tracking layout so the polling loop does less Python container churn each engine step. It is also distinct from batched C-side status queries, adaptive cadence, or scan prioritization because it preserves the polling cadence and check order semantics while reducing bookkeeping overhead.

---
