# EngineCoreProc._process_engine_step

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/core.py`](vllm/v1/engine/core.py) (lines 1205–1223)
- **Symbol:** `EngineCoreProc._process_engine_step`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0005`

## Description
Runs one engine step, enqueues outputs, calls post_step, and sleeps briefly when no model execution happened while unfinished requests remain, such as when waiting for remote KV transfer.

## Current approach
Hard-coded time.sleep(0.001) whenever not model_executed and scheduler.has_unfinished_requests() is true, independent of consecutive empty-step count, remote-KV progress signals, or background-thread readiness.

## Estimated impact explanation
In remote-KV or partially blocked agentic deployments, this sleep is on the critical path for resumed requests. Tuning it can reduce TTFT after KV arrival and avoid unnecessary TPOT stalls, but it only matters for workloads that hit the no-model-executed state.

## Evolve rationale
The concrete construct is the fixed 1 ms sleep at line 1221. Adaptive backoff, bounded zero-sleep spinning, or event-driven wakeup from KV-transfer threads can be evaluated without changing request semantics. Correctness oracle: distributed KV/disaggregated-prefill tests must still make forward progress and produce identical outputs; a targeted benchmark can measure time from remote-KV readiness to resumed decode.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace fixed 1 ms sleep with a bounded wait that also wakes on new client input
- **Agent:** claude

**Detailed description.**

At vllm/v1/engine/core.py:1220-1221, replace the unconditional `time.sleep(0.001)` inside `EngineCoreProc._process_engine_step` with a bounded wait that wakes on either (a) the same ~1 ms timeout, or (b) arrival of a new item on `self.input_queue`. Concretely, swap `time.sleep(0.001)` for `with self.input_queue.not_empty: self.input_queue.not_empty.wait(timeout=0.001)` (using `queue.Queue`'s existing internal Condition), or equivalently expose a small `_idle_wakeup` `threading.Event` that `put`/`put_nowait` paths set and that this site `wait(timeout=0.001)`s on. Behavior is unchanged when the engine is purely stalled on remote KV (it still yields the GIL for ~1 ms so NIXL/handshake threads progress, satisfying the existing comment), but a new client request — including the next turn of an in-flight agentic conversation or an abort — preempts the wait immediately instead of being delayed by up to 1 ms per stalled step. This keeps the call site local: no scheduler/connector API changes, no spin loop, no adaptive timing logic.

**Novelty rationale.**

The candidate's `evolve_rationale` lists adaptive backoff, bounded zero-sleep spinning, and event-driven wakeup from KV-transfer threads — all framed around the remote-KV side. None of them address the orthogonal cost that the same `time.sleep` also blocks the engine thread from servicing `input_queue`, which is exactly the path multi-turn agentic workloads take when a new turn arrives during a remote-KV stall. Waking on input-queue activity is a different, additive lever that targets the stated TTFT objective for multi-turn agentic traffic without requiring any KV-connector redesign, and is not implied by any listed proposal (deep_research_proposals is empty).

---

### 2. Skip the idle sleep after a no-token step that just observed KV receive completion
- **Agent:** codex

**Detailed description.**

In vllm/v1/engine/core.py around EngineCoreProc._process_engine_step, keep the existing 1 ms yield for true no-progress empty polls, but bypass it when the just-completed step has already learned that a remote KV load finished. Concretely, expose a small scheduler query such as `has_ready_remote_kv_waiters()` backed by `finished_recving_kv_req_ids`, or return a `kv_recv_progress` flag from `step()` / `step_with_batch_queue()` when `model_output.kv_connector_output.finished_recving` is non-empty. Then change the sleep condition to only sleep when `not model_executed`, unfinished requests remain, and no remote-KV waiter became schedulable in that same step. This avoids adding a fixed 1 ms delay between recording `finished_recving` in `scheduler.update_from_output()` and the next scheduler pass that promotes the request back to WAITING/PREEMPTED and resumes decode.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes waking the bounded wait on new client input or aborts arriving through `input_queue`; this proposal is orthogonal because it uses KV progress that the engine has already observed during the current no-model step, and skips the sleep before the next scheduling pass rather than waiting for client-side activity.

---
