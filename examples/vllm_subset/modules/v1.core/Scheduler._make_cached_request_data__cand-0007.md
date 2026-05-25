# Scheduler._make_cached_request_data

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 1043–1101)
- **Symbol:** `Scheduler._make_cached_request_data`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0007`

## Description
Builds CachedRequestData for scheduled running and resumed requests, including req_ids, PP token-id slices, resumed_req_ids, all_token_ids snapshots, new_block_ids, num_computed_tokens, and num_output_tokens.

## Current approach
Single Python loop over itertools.chain(running_reqs, resumed_reqs) that appends to several lists, copies all_token_ids for requests not scheduled in the previous step, and calls get_block_ids per request. Outputs are not pre-sized or reused.

## Estimated impact explanation
This is per-step allocation work proportional to scheduled request count. Optimizing it improves TPOT in high-concurrency decode, but it is secondary to scheduling decisions and update_from_output.

## Evolve rationale
The optimization unit is the per-step payload construction loop. With many scheduled requests, list growth, dict membership probes, all_token_ids.copy(), and per-request block-id materialization add scheduler CPU and allocation pressure. Headroom includes pre-sizing outputs, avoiding all_token_ids copies when downstream paths do not consume them, using stable cached block-id tuples where safe, and splitting the PP path from the common non-PP path. Correctness oracles include tests/v1/core/test_scheduler.py and tests/v1/worker/test_gpu_input_batch.py, which validate SchedulerOutput/CachedRequestData contracts.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Eliminate all_token_ids.copy() via split prompt/output token storage on Request
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/sched/scheduler.py:1043-1101, the only O(seq_len) work in the per-step loop is `all_token_ids[req_id] = req.all_token_ids.copy()` taken whenever a request was not scheduled in the previous step (resumed reqs, preempted-then-rescheduled, or first-step running). For long contexts in multi-turn agentic workloads this snapshot can be tens of thousands of ints per affected request, dominating scheduler CPU on the steps that admit/resume requests. Change `Request.all_token_ids` from a single growing list to two backing stores: `prompt_token_ids` (kept as the existing immutable tuple/sequence already provided by the engine) and `output_token_ids` (a small list that only grows by sampled/spec tokens). Have `Request.all_token_ids` remain a property (a lightweight `ChainSequence`/`_AllTokenIdsView`) that supports the existing `__getitem__`, slicing, and `len()` callers without copying. Then in `_make_cached_request_data`, replace `req.all_token_ids.copy()` with shipping a `(prompt_token_ids_ref, output_token_ids_snapshot)` pair: the prompt reference is shared (immutable, zero-copy), and only the much-smaller `output_token_ids` list is copied (or even passed as a `tuple(output_token_ids)` since output growth is append-only and the consumer reads it within the same step under the GIL). Update `CachedRequestData.all_token_ids` consumers in the worker (`gpu_input_batch`/`block_table` rebuild paths) to accept the split form, falling back to the chained view where they currently iterate. This removes the dominant allocation in the loop, leaves the existing pre-sizing and PP-split opportunities to the listed headroom, and is independent of caching block-id tuples.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's evolve_rationale lists generic headroom (pre-sizing lists, avoiding copies when downstream does not consume, cached block-id tuples, PP/non-PP split) but does not propose changing `Request`'s token storage representation to make the snapshot zero-copy by construction. This proposal targets the same hotspot from a data-structure angle rather than a loop-shape angle, and is orthogonal to and complementary with the listed micro-optimizations.

---

### 2. Replace prev-step request-id set with per-request scheduling epochs
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/sched/scheduler.py:_make_cached_request_data`, remove the `req_id in self.prev_step_scheduled_req_ids` string-set lookup from the per-request loop. Add a monotonically increasing scheduler epoch and a `last_scheduled_epoch` integer on `Request`, initialized to an invalid value. During `_make_cached_request_data`, treat a request as scheduled in the previous step when `req.last_scheduled_epoch == self._last_completed_schedule_epoch`; after the `SchedulerOutput` is built, advance the epoch and stamp all requests in `num_scheduled_tokens` with the new epoch. Reset paths that currently clear `prev_step_scheduled_req_ids` should instead invalidate the epoch by advancing it without stamping requests. This preserves the existing `all_token_ids` resend behavior for resumed or skipped-step requests while avoiding one hash lookup per cached request and the per-step `clear()`/`update()` churn on `prev_step_scheduled_req_ids`. Validate with the existing scheduler preemption/resume tests in `tests/v1/core/test_scheduler.py`, plus a focused test where a request is scheduled, skipped for one step, then scheduled again and still appears in `CachedRequestData.all_token_ids`.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes changing token storage so `all_token_ids.copy()` becomes cheaper or unnecessary; this proposal leaves token storage and copy semantics unchanged and instead removes the previous-step membership set used to decide whether that copy is required. It is also distinct from the listed generic headroom around pre-sizing outputs, PP/non-PP splitting, and cached block-id tuples.

---
