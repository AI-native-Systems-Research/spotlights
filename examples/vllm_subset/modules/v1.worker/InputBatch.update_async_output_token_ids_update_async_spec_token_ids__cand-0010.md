# InputBatch.update_async_output_token_ids/update_async_spec_token_ids

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_input_batch.py`](vllm/v1/worker/gpu_input_batch.py) (lines 1019–1079)
- **Symbol:** `InputBatch.update_async_output_token_ids/update_async_spec_token_ids`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0010`

## Description
Repairs async-scheduling sampling metadata by replacing placeholder output token IDs and speculative draft-token IDs with results copied from the previous step.

## Current approach
The output-token path loops over request IDs, synchronizes the async CPU copy on first use, converts the whole sampled-token tensor to nested Python lists, scans lists with index(-1), deletes list suffixes, and splices replacements. The spec-token path performs another per-request loop and mutates draft/spec lists in place.

## Estimated impact explanation
The path is conditional on async scheduling plus token-history consumers, but that combination is important for low-TPOT spec decode with penalties or bad words. Reducing synchronization-adjacent Python list work improves median TPOT in that configuration.

## Evolve rationale
This region is called immediately before logits processors or rejection sampling need token history. Headroom is in tracking first-placeholder offsets, slicing only active rows from the copied tensor, avoiding full tolist conversion, and consolidating output/spec repairs from one prev_req_id_to_index traversal. Correctness oracle: per-step output_token_ids/spec_token_ids equality in async scheduling and async spec-decode tests, including KV-load discard and placeholder over-acceptance cases.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Consolidate async output/spec repair into a single pass with cached first-placeholder offsets and per-row tensor slicing
- **Agent:** claude

**Detailed description.**

Refactor `InputBatch.update_async_output_token_ids` and `update_async_spec_token_ids` (vllm/v1/worker/gpu_input_batch.py:1019-1079) into one consolidated repair pass that eliminates the three dominant Python costs in the current code (full nested-list materialization of the sampled-token tensor, two `list.index(-1)` linear scans per request, and two separate per-request loops):

1. Maintain a `first_placeholder_offsets: list[int | None]` mirror parallel to `sampling_metadata.output_token_ids`, written at the same site that appends `-1` placeholders during async scheduling. This replaces the per-step `req_output_token_ids.index(-1)` scan at line 1054 with an O(1) lookup. Reset/clear entries when a row is repaired or when the slot is freed in the existing add/remove/condense paths so the offset stays in sync with the placeholder.

2. After `async_copy_ready_event.synchronize()`, compute per-row valid sample counts once via a single vectorized op on the already-resident CPU tensor — e.g. `valid_counts = (self.sampled_token_ids_cpu != -1).sum(dim=1).tolist()` — and skip the `sampled_token_ids_cpu.tolist()` whole-tensor materialization at line 1044. The current code converts every row to a nested Python list even though only rows whose `req_id` appears in `prev_req_id_to_index` (and whose current row still has a placeholder tail) are actually consumed; this is wasted Python-object allocation that dominates the call's wall time when the batch is large but the active-async subset is small.

3. Iterate `prev_req_id_to_index.items()` exactly once, mapping each `prev_index` to the current row via `self.req_id_to_index.get(req_id)`. For each match, in the same iteration: (a) read `count = valid_counts[prev_index]`, slice only the needed row with `new_ids = self.sampled_token_ids_cpu[prev_index, :count].tolist()`, and splice into `output_token_ids[curr_idx][first_placeholder_offsets[curr_idx]:]` clipped by `num_placeholders`; (b) if `draft_token_ids` was passed in (today's `update_async_spec_token_ids` path) and `spec_token_ids[curr_idx]` is non-empty, repair it from `draft_token_ids[prev_index]` in the same step. This collapses the two methods that today each iterate `self.req_ids` independently into a single pass keyed off the (typically smaller) `prev_req_id_to_index` dict, and removes the redundant `for req_id, spec_ids in zip(self.req_ids, spec_token_ids)` scan at line 1071 over rows that have no async-copy work.

4. Keep public callsites unchanged — expose a thin shim `update_async_output_token_ids()` and `update_async_spec_token_ids(draft_token_ids)` that route through one private `_repair_async_token_state(draft_token_ids=None)` helper so the second call becomes a near no-op when the first already did the heavy work in the same step.

Correctness oracle remains the existing async-scheduling and async-spec-decode tests (KV-load discard with fewer placeholders than sampled, optimistic over-acceptance with more placeholders than sampled, and `output_token_ids[-1] != -1` short-circuit when tokens were rolled back). Because step 1 changes the placeholder bookkeeping invariant, audit every site that mutates `sampling_metadata.output_token_ids` (placeholder append, KV-load failure rollback, request add/remove/condense) and add asserts in debug mode that `first_placeholder_offsets[i]` is None iff `output_token_ids[i]` has no `-1` tail.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. While the candidate's own evolve_rationale lists the general headroom areas (offset tracking, row slicing, avoiding tolist, consolidation), this proposal commits to specific mechanisms that are not entailed by that prose: (a) a parallel `first_placeholder_offsets` mirror list updated at placeholder-append time rather than recomputed via `.index(-1)`, with explicit invariant maintenance across the add/remove/condense/rollback paths; (b) a single vectorized `(tensor != -1).sum(dim=1).tolist()` valid-count derivation that replaces both the per-row `new_ids.index(-1)` scan and the full-tensor `.tolist()`; and (c) routing both public methods through one shared `_repair_async_token_state` helper keyed off `prev_req_id_to_index.items()` (the smaller dimension) instead of `self.req_ids`, which is the consolidation step the rationale gestures at but does not specify. Each of these is a concrete, reviewable code change rather than a restatement of the headroom.

---

### 2. Share one lazy async-sampled-token parse between repair and output finalization
- **Agent:** codex

**Detailed description.**

Introduce a small shared async sampled-token cache owned by `AsyncGPUModelRunnerOutput` and stored on `InputBatch` by `set_async_sampled_token_ids`, replacing the raw `sampled_token_ids_cpu`/`async_copy_ready_event` pair used by `InputBatch.update_async_output_token_ids` in `vllm/v1/worker/gpu_input_batch.py:1019-1058`. The cache should synchronize the copy event once, parse valid sampled token rows once, and expose read-only row data to the repair path. `AsyncGPUModelRunnerOutput.get_output()` should consume the same cache when building `ModelRunnerOutput.sampled_token_ids`, returning fresh mutable list copies to the scheduler so scheduler stop-trimming cannot mutate the cached rows. In `update_async_output_token_ids`, splice from a copied prefix of the cached row rather than mutating the cached row with `del new_ids[...]`. Guard the cache with a lock because `get_output()` can run on the `WorkerAsyncOutput` thread while the next sampling step repairs async token history. This removes the current duplicated CPU tensor parsing where `update_async_output_token_ids()` calls `tolist()` for token-history repair and `AsyncGPUModelRunnerOutput.get_output()` later calls `tolist()` or `RejectionSampler.parse_output()` again for the same previous-step tensor.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A focuses on making the repair loop itself cheaper by caching placeholder offsets, slicing active rows, vectorizing valid-count detection, and consolidating output/spec repairs. This proposal targets a different duplication boundary: the same async CPU tensor is parsed independently by `InputBatch.update_async_output_token_ids` and by `AsyncGPUModelRunnerOutput.get_output`, potentially on different threads. A shared, thread-safe lazy parse cache is orthogonal to Agent A's loop restructuring and can coexist with it by serving the row slices that Agent A's repaired loop would consume.

---
