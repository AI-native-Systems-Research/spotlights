# GPUModelRunner._update_states

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 1065–1419)
- **Symbol:** `GPUModelRunner._update_states`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0001`

## Description
Reconciles cached request state and the persistent input batch with SchedulerOutput: removes finished and unscheduled requests, creates CachedRequestState entries, updates running request token, block, and speculative-decode state, admits resumed requests, condenses the batch, and refreshes metadata.

## Current approach
Several sequential Python passes over finished, unscheduled, new, and cached requests. The hot cached-request loop performs repeated dict lookups, list mutations, block-table updates, feature-flag checks, prev-index lookups, and per-request spec-decode bookkeeping before add_request, condense, and refresh_metadata run.

## Estimated impact explanation
The routine is on every decode and prefill scheduling step. In multi-turn agentic workloads with frequent request admission, preemption, and completion, reducing its Python overhead can move both median TPOT and request TTFT before GPU work starts.

## Evolve rationale
This method runs once per model step before forward execution, so batch-size-linear Python work directly affects TTFT and TPOT under high request churn. Headroom is in hoisting invariant flags, reducing repeated req_id_to_index and prev_req_id_to_index lookups, batching updates to persistent arrays, and coordinating additions/removals with condense. Correctness oracle: compare self.requests, InputBatch row mappings, block tables, spec-token state, deferred correction effects, and sampling metadata against tests such as tests/v1/worker/test_gpu_model_runner.py and tests/v1/worker/test_gpu_input_batch.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Classify-then-apply the cached-request loop with vectorized NumPy updates to per-row persistent arrays
- **Agent:** claude

**Detailed description.**

Restructure the per-cached-request Python loop in GPUModelRunner._update_states (vllm/v1/worker/gpu_model_runner.py:1199-1366) into two phases: (1) a single thin Python pass that classifies each entry of scheduler_output.scheduled_cached_reqs into mutually-exclusive buckets — running-in-batch (has req_index), resumed-from-preemption (req_index is None), running-with-new-blocks, and spec-decode-correction-needed — while gathering parallel NumPy arrays of req_indices, num_computed_tokens, num_output_tokens, and (when not is_last_rank) start/end token-write ranges; (2) a vectorized phase that applies the running-request majority updates in bulk: a single fancy-index assignment for self.input_batch.num_computed_tokens_cpu[indices] = ncs_array, a single batched block_table.append_rows call (new method that accepts a list of (row, blocks) and folds them into one CPU buffer fill), and a vectorized num_tokens_no_spec[indices] = end_idx_array write for the truncate path (lines 1296-1306). The rare paths — resumed_from_preemption (line 1314), prev_num_draft_len async-spec correction (lines 1230-1270), and per-request update_req_spec_token_ids (line 1355) — remain in Python but are iterated only over their pre-bucketed sublists. Hoist invariant flags (is_ngram_gpu, self.uses_mrope, self.uses_xdrope_dim, self.use_async_scheduling, self.use_async_spec_decode, is_last_rank, has-original_num_spec) into local bools above both loops, and bind self.input_batch.req_id_to_index / prev_req_id_to_index to local names so the hot dict.get becomes a local LOAD_FAST. Additionally, fold the unscheduled-request removal (lines 1119-1120) and the new/resumed admit-and-condense pair (lines 1364-1369) into a single index-reuse pass: directly slot reqs_to_add into indices freed by removed unscheduled+finished requests in this step, eliminating the condense() call when the freed-set size ≥ added-set size. Correctness is preserved because add_request already prefers smallest empty indices and condense is only required when trailing gaps remain; a final condense() is invoked only if that condition is detected. The classify pass also lets us short-circuit when scheduled_spec_tokens, original_num_spec_per_req, and resumed_req_ids are all empty AND there are no new/finished/unscheduled — a steady-state pure-decode fast path that skips reqs_to_add iteration, condense, and the deferred-correction closure construction entirely. Validate against tests/v1/worker/test_gpu_model_runner.py and tests/v1/worker/test_gpu_input_batch.py, plus the spec-decode path coverage in tests/v1/spec_decode and tests/v1/e2e for async scheduling. Microbench: time _update_states under a synthetic 256-request decode-heavy step to confirm the median-step Python time drop, and end-to-end measure TTFT/TPOT on a multi-turn agentic trace where churn rate varies.

**Novelty rationale.**

The candidate has no existing deep_research_proposals, so any concrete proposal is novel by construction. Beyond that, the evolve_rationale only gestures at headroom ("hoisting invariant flags", "reducing repeated lookups", "batching updates to persistent arrays", "coordinating additions/removals with condense") without prescribing a structural change. This proposal commits to a specific two-phase classify-then-apply restructuring with named NumPy fancy-index targets (num_computed_tokens_cpu, num_tokens_no_spec, token_ids_cpu ranges), a new batched block_table.append_rows API surface, an explicit index-reuse path that elides condense, and a steady-state fast-path predicate — a concrete refactor rather than the rationale's enumeration of headroom areas.

---

### 2. Add a parked-row fast path for resumed cached requests
- **Agent:** codex

**Detailed description.**

In GPUModelRunner._update_states, make unscheduled cached-request removal preserve reusable CPU batch rows for requests that are removed from the persistent batch but kept in self.requests. Extend InputBatch.remove_request with a preserve_row mode used only for unscheduled/preempted removals, recording a small parked-row descriptor on the CachedRequestState or InputBatch: row index, active token count, prompt length, and a row epoch. Add InputBatch.try_add_parked_request for reqs_to_add before the normal add_request path: if the parked row was not overwritten by condense, swap, move, or another add, restore req_id_to_index, req_output_token_ids, sampling metadata, LoRA/generator state, and the block table without rewriting the full prompt/output token_ids_cpu and is_token_ids slices; if the destination row differs but the source row is still valid, copy the active NumPy slice row-to-row; otherwise fall back to add_request. Invalidate row epochs on every row overwrite/move. Finished requests and streaming updates with changed prompt content should never use the parked fast path. This targets long-context preemption/resume churn where the current resumed path replays add_request and converts/copies the whole prompt plus historical outputs back into token_ids_cpu even though the worker already had that row recently. Validate with tests/v1/worker/test_gpu_model_runner.py resume/unscheduled cases, tests/v1/worker/test_gpu_input_batch.py condense/reorder cases, and an added forced-preemption test with long prompts that compares req_id_to_index, token_ids_cpu, is_token_ids, block_table, sampling_metadata, and spec_token_ids against the existing slow path.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A focuses on vectorizing the scheduled-cached loop, batching block-table appends, hoisting invariants, reusing same-step freed indices, and eliding condense in certain cases. This proposal addresses a different cost center: the O(sequence length) row repopulation performed by InputBatch.add_request when a cached request returns after being off-batch. It does not depend on vectorizing the cached loop or changing condense policy; it adds a validated row-cache/reattach path for resumed requests that Agent A did not cover.

---
