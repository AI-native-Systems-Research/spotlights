# GPUModelRunner._update_states

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 1233–1607)
- **Symbol:** `GPUModelRunner._update_states`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0002`

## Description
Applies scheduler deltas for finished, unscheduled, new, resumed, and scheduled requests to cached request state and the persistent legacy GPU input batch.

## Current approach
Uses nested Python loops over scheduler request groups, per-request dict lookups, list extensions for block_ids and output_token_ids, per-request CPU tensor slice writes, and per-request block_table.append_row/add_row calls before condensing the batch.

## Estimated impact explanation
Admission, preemption, and finish handling gate TTFT for new turns and affect churn-heavy decode steps; reducing Python O(active requests) work lowers median TTFT and TPOT under high request turnover.

## Evolve rationale
Request churn is common in multi-turn agent workloads and this method runs before every legacy model step. Batched/vectorized state updates and block-table row mutations can preserve semantics covered by tests/v1/worker/test_gpu_model_runner.py::test_update_states_new_request, test_update_states_request_finished, test_update_states_request_resumed, test_update_states_no_changes, and test_update_states_request_unscheduled.

## Deep research proposals

### 1. Adopt MRV2-style decoupled persistent/per-step state in _update_states
- **Finding:** `find-vllm_v1_worker-0003` — *Model Runner V2 Design Document*
- **Source URL:** <https://docs.vllm.ai/en/v0.17.0/design/model_runner_v2/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor GPUModelRunner._update_states (vllm/v1/worker/gpu_model_runner.py:1233-1607) to apply the Model Runner V2 pattern: decouple persistent per-request state from per-step input tensors. Concretely, (1) keep persistent state (block_table rows, output_token_ids, cached prompt/sampling metadata) at stable row indices assigned when a request is admitted, and stop condensing/reordering the batch on finish/unschedule; instead mark rows free and reuse row slots on new admissions. (2) Replace the per-request Python loops that do dict lookups, list extensions, and per-request CPU tensor slice writes with vectorized/batched updates: collect all newly scheduled block-id extensions and new-token appends across requests, then apply them via a single scatter/index_copy into the GPU-resident block_table and output_token_ids buffers (mirroring MRV2's 'gather runs in parallel on the GPU with low overhead'). (3) Build per-step input tensors by gathering from the persistent GPU-resident state using an active-row index tensor, rather than rewriting the persistent batch every step. Preserve semantics validated by tests/v1/worker/test_gpu_model_runner.py::test_update_states_{new_request,request_finished,request_resumed,no_changes,request_unscheduled} by keeping the same observable effects on cached_request_states and the input batch view.

**Proposal rationale.**

The finding directly targets the exact hot path in this candidate: MRV2's design explicitly addresses request-churn bookkeeping in the persistent batch, which is what _update_states does before every legacy step. The candidate's current_approach (per-request dict lookups, list extensions, per-request CPU tensor slice writes, block_table.append_row/add_row per request, followed by condense) is the pattern MRV2 was designed to replace with stable rows plus GPU-side gather. Under the caller's multi-turn agentic workload, admission/preemption/finish churn is frequent, so removing per-request Python work and batch-wide reordering plausibly reduces the Python overhead that gates TTFT for new turns and TPOT during churn-heavy decode, matching the candidate's stated impact mechanism.

---

## Agent proposals

### 1. Fuse remove/condense/add churn into in-step slot-swap admission in _update_states
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py:1233-1607, replace the three-pass churn pattern (remove finished + unscheduled → add new + resumed → condense) with a single in-step slot-swap pass. Concretely: (1) Build two ordered lists in one traversal — vacated_slots (row indices freed by finished_req_ids and unscheduled_req_ids) and to_admit (CachedRequestState objects for scheduled_new_reqs and resumed_req_ids). (2) Pair them index-wise so each admission directly overwrites a vacated row (block_table row, num_computed_tokens_cpu, num_tokens_no_spec, token_ids_cpu slice, sampling/pooling metadata, req_id_to_index) instead of going through InputBatch.remove_request → free-index heap → add_request → condense. Only truly leftover vacated slots need condensing, and only truly leftover admissions need free-slot allocation, so both `remove_request` loops and the final `condense()` pass shrink to |free_slots − admits| entries. (3) Batch the sampling-metadata and LoRA "dirty" marking into a single delta apply at the end of the swap pass rather than per-request inside remove_request/add_request. (4) Preserve the abort-then-resubmit edge case (same req_id in finished_req_ids and scheduled_new_reqs) by processing finished removals into vacated_slots before matching admits. Keep observable effects identical for test_update_states_new_request, test_update_states_request_finished, test_update_states_request_resumed, test_update_states_no_changes, and test_update_states_request_unscheduled by asserting req_id_to_index matches the equivalent remove/add/condense outcome (any stable pairing satisfies the current tests, which check per-slot correctness rather than slot identity).

**Novelty rationale.**

The existing deep_research proposal (find-vllm_v1_worker-0003) proposes MRV2-style *persistent* stable rows that live across steps, eliminating condense by never reordering and building input tensors via GPU-side gather from persistent buffers. That is a cross-step architectural change requiring new GPU-resident state and a gather path. This proposal is orthogonal and complementary: it keeps the current per-step InputBatch layout and semantics intact, and instead targets the *within-step* Python churn cost by pairing removals with admissions into direct slot overwrites — collapsing three sequential passes (remove loop, add loop, condense) into one, and batching dirty-flag maintenance. It requires no GPU-side data structure changes and lands independently of (and stacks with) MRV2 stable rows; even under MRV2 you still admit and free rows, so slot-swap admission remains a useful reduction of per-request Python overhead. It also does not overlap with MRV2's vectorized block-id scatter — the block-table row still gets written per admission here, just once instead of remove-then-append-row.

---

### 2. Cache per-step scheduler membership to avoid repeated set/dict probes
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu_model_runner.py:1233-1607`, restructure `_update_states` so the scheduler delta is normalized once at the top into compact membership maps/lists used by all later branches. Concretely, build local aliases for `cached_request_states`, `req_id_to_index`, and the scheduler-output collections, then precompute: `finished_or_unscheduled_req_ids`, a `resumed_req_id_to_new_blocks` map from `scheduler_output.scheduled_cached_reqs`, and an ordered list of scheduled-running request ids that excludes new admissions. Use these normalized structures for the cached-state updates and the legacy `InputBatch` mutations instead of repeatedly testing membership across `finished_req_ids`, `scheduled_new_reqs`, `scheduled_cached_reqs`, and `req_id_to_index` inside separate loops. This is a narrow Python hot-path cleanup: fewer repeated hash lookups and less branch work per active request, while preserving the same remove/add/condense behavior and the existing block-table row mutation API. Add a focused microbenchmark or timing assertion around `_update_states` with a churn-heavy synthetic scheduler output in `tests/v1/worker/test_gpu_model_runner.py` or a nearby benchmark to verify lower overhead without changing semantics.

**Novelty rationale.**

The deep_research proposal is an architectural MRV2-style redesign with stable persistent rows, GPU-resident scatter/gather, and no per-step condensing. Claude's proposal changes the mutation algorithm by directly pairing vacated slots with admissions to avoid remove/add/condense churn. This proposal deliberately avoids both: it keeps the existing data structures, row lifecycle, block-table calls, and condense semantics, and only reduces Python overhead from repeated scheduler-delta membership/dict work within the current algorithm. It is therefore independently landable as a small hot-path optimization and does not duplicate either stable-row architecture or slot-swap admission.

---
