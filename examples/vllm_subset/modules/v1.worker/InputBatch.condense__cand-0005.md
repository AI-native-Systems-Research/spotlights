# InputBatch.condense

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_input_batch.py`](vllm/v1/worker/gpu_input_batch.py) (lines 684–810)
- **Symbol:** `InputBatch.condense`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0005`

## Description
Compacts the persistent batch after removals by moving active high-index rows into lower empty slots and updating parallel token, sampling, LoRA, block-table, and logits-processor state.

## Current approach
A while-loop pairs each empty index with the largest non-empty row and copies many arrays one row at a time. It also performs per-row list swaps, dict pop/insert operations, block_table.move_row calls, and optional mask, bad-word, and generator remapping.

## Estimated impact explanation
Request churn creates TPOT spikes when compaction does many serial row copies. A batched compaction path can reduce those spikes and improve median TPOT for high-concurrency agentic workloads.

## Evolve rationale
Condense is triggered whenever request removals leave holes, which is common in multi-turn agentic traffic. Headroom is in computing a single row permutation, applying batched numpy/torch indexed moves across row-major arrays, and remapping dictionaries from one move table instead of per-row operations. Correctness oracle: compare req_id_to_index, req_ids ordering, block table rows, sampling metadata, LoRA mappings, bad-word masks, and generator associations using tests/v1/worker/test_gpu_input_batch.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace condense() with a slot-indirection layer so removals leave holes that consumers gather through, eliminating per-churn data movement
- **Agent:** claude

**Detailed description.**

Stop physically compacting the persistent batch in `InputBatch.condense` (vllm/v1/worker/gpu_input_batch.py:684-810). Introduce a small `self.active_to_slot: np.ndarray` (length up to `max_num_seqs`) that maps the dense logical position `[0, num_reqs)` used by the forward pass to the physical persistent-batch row in `token_ids_cpu`, `is_token_ids`, `num_tokens_no_spec`, `num_prompt_tokens`, `num_computed_tokens_cpu`, `block_table.block_table`, `request_lora_mapping`, `temperature_cpu`/`top_p_cpu`/`top_k_cpu`/`frequency_penalties_cpu`/`presence_penalties_cpu`/`repetition_penalties_cpu`/`num_accepted_tokens_cpu`, and the dict-keyed state (`generators`, `bad_words_token_ids`, `req_prompt_embeds`). Removals from `batch_update_builder.removed` just mark slots free in a bitmap and pop from `active_to_slot`; adds reuse the lowest free slot (already the existing add path); `condense()` becomes a no-op aside from `del self._req_ids[num_reqs:]` style trimming and a single `np.flatnonzero` to refresh the active vector at the end of `_make_batch_update_metadata`/before `refresh_metadata`. Then port consumers: (a) anywhere a kernel currently reads block_table or per-row CPU/GPU buffers as `[:num_reqs]`, plumb `active_to_slot[:num_reqs]` and use it as a gather index when uploading to GPU (one `np.take` into the contiguous GPU staging buffer per step instead of N row-copies plus a contiguous H2D); (b) `req_id_to_index` already returns physical slot indices, so internal callers stay correct; only the dense-position consumers (sampler, attention metadata builders) need the indirection. Logitsprocs that depend on the `batch_update_builder.moved` log can be fed the synthesized permutation from the previous active vector to the new one — computed once via `np.where`/argsort on the bitmap. Outcome: the cost of churn drops from O(num_reqs * num_arrays) Python+numpy work per step to O(removed) bitmap edits plus a single staging-gather that replaces an already-existing H2D copy, and median-TPOT spikes after removals in multi-turn agentic traffic disappear.

**Novelty rationale.**

The candidate has no listed deep_research_proposals, and the evolve_rationale frames the win as computing a single permutation and replacing per-row numpy/torch copies with batched indexed moves — i.e., make the existing copy pattern faster. This proposal is qualitatively different: it removes the copies altogether by introducing a logical→physical slot indirection so the persistent batch is allowed to stay sparse, and the only per-step cost is producing a small gather vector consumed when state is staged to GPU. None of the rationale's bullets (single permutation, batched indexed numpy/torch moves, dict remapping from one move table) describe avoiding compaction itself or repointing forward-pass consumers through an indirection layer.

---

### 2. Add a suffix-removal fast path to condense()
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu_input_batch.py:684-810`, add a local fast path for the common case where all remaining `batch_update_builder.removed` indices are outside the new dense prefix, i.e. `min(removed) >= self.num_reqs`. In that case `condense()` does not need to move any active row; it can just trim `_req_ids`, `req_output_token_ids`, and `spec_token_ids` to `num_reqs` and return, leaving `batch_update_builder.removed` intact for logits processors. For mixed removals, keep the existing physical move semantics but build a small `removed_set` or boolean mask from `empty_req_indices` and use it for the `while last_req_index in empty_req_indices` scan so large removal bursts do not pay repeated O(R) list-membership checks. Add a test case in `tests/v1/worker/test_gpu_input_batch.py` that removes a large suffix and asserts no moved entries are recorded, tail removals remain visible to `refresh_metadata()`, and the lists are trimmed correctly.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. This is also distinct from Claude's proposal: it does not introduce logical-to-physical slot indirection, sparse persistent rows, or consumer-side gather indices. It keeps the current dense physical layout and row-move behavior, but avoids unnecessary work when removals are already a suffix and removes the current O(R^2)-style membership scan in the existing loop.

---
