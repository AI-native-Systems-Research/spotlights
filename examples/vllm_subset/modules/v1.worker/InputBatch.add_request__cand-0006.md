# InputBatch.add_request

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_input_batch.py`](vllm/v1/worker/gpu_input_batch.py) (lines 336–482)
- **Symbol:** `InputBatch.add_request`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0006`

## Description
Admits a new or resumed request into a persistent-batch row, copying token IDs, computed-token counts, block-table rows, sampling parameters, allowed-token masks, bad-word state, pooling state, and LoRA mappings.

## Current approach
A long Python sequence writes individual numpy slots, mutates multiple req_id sets and dicts, conditionally allocates and fills the allowed_token_ids mask, and performs per-request LoRA and sampling bookkeeping.

## Estimated impact explanation
Admission overhead contributes directly to TTFT for new turns. The absolute per-request saving is smaller than full state reconciliation, but it compounds under high request churn.

## Evolve rationale
The method is called once for every admitted or resumed request. Headroom is in batching admissions collected by GPUModelRunner._update_states, vectorizing sampling-parameter writes, updating feature sets from boolean masks, and allocating/clearing allowed-token masks once per batch update. Correctness oracle: InputBatch invariants and sampling metadata equality in tests/v1/worker/test_gpu_input_batch.py plus sampling-output parity for requests with penalties, allowed tokens, bad words, and LoRA.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Add vectorized InputBatch.add_requests that batches scalar writes via numpy fancy indexing
- **Agent:** claude

**Detailed description.**

Introduce a new method `InputBatch.add_requests(requests: list[CachedRequestState]) -> list[int]` on `vllm/v1/worker/gpu_input_batch.py` (alongside the existing `add_request` at lines 336-482) that performs all per-row scalar bookkeeping for admitted requests in a vectorized way, then call it once from `GPUModelRunner._update_states` (`vllm/v1/worker/gpu_model_runner.py:1362-1366`) instead of looping `for request in reqs_to_add: self.input_batch.add_request(request)`.

Concretely, the batched path should:

1. Resolve `req_indices` for all admitted requests up front using the existing free-slot machinery (`_register_add_request`), then build small numpy arrays of length `n = len(requests)` for: `temperature`, `top_p`, `top_k` (with the `0 < top_k < vocab_size` clamp expressed as a vectorized `np.where`), `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `num_prompt_tokens`, `num_tokens_no_spec`, `num_computed_tokens`, and a constant `1` for `num_accepted_tokens_cpu`. Issue one assignment per CPU tensor using fancy indexing, e.g. `self.temperature_cpu[req_indices] = temps`, replacing the ten-plus per-row scalar stores at lines 384-407 and 358-378 and 467.
2. Replace the per-request `set.add(req_id)` calls (lines 385/388/392/395/401/404/409/428) with bulk set updates derived from boolean masks over the gathered arrays: e.g. `self.greedy_reqs.update(itertools.compress(req_ids, is_greedy_mask))` and similarly for `random_reqs`, `top_p_reqs`, `top_k_reqs`, `frequency_penalties_reqs`, `presence_penalties_reqs`, `repetition_penalties_reqs`, and `has_allowed_token_ids`. This avoids the per-request branch overhead and Python attribute lookups inside the hot loop.
3. Allocate `allowed_token_ids_mask` and `allowed_token_ids_mask_cpu_tensor` once for the whole admission (only if any admitted request carries `allowed_token_ids`), then set the rows for affected requests in one `cpu_tensor[indices_with_allowed] = True` followed by a targeted False-fill scatter for the allowed columns — instead of the per-request lazy-allocation + full-row True-fill at lines 429-448, which currently re-checks the allocation predicate on every admission.
4. Batch the block-table updates by extending `BlockTable` with an `add_rows(block_ids_list, req_indices)` call (or an internal numpy gather) rather than calling `block_table.add_row` once per request at line 379.
5. Keep the per-request paths that are inherently dict/list-shaped (token_ids_cpu/is_token_ids slice writes, prompt_embeds, generators, num_logprobs, logprob_token_ids, bad_words_token_ids, pooling_states, lora bookkeeping) inside a tight Python loop over `(req_index, request)` pairs but free of the scalar-tensor write overhead. The LoRA bookkeeping in particular benefits because `request_lora_mapping[req_indices] = lora_ids_array` (with 0 for no-LoRA) can also be vectorized.

A single `add_request` call should remain as a thin wrapper that calls `add_requests([request])[0]` to preserve the public signature used elsewhere. Correctness is verifiable against the existing oracle in `tests/v1/worker/test_gpu_input_batch.py` (InputBatch invariants and sampling-metadata equality) plus end-to-end sampling parity for requests that exercise penalties, allowed_token_ids, bad words, and LoRA.

Expected payoff under multi-turn agentic workloads: admission cost is paid on every new turn for many concurrent sessions, so collapsing ~20 scalar tensor stores and ~7 conditional set ops per request into ~7 vectorized numpy stores and ~7 bulk set updates per admission batch reduces per-request Python overhead and shortens the synchronous portion of `_update_states`, directly trimming median TTFT for newly admitted turns.

**Novelty rationale.**

The candidate currently has zero existing deep_research_proposals, so any concrete proposal is novel by construction. Beyond that, this proposal goes past the high-level evolve_rationale by specifying (a) a concrete `add_requests` plural API contract callable from a single call site in `_update_states`, (b) the exact set of CPU tensors that should be converted from scalar stores to numpy fancy-indexed assignments, (c) the specific replacement of per-request `set.add` calls with mask-driven `set.update(itertools.compress(...))` operations, (d) one-shot allocation of the allowed_token_ids mask plus a scatter-based clear/set instead of per-row True-fill, and (e) a `BlockTable.add_rows` batched companion. None of these are stated as proposals already attached to this candidate.

---

### 2. Clear stale prompt-embedding row state when reusing InputBatch slots
- **Agent:** codex

**Detailed description.**

Update `InputBatch.add_request` in `vllm/v1/worker/gpu_input_batch.py` so `req_prompt_embeds` is treated like other per-row state: when a request with `prompt_embeds is None` is admitted into an existing `req_index`, explicitly `pop(req_index, None)`; when `prompt_embeds` is present, keep the current assignment. Also clear `req_prompt_embeds` in `remove_request` for the removed row so a later add into that slot cannot inherit stale embedding tensors. This prevents `_prepare_inputs` from entering the prompt-embedding copy path for rows that no longer have embeddings, and avoids copying an old request's embeddings into a new text-only request after slot reuse. Add a focused test in `tests/v1/worker/test_gpu_input_batch.py` that adds a request with `prompt_embeds`, removes it, adds a non-embedding request into the freed row, and asserts `req_prompt_embeds` has no entry for that row and sampling/input-batch metadata remains consistent.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal is about adding a batched `add_requests` API, vectorizing scalar numpy writes, bulk-updating feature sets, batching allowed-token mask work, and batching block-table rows; it explicitly leaves dict/list-shaped state such as `prompt_embeds` in a per-request loop. This proposal is different: it targets stale per-row prompt-embedding state cleanup on slot reuse/removal, a correctness and avoidable-work issue not covered by the vectorized admission proposal.

---
