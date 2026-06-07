# GPUModelRunner._prepare_inputs

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 1787–2096)
- **Symbol:** `GPUModelRunner._prepare_inputs`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0015`

## Description
Prepares token, position, sequence-length, discard-mask, slot-mapping, speculative-logit, and LoRA inputs for a scheduled model step before attention metadata and forward execution.

## Current approach
The method performs numpy repeat/cumsum work, torch.from_numpy wrapping, CPU index_select into staging tensors, prompt-embed per-request copies, full-tail query_start_loc fills, multiple small copy_to_gpu transfers, an optional num_accepted_tokens event synchronize, and separate GPU kernels/copies for slot mappings, positions, sequence lengths, rotary positions, and spec-decode side metadata.

## Estimated impact explanation
The method sits directly before every target-model forward. Reducing Python array construction, small H2D transfers, and synchronization in this path moves median TPOT for decode-heavy agentic workloads and TTFT for resumed or chunked-prefill turns.

## Evolve rationale
This is the classic GPUModelRunner input-prep hot path and runs once per forward pass. Headroom is in reusing scratch tensors for req_indices/token_indices/num_tokens, coalescing H2D copies for query_start_loc, discard masks, req indices and scheduled-token counts, avoiding whole-tail fills when padding shape is unchanged, and fusing adjacent position/seq-len/slot-mapping preparation where contracts allow. Correctness oracle: field-by-field equality of input_ids, positions, seq_lens, query_start_loc, discard_request_mask, num_computed_tokens, slot mappings, logits_indices, spec metadata, and LoRA activation in tests/v1/worker/test_gpu_model_runner.py plus e2e logits/sampling parity.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace per-step num_tokens Python list-comp with an incrementally maintained input_batch.num_total_tokens_cpu array
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py:1925-1932, _prepare_inputs builds `num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]` and then `num_tokens_np = np.array(num_tokens, dtype=np.int32)` every forward pass solely to compute `discard_request_mask` via `optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np`. This pays a Python-level dict lookup and attribute access into the separate `self.requests` mapping for every active request on every step, plus a fresh small allocation. Replace this with a pre-allocated `input_batch.num_total_tokens_cpu` numpy array (and matching pinned tensor view) maintained incrementally: initialize on `add_request` to `len(prompt_token_ids)` (or full restored length on resume), increment by the number of newly sampled output tokens at the same site that already updates `num_computed_tokens_cpu` after each step, and compact in `condense()` alongside the other per-request CPU arrays. Then _prepare_inputs becomes `np.less(self.input_batch.optimistic_seq_lens_cpu_np[:num_reqs], self.input_batch.num_total_tokens_cpu[:num_reqs], out=self.discard_request_mask.np[:num_reqs])` followed by the existing `copy_to_gpu(num_reqs)`, eliminating both the Python iteration over req_ids/self.requests and the per-step `np.array` allocation. Correctness oracle is unchanged (`tests/v1/worker/test_gpu_model_runner.py` discard-mask assertions and e2e sampling parity), and the maintenance hooks live in the same `add_request` / `condense` / output-append paths that already maintain `num_computed_tokens_cpu`.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's evolve_rationale enumerates scratch-tensor reuse for req_indices/token_indices/num_tokens, H2D copy coalescing, avoiding whole-tail fills, and fusing position/seq-len/slot-mapping kernels — it does not call out the Python-level cross-structure lookup into `self.requests` to rebuild `num_tokens` each step, nor the resulting per-step `np.array` allocation, nor the architectural shift from per-step recomputation to incremental maintenance on the input_batch. This proposal targets that specific Python-loop dependency rather than the GPU/H2D-copy mechanics already covered.

---

### 2. Add a token-only fast path for prompt-embeds-enabled decode steps
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu_model_runner.py:_prepare_inputs` around lines 1853-1899, derive a per-step `has_prompt_embed_tokens` flag from `input_batch.req_prompt_embeds` and each request's scheduled `[num_computed_tokens_cpu, num_computed_tokens_cpu + num_scheduled_tokens)` range, treating mixed prompts as active only when that range contains `is_token_ids == False`. When the flag is false, skip the `is_token_ids` index_select and prompt-embed CPU slice-copy loop, and have `_prepare_input_ids` skip `inputs_embeds.copy_to_gpu()` / `is_token_ids.copy_to_gpu()` for that step. Thread the flag to the model-input selection so prompt-embeds-capable models use the normal `input_ids` path for token-only decode/resume steps, with a separately captured token-id CUDA graph for those models; keep the existing `inputs_embeds` path when any scheduled token is actually an embed. This avoids hidden-size embedding H2D copies and out-of-graph token embedding on the long generated tail after media or prompt-embed prefills. Validate with parity tests for a prompt-embeds prefill followed by decode steps, plus a mixed batch where one request still has embed positions and another is token-only.

**Novelty rationale.**

There are no existing deep_research_proposals for this candidate. Agent A's proposal targets the discard-mask path at lines 1925-1932 by incrementally maintaining total token counts; it does not touch prompt-embed scheduling, `inputs_embeds` transfers, or runtime selection between `input_ids` and `inputs_embeds`. This proposal is also distinct from generic scratch reuse, metadata H2D coalescing, tail-fill avoidance, and slot/position fusion ideas because it changes the prompt-embed branch behavior for token-only decode steps.

---
