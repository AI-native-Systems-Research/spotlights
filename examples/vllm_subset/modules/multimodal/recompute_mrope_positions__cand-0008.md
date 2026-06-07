# recompute_mrope_positions

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/evs.py`](vllm/multimodal/evs.py) (lines 154–356)
- **Symbol:** `recompute_mrope_positions`
- **Kind:** function
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0008`

## Description
Recomputes mrope positions for chunked-prefill multimodal inputs after media-token pruning, including Qwen2.5-VL and Qwen3-VL timestamp/video layouts.

## Current approach
A Python loop over multimodal_positions repeatedly slices tensors, computes count_nonzero/nonzero, calls .item() for boolean and index decisions, recomputes seen vision-start slices, and rebuilds trailing text positions with torch.cumsum for each media segment.

## Estimated impact explanation
This function is called during prefill for image/video-bearing Qwen VL requests. If tensors are on GPU, .item() and Python-controlled tensor decisions serialize the prefill path; even on CPU, repeated scans add overhead. Vectorizing the segment math cuts media TTFT for long multimodal prompts.

## Evolve rationale
The loop at lines 238-353 contains device-to-Python scalar conversions and repeated prefix-count work. Headroom includes precomputing vision_start_indices and media prefix sums once, replacing per-iteration count_nonzero with cumsum lookups, batching segment offsets, and keeping decisions on-device until the final delta. Oracle: tests/model_executor/test_qwen3_vl_mrope.py plus Qwen2.5/Qwen3 VL recompute_mrope_positions callers must produce identical positions and mrope_position_delta for full and chunked prefill cases.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Cache per-request prefix scans across chunked-prefill calls
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/evs.py:154-356, recompute_mrope_positions is invoked once per chunked-prefill step from gpu_model_runner.py:3035, always against the same full prompt_token_ids on req_state. Every call re-derives sequence-level quantities that depend only on input_ids and the model's token IDs: image_mask, video_mask, media_mask, text_mask, total_mm_tokens, vision_start_indices, and (after the loop refactor) the full-sequence text cumsum. For long video prompts split into many prefill chunks, this is repeated O(num_chunks) times. Proposal: introduce a small cached struct (e.g. attached to CachedRequestState as mrope_evs_cache, keyed by id(prompt_token_ids) plus the (image_token_id, video_token_id, vision_start_token_id) triple to invalidate on token-id mismatch) holding precomputed: media_mask, text_mask, vision_start_indices, total_mm_tokens, and a precomputed cumulative media-prefix-count tensor used by the loop's seen_mm_tokens / seem_mm_tokens_before_last_vision_start lookups (each becomes an O(1) gather instead of count_nonzero over a slice). recompute_mrope_positions gains an optional cache argument (or builds-and-stores on first call); seen_mm_tokens for a given num_computed_tokens reduces to media_prefix_count[num_computed_tokens]; the bisect over vision_start_indices becomes a torch.searchsorted on the cached tensor. The function still produces identical positions/delta because the cached values are pure functions of input_ids and token ids that are constant across chunks. Validation: tests/model_executor/test_qwen3_vl_mrope.py plus the Qwen2.5/Qwen3 VL chunked-prefill paths in qwen2_5_vl.py / qwen3_vl.py / qwen3_5.py must produce byte-identical mrope_positions and mrope_position_delta for both single-shot and multi-chunk prefill; add a microbench measuring the per-chunk cost on a 32k-token video prompt split into 4-8 chunks to confirm the saved work translates to TTFT improvement. Implementation should keep the cache lifetime bounded to the request (cleared in the same place CachedRequestState is dropped) so it does not retain prompt tensors after the request finishes.

**Novelty rationale.**

The candidate has no listed deep_research_proposals, and the evolve_rationale on the candidate itself targets within-call vectorization (precomputing vision_start_indices, replacing per-iteration count_nonzero with cumsum lookups, batching offsets, keeping decisions on-device). It treats each invocation in isolation. This proposal is orthogonal: it eliminates redundant work *across* the repeated invocations that chunked prefill performs against the same prompt_token_ids, by hanging a per-request cache off CachedRequestState. The two ideas compose (the cached media_prefix_count is exactly the structure the in-loop vectorization wants), but caching across chunks is not implied by, or covered in, the candidate's stated headroom.

---

### 2. Update request M-RoPE positions in place
- **Agent:** codex

**Detailed description.**

Add an explicit in-place path to `vllm/multimodal/evs.py:154-356` for callers that own the passed `mrope_positions` tensor. Instead of unconditionally cloning `mrope_positions` at function entry, let the pruning path update the request tensor directly, return it unchanged for the empty/no-remaining-media fast paths, and have the caller skip the follow-up `req_state.mrope_positions.copy_(new_mrope_positions)` when the returned tensor aliases the original. Keep the existing clone-preserving behavior as the default if needed for external callers. This removes one full `(3, N)` allocation/copy per chunked-prefill recomputation while preserving the current span and suffix update logic. Validation should assert byte-identical `mrope_positions` and `mrope_position_delta` for Qwen2.5-VL and Qwen3-VL, plus an aliasing test that confirms the opt-in path mutates only the provided per-request tensor.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes caching input-dependent prefix scans across chunked-prefill calls; it does not address the unconditional output clone or the full copy back into `req_state.mrope_positions` after each call. This proposal targets tensor ownership and materialization overhead inside each invocation, so it composes with Agent A's cache without duplicating it.

---
