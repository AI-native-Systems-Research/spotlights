# GPUModelRunner._compute_prev_positions/_prepare_input_ids

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 1598–1740)
- **Symbol:** `GPUModelRunner._compute_prev_positions/_prepare_input_ids`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0002`

## Description
Builds the current-to-previous batch row mapping and uses it to populate input_ids in async scheduling, copying prior sampled tokens and prior draft tokens from cached GPU tensors into the next step input buffer.

## Current approach
A Python loop fills prev_positions from prev_req_id_to_index, then another per-request loop builds Python lists for sampled-token indices, draft-token indices, previous row indices, and flattened positions. The slow path materializes four small pinned CPU tensors and launches separate H2D transfers and scatter_ operations, and draft_token_ids is cast to int32 on each call.

## Estimated impact explanation
When async scheduling is enabled this runs every decode step. Removing per-request Python list construction and several tiny H2D transfers targets median TPOT, especially for small decode batches common in multi-turn agentic traffic.

## Evolve rationale
This is one async-scheduling input-repair hot path. Headroom is in maintaining prev_positions incrementally during state updates, producing scatter indices in one reusable pinned buffer, coalescing H2D copies, fusing sampled/draft scatters, and caching the int32 draft-token view. Correctness oracle: for unchanged, reordered, newly admitted, and spec-decode batches, compare input_ids and is_token_ids against the current implementation and verify downstream logits/sampling parity in tests/v1/worker/test_gpu_model_runner.py and async-scheduling e2e tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Vectorize prev/draft index construction and fuse into one H2D copy + one scatter
- **Agent:** claude

**Detailed description.**

Rewrite the per-request Python loop in `GPUModelRunner._prepare_input_ids` (vllm/v1/worker/gpu_model_runner.py:1651-1740) so that all four index arrays — `sample_flattened_indices`, `spec_flattened_indices`, `prev_indices`, and `prev_draft_token_indices` — are produced with NumPy on top of the already-numpy `prev_positions[:num_reqs]` and `cu_num_tokens`, then concatenated into a single pinned-CPU staging tensor that is uploaded with one `non_blocking=True` H2D copy and consumed by a single fused `scatter_`. Concretely: (1) `valid_mask = prev_positions >= 0`, `prev_indices_np = prev_positions[valid_mask]`, and `flattened_last = cu_num_tokens[valid_mask] - 1`; (2) build `draft_lens_np` as an `np.int32` array indexed from `scheduler_output.scheduled_spec_decode_tokens` for the masked req_ids (single Python comprehension, no per-step list-extending); (3) `sample_flattened_indices_np = flattened_last - draft_lens_np`; (4) reuse the existing `_get_cumsum_and_arange` helper (vllm/v1/worker/gpu_model_runner.py:1572-1596) to compute the batched arange and add `np.repeat(sample_flattened_indices_np + 1, draft_lens_np)` to get `spec_flattened_indices_np`, and similarly `np.repeat(prev_indices_np * self.num_spec_tokens, draft_lens_np) + arange` for `prev_draft_token_indices_np`; (5) compute `common_indices_match` and `max_flattened_index` with `np.array_equal` / `np.max` over `prev_indices_np` and `flattened_last` instead of per-iteration scalar tracking — keeping the existing fast path on line 1693 intact; (6) pack the four index arrays plus their offsets into one contiguous int64 pinned buffer (pre-allocated as a `self._prep_input_ids_index_buf` that is grown on demand), do one `.to(self.device, non_blocking=True)`, and slice into per-array GPU views; (7) fuse the two `scatter_` calls (lines 1712-1718 and 1736-1740) — both target `self.input_ids.gpu` along dim 0 — into one `scatter_` whose index is `torch.cat([sampled_idx, spec_idx])` and whose src is `torch.cat([prev_sampled_token_ids[prev_idx, 0], draft_token_ids_int32.flatten()[prev_draft_idx]])`; (8) cache the int32 cast of `self._draft_token_ids` (line 1734) alongside it when it is set after sampling, so the per-step `.to(dtype=torch.int32)` becomes a no-op view lookup. Also vectorize `_compute_prev_positions` (lines 1598-1611) using a precomputed numpy array for `prev_req_id_to_index` values keyed by an iteration over `req_ids[:num_reqs]` once rather than the python `dict.get` loop, when profitable. Correctness oracle: parity tests in tests/v1/worker/test_gpu_model_runner.py covering unchanged batches, reordered batches, newly admitted requests, and spec-decode batches; assert that `input_ids.gpu[:total_num_scheduled_tokens]` and `is_token_ids` match the existing implementation byte-for-byte; plus async-scheduling end-to-end TPOT measurements on a multi-turn agentic decode workload.

**Novelty rationale.**

The candidate has no existing deep_research_proposals to overlap with. While the candidate's evolve_rationale lists the general headroom (incremental prev_positions, reusable pinned buffer, coalesced H2D, fused scatters, cached int32 view), it does not specify a concrete implementation. This proposal pins down (a) reuse of the already-present `_get_cumsum_and_arange` helper to build `spec_flattened_indices` and `prev_draft_token_indices` via `np.repeat` + batched arange instead of Python `range(...).extend`, (b) a single packed-offset pinned buffer with index views rather than four separate `torch.tensor(..., pin_memory=True)` allocations, and (c) a single fused `scatter_` over concatenated index/src tensors targeting `self.input_ids.gpu` — none of which are spelled out in the rationale, and (a)+(c) in particular are non-obvious because they depend on noticing that both scatters share dim 0 and that the existing helper already does exactly the batched-arange shape needed.

---

### 2. Hoist the unchanged no-spec decode fast path before index repair
- **Agent:** codex

**Detailed description.**

Add an explicit identity-batch fast path for the common async decode case in `GPUModelRunner._compute_prev_positions` / `_prepare_input_ids` before any sampled/draft index arrays are built. When `prev_sampled_token_ids` is cached, also store a lightweight previous-batch order stamp on `InputBatch` (for example an `order_version` incremented by add/remove/condense/swap/reorder operations, plus the count of valid previous sampled rows). In `_compute_prev_positions`, if the previous order stamp still matches the current batch and `len(prev_req_id_to_index) == num_reqs`, fill `self.prev_positions.np[:num_reqs]` from `self.arange_np[:num_reqs]` and set a private identity flag instead of doing per-request dict lookups. Then, at the top of the async branch in `_prepare_input_ids`, if that identity flag is set, `scheduled_spec_decode_tokens` is empty, and `total_num_scheduled_tokens == num_reqs`, directly copy `self.input_batch.prev_sampled_token_ids[:num_reqs, 0]` into `self.input_ids.gpu[:num_reqs]`, set `is_token_ids.gpu[:num_reqs] = True` when prompt embeds are enabled, and return. Keep the existing path for reordered batches, newly admitted requests, invalid/discarded rows, prompt chunks, and spec decode. Add parity coverage in `tests/v1/worker/test_gpu_model_runner.py` for unchanged async decode taking the new path, plus reordered/new/spec cases proving fallback behavior remains byte-for-byte identical.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Claude's proposal optimizes the generic slow path by vectorizing index construction, packing H2D index copies, fusing scatters, and caching the draft-token int32 view; it explicitly keeps the existing fast path after index analysis. This proposal targets a different optimization: prove the stable no-spec decode case from batch-order metadata and return before constructing any repair indices or walking requests, eliminating both `_compute_prev_positions` dict lookups and `_prepare_input_ids` per-request repair bookkeeping for the median steady-state agentic decode path.

---
