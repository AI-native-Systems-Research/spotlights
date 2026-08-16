# GPUModelRunner._calc_mrope_positions

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 2795–2842)
- **Symbol:** `GPUModelRunner._calc_mrope_positions`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0008`

## Description
Populates legacy M-RoPE position buffers from per-request precomputed prompt positions and computes continuation positions.

## Current approach
Loops over input_batch.req_ids, performs per-request dict and scheduler lookups, computes prompt lengths, and writes request slices into a shared CPU/NumPy tensor.

## Estimated impact explanation
The impact is concentrated in multimodal agent workloads, where per-turn image/video requests make M-RoPE active; reducing this loop lowers multimodal TPOT and TTFT.

## Evolve rationale
This per-request loop runs every step for M-RoPE multimodal models. Vectorized offset construction or cached prompt lengths can preserve behavior covered by tests/v1/worker/test_mrope_prompt_embeds.py and tests/v1/worker/test_gpu_model_runner_mm_gather.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Vectorize decode-branch M-RoPE writes and drop per-request dict lookups
- **Agent:** claude

**Detailed description.**

In `GPUModelRunner._calc_mrope_positions` (vllm/v1/worker/gpu_model_runner.py:2795-2842), replace the per-request Python loop with a two-phase vectorized fill against `self.mrope_positions.np`.

Phase 1 (setup, no Python-level dict lookups on the hot path): during `input_batch` state maintenance, keep two dense CPU arrays sized to `max_num_reqs`: `mrope_position_delta_cpu[i]` and `mrope_num_prompt_tokens_cpu[i]`, populated when a request is added/updated (mirroring the existing `num_computed_tokens_cpu` pattern). This removes the `self.requests[req_id]` and `scheduler_output.num_scheduled_tokens[req_id]` dict lookups inside `_calc_mrope_positions`, which currently run once per request per step. Reuse `input_batch.num_scheduled_tokens` (or the array produced by `_prepare_inputs`) so `num_scheduled_tokens` is also read from a contiguous array rather than the scheduler dict.

Phase 2 (vectorized fill): compute per-request `prompt_part_len` and `completion_part_len` for the whole batch with NumPy (`np.minimum`/`np.maximum` on `num_computed_tokens_cpu`, `num_scheduled_tokens`, `mrope_num_prompt_tokens_cpu`) and a running `mrope_pos_ptr` cumulative sum. Split the batch into two masks:
  - Prompt-side writes: only requests with `prompt_part_len > 0` still need per-request slice copies from `req.mrope_positions[:, src_start:src_end]` because the source tensor lives on each Request object. For those, iterate only over the (typically small) prefill subset instead of the entire batch. In multi-turn agentic decode-heavy steady state this collapses to zero iterations.
  - Decode-side writes: for every request with `completion_part_len > 0`, replace the current per-request `MRotaryEmbedding.get_next_input_positions_tensor` calls with a single vectorized fill. Build a per-decode-token base array with `np.repeat(mrope_position_delta_cpu + num_computed_tokens_cpu + prompt_part_len, completion_part_len)` and add a per-token offset produced by a fused `np.concatenate([np.arange(n) for n in completion_part_len])` (or use the existing `arange`-style helper vLLM already uses for building position ids in `_prepare_inputs`). Broadcast-assign the resulting 1-D array into `self.mrope_positions.np[:, decode_dst_start:decode_dst_end]` on all three M-RoPE rows at once (they are identical for pure-text continuation tokens, matching `get_next_input_positions_tensor`'s current per-row copy).

The destination offsets (`decode_dst_start`, prompt `dst_start`) are computed from the cumulative sum of `num_scheduled_tokens` masked by phase, so no `mrope_pos_ptr` running Python counter is needed. The output tensor `self.mrope_positions.np` and its dtype are unchanged, preserving the contract exercised by `tests/v1/worker/test_mrope_prompt_embeds.py` and `tests/v1/worker/test_gpu_model_runner_mm_gather.py`.

Apply the same transformation, one for one, to `_calc_xdrope_positions` immediately below (vllm/v1/worker/gpu_model_runner.py:2844+), since it is a structural clone.

Expected win on the caller's multi-turn agentic workload: at steady-state most active requests are in decode with `completion_part_len == 1`, so the current implementation makes O(batch_size) NumPy `arange`+slice-assign calls and O(batch_size) Python-level dict lookups per step; the vectorized version reduces that to O(1) NumPy calls plus a single small prefill loop, cutting M-RoPE bookkeeping overhead from the per-token TPOT critical path.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. Beyond that, the proposal is materially more specific than the candidate's own `evolve_rationale`, which only vaguely suggests 'vectorized offset construction or cached prompt lengths.' This proposal (a) identifies that the decode branch dominates in multi-turn agentic workloads and can be collapsed to a single batched NumPy fill using `np.repeat` + concatenated `np.arange`, (b) proposes concrete new dense CPU-side companion arrays (`mrope_position_delta_cpu`, `mrope_num_prompt_tokens_cpu`) attached to `input_batch` to eliminate dict lookups on `self.requests` and `scheduler_output.num_scheduled_tokens` inside the hot loop, and (c) explicitly extends the transformation to the structurally identical `_calc_xdrope_positions` neighbor.

---

### 2. Add an all-decode GPU fast path for M-RoPE positions
- **Agent:** codex

**Detailed description.**

In `GPUModelRunner._calc_mrope_positions` and its call site around `gpu_model_runner.py:2041`/`2254`, add a guarded fast path for the common steady-state case where every scheduled request is past its prompt (`num_computed_tokens_cpu >= num_prompt_tokens`) and therefore `prompt_part_len == 0` for the whole batch. In that case, skip CPU staging in `self.mrope_positions.cpu` entirely and fill `self.mrope_positions.gpu[:, :total_num_scheduled_tokens]` directly from the already-built GPU token positions plus each request's `mrope_position_delta`. For pure continuation tokens, the three M-RoPE rows are identical, so the target values are `positions[:total_num_scheduled_tokens] + per_token_delta`, broadcast across the three rows. The per-token delta can be gathered by request index from a dense per-batch delta tensor and repeated using the same `req_indices`/`req_indices_gpu` machinery already used to compute regular positions. Keep the existing CPU path for any mixed prefill/decode step, prompt slice copying, async-spec-ddecode drift handling, and XD-RoPE unless a separate equivalent fast path is validated. This removes both the `_calc_mrope_positions` Python work and the host-to-device copy for decode-only M-RoPE steps, which is the likely median TPOT path in multi-turn agentic workloads.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes vectorizing `_calc_mrope_positions` on CPU/NumPy and still writing `self.mrope_positions.np` followed by the existing CPU-to-GPU copy, plus applying the same pattern to XD-RoPE. This proposal is different: it adds a narrower all-decode fast path that bypasses the legacy CPU buffer and H2D copy entirely by materializing M-RoPE continuation positions directly on GPU from the already-computed token positions and per-request deltas.

---
