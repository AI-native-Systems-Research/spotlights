# GPUModelRunner._calc_mrope_positions/_calc_xdrope_positions

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 2500–2594)
- **Symbol:** `GPUModelRunner._calc_mrope_positions/_calc_xdrope_positions`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0004`

## Description
Computes per-token M-RoPE and XD-RoPE positions for multimodal requests by mixing precomputed prompt positions with on-the-fly completion positions in CPU staging buffers.

## Current approach
Two nearly identical Python per-request loops walk input_batch.req_ids, look up request state, compute prompt/completion splits, copy prompt-side slices from per-request tensors, and call MRotaryEmbedding or XDRotaryEmbedding helpers for completion-side positions.

## Estimated impact explanation
The optimization only applies to multimodal rotary models, but those are plausible agentic workloads. Replacing per-request Python slicing with batched tensor/numpy operations reduces per-step CPU input-prep time and therefore median TPOT for those models.

## Evolve rationale
These loops run on every step for multimodal rotary models such as Qwen-VL and Hunyuan-VL. Headroom is in factoring the duplicated M-RoPE/XD-RoPE logic, batching prompt-slice gathers, vectorizing completion-side offsets, and pre-staging per-request position tensors into a contiguous layout. Correctness oracle: element-wise equality of mrope_positions/xdrope_positions and downstream logits parity, covered by mrope kernel/model tests and multimodal v1 model-runner tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Pre-stage M-RoPE/XD-RoPE positions into a persistent batch-slot buffer at request-admit time
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py:2500-2594, replace the per-step prompt-side slice copy by pre-staging each request's full precomputed prompt mrope_positions / xdrope_positions into a persistent per-slot CPU+GPU buffer when the request is first added to input_batch (analogous to how token_ids_cpu and num_computed_tokens are persistently slotted). The slot buffer has shape [3, num_slots, max_model_len] (or a chunked variant) and is written exactly once per request, eliminating the prompt-side memcpy loop on every step. Per step, _calc_mrope_positions/_calc_xdrope_positions then becomes (a) a single vectorized torch.gather / advanced-indexing read of the prompt-side slice for all requests using cu_num_scheduled_tokens and num_computed_tokens_cpu (already numpy arrays on input_batch) into self.mrope_positions.cpu, plus (b) a single batched numpy computation of the completion-side positions across all requests with completion_part_len > 0, parameterized by mrope_position_delta (None for XD-RoPE). The two methods collapse into one shared helper _calc_rope_positions(kind) that selects the slot buffer, the staging buffer, and the completion-position formula, removing the duplicated control flow. For sequences too large to slot fully, fall back to the existing per-request path only on overflow. Verify with the existing mrope kernel/model tests and the multimodal v1 model-runner tests for element-wise equality of mrope_positions/xdrope_positions and downstream logits parity.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's evolve_rationale only lists general headroom directions (factoring duplication, batching gathers, vectorizing offsets, pre-staging tensors) without prescribing how. This proposal commits to a specific architectural change — extending input_batch's persistent-slot pattern to mrope/xdrope positions so the prompt-side path becomes a one-time admit-time write rather than a per-step copy — and unifies the two methods through a single kind-parameterized helper, which is more concrete and structurally different from a pure 'vectorize the loop' rewrite.

---

### 2. Move M-RoPE/XD-RoPE position assembly onto the existing GPU RopeState path
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu_model_runner.py:2500-2594`, replace the CPU staging loops with the existing GPU-side `RopeState` machinery from `vllm/v1/worker/gpu/mm/rope.py`. At request admission, stage each request's precomputed prompt positions and M-RoPE delta into a `RopeState`-style UVA-backed prefill store keyed by the current input-batch slot. After `query_start_loc`, `num_scheduled_tokens`, `num_prompt_tokens`, and `num_computed_tokens` are available on GPU, call a Triton kernel equivalent to `_prepare_rope_positions_kernel` to fill `self.mrope_positions.gpu` or `self.xdrope_positions.gpu` directly for the current scheduled token span. The kernel can choose prompt positions for `orig_pos < num_prompt_tokens` and otherwise compute decode positions as `orig_pos + delta` for M-RoPE or `orig_pos` for XD-RoPE, handling mixed prompt/completion chunks in one pass. This removes the per-step Python request walk, the per-request helper calls, and the CPU-to-GPU copy of the assembled positions; `_get_positions` and downstream model calls can keep consuming the same GPU tensor shape.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes a persistent per-slot CPU+GPU buffer plus vectorized CPU/torch gather and batched numpy completion computation, followed by copying the assembled staging buffer to GPU. This proposal is different: it reuses the repository's existing GPU `RopeState`/Triton design to assemble the final positions directly on device, so the per-step CPU staging path is eliminated rather than optimized.

---
