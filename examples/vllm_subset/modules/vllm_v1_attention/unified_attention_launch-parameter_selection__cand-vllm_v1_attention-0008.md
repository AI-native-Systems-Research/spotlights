# unified_attention launch-parameter selection

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/ops/triton_unified_attention.py`](vllm/v1/attention/ops/triton_unified_attention.py) (lines 784–1189)
- **Symbol:** `unified_attention launch-parameter selection`
- **Kind:** region
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_attention-0008`

## Description
Selects tile sizes, BLOCK_M/BLOCK_Q, tensor-descriptor gates, prefill/decode launch shape, special large-head tuning, and 2D versus 3D unified-attention launch mode.

## Current approach
Uses chained heuristics: _get_tile_size returns 32 for Gemma3 or prefill, 16 for bf16/fp16 decode, and 32 for fp8 decode; BLOCK_M is 16 or next_power_of_2(num_queries_per_kv); an SM100 head_size==256 branch sets BLOCK_M=32, TILE_SIZE_PREFILL=128, 8 warps, and 2 stages; use_3d is disabled for prefills, large batches, batch invariance, or missing segment buffers.

## Estimated impact explanation
These launch choices directly control occupancy, split-K work, and memory traffic. Better per-shape settings move TTFT for prefills and median TPOT for repeated decode in multi-turn agentic serving.

## Evolve rationale
Tunable constructs are _get_tile_size, BLOCK_M, BLOCK_Q, TILE_SIZE_PREFILL, TILE_SIZE_DECODE, launch_num_warps, launch_num_stages, use_td_qo, use_3d, and NUM_SEGMENTS_PER_SEQ. Correctness oracle is invariant attention output for any legal launch configuration, verified by existing Triton attention correctness tests.

## Deep research proposals

### 1. Adaptive Flash-Decoding split-KV: choose NUM_SEGMENTS_PER_SEQ and 3D gate from seqlen and occupancy
- **Finding:** `find-vllm_v1_attention-0002` — *Flash-Decoding for Long-Context Inference*
- **Source URL:** <https://princeton-nlp.github.io/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/ops/triton_unified_attention.py at the launch-parameter selection block (lines ~1036-1082), replace the current 3D-launch gate and fixed segment count with a Flash-Decoding-style adaptive split-KV policy for the decode path. Concretely: (1) When max_seqlen_q == 1 and 3D buffers are available, compute an adaptive split degree segs = clamp(ceil(max_seqlen_k / TARGET_KV_CHUNK), 1, num_par_softmax_segments) so long contexts fan out more KV work while short contexts do not pay the reduction overhead; select TARGET_KV_CHUNK from TILE_SIZE_DECODE and block_size (e.g. a multiple of block_size that keeps each segment >= a few tiles). (2) Also gate on effective occupancy: only enable 3D when num_seqs * num_kv_heads * segs would meaningfully exceed available SMs (i.e., when the 2D grid (total_num_q_blocks, num_kv_heads) is too small to fill the device). This turns the current coarse `num_seqs > seq_threshold_3D` threshold into a joint (batch, context, device) decision that matches Flash-Decoding's target regime: small batch, long context. (3) Pass the adaptive `segs` into both the kernel launch grid dimension and reduce_segments (via NUM_SEGMENTS_PER_SEQ and num_segments) so the log-sum-exp merge reduction pass processes exactly the segments produced. Preserve current fallbacks (skip when segm buffers are None, when there is any prefill row, when batch invariance is on, or when max_seqlen_q > 1) so correctness/invariance guarantees are unchanged. The existing kernel already implements the partial-attention + LSE-merge reduction; only the launch-parameter selection changes.

**Proposal rationale.**

The candidate already contains the machinery Flash-Decoding requires — 3D launch with per-segment partial output/max/expsum and a reduce_segments merge pass — but its gate and segment count are static. Flash-Decoding explicitly motivates adding the KV-sequence-length parallelization dimension precisely for the small-batch, long-context decode regime, which is exactly the multi-turn agentic TPOT workload the caller targets. Making NUM_SEGMENTS_PER_SEQ a function of max_seqlen_k and available parallelism (rather than a fixed configured value) fills SMs when batch is small and long contexts dominate, while the num_seqs threshold alone under-uses split-KV for long single-request decodes and can over-split short-context large-batch decodes. The change is confined to the launch-parameter selection region (candidate lines 784-1189) and preserves the existing correctness oracle: for any legal (segs, gate) choice the 3D partial+reduce path yields the same attention output as the 2D path, and this is covered by the existing Triton attention correctness tests.

---

### 2. Gate 3D split-KV decode via an occupancy-aware policy over SMs, batch, and KV heads
- **Finding:** `find-vllm_v1_attention-0005` — *Multi-Head, Multi-Query, and Group-Query Attention*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the fixed `use_3d` and `NUM_SEGMENTS_PER_SEQ` gating in vllm/v1/attention/ops/triton_unified_attention.py:1041-1088 with an occupancy-aware heuristic modeled after TensorRT-LLM's generation-phase multi-block policy. Concretely: (1) compute an estimated `base_occupancy = num_seqs * num_kv_heads` (the natural 2D grid) and compare it to the device's SM count (via `current_platform.get_device_properties().multi_processor_count` or an equivalent already exposed by vLLM); (2) enable `use_3d` only for decode batches (`max_seqlen_q == 1`) where `base_occupancy < sm_count * target_waves` — i.e., a single non-split launch would leave SMs idle; (3) when 3D is enabled, derive `NUM_SEGMENTS_PER_SEQ` at call time from `ceil(sm_count * target_waves / base_occupancy)` clamped by `max_seqlen_k / TILE_SIZE_DECODE` so segments are never smaller than one KV tile, instead of relying on the externally-supplied `num_par_softmax_segments`; (4) retain the existing hard-disable conditions (prefill batches, batch invariance, missing segm buffers) as fail-closed guards. Keep the `seq_threshold_3D` path as a fallback override but stop using it as the primary switch. The correctness oracle is unchanged (existing Triton attention correctness tests) since only launch shape and segment count change, not kernel semantics — the `reduce_segments` pass at line 1169 already handles arbitrary `NUM_SEGMENTS_PER_SEQ`.

**Proposal rationale.**

The candidate's `use_3d` / `NUM_SEGMENTS_PER_SEQ` machinery is functionally the same optimization TRT-LLM describes as the 'multi-block version of the GPU kernel' guided by an 'internal heuristic' over batch size, head count, and SM count. vLLM's current gate uses only a fixed `seq_threshold_3D` and buffer-presence checks — it does not consider SM count or KV-head count, so it under-splits on decode-heavy multi-turn agentic workloads with small batches (few sequences × few KV heads leaves many SMs idle) and over-splits when the natural grid already saturates the device. Multi-turn agentic serving is exactly the low-occupancy decode regime this heuristic targets: median TPOT is dominated by decode latency, and split-KV parallelism reclaims SM occupancy when `num_seqs * num_kv_heads < sm_count`. Making the split decision occupancy-aware directly addresses the estimated_impact goal of reducing median TPOT without introducing a new correctness path.

---

## Agent proposals

### 1. Extend 3D split-KV to long chunked prefills to attack TTFT, not just TPOT
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/ops/triton_unified_attention.py at the launch-parameter selection block (lines ~1036-1082), relax the `max_seqlen_q > 1` hard-disable in the `use_3d` gate so chunked-prefill launches can also use the 3D split-KV path when the natural 2D grid under-fills the device and per-program KV work is long. Concretely: (1) Replace `max_seqlen_q > 1` in the disqualifier list with a finer test — `is_prefill_shaped = max_seqlen_q > 1`, `base_grid = total_num_q_blocks * num_kv_heads`, `avg_kv_per_prog = max_seqlen_k / max(1, TILE_SIZE_PREFILL)` — and allow 3D on prefill only when `base_grid < sm_count * target_waves` AND `avg_kv_per_prog >= min_kv_tiles_for_split` (e.g. 8) AND every entry in `seqused_k` for prefill rows shares a common bound so segments cover a well-defined KV extent. Practically the cleanest realization: for chunked-prefill batches where the query rows are small (few BLOCK_Q programs) but each program iterates over a long KV history (very common in the first-turn prefill of long multi-turn agentic sessions), the current 2D grid is (few_q_blocks * num_kv_heads) and can leave the device under-loaded — 3D splits the KV dimension across a `NUM_SEGMENTS_PER_SEQ` axis and finishes with the existing `reduce_segments` LSE-merge pass. (2) Use `tile_size = TILE_SIZE_PREFILL` (not TILE_SIZE_DECODE) for the prefill-3D branch so BLOCK_M/BLOCK_Q/tile choices continue to match the prefill regime; only the KV axis is split. (3) Compute `segs_prefill = clamp(ceil(sm_count * target_waves / base_grid), 1, floor(max_seqlen_k / (TILE_SIZE_PREFILL * min_kv_tiles_for_split)))` and pass it as `num_segments`. (4) Preserve all remaining hard-disables (missing segm buffers, batch invariance, per-seq causal masks that would make partial-attention LSE merge unsafe — verify the existing `reduce_segments` kernel handles the per-token causal offsets correctly for max_seqlen_q > 1; if it does not, fall back to 2D as today). The correctness oracle is unchanged (existing Triton attention correctness tests exercise both 2D and 3D paths); this change only widens the regime in which 3D is chosen. Fail-closed: if `reduce_segments` proves not to handle mixed-length prefill rows correctly (its current call site assumes decode-shaped BLOCK_Q loads at line 1170), keep the guard and instead special-case `is_prefill_shaped and num_seqs == 1` (single-request long prefill), which is the highest-value slice for multi-turn agentic TTFT and is the simplest to verify — the reduce kernel then only merges segments for one sequence's query rows.

**Novelty rationale.**

Both listed deep_research_proposals (find-vllm_v1_attention-0002 and find-vllm_v1_attention-0005) modify the decode-only 3D split-KV path and explicitly preserve `max_seqlen_q > 1` as a hard disable, so they target TPOT only. The caller objective explicitly names *median TTFT* alongside TPOT, and the workload hint (multi-turn agentic serving) makes first-turn long-prefill TTFT a first-class metric. This proposal extends split-KV to a regime the existing proposals explicitly exclude — the prefill/chunked-prefill path — using the same 3D machinery (segm buffers + reduce_segments) that is already in-tree. Its gate/segment policy is orthogonal to and composable with either find-0002 (adaptive segment count) or find-0005 (SM-aware gating), because those policies only fire when max_seqlen_q == 1 while this one only fires when max_seqlen_q > 1.

---

### 2. Add per-shape autotuned launch-parameter buckets for 2D attention
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/ops/triton_unified_attention.py` around the launch-parameter selection block, replace the fixed 2D-path choices for `BLOCK_M`, `BLOCK_Q`, `TILE_SIZE_PREFILL`, `TILE_SIZE_DECODE`, `launch_num_warps`, and `launch_num_stages` with a small cached policy keyed by stable shape/backend buckets: `(device capability, head_size, q dtype element size, block_size, num_queries_per_kv, is_prefill, max_seqlen_q bucket, max_seqlen_k bucket, use_td, kv_quant_mode)`. Keep the current heuristics as the default entry, but allow a short candidate set such as `BLOCK_M in {16, 32}`, `TILE_SIZE_PREFILL in {32, 64, 128}` for prefill, `TILE_SIZE_DECODE in {16, 32, 64}` for decode where legal, and `num_warps in {4, 8}`. The implementation can start as an opt-in environment flag that records benchmark-selected choices in an in-process LRU cache after a few warmup launches, then reuses the best configuration for the same bucket. Fail closed to the existing static selection whenever tensor-descriptor constraints, block-size divisibility, compilation budget, or unsupported quantization modes make a candidate illegal. Correctness validation remains the existing Triton attention correctness suite because all candidates only change tiling/launch constexprs, not attention semantics; performance validation should compare median TTFT/TPOT on representative long-prefill and repeated-decode buckets against the current hard-coded policy.

**Novelty rationale.**

The existing deep_research proposals both focus on the 3D split-KV decode gate and adaptive `NUM_SEGMENTS_PER_SEQ`; Agent A focuses on extending that same 3D split-KV machinery to long prefills. This proposal is separate: it targets the ordinary 2D launch-parameter selection that remains active for most prefills, large batches, batch-invariant requests, and any case without segment buffers. It does not change `use_3d` or segment count, and it specifically addresses the other tunables named in the candidate (`BLOCK_M`, `BLOCK_Q`, tile sizes, warps, stages) that the listed proposals leave essentially static.

---
