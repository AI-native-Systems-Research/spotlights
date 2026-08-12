# TritonAttentionMetadataBuilder 2D/3D tuning constants

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/triton_attn.py`](vllm/v1/attention/backends/triton_attn.py) (lines 54–175)
- **Symbol:** `TritonAttentionMetadataBuilder 2D/3D tuning constants`
- **Kind:** config_block
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0009`

## Description
Defines the TritonAttention 2D/3D switch threshold and allocates segmented softmax buffers for the 3D decode path.

## Current approach
MIN_LAUNCH_GRID_SIZE_2D=128 and NUM_PAR_SOFTMAX_SEGMENTS=16. seq_threshold_3D is 128 // num_heads_kv, optionally snapped to the nearest CUDA graph capture size, and buffers are allocated using that threshold and segment count.

## Estimated impact explanation
Small-batch decode, common in agentic serving, is where the 2D/3D switch matters. Tuning the threshold and segment count can improve SM utilization and median TPOT.

## Evolve rationale
Concrete tunables are MIN_LAUNCH_GRID_SIZE_2D, NUM_PAR_SOFTMAX_SEGMENTS, and the CUDA-graph snapping policy. Correctness oracle is equality of Triton attention outputs because these values only change launch partitioning and intermediate buffer shape.

## Deep research proposals

### 1. Adaptive Flash-Decoding split count for the Triton 3D decode path
- **Finding:** `find-vllm_v1_attention-0002` — *Flash-Decoding for Long-Context Inference*
- **Source URL:** <https://princeton-nlp.github.io/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the fixed NUM_PAR_SOFTMAX_SEGMENTS = 16 in vllm/v1/attention/backends/triton_attn.py (lines 54-56, applied in TritonAttentionMetadataBuilder.__init__ at lines 154-175) with a workload-aware split count derived from (batch_size <= seq_threshold_3D, max_seq_len, num_heads_q, num_heads_kv, and the number of SMs on the current device). Concretely: (1) compute a target number of KV splits so that the 3D launch grid (batch * num_heads_q * splits) roughly matches or modestly oversubscribes the SM count, following the Flash-Decoding recipe of parallelizing over the KV sequence-length dimension; (2) clamp the split count to a power-of-two range (e.g. 4-64) and to a value that keeps per-split KV tile size above a minimum (so reduction overhead does not dominate short contexts); (3) allocate softmax_segm_output/max/expsum using this adaptive split count rather than the fixed 16, sized for the CUDA-graph-captured maximum so buffers remain graph-compatible. Also revisit the seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv rule (line 139) so the 2D->3D crossover accounts for KV length, not just batch size: when max_seq_len is large enough that the KV-parallel 3D path can saturate SMs even at moderate batch, prefer 3D. Because these values only change launch partitioning and intermediate buffer shape, the correctness oracle (equality of Triton attention outputs) is preserved.

**Proposal rationale.**

The candidate's 3D decode path already implements the structural pattern Flash-Decoding advocates - parallelize over KV sequence length and merge partial (max, expsum, output) via a log-sum-exp reduction - but it hard-codes the parallelization width at NUM_PAR_SOFTMAX_SEGMENTS = 16 and gates entry to the 3D path solely on batch size (MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv). Flash-Decoding's core contribution is precisely that this split count must scale with context length and available parallelism to be effective on small-batch, long-context decode - the exact regime called out in the caller context (multi-turn agentic workload, minimize TPOT). Making the split count and the 2D/3D switch adaptive addresses a concrete constraint in the current approach (fixed tunables that under-parallelize long contexts and over-parallelize short ones) and is the direct application of the finding's transferable idea.

---

### 2. Replace fixed 2D/3D switch threshold with occupancy-aware heuristic incorporating SM count
- **Finding:** `find-vllm_v1_attention-0005` — *Multi-Head, Multi-Query, and Group-Query Attention*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/triton_attn.py (lines 54-175), replace the fixed constants MIN_LAUNCH_GRID_SIZE_2D=128 and NUM_PAR_SOFTMAX_SEGMENTS=16, and the derived seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv, with an occupancy-aware policy modeled on TensorRT-LLM's multi-block generation-phase heuristic. Concretely: (1) derive the launch-grid target from the device's SM count (via current_platform / torch.cuda.get_device_properties().multi_processor_count) rather than a hardcoded 128, so seq_threshold_3D becomes roughly ceil(alpha * num_SMs / num_heads_kv) with a tunable alpha targeting reasonable occupancy on the specific GPU; (2) derive NUM_PAR_SOFTMAX_SEGMENTS as a function of expected per-head work (max_seq_len, num_heads_kv) and SM count, so the 3D split-KV parallelism scales with how much room there is on the device rather than being pinned at 16; (3) keep the CUDA-graph snapping step, but snap the SM-derived threshold to the nearest capture size instead of a constant-derived one. The softmax_segm_output/max/expsum buffers continue to be allocated using the resulting (seq_threshold_3D, num_heads_q, num_par_softmax_segments) shape. Output correctness is preserved because these changes only affect launch partitioning and intermediate buffer shape, matching the candidate's stated correctness oracle.

**Proposal rationale.**

The candidate explicitly identifies MIN_LAUNCH_GRID_SIZE_2D, NUM_PAR_SOFTMAX_SEGMENTS, and the CUDA-graph snapping policy as the tunables, and calls out small-batch decode as the regime where the 2D/3D switch matters most for TPOT in agentic serving. The finding contributes exactly the missing ingredient: an occupancy-aware policy — using batch size, head count, and SM count — for deciding when to enable a multi-block (3D split-KV) generation kernel rather than the single-block (2D) one. Replacing the hardcoded 128 with an SM-count-derived threshold addresses the gap that the current constant was tuned for one class of GPU; on devices with more SMs the 2D path is chosen too aggressively at small batch, and on devices with fewer SMs the 3D path allocates more segment buffers than useful. This is a concrete, transferable idea that directly targets small-batch decode TPOT for the multi-turn agentic workload named in the caller context.

---

## Agent proposals

### 1. Per-CUDA-graph-capture-size 2D/3D dispatch table replacing single scalar seq_threshold_3D
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/triton_attn.py, replace the single scalar `self.seq_threshold_3D` (line 139) and its downstream snapping (lines 149-152) with a per-capture-size 2D-vs-3D dispatch table computed once at builder construction. Concretely: (1) When `self.decode_cudagraph_enabled`, iterate over `capture_sizes` and, for each captured batch size `bs`, precompute a boolean `use_3D[bs]` = `bs * num_heads_kv < MIN_LAUNCH_GRID_SIZE_2D` (i.e., the same predicate that today's threshold encodes, but evaluated against each capture size directly rather than against a single snapped scalar). Store this as a small dict/list on the builder and, in `build()` (lines 195-268), attach the corresponding boolean/threshold for the current `num_reqs` to `TritonAttentionMetadata` so the kernel dispatch picks 2D vs 3D per-graph rather than via a single global comparison. Keep a non-CUDA-graph fallback that uses the current formula for eager execution. (2) Size the `softmax_segm_output/max/expsum` buffers (lines 156-175) to the largest capture size for which `use_3D[bs]` is true — not to the snapped scalar — so buffers are exactly as small as the 3D path actually needs, and no capacity is wasted on capture sizes that will only ever run the 2D path. If no capture size takes the 3D path (e.g., very high num_heads_kv or small capture set), skip allocation entirely and mark the 3D buffers as unused. (3) Fix the current snap-to-nearest pathology at lines 149-152: `min(..., key=lambda x: abs(x - self.seq_threshold_3D))` can round the threshold *up* to a capture size larger than the intended crossover, causing captured graphs at batch sizes above the original 128 // num_heads_kv boundary to be forced onto the 3D path (with its extra segmented-softmax reduction cost) when 2D would have won. The per-capture dispatch table eliminates this by never quantizing the boundary. Correctness is preserved: the 2D and 3D kernels compute the same attention outputs and the per-capture predicate is a strict per-graph refinement of the current global comparison; only the launch partitioning and buffer shape change, matching the candidate's stated correctness oracle.

**Novelty rationale.**

The two listed deep_research_proposals both keep the design of a single scalar `seq_threshold_3D` computed at builder construction (finding-0002 makes it depend on max_seq_len/SM count and rescales NUM_PAR_SOFTMAX_SEGMENTS; finding-0005 replaces the 128 constant with an SM-count-derived value and keeps the CUDA-graph snap step). Neither proposes a *per-capture-size* dispatch table, neither addresses the snap-to-nearest asymmetry (rounding up to a larger capture size than the true crossover), and neither observes that segment buffers can be sized to the largest 3D-taking capture size rather than to the (possibly upward-snapped) threshold. This proposal is orthogonal to both: it can compose with either occupancy-aware or Flash-Decoding-inspired threshold formulas, but the mechanism it changes — turning a single scalar comparison into a per-graph precomputed choice — is new and specifically targets the CUDA-graph interaction that both existing proposals leave intact.

---

### 2. Add measured autotuning for Triton 2D/3D decode cutover
- **Agent:** codex

**Detailed description.**

Add an optional autotuning path around `TritonAttentionMetadataBuilder` in `vllm/v1/attention/backends/triton_attn.py` that benchmarks the actual Triton 2D and 3D decode launches for this model/device instead of relying only on static constants. At builder construction, when enabled by a config/env flag, run a bounded warmup over representative decode-only shapes derived from `cudagraph_capture_sizes`, `num_heads_q`, `num_heads_kv`, `headdim`, `block_size`, and a small set of KV lengths. For each capture batch size, time the current 2D path versus candidate 3D configurations using a small grid of `num_par_softmax_segments` values, verify outputs against the existing equality oracle, and cache the winning dispatch decision plus segment count under a key containing device name/SM version, model head geometry, dtype, block size, and vLLM version. Use the cached result to set the builder’s threshold/segment metadata and allocate the segmented-softmax buffers only for the selected maximum 3D shape. Provide a conservative fallback to today’s constants when autotuning is disabled, fails, or runs under unsupported conditions.

**Novelty rationale.**

The existing deep research proposals replace the constants with analytical heuristics based on SM count, KV length, and workload shape, while Agent A proposes a per-CUDA-graph dispatch table to avoid threshold snapping errors. This proposal is different: it chooses the 2D/3D cutover and segment count from measured kernel latency on the actual deployment hardware and caches that decision. It can incorporate Agent A’s per-capture representation or the research heuristics as defaults/search seeds, but the core change is empirical autotuning rather than a new fixed or formula-derived policy.

---
