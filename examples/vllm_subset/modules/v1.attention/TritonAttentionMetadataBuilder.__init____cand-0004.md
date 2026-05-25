# TritonAttentionMetadataBuilder.__init__

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/backends/triton_attn.py`](vllm/v1/attention/backends/triton_attn.py) (lines 48–199)
- **Symbol:** `TritonAttentionMetadataBuilder.__init__`
- **Kind:** config_block
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0004`

## Description
TRITON_ATTN 3D split-path sizing defaults: MIN_LAUNCH_GRID_SIZE_2D, NUM_PAR_SOFTMAX_SEGMENTS, seq_threshold_3D, and softmax segment workspace allocation.

## Current approach
MIN_LAUNCH_GRID_SIZE_2D = 128 and NUM_PAR_SOFTMAX_SEGMENTS = 16 are fixed module constants. seq_threshold_3D is MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv, optionally snapped to the nearest CUDA graph capture size, and segment buffers are allocated as (seq_threshold_3D, num_heads_q, num_par_softmax_segments, headdim_padded).

## Estimated impact explanation
The knobs matter most for small decode batches where 2D launch grids underfill the GPU. Tuning can improve median TPOT on those shapes, but impact is limited to TRITON_ATTN's 3D decode path.

## Evolve rationale
These constants directly determine when unified_attention takes the split-KV 3D path and how much parallel reduction workspace and reduction work it pays. They are policy-defining defaults with no model-aware or device-aware tuning. Correctness oracle: tests/kernels/attention/test_triton_unified_attention.py and tests/v1/attention/test_attention_backends.py verify numerical equivalence; batch-invariant mode must preserve single-segment behavior.

## Deep research proposals

### 1. Make split-KV segment count and threshold adaptive to KV length and SM occupancy
- **Finding:** `find-0005` — *Flash-Decoding for long-context inference – PyTorch Blog*
- **Source URL:** <https://pytorch.org/blog/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/triton_attn.py:48-199 (TritonAttentionMetadataBuilder.__init__), replace the fixed module constants MIN_LAUNCH_GRID_SIZE_2D=128 and NUM_PAR_SOFTMAX_SEGMENTS=16 with values derived from device properties (e.g. SM count) and runtime workload signals, in line with Flash-Decoding's core idea of splitting along the KV sequence dimension only when the 2D launch grid (batch * num_heads_kv) underfills the GPU. Concretely: (1) compute a target launch-grid size from torch.cuda.get_device_properties().multi_processor_count (e.g. 1-2x SM count) instead of the hard-coded 128, so seq_threshold_3D = target_grid // num_heads_kv tracks the actual device; (2) make NUM_PAR_SOFTMAX_SEGMENTS scale with max_seq_len/seq_threshold_3D and remaining SM headroom rather than always being 16, so long contexts with small batches get more KV-length parallelism while short contexts avoid wasted reduction work; (3) keep the existing CUDA-graph capture-size snap and the batch-invariant single-segment fallback so numerical equivalence under tests/kernels/attention/test_triton_unified_attention.py and tests/v1/attention/test_attention_backends.py is preserved. The softmax segment workspace allocation (seq_threshold_3D, num_heads_q, num_par_softmax_segments, headdim_padded) is sized from the same derived values, so peak workspace stays bounded.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out that these knobs are policy-defining defaults with no model-aware or device-aware tuning, and that they govern when unified_attention takes the split-KV 3D path - which is exactly the Flash-Decoding decode-path optimization the finding describes. The finding's transferable insight is that the right amount of KV-length splitting depends on whether the 2D grid underfills the GPU, which happens precisely in the multi-turn agentic workload (long contexts, small effective batch) named in the caller objective for median TPOT. Replacing fixed constants with device- and workload-aware values targets that gap directly without changing the kernel's mathematical behavior.

---

### 2. Replace fixed split-KV thresholds with FlashDecoding++-style hardware/workload-adaptive heuristics
- **Finding:** `find-0008` — *FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics*
- **Source URL:** <https://proceedings.mlsys.org/paper_files/paper/2024/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/triton_attn.py:48-199 (TritonAttentionMetadataBuilder.__init__), replace the hard-coded MIN_LAUNCH_GRID_SIZE_2D = 128 and NUM_PAR_SOFTMAX_SEGMENTS = 16 module constants and the derived seq_threshold_3D with a heuristic-based selector that consults device properties (SM count / wave size) and decode-shape signals (num_heads_kv, num_heads_q, headdim_padded, expected per-request KV length) at builder construction time. Concretely: (1) compute MIN_LAUNCH_GRID_SIZE_2D from torch.cuda.get_device_properties(...).multi_processor_count so the 2D-vs-3D crossover targets a roughly full SM wave on the actual GPU rather than a fixed 128; (2) pick NUM_PAR_SOFTMAX_SEGMENTS from a small lookup keyed on (SM count, num_heads_q, headdim_padded), bounded so the (seq_threshold_3D, num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS, headdim_padded) workspace stays within the current allocation envelope; (3) keep the existing CUDA-graph capture-size snapping for seq_threshold_3D so graph capture remains stable; (4) leave the single-segment path unchanged when batch-invariant mode is requested so the test_triton_unified_attention.py and test_attention_backends.py equivalence oracles keep passing. The rest of the module (kernel call site, segment buffer allocation shape) stays the same — only the policy that picks these four values changes.

**Proposal rationale.**

The candidate explicitly flags that these constants are 'policy-defining defaults with no model-aware or device-aware tuning,' and impact is largest on small decode batches where 2D grids underfill the GPU. FlashDecoding++ contributes exactly the missing ingredient: hardware-adaptive dataflow heuristics for split/segmented decode kernels that pick split factors based on device occupancy and decode shape rather than fixed constants. Translating that idea to this candidate is a constant-policy change (no kernel rewrite) that targets the median TPOT on the multi-turn agentic workload, where decode shapes vary across turns and a single fixed crossover is unlikely to be optimal across devices. The async partial-softmax merging from the same paper would be a separate kernel-level change and is intentionally out of scope here.

---

## Agent proposals

### 1. Empirically autotune per-CUDA-graph (seq_threshold_3D, num_par_softmax_segments) during capture warmup
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/triton_attn.py:48-199 (TritonAttentionMetadataBuilder.__init__ and build_for_cudagraph_capture at lines 201-209), replace the single global pair (seq_threshold_3D, num_par_softmax_segments) with a tiny per-capture-size lookup populated by an empirical micro-benchmark run during the existing CUDA graph capture warmup. Concretely: (1) at __init__, compute a small candidate set for num_par_softmax_segments (e.g. {1, 4, 8, 16, 32}) and for seq_threshold_3D the small set of CUDA graph capture sizes that bracket MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv; (2) allocate softmax_segm_output / softmax_segm_max / softmax_segm_expsum sized to the worst-case (max candidate seq_threshold_3D, max candidate num_par_softmax_segments) so the workspace envelope is fixed and CUDA-graph-friendly, and have the kernel use a runtime-bound slice; (3) before or at the start of CUDA graph capture, for each capture_size in vllm_config.compilation_config.cudagraph_capture_sizes, call unified_attention a handful of times under torch.cuda.Event timing on synthetic inputs of that shape (using the actual KV-cache layout and dtype from kv_cache_spec) to measure each candidate config and record the winner in self._capture_size_to_split_cfg: dict[int, tuple[int, int]]; (4) in build()/build_for_cudagraph_capture(), look up the chosen (seq_threshold_3D, num_par_softmax_segments) by the captured batch size and write them onto TritonAttentionMetadata, keeping the single-segment fallback when batch-invariant mode is requested so the test_triton_unified_attention.py and test_attention_backends.py equivalence oracles still pass; (5) gate the warmup behind an env flag (default on for FULL/FULL_DECODE_ONLY cudagraph modes, off otherwise) and persist the resulting table in an in-process cache keyed on (device_capability, num_heads_q, num_heads_kv, headdim_padded, kv_dtype, block_size) so subsequent process restarts on the same shape do not re-pay warmup. The kernel call site, metadata dataclass shape, and overall workspace allocation envelope are unchanged — only the policy that picks the two integers (and now varies per captured graph size) changes.

**Novelty rationale.**

find-0005 and find-0008 both select the constants analytically at builder __init__ using static signals (SM count, num_heads, headdim, max_seq_len) or a small lookup table, and both produce a single global policy. This proposal is orthogonal in three ways: (a) it is empirical — it actually times unified_attention on candidate configs rather than predicting via an SM/wave model, which captures Triton autotune state, L2 contention with neighboring layers, and driver/Triton-version-specific launch costs that analytic heuristics inherently miss; (b) it is per-CUDA-graph-capture-size rather than global, so different captured decode batch sizes can take different split policies, which directly matches the multi-turn agentic workload where decode shapes vary across turns; (c) it leverages the existing CUDA graph capture warmup window so there is no new steady-state cost, and it is composable with either find-0005 or find-0008 (their analytic estimate can be used as the candidate-set seed). Neither existing proposal mentions empirical timing, per-capture-size policies, or warmup-time autotuning.

---

### 2. Keep raw and graph-snapped 3D thresholds separate
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/backends/triton_attn.py:126-199`, stop overwriting `self.seq_threshold_3D` with the nearest CUDA graph capture size. Instead, compute and store the raw policy threshold from `MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv`, then compute a separate graph-safe threshold used only when building metadata for captured decode graphs. The regular `build()` path should keep the raw threshold so eager fallback and non-captured batch sizes are not pushed onto the 3D split path merely because the nearest capture size rounded upward. `build_for_cudagraph_capture()` can use the capture-aligned value to keep graph path selection stable. Size the segment workspace to the max of the raw and capture-aligned thresholds that can actually be emitted, and preserve the batch-invariant fallback unchanged.

**Novelty rationale.**

The deep research proposals replace the fixed constants with hardware/workload-adaptive or lookup-based policies, but they still treat the resulting threshold as a single global value and explicitly keep the existing CUDA graph snapping behavior. Agent A proposes empirical per-capture-size autotuning during graph warmup. This proposal is narrower and orthogonal: it fixes the current coupling where CUDA graph alignment mutates the eager/default split-path policy, without changing the constants, adding device heuristics, or timing candidate configs.

---
