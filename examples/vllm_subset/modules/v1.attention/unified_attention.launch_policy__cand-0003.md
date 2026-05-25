# unified_attention.launch_policy

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/ops/triton_unified_attention.py`](vllm/v1/attention/ops/triton_unified_attention.py) (lines 578–748)
- **Symbol:** `unified_attention.launch_policy`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0003`

## Description
Launch-policy region in unified_attention that chooses BLOCK_M, BLOCK_Q, q-block overlaunch, the 2D-vs-3D split path, the kernel grid, TILE_SIZE, and the reduce_segments launch.

## Current approach
BLOCK_M is 16 unless num_queries_per_kv exceeds it; BLOCK_Q is BLOCK_M // num_queries_per_kv. total_num_q_blocks uses the upper bound q.shape[0] // BLOCK_Q + num_seqs. 3D mode is a binary predicate over buffer availability, max_seqlen_q, num_seqs, and batch invariance. The grid is (q_blocks, num_kv_heads) for 2D and (q_blocks, num_kv_heads, num_par_softmax_segments) for 3D; reduce_segments runs only for 3D.

## Estimated impact explanation
Multi-turn agentic batches often combine one-token decodes with short extends. Reducing overlaunched blocks and choosing 2D vs 3D from actual shape and device occupancy improves SM utilization, directly moving median TPOT.

## Evolve rationale
These launch parameters set CTA count, per-CTA work, and whether an extra reduction kernel is paid. They are owned in-repo tunables: BLOCK_M, BLOCK_Q, q-block overlaunch, the 2D/3D predicate, grid shape, and segment count. Correctness oracle: tests/kernels/attention/test_triton_unified_attention.py covers 2D and 3D outputs against a reference, and tests/v1/attention/test_attention_backends.py exercises backend equivalence.

## Deep research proposals

### 1. Replace static launch heuristics with load-balanced scheduling for 2D/3D split
- **Finding:** `find-0004` — *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*
- **Source URL:** <https://openreview.net/forum?id=RXPofAsL8F>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/ops/triton_unified_attention.py at lines 578-748 (unified_attention.launch_policy), replace the current static rules — BLOCK_M=16 unless num_queries_per_kv exceeds it, BLOCK_Q=BLOCK_M//num_queries_per_kv, total_num_q_blocks=q.shape[0]//BLOCK_Q + num_seqs (an upper-bound overlaunch), and a binary 2D-vs-3D predicate keyed on buffer availability, max_seqlen_q, num_seqs, and batch invariance — with a load-balanced scheduling policy that takes per-request shape (per-sequence q_len and kv_len), num_kv_heads, and target SM occupancy as inputs. Concretely: compute a true (not upper-bounded) q-block count per sequence using its actual q_len and BLOCK_Q, then choose num_par_softmax_segments and the 2D-vs-3D path so the resulting CTA count is at least a multiple of device SM count and per-CTA KV work is roughly balanced across CTAs. Keep all control values either CUDA-Graph-static (fixed BLOCK_M, BLOCK_Q, max segment count per shape bucket) or sourced from the existing seq_lens/cu_seqlens metadata tensors that are already captured into the graph, so the new policy preserves CUDA Graph compatibility. Bucket per-call shape signatures (e.g., by max_seqlen_q range and num_seqs range) and cache the chosen (BLOCK_M, BLOCK_Q, mode, segments, grid) so repeated agentic batches hit the same precompiled kernel.

**Proposal rationale.**

The candidate is exactly a launch-time scheduling decision (CTA count, per-CTA work, and whether to pay for an extra reduce_segments kernel), and its current rules are coarse: BLOCK_M is essentially fixed at 16, q-block count is an overestimate, and the 2D/3D switch is a binary predicate that does not look at how unevenly q-blocks fall across sequences. Multi-turn agentic batches mix one-token decodes with short extends, which is exactly the dynamism FlashInfer's load-balanced scheduler targets. The finding's central contribution — treating load-balanced scheduling as first-class backend policy while keeping CUDA-Graph compatibility — directly addresses the candidate's gap of static heuristics that overlaunch on near-uniform decode batches and underuse SMs on skewed extend batches, both of which lift median TPOT. The supporting quote that the scheduler 'adjusts to dynamism of user requests while maintaining compatibility with CUDAGraph' shows the technique is compatible with vLLM's existing graph-capture path, which is the main constraint on changes in this region.

---

### 2. Make 2D-vs-3D split and segment count occupancy-driven via Flash-Decoding heuristic
- **Finding:** `find-0005` — *Flash-Decoding for long-context inference – PyTorch Blog*
- **Source URL:** <https://pytorch.org/blog/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/ops/triton_unified_attention.py around lines 578-748, replace the current binary 2D-vs-3D predicate (buffer availability, max_seqlen_q, num_seqs, batch invariance) with an occupancy-aware decision modeled on Flash-Decoding. Concretely: (1) compute an estimated active-CTA count for the 2D grid (q_blocks * num_kv_heads) and compare it against device SM count times a target waves-per-SM constant; (2) when the 2D grid leaves SMs under-occupied — the multi-turn agentic case where num_seqs and max_seqlen_q are both small but per-sequence KV length is long — switch to the 3D split-KV path and pick num_par_softmax_segments so that q_blocks * num_kv_heads * num_par_softmax_segments fills the SMs without massive over-subscription, with TILE_SIZE adjusted so each segment still contains enough KV work to amortize the reduce_segments launch; (3) keep the existing buffer-availability and batch-invariance gates as hard constraints, but make max_seqlen_q and num_seqs inputs to the segment-count formula rather than thresholds in a binary predicate. Validate against tests/kernels/attention/test_triton_unified_attention.py (2D and 3D outputs vs reference) and tests/v1/attention/test_attention_backends.py.

**Proposal rationale.**

Flash-Decoding's central idea — add a KV-sequence-length parallelization dimension and merge with log-sum-exp when small effective batch under-occupies the GPU — is exactly the regime the 3D path in this candidate already implements, but the candidate selects it via a coarse binary predicate rather than from actual occupancy. For multi-turn agentic workloads (small num_seqs, mixed one-token decodes and short extends, long KV histories) the 2D grid frequently leaves SMs idle while the current predicate may still pick 2D, or picks 3D with a segment count untuned to device occupancy. Treating the 2D-vs-3D choice and num_par_softmax_segments as a function of (q_blocks, num_kv_heads, KV length, SM count) — the Flash-Decoding framing — directly targets the candidate's evolve_rationale of choosing 2D vs 3D from actual shape and device occupancy, and addresses median TPOT, which is dominated by exactly this decode regime.

---

### 3. Adopt FlashDecoding++ heuristics for 2D/3D split and segment count in unified_attention launch policy
- **Finding:** `find-0008` — *FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics*
- **Source URL:** <https://proceedings.mlsys.org/paper_files/paper/2024/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the current binary 2D-vs-3D predicate and the fixed derivation of num_par_softmax_segments in vllm/v1/attention/ops/triton_unified_attention.py:578-748 with a hardware-adaptive heuristic inspired by FlashDecoding++. Concretely: (1) compute a target CTA count from device SM count and per-CTA occupancy estimates and pick num_par_softmax_segments so that q_blocks * num_kv_heads * segments saturates SMs without over-splitting; (2) make the 2D->3D switch a function of measured occupancy of the 2D launch (q_blocks * num_kv_heads vs SM count weighted by KV length per segment) instead of only buffer availability and max_seqlen_q; (3) when emitting 3D + reduce_segments, exploit the 'unified max' formulation so per-segment partial softmax outputs share a precomputed max and reduce_segments does a synchronization-light merge, mirroring FlashDecoding++'s asynchronous partial-softmax merging. Keep BLOCK_M / BLOCK_Q derivation but allow them to vary with the chosen segment count when num_queries_per_kv is small, and gate the new path behind the existing batch-invariance flag. Validate with tests/kernels/attention/test_triton_unified_attention.py (2D and 3D paths) and tests/v1/attention/test_attention_backends.py for backend equivalence.

**Proposal rationale.**

The candidate's launch policy currently chooses 2D vs 3D and segment count from a coarse predicate that ignores actual SM occupancy, which is exactly the underutilization regime FlashDecoding++ targets with its hardware-adaptive dataflow heuristics. Multi-turn agentic batches mix short decodes and short extends, producing many shapes where a fixed predicate either leaves SMs idle (2D too few CTAs) or pays an unnecessary reduce_segments kernel (3D over-split). Importing FlashDecoding++'s occupancy-driven split decision and unified-max partial-softmax merge gives a concrete, transferable mechanism to right-size the launch and to make the 3D reduction cheaper, directly addressing the candidate's stated estimated_impact on median TPOT.

---

## Agent proposals

### 1. Closed-loop online autotuner for unified_attention launch configs per shape bucket
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/ops/triton_unified_attention.py at lines 578-748 (unified_attention.launch_policy), replace the static analytical derivation of (BLOCK_M, BLOCK_Q, 2D-vs-3D mode, num_par_softmax_segments, TILE_SIZE, q-block overlaunch factor) with a closed-loop online autotuner that observes actual kernel latency on the deployed device. Concretely: (1) at backend init, enumerate a small fixed set of candidate configs per shape-bucket, where the bucket key is (num_seqs decile, max_seqlen_q decile, max_kv_len decile, num_kv_heads, num_queries_per_kv); (2) maintain an in-process table mapping bucket_id -> list of (config, EMA latency, sample count, last-used step); (3) on each call hash the current shape, then either pick the current best-EMA config (greedy) or, with an exploration probability that decays with total sample count for the bucket, sample an under-explored candidate using UCB1; (4) instrument each launch with a pair of CUDA events captured around the unified_attention kernel (and reduce_segments when 3D), read the elapsed time after a downstream sync point that already exists in the worker loop, and update the chosen entry's EMA; (5) freeze the choice once the best config's EMA gap to the second-best exceeds a noise-threshold for N consecutive samples, after which the bucket short-circuits to the frozen config with zero exploration overhead. Preserve CUDA-Graph compatibility by either (a) pre-capturing each candidate config in a bucket as its own captured graph at warmup and dispatching among them via an index tensor, or (b) running the autotuner only during eager execution and committing the converged choice as the captured-graph config on first capture for that bucket. Persist the converged table to disk keyed by (gpu_uuid, model_id, dtype, head_dim, sliding_window, softcap) so subsequent runs on the same hardware skip the warm-up phase.

**Novelty rationale.**

All three existing deep_research_proposals (find-0004 load-balanced scheduling a la FlashInfer, find-0005 Flash-Decoding occupancy targeting, find-0008 FlashDecoding++ heuristics) replace the current rules with a different *static analytical* heuristic derived from published mechanisms: each computes the launch parameters open-loop from shape inputs and SM count. None of them measure actual kernel latency on the deployed device or adapt the choice from feedback. find-0004 mentions caching chosen configs per shape bucket, but the cached value is still derived from its analytical formula with no measurement loop. This proposal is mechanistically distinct: it is a closed-loop bandit-style learner over the same tunable space, capturing GPU-generation-specific, dtype-specific, head_dim-specific, and even firmware-specific effects (L2 residency, memory-bandwidth saturation thresholds, warp scheduler quirks) that no static heuristic can encode. It also composes with any of the three: any of those heuristics can supply the candidate-config seed set, with the online tuner picking among them (and minor perturbations) using observed latency rather than predicted occupancy. The CUDA-Graph-compatible dispatch via pre-captured per-config graphs and the cross-run on-disk persistence keyed by GPU UUID are also not raised in any existing proposal.

---

### 2. Use decode tile sizing for 2D decode launches
- **Agent:** codex

**Detailed description.**

In vllm/v1/attention/ops/triton_unified_attention.py:604-659, decouple TILE_SIZE selection from the 2D-vs-3D mode. Today every 2D launch uses TILE_SIZE_PREFILL, so large decode batches that stay on the 2D path because num_seqs > seq_threshold_3D use the prefill tile size even when max_seqlen_q == 1. Add a decode-only 2D branch such as: keep TILE_SIZE_PREFILL for true prefill/extend, but when max_seqlen_q == 1 and use_3d is false, launch the same 2D grid with TILE_SIZE_DECODE. This keeps BLOCK_M/BLOCK_Q, q-block indexing, segment buffers, and CUDA Graph shape unchanged, while reducing the per-CTA score tile footprint for the common median-TPOT path. Validate by extending tests/kernels/attention/test_triton_unified_attention.py with a decode case that forces 2D, for example num_seqs above the 3D threshold or seq_threshold_3D=0, and compare against the existing reference.

**Novelty rationale.**

The listed deep_research_proposals focus on load-balanced q-block scheduling, occupancy-driven 2D-vs-3D selection, segment-count selection, and cheaper 3D reduction. They do not call out that the current 2D decode path unconditionally inherits the prefill TILE_SIZE. Claude's proposal is an online autotuner over many launch configs; this is a narrow deterministic code change that fixes a specific existing heuristic mismatch without adding measurement, persistence, new graph variants, or a new scheduling policy.

---
