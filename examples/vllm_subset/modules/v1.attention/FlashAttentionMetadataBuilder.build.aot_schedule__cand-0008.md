# FlashAttentionMetadataBuilder.build.aot_schedule

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/backends/flash_attn.py`](vllm/v1/attention/backends/flash_attn.py) (lines 408–555)
- **Symbol:** `FlashAttentionMetadataBuilder.build.aot_schedule`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0008`

## Description
FlashAttention metadata-build region that decides whether to compute FA3 AOT scheduler metadata, which max_num_splits value to pass, how to handle sliding-window AOT support, and how scheduler metadata is copied for full CUDA graphs.

## Current approach
AOT scheduling is disabled for fast_build and VLLM_BATCH_INVARIANT. max_num_splits defaults to 0, switches to self.max_num_splits for full CUDA graph capture when num_actual_tokens fits the capture size, and is forced to 1 under batch invariance. AOT is disabled entirely when multiple sliding-window configs are observed across layers; cascade and DCP branches can invoke schedule() on split subproblems.

## Estimated impact explanation
AOT scheduling and split count affect FlashAttention wall time on decode and mixed prefill/decode batches. The external FA kernel still dominates, so impact is medium, but a better policy can shave TTFT and median TPOT without changing math.

## Evolve rationale
This is in-repo scheduling policy around an external kernel: it chooses when scheduler metadata is paid for, whether split-KV is bounded for CUDA graphs, and whether heterogeneous sliding-window models lose AOT entirely. Correctness oracle: tests/kernels/attention/test_cascade_flash_attn.py, tests/v1/attention/test_attention_backends.py, and CUDA graph tests can compare outputs with and without scheduler metadata.

## Deep research proposals

### 1. Adopt load-balanced AOT scheduling that preserves CUDA Graph compatibility for heterogeneous and dynamic batches
- **Finding:** `find-0004` — *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*
- **Source URL:** <https://openreview.net/forum?id=RXPofAsL8F>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In FlashAttentionMetadataBuilder.build (vllm/v1/attention/backends/flash_attn.py:408-555), replace the current all-or-nothing AOT scheduling policy with a load-balanced scheduler decision modeled after FlashInfer's CUDA-Graph-compatible scheduling. Concretely: (1) when multiple sliding-window configs are observed across layers, instead of disabling AOT entirely, compute per-window-class schedule metadata (or a composed schedule keyed by window) and reuse it across layers sharing the same window — keeping AOT benefits for heterogeneous models; (2) make max_num_splits a function of the request-length distribution (e.g., load-balance splits across SMs based on per-request KV lengths) rather than a binary 0/self.max_num_splits/1 choice, while still snapping to the captured CUDA-graph bucket so graph replay is preserved; (3) gate the schedule() call by an inexpensive cost predictor (token count × variance of seq lens) so that uniform decode batches skip AOT (where it's overhead) while skewed prefill+decode mixes pay for it. The cascade/DCP branches that already split into subproblems should reuse the same load-balanced policy on each split. Validate output equivalence using tests/kernels/attention/test_cascade_flash_attn.py, tests/v1/attention/test_attention_backends.py, and existing CUDA graph tests to ensure scheduler metadata changes never alter math.

**Proposal rationale.**

The candidate's current approach treats AOT scheduling as a coarse on/off policy and falls back to no-AOT whenever sliding-window configs differ across layers — exactly the dynamism case FlashInfer's paper targets. The finding's core contribution is that load-balanced scheduling can adapt to request-level dynamism while remaining CUDA-Graph compatible, which directly addresses two named gaps in this region: (a) heterogeneous sliding windows disabling AOT entirely, and (b) max_num_splits being chosen by capture-size fit rather than by load balance. For the multi-turn agentic workload in the caller context — where decode-heavy batches with variable history lengths are common — a load-balanced split count and per-window AOT can shave TTFT (better prefill scheduler metadata) and median TPOT (better decode split balance) without changing the kernel itself, matching the candidate's own medium-impact framing.

---

### 2. Make max_num_splits adaptive to KV length and effective batch in FA metadata build
- **Finding:** `find-0005` — *Flash-Decoding for long-context inference – PyTorch Blog*
- **Source URL:** <https://pytorch.org/blog/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In FlashAttentionMetadataBuilder.build (vllm/v1/attention/backends/flash_attn.py:408-555), replace the largely static max_num_splits policy (0 by default, self.max_num_splits only on full-CUDA-graph capture when num_actual_tokens fits, and 1 under VLLM_BATCH_INVARIANT) with a workload-aware policy that mirrors Flash-Decoding's split-KV parallelization heuristic. Concretely, when AOT scheduling is enabled (i.e., not fast_build, not batch-invariant, no heterogeneous sliding-window), compute a target split count from the decode-heavy signals already available in the builder: number of decode rows (num_actual_tokens minus prefill tokens), maximum/average context length per request (max_seq_len, seq_lens), and an estimate of SM occupancy (effective_batch * num_kv_heads vs device SM count). When the product of effective decode batch and KV heads is well below SM count and per-request KV length is large, raise max_num_splits toward self.max_num_splits (or a tuned cap) so the FA3 scheduler partitions along the KV sequence dimension and merges partials with log-sum-exp; when the GPU is already saturated by batch*heads, keep splits=1 to avoid the partial-merge overhead. Apply this on both the non-graph path and the existing full-CUDA-graph capture branch (the captured value should still be the upper bound). Leave the cascade and DCP schedule() invocations unchanged in shape but feed them the same adaptive split bound. Keep AOT-disabled fallbacks (heterogeneous sliding window, fast_build, batch-invariant) as today.

**Proposal rationale.**

The candidate's main lever in this region is exactly the parameter Flash-Decoding targets: split-KV parallelism, exposed here as max_num_splits passed into the FA3 AOT scheduler. The current policy treats max_num_splits as a CUDA-graph capacity knob rather than a workload knob, so long-context multi-turn decode batches (the caller's stated workload) under-utilize SMs even though FA3 supports KV-length splits. The finding contributes a concrete, transferable rule - parallelize across KV length when batch*heads is small relative to SMs, then merge with LSE - that maps directly onto choosing max_num_splits in this builder. This addresses median TPOT on long-context decode without changing kernel math, which matches the candidate's medium-impact, correctness-preserving framing (validatable via tests/kernels/attention/test_cascade_flash_attn.py and tests/v1/attention/test_attention_backends.py).

---

## Agent proposals

### 1. Memoize FA3 scheduler metadata across builds via a small shape-keyed LRU to elide redundant schedule() CPU work in steady-state decode
- **Agent:** claude

**Detailed description.**

In FlashAttentionMetadataBuilder.build (vllm/v1/attention/backends/flash_attn.py:408-555), wrap the get_scheduler_metadata / schedule() invocation behind a small process-local LRU cache keyed by a cheap signature of its inputs: (num_actual_tokens, num_query_heads, num_kv_heads, head_dim, page_size, causal flag, sliding-window value, max_num_splits chosen for this build, and a hash of the cu_seqlens_q and seqused_k tensors restricted to their valid prefix). On a hit, reuse a persistent device-side scheduler_metadata buffer instead of paying schedule() CPU cost and the HtoD copy; on a miss, run schedule() as today, copy the result into a slot in a pre-allocated pool, and store a CUDA event / step counter so the slot is only recycled after the previous kernel using it has retired. Keep the existing full-CUDA-graph path correct by only serving cache hits whose captured max_num_splits and batch-shape bucket exactly match the captured graph's bucket; otherwise fall through to the current copy-into-the-graph-buffer path. Apply the same cache to the cascade and DCP sub-schedule() calls, keying additionally on which split (prefix vs suffix, local vs remote KV) is being scheduled. Leave fast_build, VLLM_BATCH_INVARIANT, and heterogeneous-sliding-window fallbacks untouched. Validate that cached metadata produces bit-identical kernel outputs using tests/kernels/attention/test_cascade_flash_attn.py and tests/v1/attention/test_attention_backends.py, and verify CUDA graph replay correctness with the existing graph tests.

**Novelty rationale.**

Both existing deep_research_proposals (find-0004 and find-0005) change the *policy* of AOT scheduling and split-count selection — what max_num_splits to pick, whether to AOT under heterogeneous sliding windows, when to skip AOT for uniform decode. Neither addresses the orthogonal cost of *recomputing* the same scheduler metadata across consecutive builds. In multi-turn agentic decode, where successive steps share batch shape and KV-length structure within a turn, the schedule() CPU pass plus its HtoD copy are repeated work whose result is a deterministic function of inputs. Memoizing that result with a shape-keyed LRU is independent of (and composes with) any chosen splits policy, so it is not subsumed by either listed finding; it targets reduction of per-step build overhead rather than kernel-time balance, complementing rather than duplicating the existing proposals.

---

### 2. Defer FA3 AOT sliding-window initialization until AOT is actually scheduled
- **Agent:** codex

**Detailed description.**

In `FlashAttentionMetadataBuilder.build` (`vllm/v1/attention/backends/flash_attn.py:410-428`), change the lazy sliding-window setup so `self.aot_sliding_window` is not initialized during builds where `aot_schedule` is false, such as `fast_build=True` spec-decode proposer passes or `VLLM_BATCH_INVARIANT`. Concretely, make the block `if aot_schedule and self.aot_sliding_window is None:` and assign the default `(-1, -1)` only inside that branch after reading `_get_sliding_window_configs`; keep the existing heterogeneous-window fallback that disables AOT when more than one config is found. This prevents a first fast build from permanently pinning `self.aot_sliding_window` to no-window and skipping later heterogeneous-window detection, which can make subsequent normal builds pass scheduler metadata computed for the wrong sliding-window policy. Add a regression test that calls `build(..., fast_build=True)` before a normal `build(...)` for a sliding-window config and asserts the later scheduler sees the real window; add a second case for multiple layer window configs to assert AOT is still disabled after the first normal build.

**Novelty rationale.**

The deep-research proposals change scheduling policy: adaptive/load-balanced split selection and preserving AOT across heterogeneous sliding-window classes. This proposal does not change the policy; it fixes the initialization lifecycle so the current policy is applied correctly after an AOT-disabled first build. Agent A's proposal memoizes scheduler metadata and explicitly leaves fast_build and sliding-window fallbacks untouched, so it also does not cover this sentinel-state issue.

---
