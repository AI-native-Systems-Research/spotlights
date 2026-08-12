# rejection_sample

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`](vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py) (lines 922–1188)
- **Symbol:** `rejection_sample`
- **Kind:** function
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0029`

## Description
New GPU runner speculative-decoding rejection sampler pipeline that computes target/draft statistics, verifies draft tokens, resamples rejected/bonus tokens, and inserts outputs.

## Current approach
Allocates several per-call temporary tensors, uses fixed VOCAB_BLOCK_SIZE=8192 and RESAMPLE_BLOCK_SIZE=1024, and launches separate kernels for local logits stats, optional block-verification residuals, rejection, resampling, and insertion.

## Estimated impact explanation
Spec decode is a core TPOT lever for agent serving; reducing temporary traffic, launch count, or poorly matched vocab-block tiling improves median TPOT for speculative workloads.

## Evolve rationale
The VOCAB_BLOCK_SIZE and RESAMPLE_BLOCK_SIZE constants plus the multi-kernel rejection_sample launch sequence are owned kernel-fusion and tiling targets. tests/v1/spec_decode/test_rejection_sampler_utils.py, tests/v1/spec_decode/test_synthetic_rejection_sampler_utils.py, tests/v1/worker/test_gpu_rejection_sampler_chunking.py, and spec-decode e2e tests validate accepted-token and resampling semantics.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Persist and autotune rejection-sampler scratch tensors + block sizes across calls
- **Agent:** claude

**Detailed description.**

In `rejection_sample` (vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:922-1188), every invocation allocates six-to-eight fresh scratch tensors — `target_local_argmax`, `target_local_max`, `target_local_sumexp`, `draft_local_max`, `draft_local_sumexp`, optional `cumulative_log_p` and `local_residual_mass`, plus `sampled`, `num_sampled`, `target_rejected_logsumexp`, `draft_rejected_logsumexp`, `resampled_local_argmax`, and `resampled_local_max` — all with shapes derived from `num_logits`, `vocab_num_blocks`, and `resample_num_blocks`. On multi-turn agentic serving, this function runs once per decode step, and CUDACachingAllocator churn on these small [num_logits, vocab_num_blocks] tensors shows up as per-step allocator overhead (cudaMalloc/free bookkeeping, stream sync, and cache fragmentation) that inflates TPOT. Introduce a module-level `_RejectionSamplerScratch` cache keyed on `(device, dtype, has_draft_logits, use_block_verification, use_fp64)` that lazily allocates scratch buffers sized to `max(num_logits, vocab_num_blocks)` seen so far, grows in geometric steps (e.g., 1.5x with a small headroom), and returns narrowed views (`.narrow(0, 0, num_logits)`) for the current call. Because the buffers are overwritten in every kernel before being read, they can be safely reused without zeroing. Additionally, replace the two hard-coded constants `VOCAB_BLOCK_SIZE=8192` and `RESAMPLE_BLOCK_SIZE=1024` with a small env-driven autotune table (`VLLM_SPEC_DECODE_VOCAB_BLOCK` / `_RESAMPLE_BLOCK`) that picks tile sizes from `{4096, 8192, 16384}` and `{512, 1024, 2048}` based on `(vocab_size, num_logits, num_reqs)` bucket via a one-time timing sweep cached in the same scratch object — Llama-family vocabs (~128k) prefer larger vocab blocks (fewer `_compute_local_logits_stats_kernel` grid rows and smaller `target_local_*` scratch), while smaller draft vocabs benefit from 4096. The combined change reduces per-decode-step allocator traffic to zero for the steady state and improves kernel occupancy for the two grid launches that scale in `vocab_num_blocks`. Validate with `tests/v1/spec_decode/test_rejection_sampler_utils.py` and `tests/v1/worker/test_gpu_rejection_sampler_chunking.py` for correctness, and with an end-to-end spec-decode agentic benchmark (multi-turn, medium context) measuring median TPOT before/after.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so this idea cannot overlap with them. The proposal is also narrowly scoped to two mechanisms — (a) a persistent, geometrically-grown scratch-tensor cache to eliminate the ~11 per-call `new_empty` allocations, and (b) workload-aware autotuning of `VOCAB_BLOCK_SIZE`/`RESAMPLE_BLOCK_SIZE` — rather than the more obvious kernel-fusion angle called out in `evolve_rationale`, which a research proposal is more likely to pursue. It targets allocator/launch overhead that specifically dominates in short-decode steady state (agentic multi-turn), where each rejection_sample call handles only a handful of logits and per-call allocator cost is a larger fraction of runtime than the kernel math itself.

---

### 2. Add a greedy-only rejection sampler fast path
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:922-1188`, add an early specialized path for the common serving case where every active request has `temperature == 0.0` and `synthetic_conditional_rates is None`. The current generic path still allocates softmax/logsumexp scratch, launches `_compute_local_logits_stats_kernel`, `_rejection_kernel`, `_resample_kernel`, and `_insert_resampled_kernel`, and the resample launch fans out over `(num_reqs, resample_num_blocks)` even though greedy non-bonus rejections already have the target argmax. Implement a `_greedy_target_argmax_kernel` that computes target argmax local reductions for all `num_logits`, including bonus-token rows that the existing stats kernel explicitly skips, and a `_greedy_rejection_and_bonus_kernel` that loops per request, compares each draft token against the reduced target argmax, writes either accepted draft tokens or the first target argmax, writes the bonus argmax when all drafts are accepted, and stores `num_sampled = accepted_length + 1`. This path can skip draft max/sumexp, target sumexp, rejected logsumexp, residual mass, resample scratch, `_resample_kernel`, and `_insert_resampled_kernel` entirely. Gate it narrowly so stochastic, synthetic-acceptance, and residual/block-verification sampling semantics continue through the existing implementation. Validate with the existing rejection sampler tests plus a new mixed-batch test where all temperatures are zero and some requests reject early while others accept all drafts and need the bonus token.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposed persistent scratch allocation and block-size autotuning while preserving the generic multi-kernel algorithm. This proposal is a separate semantic specialization for all-greedy batches: it removes probability-stat and resampling work by changing the control flow for a narrower workload case, rather than caching the existing temporaries or tuning existing tile constants.

---
