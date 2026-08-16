# FlashInferMetadataBuilder fixed split policy

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flashinfer.py`](vllm/v1/attention/backends/flashinfer.py) (lines 645–652)
- **Symbol:** `FlashInferMetadataBuilder fixed split policy`
- **Kind:** config_block
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0019`

## Description
Sets FlashInfer fixed split sizes and split-KV disabling policy under batch-invariant mode.

## Current approach
When VLLM_BATCH_INVARIANT is enabled, decode_fixed_split_size is hardcoded to 2048, prefill_fixed_split_size to 4096, and disable_split_kv to True; otherwise both split sizes are -1 and split-KV is enabled.

## Estimated impact explanation
These split settings affect FlashInfer planning and kernel partitioning for both TTFT and TPOT. Better values by model, page size, and GPU architecture can improve median latency while preserving outputs.

## Evolve rationale
Concrete tunables are the 2048 and 4096 fixed_split_size constants and disable_split_kv boolean passed later into FlashInfer plan calls. Correctness oracle is output equality for decode and prefill across split settings plus FlashInfer backend regression tests.

## Deep research proposals

### 1. Replace fixed split constants with occupancy-aware split policy for batch-invariant FlashInfer decode/prefill
- **Finding:** `find-vllm_v1_attention-0005` — *Multi-Head, Multi-Query, and Group-Query Attention*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flashinfer.py at lines 645-652, replace the hardcoded decode_fixed_split_size=2048, prefill_fixed_split_size=4096, and disable_split_kv=True constants used under VLLM_BATCH_INVARIANT with an occupancy-aware policy inspired by TRT-LLM's multi-block generation-phase heuristic. Specifically, compute split sizes (and whether to keep split-KV enabled) from the actual planning inputs already available to FlashInferMetadataBuilder: batch size, number of KV heads, GPU SM count, and KV page size. When the batch*heads product is large enough to saturate SMs, prefer larger split sizes (or leave split_size=-1 and split-KV enabled) so decode does not pay the multi-block reduction overhead; when batch*heads is small relative to SMs (typical single-request or small-batch multi-turn agentic decode), pick a smaller decode_fixed_split_size that yields enough KV chunks to keep SMs busy, and keep disable_split_kv=True only when required for batch-invariance. Keep prefill sizing separate but use the same occupancy signal for prefill_fixed_split_size. Preserve exact output equality under VLLM_BATCH_INVARIANT by making the policy a pure function of batch/model/hardware (not per-step KV length), so plan inputs remain deterministic across identical requests. Correctness is checked with the existing FlashInfer backend regression tests plus decode/prefill output-equality against the current 2048/4096 constants; performance is checked with vllm bench on the multi-turn agentic workload.

**Proposal rationale.**

The candidate's current approach uses two hardcoded constants (2048 for decode, 4096 for prefill) chosen without reference to batch size, head count, or SM count, and unconditionally disables split-KV under batch-invariant mode. TRT-LLM's documented generation-phase behavior explicitly says the multi-block/split path should only be preferred when one-block-per-head under-occupies the SMs, and gates it on batch size, head count, SM count via an internal heuristic. That is exactly the missing signal here: for the caller's multi-turn agentic workload, decode batches are often small, so a fixed 2048 split can either over-split (adding reduction overhead) or under-split (leaving SMs idle) depending on head count and GPU. Turning the two constants into an occupancy-aware policy is a transferable, concrete idea that directly targets the candidate's stated tunables (decode_fixed_split_size, prefill_fixed_split_size, disable_split_kv) and its stated impact goals (median TTFT and TPOT), while preserving the batch-invariance oracle because the policy inputs (batch, heads, SMs, page size) are stable across identical requests.

---

## Agent proposals

### 1. Keep FlashInfer split-KV enabled under batch-invariant mode by pinning a page-size-aligned fixed split factor
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/flashinfer.py at lines 645-652, replace the current `disable_split_kv=True` branch under `VLLM_BATCH_INVARIANT` with a policy that keeps split-KV enabled while still preserving determinism. The current code disables split-KV entirely in batch-invariant mode, which sacrifices FlashInfer's multi-block reduction path exactly where the caller's multi-turn agentic workload needs it most (small decode batches where one-block-per-head under-occupies SMs). The determinism-preserving replacement is: (1) always set `self.disable_split_kv = False` under `VLLM_BATCH_INVARIANT`, because a positive `decode_fixed_split_size` / `prefill_fixed_split_size` already forces FlashInfer to compute the number of KV chunks as `ceil(kv_len / fixed_split_size)` deterministically — the non-determinism that split-KV normally introduces comes from the runtime-chosen split factor, not from the reduction itself, and pinning the split size removes that source. (2) Choose the fixed sizes as page-size-aligned integers based on `self.kv_cache_spec.block_size` (already available at __init__), e.g. `decode_fixed_split_size = block_size * decode_pages_per_split` where `decode_pages_per_split` is a small power of two (e.g. 128 for block_size=16 → 2048) so the last split of any KV length is a whole number of pages and reduction work per split is uniform. (3) Fall back to `disable_split_kv = True` only when the FlashInfer plan path genuinely rejects a positive fixed size (e.g. certain KV-quant configurations already gated further down in this file). Preserve exact output equality by making all three tunables pure functions of `block_size`, model config, and hardware — none of them may depend on per-request KV length or per-step scheduling state.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_attention-0005) proposes an occupancy-aware policy that computes split sizes and `disable_split_kv` from batch × heads × SM × page_size, and explicitly retains `disable_split_kv=True` 'only when required for batch-invariance'. It treats the split-KV disable flag as coupled to invariance. This proposal is materially different: its core claim is that split-KV can remain *enabled* under `VLLM_BATCH_INVARIANT` because a positive fixed_split_size already makes FlashInfer's chunk count deterministic (`ceil(kv_len/fixed_split_size)`), so the invariance guarantee does not require disabling the reduction path at all. It also adds a distinct page-size-alignment rule (fixed_split_size = block_size × pow2 pages_per_split) to keep reduction work per split uniform, which the existing proposal does not discuss — that proposal drives sizes from occupancy, not from KV page geometry.

---

### 2. Short-circuit fixed splitting when configured context bounds cannot exceed one chunk
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/backends/flashinfer.py` around lines 645-652, replace the unconditional batch-invariant `decode_fixed_split_size=2048` and `prefill_fixed_split_size=4096` assignments with a bounded policy that first derives the maximum effective KV length the backend can see: `effective_max_kv_len = min(model_config.max_model_len, window_left)` when `window_left > 0`, otherwise `model_config.max_model_len`. If `effective_max_kv_len <= 2048` for decode or `<= 4096` for prefill, set that phase's `fixed_split_size` to `-1` because FlashInfer can never form more than one KV split for that phase under the configured model/window anyway. Keep the existing fixed values only when the configured context/window can actually produce multiple chunks. This avoids paying fixed-split planning/reduction bookkeeping for short-context or sliding-window models common in multi-turn agentic serving, while preserving batch invariance because the branch depends only on static model configuration, not per-request sequence lengths or scheduler state. Add a focused regression that constructs `FlashInferMetadataBuilder` for batch-invariant mode with `max_model_len` below and above the thresholds, including a positive `window_left`, and asserts the phase split sizes are disabled only when multi-split execution is impossible.

**Novelty rationale.**

The deep research proposal changes the split policy based on occupancy signals such as batch size, KV heads, SM count, and page size, and Agent A focuses on keeping split-KV enabled with page-size-aligned fixed split factors. This proposal is different: it adds a static reachability guard based on configured maximum effective KV length, including sliding-window attention, so vLLM does not request a fixed split path when the model configuration proves no second split can ever exist. It neither introduces an occupancy heuristic nor argues for enabling split-KV under batch-invariant mode.

---
