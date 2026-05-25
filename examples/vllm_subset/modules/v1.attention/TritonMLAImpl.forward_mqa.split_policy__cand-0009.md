# TritonMLAImpl.forward_mqa.split_policy

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/backends/mla/triton_mla.py`](vllm/v1/attention/backends/mla/triton_mla.py) (lines 147–203)
- **Symbol:** `TritonMLAImpl.forward_mqa.split_policy`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0009`

## Description
Triton MLA decode split-count policy and per-call intermediate allocation before decode_attention_fwd.

## Current approach
Batch-invariant mode forces num_kv_splits = 1. Otherwise min_work_per_split = 512, ideal_splits = next_power_of_2(max_seq_len // 512), and the result is capped by self._sm_count * occupancy_multiplier with occupancy_multiplier = 2. attn_logits is allocated with torch.empty on every forward call.

## Estimated impact explanation
MLA decode is a dominant TPOT path for DeepSeek-style models on long agentic contexts. A better split policy improves occupancy without excessive reduction work, and reusing attn_logits removes per-step allocation overhead, directly improving median TPOT.

## Evolve rationale
num_kv_splits controls split-KV parallelism and the stage1/stage2 intermediate size for the Triton MLA decode kernel. min_work_per_split, next-power-of-two rounding, occupancy_multiplier, and the per-call torch.empty are explicit hot-path choices. Correctness oracle: tests/kernels/attention/test_triton_decode_attention.py compares decode_attention_fwd against a reference across split counts, and tests/v1/attention/test_mla_backends.py exercises TRITON_MLA end to end.

## Deep research proposals

### 1. Replace max_seq_len heuristic with load-balanced num_kv_splits policy for Triton MLA decode
- **Finding:** `find-0004` — *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*
- **Source URL:** <https://openreview.net/forum?id=RXPofAsL8F>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/mla/triton_mla.py:147-203 (TritonMLAImpl.forward_mqa.split_policy), replace the current max_seq_len-driven heuristic (min_work_per_split=512, next_power_of_2(max_seq_len // 512), capped by sm_count * occupancy_multiplier=2) with a load-balanced split-count policy that takes the actual per-request seq_len distribution into account. Concretely: (1) compute total decode work as sum(seq_lens) rather than using only max_seq_len so a few long sequences in an agentic batch do not over-split the short ones; (2) choose num_kv_splits so that average per-split work is >= min_work_per_split AND the largest sequence still saturates the SMs, which is the load-balancing condition the FlashInfer paper describes; (3) keep the next-power-of-2 rounding only when required by the kernel, otherwise use the exact value to avoid the 2x-padded reduction tail; (4) preserve CUDA Graph / batch-invariant compatibility by deriving num_kv_splits purely from shapes already captured in the attention metadata (so it is constant per captured graph) and keeping the num_kv_splits=1 path for batch_invariant mode untouched; (5) hoist the attn_logits torch.empty allocation out of the hot path by caching a buffer sized to (max_batch, num_heads, max_num_kv_splits, head_dim+1) on the impl instance and resizing/re-viewing it instead of reallocating every forward.

**Proposal rationale.**

The candidate's gap is exactly what FlashInfer's load-balanced scheduling targets: the current policy bases the split count on max_seq_len, which on a multi-turn agentic batch with skewed seq_lens either over-splits (huge stage-2 reduction) or under-utilizes SMs. The finding's concrete contribution -- treating load-balanced scheduling as a first-class backend policy that stays CUDA-Graph-compatible -- maps directly onto picking num_kv_splits from the seq_len distribution while keeping the value constant per captured graph. The buffer-reuse half of the proposal is consistent with the finding's framing of metadata/kernel efficiency for dynamic batches and removes the per-step torch.empty highlighted in the candidate description, both of which target the median TPOT objective stated in the caller context.

---

### 2. Adopt Flash-Decoding-style adaptive KV-length split policy and reuse attn_logits buffer in Triton MLA decode
- **Finding:** `find-0005` — *Flash-Decoding for long-context inference – PyTorch Blog*
- **Source URL:** <https://pytorch.org/blog/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/mla/triton_mla.py at TritonMLAImpl.forward_mqa.split_policy (lines 147-203), replace the current static heuristic (min_work_per_split=512, ideal_splits = next_power_of_2(max_seq_len // 512), capped by self._sm_count * occupancy_multiplier=2) with an adaptive split-count policy modeled on Flash-Decoding. Concretely: derive num_kv_splits as a function of effective parallel work outside the KV dimension (batch * num_heads) versus available SM capacity (self._sm_count * occupancy_multiplier), so that splits grow when batch*heads alone under-occupies the GPU (typical of long-context, small-batch decode) and shrink when batch*heads already saturates SMs. Bound splits both above (so stage-2 LSE merge cost does not dominate, e.g. ceil(max_seq_len / min_work_per_split)) and below (1, preserving the batch-invariant short-circuit). In addition, hoist the per-call torch.empty for attn_logits out of the hot path: cache the stage-1 logits/LSE buffer on the impl as an attribute keyed by (num_kv_splits, max_batch, num_heads, head_dim_v + 1) and reallocate only on shape growth, mirroring how Flash-Decoding's two-pass design implies a reusable partial-output workspace. Validate with tests/kernels/attention/test_triton_decode_attention.py (split-count sweep against the reference) and tests/v1/attention/test_mla_backends.py (TRITON_MLA end-to-end), and benchmark median TPOT on long multi-turn agentic traces.

**Proposal rationale.**

The candidate's current split policy decides num_kv_splits purely from max_seq_len (next_power_of_2(max_seq_len // 512)) and a fixed sm_count*2 cap, ignoring how much parallel work already exists in the batch*num_heads dimension. Flash-Decoding's central observation is exactly that, in decode, batch*heads is frequently too small to saturate SMs on long contexts, and that adding a KV-length parallelization dimension with an LSE-based merge fills that gap — which is the dominant regime for the stated multi-turn agentic, long-context TPOT objective. This makes the finding a concrete, transferable algorithmic prescription for how the split count should depend on (batch*heads, seq_len, sm_count), rather than a restatement of the existing heuristic. Coupling that policy change with caching the stage-1 attn_logits workspace removes the per-forward torch.empty that the candidate explicitly flags, directly improving median TPOT without changing the kernel contract that decode_attention_fwd already validates against the reference.

---

### 3. Adopt FlashDecoding++ shape/hardware-adaptive heuristic for num_kv_splits in Triton MLA decode
- **Finding:** `find-0008` — *FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics*
- **Source URL:** <https://proceedings.mlsys.org/paper_files/paper/2024/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/mla/triton_mla.py around lines 147-203 (TritonMLAImpl.forward_mqa split-count policy), replace the current fixed heuristic (min_work_per_split=512, next_power_of_2(max_seq_len // 512), capped by self._sm_count * occupancy_multiplier with occupancy_multiplier=2) with a hardware- and shape-adaptive heuristic in the spirit of FlashDecoding++. Concretely: (1) compute the candidate split count from both the per-split work target and an SM-occupancy target derived from the actual decode batch size B and head group, choosing num_kv_splits so that B * num_kv_splits is close to (but not far above) self._sm_count * occupancy_multiplier, since when B already saturates the SMs further splitting only adds reduction overhead; (2) tune occupancy_multiplier and min_work_per_split per device class (e.g., differentiate H100/H200/MI300 via torch.cuda.get_device_capability or a small lookup) instead of using a single constant of 2 and 512; (3) drop the next_power_of_2 rounding when the resulting split count is already bounded by the SM cap, since power-of-two rounding can either under- or over-split for non-power-of-two sequence lengths typical in agentic workloads. In addition, hoist the torch.empty for attn_logits out of the per-call hot path: cache a buffer on self keyed by (max_num_reqs, num_heads, max_num_kv_splits, head_dim_v) and grow it monotonically, mirroring FlashDecoding++'s emphasis on removing synchronization/allocation overhead in the decode loop. Validate with tests/kernels/attention/test_triton_decode_attention.py and tests/v1/attention/test_mla_backends.py to confirm numerical parity across split counts.

**Proposal rationale.**

The candidate's current policy uses two coarse constants (min_work_per_split=512, occupancy_multiplier=2) plus next_power_of_2 rounding, which is exactly the kind of shape-specific underutilization FlashDecoding++ targets via its hardware-adaptive dataflow heuristics. The paper explicitly motivates avoiding shape-mismatched splits and synchronization costs in decode, which maps directly onto num_kv_splits selection for the Triton MLA decode kernel and the per-call attn_logits allocation. The finding does not require kernel rewrites to apply here: the transferable idea is the heuristic discipline (consider B, SM count, and device class jointly, and remove avoidable per-step overhead), which is implementable inside the lines 147-203 region without touching decode_attention_fwd, addressing a concrete gap (one-size-fits-all constants) on a hot TPOT path for multi-turn agentic workloads.

---

## Agent proposals

### 1. Make Triton MLA decode num_kv_splits page-aligned to the paged-KV PAGE_SIZE
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/mla/triton_mla.py:147-203 (TritonMLAImpl.forward_mqa split policy), the current heuristic chooses num_kv_splits from max_seq_len // min_work_per_split (with min_work_per_split=512 tokens), rounds up via triton.next_power_of_2, and caps by self._sm_count * 2. The KV cache, however, is paged: PAGE_SIZE = kv_c_and_k_pe_cache.size(1) at line 184 (typically 16/32/64), and decode_attention_fwd walks the cache one block_table entry at a time. The split policy is completely unaware of this granularity, so split boundaries routinely fall mid-page, which (a) forces splits to fetch but mask partial pages at their endpoints, and (b) produces uneven per-split token counts that imbalance the stage-2 LSE merge.

Concrete change, scoped strictly to lines 147-165 in this file:
  1. Read PAGE_SIZE before computing num_kv_splits (move the line-184 read up; it depends only on kv_c_and_k_pe_cache, already in scope).
  2. Replace the token-based min_work_per_split=512 with a page-based min_pages_per_split (e.g., 8 for PAGE_SIZE=64 / 16 for PAGE_SIZE=32, calibrated so min_pages_per_split * PAGE_SIZE ≈ 512 to preserve current per-split work intent).
  3. Compute pages_per_seq = cdiv(attn_metadata.max_seq_len, PAGE_SIZE), and ideal_splits = max(1, pages_per_seq // min_pages_per_split).
  4. Replace next_power_of_2 rounding with the largest divisor of pages_per_seq that is <= min(ideal_splits, sm_count * occupancy_multiplier); when no clean divisor exists, pick num_kv_splits so pages_per_split = cdiv(pages_per_seq, num_kv_splits) is uniform across all but at most one split (i.e., minimize the tail-split shortfall in pages, not in tokens).
  5. Keep the VLLM_BATCH_INVARIANT short-circuit (num_kv_splits=1) unchanged, and keep sm_count * occupancy_multiplier as the upper bound.

The net effect is that each split processes a whole number of KV pages, eliminating mid-page boundary work inside decode_attention_fwd and giving the stage-2 reducer uniform partial sums to merge. This is a strictly local edit; the kernel contract is unchanged, so tests/kernels/attention/test_triton_decode_attention.py and tests/v1/attention/test_mla_backends.py (TRITON_MLA) remain valid oracles.

**Novelty rationale.**

find-0004 reformulates the heuristic around sum(seq_lens) for load balancing across requests; find-0005 derives splits from batch*num_heads vs SM occupancy in the Flash-Decoding spirit; find-0008 proposes per-device-class tuning of min_work_per_split and occupancy_multiplier. All three operate purely in the (seq_len, batch, heads, sm_count) space and explicitly retain or replace the next_power_of_2 rounding, but none of them references PAGE_SIZE, the block_table, or the paged-KV layout. Page alignment is an orthogonal axis: it targets intra-split boundary work and stage-2 reduction balance that persist no matter which seq-len-based heuristic is chosen, and it composes cleanly with any of the three proposed seq_len/SM heuristics rather than competing with them. The buffer-reuse half of the existing proposals is intentionally not duplicated here.

---

### 2. Add a direct-output fast path when num_kv_splits is 1
- **Agent:** codex

**Detailed description.**

In vllm/v1/attention/backends/mla/triton_mla.py:147-203, branch after computing num_kv_splits. When num_kv_splits == 1, skip creating attn_logits and call a single-split decode wrapper that writes the stage-1 accumulator and LSE directly to o and lse. The wrapper can be implemented by adding a DIRECT_OUTPUT constexpr to the Triton decode stage-1 kernels in vllm/v1/attention/ops/triton_decode_attention.py, including the grouped MLA path, so the existing loop body is reused but its final store goes to o/lse instead of Mid_O and _decode_softmax_reducev_fwd is not launched. Keep the existing attn_logits allocation and two-stage decode_attention_fwd path for num_kv_splits > 1. This accelerates VLLM_BATCH_INVARIANT, short-context decode where the heuristic already selects one split, and any future policy that intentionally declines splitting. Validate with tests/kernels/attention/test_triton_decode_attention.py using num_kv_splits=1 against the reference and tests/v1/attention/test_mla_backends.py with VLLM_BATCH_INVARIANT enabled.

**Novelty rationale.**

find-0004/find-0005/find-0008 propose better split-count selection and reusable attn_logits storage; they still assume the two-stage split-KV contract and only reduce allocator overhead. Agent A's proposal changes split boundaries to respect PAGE_SIZE. This proposal is not another heuristic or buffer cache: it removes the intermediate tensor and the stage-2 reduction launch entirely in the degenerate one-split case, which those proposals preserve as a path but do not optimize.

---
