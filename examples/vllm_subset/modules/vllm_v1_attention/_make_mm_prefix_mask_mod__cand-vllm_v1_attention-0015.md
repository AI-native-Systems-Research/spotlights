# _make_mm_prefix_mask_mod

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flash_attn.py`](vllm/v1/attention/backends/flash_attn.py) (lines 1424–1530)
- **Symbol:** `_make_mm_prefix_mask_mod`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0015`

## Description
Builds the cached CuTE-DSL mask_mod for FA4 multimodal prefill, combining causal or sliding-window masking with mm_prefix ranges.

## Current approach
Caches by sliding-window parameters, defines nested cute.jit helpers, and loads two range entries for each query row before OR-ing the multimodal range with the causal/window mask.

## Estimated impact explanation
For multimodal agent workloads this mask executes inside every FA4 prefill block. Reducing per-mask loads or simplifying the predicate can improve image-heavy TTFT.

## Evolve rationale
Owned constructs are _load_q_range and the mm_prefix_mask_mod boolean expression. Correctness oracle is FA4 multimodal attention output equality against the current mask across valid mm_prefix batches and existing mm_prefix tests.

## Deep research proposals

### 1. Precompute full/partial KV block metadata for mm_prefix so FA4 skips per-element masking on fully-included blocks
- **Finding:** `find-vllm_v1_attention-0007` — *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*
- **Source URL:** <https://pytorch.org/blog/flexattention/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flash_attn.py around lines 1424-1530 (_make_mm_prefix_mask_mod) and the caller site at lines 1010-1050, augment the mm_prefix path so it feeds FA4's existing block-sparse machinery (block_sparse_tensors with full_block_cnt / full_block_idx, already accepted by _flash_attn_fwd at flash_attn_interface.py:388-427) with per-request block-level metadata derived from the mm_prefix ranges. Concretely: when building mm_prefix_query_ranges (the range fill in AttentionMetadataBuilder), also emit, per query row (or per BLOCK_M tile), (1) the KV blocks fully covered by ((causal ∧ window) ∨ mm_prefix) and (2) the boundary/partial blocks that still need mask evaluation. Full blocks are enqueued as full_block_idx and skipped by the kernel loop (no mask_mod call, no per-element predicate); partial blocks continue to run the current mm_prefix_mask_mod. Because mm_prefix ranges are non-overlapping absolute [start,end] pairs and the causal/window edge is a single diagonal band, both boundary sets are cheap to enumerate on the CPU/GPU when the range tensor is filled (fill_mm_prefix_query_ranges) — one strided pair of block indices per row. This is the FA4 analogue of FlexAttention's custom BlockMask constructor: identical semantics, but expressed via block_sparse_tensors instead of BlockMask.from_kv_blocks. Keep use_fast_sampling=True as a fallback for gap-only skipping; the new path adds the 'full block' skip that use_fast_sampling alone does not provide.

**Proposal rationale.**

The candidate's mm_prefix_mask_mod is evaluated per (q_idx, kv_idx) inside every FA4 prefill block, and each row currently pays two range loads plus the OR-with-causal predicate. The FlexAttention blog's central optimization — 'Write a custom constructor for BlockMask' that separates full blocks from partial blocks so the kernel skips per-element masking on full blocks — is directly transferable because FA4's CuTE-DSL kernel already accepts block-sparse metadata via block_sparse_tensors (full_block_cnt / full_block_idx, per the assertion at flash_attn_interface.py:391-398). Today the mm_prefix caller (flash_attn.py:1040-1044) does not populate that structure, so every KV block runs mask_mod even when it lies entirely inside a mm_prefix range or entirely inside the causal/window region. Since mm_prefix ranges are non-overlapping absolute [start,end] intervals and are already materialized at range-fill time (mm_prefix_query_ranges_np at flash_attn.py:723-744), the full/partial split is cheap to compute once per forward. For image-heavy multimodal agent workloads (the stated workload hint), mm_prefix regions typically cover many contiguous KV blocks per query row, so skipping mask_mod on those full blocks removes a large fraction of the per-element predicate work — directly targeting median prefill TTFT while leaving the correctness oracle (FA4 output equality against the current mask across existing mm_prefix tests) intact.

---

## Agent proposals

### 1. Pack mm_prefix range into a single int64 load and use range-length subtract for the predicate
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/flash_attn.py, change the mm_prefix range representation feeding _make_mm_prefix_mask_mod (lines 1424-1530) from a per-token int32 [start, end] pair to a per-token packed int64 (start in the low 32 bits, length = end - start + 1 in the high 32 bits) and update the mask predicate accordingly. Concretely: (a) In fill_mm_prefix_query_ranges (the CPU filler that produces mm_prefix_query_ranges_np) and the surrounding persistent buffer allocation, store one 1-D int64 tensor mm_prefix_query_ranges instead of a 2-column int32 tensor; each row holds (int64(len) << 32) | uint32(start). Encode the (-1, -1) 'no range' sentinel as start = INT32_MAX and len = 0 so no valid kv_idx can satisfy the predicate. (b) In _load_q_range at lines 1469-1486, load a single int64 value from aux_tensors[0][token_idx], then split into (r_start_ssa, r_len_ssa) via bit-shift/mask (CuTE-DSL supports int64->int32 splits; alternatively expose it as a Struct read). This replaces the two scalar_to_ssa(q_ranges[token_idx, 0/1], Int32) loads with a single scalar_to_ssa(q_ranges[token_idx], Int64) load plus register-cheap shifts, halving the per-row global-memory (or L2) traffic for the range aux tensor. (c) In both mm_prefix_mask_mod branches (lines 1490-1508 and 1512-1526), replace `mm = (kv_idx >= r_start) & (kv_idx <= r_end)` with `mm = (kv_idx - r_start) < r_len` using unsigned semantics (or an equivalent CuTE-DSL uint32 cast) so that kv_idx < r_start wraps to a large value and fails the compare. This turns the two-compare-plus-AND predicate into one subtract-plus-one-compare per (q, kv) inside the FA4 unrolled mask loop, and keeps the OR with the causal/window term as-is. Because mm_prefix ranges are non-overlapping and always representable in int32, the packing is loss-free; the (-1,-1) sentinel maps unambiguously to a never-matching range. No changes are required to the FA4 kernel or to block_sparse_tensors — this is purely a data-layout and predicate rewrite inside the cached mask_mod.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_attention-0007) attacks the problem at the block level: it plumbs mm_prefix into FA4's block_sparse_tensors (full_block_cnt / full_block_idx) so full blocks skip mask_mod entirely, and leaves the mask_mod body unchanged for partial blocks. This proposal is complementary and orthogonal: it does not touch block_sparse_tensors at all, and instead attacks the per-element cost that is still paid on every partial/boundary block by (1) halving the aux-tensor load count per query row (one int64 load instead of two int32 loads — directly addressing the candidate's 'two range entries' cost hint) and (2) collapsing the two-compare-and-AND range membership predicate into a single subtract-and-compare via the unsigned-wrap trick with a length-encoded sentinel. Both optimizations remain necessary even after the block-sparse split lands, because boundary blocks (the diagonal band and any block straddling a mm_prefix edge) still must evaluate the mask_mod per element; the two ideas stack rather than overlap.

---
