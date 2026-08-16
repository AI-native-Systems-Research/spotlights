# FlexAttentionMetadata._build_block_mask_direct

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flex_attention.py`](vllm/v1/attention/backends/flex_attention.py) (lines 692–822)
- **Symbol:** `FlexAttentionMetadata._build_block_mask_direct`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0023`

## Description
Builds FlexAttention BlockMask metadata directly from paged-KV block tables for the direct-build path.

## Current approach
Indexes block_table for every query token up to cdiv(max_seq_len, block_size), applies several masked_fill_ pruning passes for sequence length, causal, sliding-window, R-SWA, and custom sparsity hints, pads and reshapes by q_block_size, deduplicates with unique_static_unsorted, and copies kv_indices and kv_num_blocks into persistent buffers.

## Estimated impact explanation
For FlexAttention decode this metadata path runs before attention and can dominate small-batch per-step overhead. Reducing over-fetch, pruning passes, or dedup work can improve median TPOT for agentic decode batches using the Flex backend.

## Evolve rationale
Owned constructs include the used_pages over-estimation, token_indices arange, masked_fill_ pruning passes, unique_static_unsorted deduplication, and copy_to_persistent calls. Correctness oracle is BlockMask equivalence against build_block_mask/create_block_mask for randomized block tables and masks, plus FlexAttention output equality.

## Deep research proposals

### 1. Cache and reuse per-step BlockMask metadata across FlexAttention layers
- **Finding:** `find-vllm_v1_attention-0004` — *flashinfer.cascade*
- **Source URL:** <https://docs.flashinfer.ai/api/cascade.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In FlexAttentionMetadata._build_block_mask_direct (vllm/v1/attention/backends/flex_attention.py:692-822), introduce a metadata-level cache keyed by stable per-step inputs (batch shape, doc_ids, num_blocks_per_seq, block_table page identities for the used slice, max_seq_len, q_block_size, block_size, causal/sliding_window/R-SWA/custom-hint config and their parameters) so that when multiple FlexAttention layers share the same decode step and identical mask configuration, the expensive computation (used_pages gather, masked_fill_ pruning passes for past_seq/causal/sliding_window/R-SWA/custom-hint, pad-and-reshape, unique_static_unsorted dedup, kv_num_blocks reduction, and copy_to_persistent into persistent_kv_indices/persistent_kv_num_blocks) is only performed once per step. Subsequent layers reuse the already-populated persistent CUDA-graph buffers and rebuild BlockMask from cached kv_indices/kv_num_blocks (still calling BlockMask.from_kv_blocks with the cached tensors and the layer's mask_mod), invalidating on shape or config change. Where different layers only differ in mask_mod but not in block-level pruning (e.g., same causal/sliding config), share the block-index tensors and rebuild only the BlockMask wrapper.

**Proposal rationale.**

The FlashInfer cascade docs explicitly note that auxiliary planning data structures can be reused across multiple batch decode attention calls in a step. The candidate's per-step, per-layer construction repeats identical block-table gathers, masked_fill_ passes, and dedup work for every layer that shares the same decode-step geometry and mask configuration, which the description calls out as a dominant per-step overhead for small-batch agentic decode. Caching keyed on stable shape/page/mask-config plans transfers the FlashInfer pattern to Flex metadata build and directly targets median TPOT on multi-turn agentic workloads without changing the correctness oracle (BlockMask equivalence and FlexAttention output equality), since cache hits reproduce the same tensors and misses fall back to the current path.

---

### 2. Split fully-unmasked KV blocks into full_kv_indices to skip per-element mask_mod
- **Finding:** `find-vllm_v1_attention-0007` — *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*
- **Source URL:** <https://pytorch.org/blog/flexattention/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In FlexAttentionMetadata._build_block_mask_direct (vllm/v1/attention/backends/flex_attention.py:692-822), populate the currently-None full_kv_num_blocks/full_kv_indices arguments of BlockMask.from_kv_blocks by classifying each (q_block, kv_block) pair as fully-inside-the-mask vs partially-masked while the pruning passes already run. Concretely: (1) In addition to the existing masked_fill_ passes that zero out entirely-excluded pages, compute a companion boolean tensor is_full_block[q_token, kv_block] that is True only when every KV token in that kv_block, for every q_token in the q_block_size group, is unmasked by causal, sliding-window, R-SWA, and custom-hint predicates. For causal this means block_ends[None,:] <= logical_q_idx[:,None] - (q_block_size - 1); for sliding_window this means block_starts[None,:] >= min_kv_idx[:,None]; for R-SWA this means the kv block lies entirely in either the prefix range or the window range for the strictest q_token in the group; and for the custom sparsity hint, extend BlockSparsityHint to optionally expose a hint_full_fn (defaulting to False, i.e. treat as partial) so backends that know their hint is block-aligned can report full blocks. (2) After the pad_to_multiple/reshape/q_block_size grouping, aggregate is_full_block by AND across the q_block_size rows within each group, then run the same unique_static_unsorted+copy_to_persistent pipeline to produce persistent full_kv_indices / full_kv_num_blocks buffers alongside the existing kv_indices / kv_num_blocks. (3) Pass them into BlockMask.from_kv_blocks. Partial blocks (the remainder of kv_indices minus full ones) still evaluate mask_mod per element; full blocks are executed by the FlexAttention kernel without per-element masking, which is the specific optimization the blog calls out.

**Proposal rationale.**

The candidate already implements the blog's high-level idea of a custom BlockMask constructor via BlockMask.from_kv_blocks — but it explicitly sets full_kv_num_blocks=None and full_kv_indices=None (lines 813-814), which means FlexAttention treats every retained kv block as partial and re-evaluates mask_mod per element. The blog's core performance claim for compact BlockMask construction is precisely the full/partial split, so materializing full-block metadata is the transferable idea from the finding that this candidate has not yet applied. For agentic decode workloads (the caller's target for TPOT), most kv blocks in a request's history are entirely below the causal frontier and entirely inside the sliding/R-SWA window, so a large fraction of retained blocks should classify as full — cutting per-token mask_mod work in the FlexAttention kernel and reducing the median TPOT this candidate contributes to. The classification reuses the same block_starts/block_ends and logical_q_idx tensors already computed in the pruning passes, so it adds a bounded amount of elementwise work to a metadata path that runs once per step rather than per token.

---

## Agent proposals

### 1. Add a decode-specialized analytic fast path in _build_block_mask_direct that skips the batch-max gather and masked_fill_ pruning chain
- **Agent:** claude

**Detailed description.**

In FlexAttentionMetadata._build_block_mask_direct (vllm/v1/attention/backends/flex_attention.py:692-822), add a specialized fast path taken when max_query_len == 1 (pure decode) and block_sparsity_hint is None. In this regime each request contributes exactly one query token and its retained KV block set is a closed-form contiguous range determined analytically from num_blocks_per_seq, decode_offset, sliding_window, rswa_window, and rswa_prefix_lens — no per-row gather-and-prune is needed.

Concretely, replace the current sequence of (a) used_pages = self.block_table[self.doc_ids, : cdiv(self.max_seq_len, self.block_size)], (b) the past_seq masked_fill_ on lines 731-733, (c) the causal future_blocks masked_fill_ on lines 755-757, (d) the sliding_window masked_fill_ on lines 759-766, (e) the R-SWA in_gap masked_fill_ on lines 767-783, and (f) unique_static_unsorted dedup on lines 801-803 with a direct construction:

1. Per row r (corresponding to doc_ids[r]), compute end_block = num_blocks_per_seq[doc_ids[r]] (this replaces both the initial cdiv(max_seq_len, block_size) over-fetch and the past_seq / future_blocks masks — for a single q token at position decode_offset[doc_ids[r]], the causal frontier is exactly num_blocks_per_seq).
2. For uniform sliding_window: start_block = max(0, (decode_offset[doc_ids[r]] - (sliding_window - 1))) // block_size. For non-sliding causal: start_block = 0. For R-SWA: produce two contiguous ranges [0, prefix_block) and [window_start_block, end_block) where prefix_block = rswa_prefix_lens[doc_ids[r]] // block_size and window_start_block = max(prefix_block, (decode_offset[doc_ids[r]] - (rswa_window - 1)) // block_size).
3. Compute kv_num_blocks[r] as the total width(s) of these range(s), vectorized across r on-device using arithmetic on num_blocks_per_seq / decode_offset / sliding_window / rswa_* — no masked_fill_ and no intermediate [N, cdiv(max_seq_len, block_size)] boolean tensors.
4. Fill kv_indices[r, :kv_num_blocks[r]] by directly slicing self.block_table with a gathered range. Because R-SWA already forces q_block_size=1 (vllm/v1/attention/backends/flex_attention.py:1081) and the non-R-SWA decode case also has one q token per row, no cross-row dedup within a q_block_size group is required — the pad_to_multiple / reshape / unique_static_unsorted pipeline collapses to a direct copy into persistent_kv_indices / persistent_kv_num_blocks. Everything downstream (BlockMask.from_kv_blocks call and its kwargs at lines 809-822) is unchanged.

The general path (max_query_len > 1 or block_sparsity_hint is not None) continues to use the existing construct-then-prune implementation, so behavior on prefill and hint-driven sparsity is untouched. Correctness is checked against the current slow path by comparing kv_num_blocks and the sorted valid-prefix of each row of kv_indices for randomized (doc_ids, num_blocks_per_seq, causal/SW/R-SWA config) inputs, and by FlexAttention output equality on a decode step.

**Novelty rationale.**

Neither existing deep_research_proposal specializes the metadata build itself for the decode-only shape. find-vllm_v1_attention-0004 caches the output of the current construct-then-prune path across FlexAttention layers within a single step; on the first layer of every step (i.e., every cache miss) it still runs the full [N, cdiv(max_seq_len, block_size)] gather and every masked_fill_ pass — this proposal reduces that miss-cost itself, and is complementary to caching (a cached fast-path result is smaller and cheaper to reuse). find-vllm_v1_attention-0007 augments the returned BlockMask with full_kv_indices / full_kv_num_blocks so the FlexAttention kernel can skip per-element mask_mod on fully-inside-mask blocks; that is a kernel-side saving that runs after BlockMask is built. This proposal is orthogonal: it saves host-side dispatch and torch allocation cost during the metadata build by replacing the construct-then-prune pattern with a closed-form contiguous-range computation when max_query_len == 1, which is the dominant per-step shape for the caller's multi-turn agentic decode workload. Neither existing proposal identifies or exploits the fact that decode's retained KV block set per row is a contiguous range (or two ranges for R-SWA) directly computable from num_blocks_per_seq / decode_offset / sliding_window / rswa_* without materializing any [N, max_blocks] boolean tensor or invoking unique_static_unsorted.

---
