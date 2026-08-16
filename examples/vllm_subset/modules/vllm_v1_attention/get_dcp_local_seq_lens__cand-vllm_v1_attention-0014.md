# get_dcp_local_seq_lens

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/utils.py`](vllm/v1/attention/backends/utils.py) (lines 958–995)
- **Symbol:** `get_dcp_local_seq_lens`
- **Kind:** function
- **Estimated impact:** low
- **Id:** `cand-vllm_v1_attention-0014`

## Description
Computes per-rank local sequence lengths for Decode Context Parallelism.

## Current approach
Converts seq_lens to int32, allocates rank_offsets with torch.arange or torch.tensor on each call, applies integer arithmetic and torch.clip, and returns a fresh tensor.

## Estimated impact explanation
Per-call cost is small, so impact is low. It still runs on the DCP per-step path, so caching offsets or fusing arithmetic can modestly improve TPOT in DCP-enabled agentic serving.

## Evolve rationale
rank_offsets allocation and the local-length formula are small but repeated in DCP metadata paths. Correctness oracle is bit-exact equality with the current function over random seq_lens, dcp_size, rank, and interleave settings.

## Deep research proposals

### 1. Cache DCP rank_offsets tensor across metadata builds in get_dcp_local_seq_lens
- **Finding:** `find-vllm_v1_attention-0004` — *flashinfer.cascade*
- **Source URL:** <https://docs.flashinfer.ai/api/cascade.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Apply the FlashInfer cascade idea of reusing auxiliary planning structures across repeated calls to get_dcp_local_seq_lens in vllm/v1/attention/backends/utils.py (lines 958-995). Today the function allocates rank_offsets on every invocation via torch.arange(dcp_size, ...) (when dcp_rank is None) or torch.tensor(dcp_rank, ...) (when a specific rank is given). Because rank_offsets depends only on (dcp_size, dcp_rank, seq_lens.device, seq_lens_i32.dim()) — all of which are stable across steps within a running DCP-enabled serve — cache these tensors in a small module-level dict keyed by (dcp_size, dcp_rank, device, ndim). On first use build the tensor with torch.arange / torch.tensor as today; on subsequent calls reuse the cached tensor directly. Optionally also cache a small preallocated int32 output buffer keyed by (max_batch, device) if it can be safely reused within a CUDA-graph capture, mirroring the cascade planner's reuse of preallocated buffers across decode calls. Behavior must remain bit-exact with the current formula (base + clipped remainder), verified against random seq_lens, dcp_size, rank, and cp_kv_cache_interleave_size settings.

**Proposal rationale.**

The finding's transferable idea — 'auxiliary data structures can be reused across multiple batch decode attention calls' — maps directly to the per-call allocation of rank_offsets in this helper. rank_offsets is a pure function of (dcp_size, dcp_rank, device) and does not vary across metadata builds in a running server, so it fits the 'stable plan reused across steps' pattern from MultiLevelCascadeAttentionWrapper. The candidate's evolve_rationale explicitly calls out caching offsets as the improvement axis, and its estimated_impact notes this runs on the DCP per-step path where reducing small allocations helps median TPOT on multi-turn agentic workloads. The change is narrow, preserves the bit-exact oracle, and addresses the concrete gap (redundant torch.arange/torch.tensor allocations) without altering the arithmetic.

---

## Agent proposals

### 1. Add fast paths in get_dcp_local_seq_lens for dcp_size==1, interleave==1, and int32 inputs
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/utils.py:958-995, add three early-return branches at the top of get_dcp_local_seq_lens to eliminate work entirely on hot DCP metadata-build paths whenever configuration makes the general formula trivial.

1) If dcp_size == 1: return seq_lens directly when seq_lens.dtype is already torch.int32, otherwise return seq_lens.to(torch.int32). DCP is disabled so the local length equals the global length; the entire arange/tile/divide/clip pipeline is skipped, including any cached rank_offsets lookup.

2) Elide the seq_lens.to(torch.int32) dtype-copy when seq_lens.dtype is already torch.int32 (alias instead of copy). Callers frequently pass int32 seq_lens already, so this avoids an unnecessary tensor allocation and D2D copy per call, independent of any offsets caching.

3) If cp_kv_cache_interleave_size == 1: the two integer divisions `// 1 // dcp_size * 1` collapse to `// dcp_size`, and torch.clip(remainder - rank_offsets, 0, 1) degenerates to (remainder > rank_offsets).to(int32). Compute base = seq_lens_tiled // dcp_size, rem = seq_lens_tiled - base * dcp_size, and return base + (rem > rank_offsets).to(torch.int32), where rank_offsets is either the arange (all-ranks case) or the scalar tensor (per-rank case). This removes one integer div, one mul, and one clip from the elementwise pipeline while remaining bit-exact with the general formula.

All three branches preserve bit-exact equality with the current function under the candidate's stated oracle (random seq_lens, dcp_size, dcp_rank, cp_kv_cache_interleave_size, plus both int32 and int64 seq_lens dtypes). The optimization composes cleanly with the existing deep_research_proposal to cache rank_offsets: when neither fast path fires (dcp_size > 1 and interleave > 1), the general branch still benefits from cached auxiliary tensors.

**Novelty rationale.**

The listed deep_research_proposal targets allocation reuse for rank_offsets (memoizing torch.arange/torch.tensor across metadata builds). This proposal is orthogonal: it targets work elimination and dtype-copy elimination, not auxiliary-tensor caching. Specifically, (a) short-circuiting dcp_size==1 to the identity, (b) aliasing instead of .to(int32) when already int32, and (c) collapsing the divide/multiply/clip chain when cp_kv_cache_interleave_size==1 into a single comparison, are not discussed in the existing proposal. Both optimizations still apply on the DCP per-step path relevant to median TPOT, and both compose with — rather than duplicate — the offsets cache.

---
