# FullAttentionManager.find_longest_cache_hit

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 448–494)
- **Symbol:** `FullAttentionManager.find_longest_cache_hit`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0016`

## Description
Full-attention prefix-cache lookup that walks block_hashes left to right up to max_length, calls block_pool.get_cached_block for each group, appends hits, stops at the first miss, then applies EAGLE and alignment pops.

## Current approach
A pure Python for loop over itertools.islice(block_hashes, max_num_blocks) performs one hash-table probe per candidate block and per-group append work through block_pool.get_cached_block. It then mutates the computed block lists for EAGLE and alignment after the scan.

## Estimated impact explanation
This moves TTFT for cached-prefix text and multimodal turns on the common full-attention path. It is medium because the scan is admission-time rather than every decode step, but long shared prefixes can make the per-admission cost substantial.

## Evolve rationale
The specific construct is the for block_hash in itertools.islice(...) lookup loop plus the post-scan EAGLE/alignment pop policy. Unitary full-attention models and the full-attention side of hybrid models pay this during prefix-cache admission for long shared agentic prompts. Headroom includes single-group fast paths, batched/grouped cache probes, pre-sized output lists, reusing a known hit boundary across retries, and avoiding post-hoc list pops when the final admissible length is known before scanning. Correctness oracles include tests/v1/core/test_prefix_caching.py, tests/v1/core/test_kv_cache_utils.py, and tests/v1/kv_connector/unit/test_nixl_connector_hma.py.

## Deep research proposals

### 1. Extend find_longest_cache_hit beyond strict prefix to admit non-prefix cached chunks with selective recompute
- **Finding:** `find-0016` — *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*
- **Source URL:** <https://www.microsoft.com/en-us/research/publication/you-only-prefill-once-combining-cached-knowledge-for-large-language-model-serving-with-cacheblend/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Generalize FullAttentionManager.find_longest_cache_hit in vllm/v1/core/single_type_kv_cache_manager.py (lines 448-494) so that it does not unconditionally stop at the first miss in the left-to-right scan over block_hashes. Following CacheBlend's idea of reusing precomputed KV blocks regardless of position, after detecting the first miss the method would continue probing block_pool.get_cached_block for subsequent block_hashes and record any non-prefix hits as a second category of admissible blocks. The return contract would be extended (or a sibling method added) to surface (i) the contiguous prefix hit length, exactly as today, and (ii) a set of additional cached block ids at known non-prefix positions, so an upper layer can decide to reuse them while marking a small subset of tokens for selective recomputation to repair cross-chunk attention, in the spirit of CacheBlend's selective-recompute step. EAGLE and alignment pops would still apply to the contiguous-prefix portion to preserve current invariants, while the non-prefix hits would be passed through unchanged to the caller. Correctness would be guarded by the existing oracles called out in evolve_rationale (tests/v1/core/test_prefix_caching.py, tests/v1/core/test_kv_cache_utils.py, tests/v1/kv_connector/unit/test_nixl_connector_hma.py), with new tests added to cover the non-prefix admission path and the selective-recompute boundary.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out long shared agentic prompts and identifies 'reusing a known hit boundary across retries' as headroom; CacheBlend directly attacks the deeper structural limit behind that headroom by reusing KV caches 'regardless prefix or not'. The caller objective is TTFT reduction for multi-turn agentic workloads, which is the exact regime (RAG/tool-use with repeated context chunks that often appear at non-leading positions) where strict leading-prefix matching leaves cache hits on the table. find_longest_cache_hit is the natural admission-time chokepoint to extend, because the per-block hash probe loop already has the information needed to identify non-prefix hits at near-zero extra cost and the function's output is what downstream scheduling consumes when deciding how much to prefill. The change is concrete and bounded to this method's contract plus a thin selective-recompute hook, and it addresses the gap that the current approach treats the first miss as a hard scan terminator rather than a signal to switch matching modes.

---

## Agent proposals

### 1. Specialize FullAttentionManager.find_longest_cache_hit with a single-group fast path and precomputed eagle/alignment truncation
- **Agent:** claude

**Detailed description.**

Refactor FullAttentionManager.find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:448-494) to (1) eliminate the post-scan EAGLE and alignment `pop` loops, (2) avoid the per-block list/tuple indirection in the common single-group case, and (3) keep semantics identical to the current implementation.

Concrete changes:

- Hoist constants once before the scan: compute `align_blocks = alignment_tokens // block_size` when `alignment_tokens % block_size == 0` (the assumed/common case) and otherwise fall back to today's per-pop loop; compute `eagle_drop = 1 if use_eagle else 0`.
- Single-group fast path (`len(kv_cache_group_ids) == 1`, today's most common shape including unitary full-attention models and most hybrid layers): bypass `block_pool.get_cached_block` entirely. Resolve `group_id` once, build the `BlockHashWithGroupId` per block inline, and call `block_pool.cached_block_hash_to_block.get_one_block(...)` directly. This drops one Python function call, one per-call list allocation, and the `zip(computed_blocks, cached_block)` work per probe. Append into a single pre-existing list rather than into a tuple-of-lists.
- Multi-group path: keep `get_cached_block`, but pre-allocate the per-group lists with `__init__` size hints and skip the tuple-rebuild per iteration.
- Replace the post-scan EAGLE+alignment pops with a single arithmetic truncation: after the scan returns `hit_count`, compute
    `final_count = ((hit_count - eagle_drop) // align_blocks) * align_blocks`
  (clamped at 0; preserves the current `block_size != alignment_tokens` short-circuit by setting `align_blocks = 1` in that case). Then slice/truncate each per-group list to `final_count` in one operation instead of iterating `.pop()` up to `align_blocks - 1` times across every group.
- Preserve the exact early-exit-on-miss invariant and the `dcp_world_size * pcp_world_size > 1` block-size adjustment.

Correctness oracles already named in the candidate's evolve_rationale apply unchanged: tests/v1/core/test_prefix_caching.py, tests/v1/core/test_kv_cache_utils.py, tests/v1/kv_connector/unit/test_nixl_connector_hma.py. The change is admission-time only and is observable to the scheduler exactly via the same returned tuple of per-group block lists.

**Novelty rationale.**

The single existing deep_research_proposal (find-0016) extends the *semantic contract* of `find_longest_cache_hit` to admit non-prefix cached blocks in the spirit of CacheBlend, including a new return signature and a selective-recompute hook in the caller. This proposal is strictly orthogonal: it keeps the prefix-only contract and the exact same return shape, and instead targets the per-block Python overhead and the post-scan `pop` loops the candidate's evolve_rationale explicitly flags as headroom (single-group fast paths, pre-sized output lists, avoiding post-hoc list pops when the final admissible length is known before scanning). None of those headroom items are addressed by find-0016, which leaves the inner probe loop and the EAGLE/alignment pop logic untouched.

---

### 2. Memoize full-attention prefix-hit retries and resume from the last boundary
- **Agent:** codex

**Detailed description.**

Add a retry-resume path for FullAttentionManager.find_longest_cache_hit in vllm/v1/core/single_type_kv_cache_manager.py:448-494. Split the method internally into the raw left-to-right cache scan and the existing EAGLE/alignment truncation. Let the coordinator/KVCacheManager pass an optional per-request memo containing the raw hit blocks, raw hit count, block_hashes length, max_length, kv_cache_group_ids, use_eagle/alignment settings, and a BlockPool eviction_epoch. BlockPool should increment the eviction epoch on cached-block eviction and reset_prefix_cache. When allocate_slots returns None and the same waiting request is retried, reuse the memo if the epoch and settings still match: initialize computed_blocks from the raw memo and continue scanning at raw_hit_count instead of probing again from block 0. If block hashes or max_length changed, slice or extend from the safe boundary; if the eviction epoch changed, fall back to the current full scan. The public return shape stays unchanged after applying EAGLE/alignment truncation. Add a focused scheduler/KV-cache test that forces allocation failure after get_computed_blocks, retries the same request, asserts the second lookup starts at the memoized boundary, and verifies eviction/reset invalidation.

**Novelty rationale.**

The deep_research proposal changes the semantic contract to admit non-prefix cached chunks and selective recompute; this keeps the strict prefix-only contract and only avoids repeating already-proven prefix probes for the same request after admission retry. Agent A optimizes the cost of one invocation with a single-group fast path and arithmetic truncation; this targets repeated invocations caused by allocation failure/preemption and skips the whole previously scanned prefix, so it is complementary rather than a duplicate.

---
