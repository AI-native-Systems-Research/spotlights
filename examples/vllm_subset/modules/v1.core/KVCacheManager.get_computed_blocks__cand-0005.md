# KVCacheManager.get_computed_blocks

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/kv_cache_manager.py`](vllm/v1/core/kv_cache_manager.py) (lines 183–223)
- **Symbol:** `KVCacheManager.get_computed_blocks`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0005`

## Description
Computes a request's locally cached prefix by calling coordinator.find_longest_cache_hit over request.block_hashes with max_cache_hit_length set to request.num_tokens - 1, then records prefix-cache stats.

## Current approach
Each admission attempt re-runs the prefix lookup up to the full prompt-minus-one boundary unless prefix caching is disabled or the request skips prefix-cache reads. There is no per-request memo of recent hit length across retries or adjacent turns.

## Estimated impact explanation
This can reduce repeated admission-time hash-table scans for long shared prompts. The workload-level signal is TTFT for queued or retried agentic turns; impact is below the scheduler loops because the heavy work is delegated to the coordinator/managers.

## Evolve rationale
The concrete policy is the get_computed_blocks max_cache_hit_length calculation and unconditional coordinator lookup. Multi-turn sessions repeatedly query nearly identical prefixes, and skipped/failed admissions can retry across steps. Headroom includes caching the last known hit/miss boundary on the request, short-circuiting after a stable miss, or carrying a rolling session prefix-hit watermark. Correctness oracle: tests/v1/core/test_prefix_caching.py exercises get_computed_blocks and allocate_slots interactions across full hits, partial hits, EAGLE, and sliding-window cases.

## Deep research proposals

### 1. Extend get_computed_blocks to consult a tiered KV cache with async layer-wise preload
- **Finding:** `find-0009` — *AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving*
- **Source URL:** <https://arxiv.org/html/2403.19708v2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify KVCacheManager.get_computed_blocks (vllm/v1/core/kv_cache_manager.py:183-223) so that when the in-GPU coordinator.find_longest_cache_hit returns fewer hit blocks than request.block_hashes implies, the manager additionally probes a lower-tier KV store (host DRAM, and optionally SSD/remote) keyed by the same block hashes to extend the computed-prefix length. Hits located in lower tiers are reported back to the caller as computed blocks while an asynchronous, layer-wise transfer is kicked off to stage those KV tensors into GPU blocks before the corresponding attention layer executes, mirroring AttentionStore's overlap-fetch-with-compute scheme. Demotion of currently-resident GPU blocks to the lower tier would happen on eviction rather than synchronously here. The change is localized to get_computed_blocks plus a small tiered-store interface; it preserves the existing prefix-cache-disabled and skip_cache_block_hash short-circuits and keeps prefix_cache_stats accounting intact (with new counters for tiered hits). Correctness can be checked against tests/v1/core/test_prefix_caching.py, with new cases for partial-tiered-hit and EAGLE/sliding-window interactions.

**Proposal rationale.**

The candidate's gap is that multi-turn agentic prompts repeatedly land at get_computed_blocks for nearly identical prefixes, but once a GPU block has been evicted the lookup reports a miss and the prefix is recomputed end-to-end, inflating TTFT. AttentionStore directly targets this regime by retaining KV across a GPU/host/storage hierarchy and using scheduler-aware, layer-wise async preload so a tier-2 hit is almost as cheap as a tier-1 hit. Wiring the coordinator lookup to fall through to such a tier and to issue an async preload is a concrete, transferable mechanism that addresses the exact workload (multi-turn) and exact metric (TTFT/median TPOT) called out in the caller context, while staying within the candidate's existing entry point.

---

### 2. Extend get_computed_blocks to admit non-prefix KV cache chunks via CacheBlend-style selective recomputation
- **Finding:** `find-0016` — *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*
- **Source URL:** <https://www.microsoft.com/en-us/research/publication/you-only-prefill-once-combining-cached-knowledge-for-large-language-model-serving-with-cacheblend/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/core/kv_cache_manager.py:183-223 (KVCacheManager.get_computed_blocks), the current logic invokes coordinator.find_longest_cache_hit over request.block_hashes bounded by max_cache_hit_length=request.num_tokens-1, which only recovers a strict leading prefix. Generalize this admission step so that, in addition to the longest leading-prefix hit, the manager also probes for non-leading block-hash matches corresponding to repeated RAG/tool-context chunks within the request's token sequence (e.g., by hashing chunk-aligned segments of request.block_hashes and querying the coordinator's block-hash index for arbitrary-position hits). For matched non-prefix chunks, mark a small subset of boundary/cross-attention-sensitive tokens for selective recomputation (per CacheBlend) while reusing the cached KV blocks for the remainder, returning the union of reused blocks alongside the recompute-mask metadata that downstream attention can honor. Preserve current behavior under disable_prefix_cache / no-cache paths and gate the new non-prefix path behind a config flag so the existing prefix-only oracle in tests/v1/core/test_prefix_caching.py remains valid; add new cases covering RAG-style repeated middle chunks. Update prefix_cache_stats accounting to separately report prefix vs. non-prefix reused tokens.

**Proposal rationale.**

The candidate's gap is that get_computed_blocks only exploits leading-prefix reuse, which leaves cache value on the table for multi-turn agentic and RAG workloads where the same retrieved chunks reappear at varying offsets across turns. CacheBlend directly targets this gap: it reuses precomputed KV regardless of prefix position and recovers quality by recomputing a small token subset, which is exactly the transferable idea this candidate needs. Because get_computed_blocks is the single admission point that decides which blocks count as 'computed' before allocate_slots runs, extending its lookup is the minimal, correctly-scoped place to introduce non-prefix reuse, and the workload hint (multi-turn agentic, TTFT-sensitive) matches CacheBlend's reported regime.

---

### 3. Cache per-session prefix-hit watermark to skip redundant find_longest_cache_hit scans across multi-turn admissions
- **Finding:** `find-0017` — *Stateful Large Language Model Serving with Pensieve*
- **Source URL:** <https://arxiv.org/abs/2312.05516>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Augment KVCacheManager.get_computed_blocks (vllm/v1/core/kv_cache_manager.py:183-223) with a session-keyed memo of the last successful cache-hit boundary, drawing on Pensieve's notion of treating a conversation as a stateful cache owner across requests. Concretely: (1) tag each Request with a session/conversation id (or derive one from a stable parent prompt hash) and track, in the manager, the last (block_hashes_prefix_signature, num_new_computed_tokens) observed for that session; (2) at the top of get_computed_blocks, if the incoming request.block_hashes share the recorded prefix signature up to the previous hit length, skip directly to validating only the suffix beyond that watermark via coordinator.find_longest_cache_hit with a starting offset (or a tighter max_cache_hit_length window) instead of re-scanning the full request.num_tokens-1 prefix; (3) on a confirmed miss at the watermark, mark the session entry stale rather than freeing it, so a subsequent retry in the same step does not re-issue an identical full-prefix lookup; (4) keep the existing code path (full lookup) as the fallback when the session memo is absent, mismatched, or invalidated by preemption (request.num_preemptions > 0 already gates one stat field and is a natural invalidation trigger). The PrefixCacheStats.record call is preserved by reporting the combined hits (memoized + suffix). No change to coordinator semantics or to the block-hash scheme is required; the memo is purely an admission-side shortcut keyed on what the coordinator already returned previously.

**Proposal rationale.**

The candidate explicitly flags multi-turn sessions repeatedly querying nearly identical prefixes and retried admissions as the headroom, and lists 'carrying a rolling session prefix-hit watermark' as a candidate mechanism. Pensieve contributes the transferable framing that multi-turn conversation state should persist across request boundaries rather than being recomputed each turn, which justifies introducing a session-scoped memo above the coordinator. This directly targets the multi-turn agentic workload in the caller context and the TTFT objective by shrinking the work done per admission attempt for queued/retried turns, without changing the underlying cache contents or correctness oracle exercised by tests/v1/core/test_prefix_caching.py (full-hit, partial-hit, EAGLE, sliding-window cases all still flow through find_longest_cache_hit on memo miss).

---

## Agent proposals

### 1. Intra-step admission-cache to deduplicate find_longest_cache_hit across concurrent requests sharing block-hash prefixes
- **Agent:** claude

**Detailed description.**

Extend KVCacheManager.get_computed_blocks (vllm/v1/core/kv_cache_manager.py:183-223) with a per-scheduler-step memo keyed on a rolling hash of leading block_hashes that is consulted before delegating to coordinator.find_longest_cache_hit. Concretely: (1) add a small structure on KVCacheManager (e.g., self._step_admission_cache) holding entries of the form prefix_signature -> (computed_blocks, num_new_computed_tokens, max_cache_hit_length_at_lookup), reset by the scheduler at the start of every step (a lifecycle hook already exists where free_block_hashes / similar per-step bookkeeping runs); (2) at the top of get_computed_blocks, walk request.block_hashes building cumulative signatures and probe the memo for the longest matching signature whose recorded max_cache_hit_length_at_lookup is >= the current request's max_cache_hit_length; on hit, reuse those computed_blocks/num_new_computed_tokens directly and only invoke find_longest_cache_hit on the residual suffix (or skip entirely when the request is a strict prefix of a memoized request); (3) on miss, run the existing find_longest_cache_hit and store the result keyed by the prefix signatures it actually traversed so subsequent requests in the same step benefit; (4) preserve the prefix-cache-disabled and skip_cache_block_hash short-circuits and continue calling prefix_cache_stats.record with the same (requests, queries, hits) accounting so existing tests/v1/core/test_prefix_caching.py oracles are unaffected. The memo is intentionally ephemeral (single scheduler step), so it does not interact with eviction/coordinator state and cannot go stale.

**Novelty rationale.**

All three existing proposals act across turns or across tiers: find-0009 falls through to a lower-tier KV store, find-0016 broadens what counts as a hit by allowing non-prefix matches with selective recompute, and find-0017 memoizes per-session hit watermarks across turns to skip the leading prefix on a subsequent admission. None of them address the orthogonal axis exploited here, which is duplicated work across distinct concurrent requests within a single scheduler step (e.g., a batch of agentic tool-call requests that share a long system prompt or shared retrieved context but have no session relationship in find-0017's sense). The mechanism is a step-scoped admission cache, which neither requires a tier-2 store (find-0009), nor changes what counts as 'computed' (find-0016), nor depends on a session id or cross-step persistence (find-0017); it is purely a fan-in deduplication of find_longest_cache_hit over the waiting queue.

---

### 2. Compact leading null blocks for sliding-window prefix hits
- **Agent:** codex

**Detailed description.**

Change KVCacheManager.get_computed_blocks in vllm/v1/core/kv_cache_manager.py:183-223 to accept a compact computed-block result that can represent leading skipped/null KV blocks by count instead of materializing long lists of block_pool.null_block. The current sliding-window and chunked-local lookup paths can return many leading null blocks for tokens that are already outside the attention window; for long agentic prompts this creates Python list allocation, iteration, and later ref-count work even though those entries carry no real KV storage. Add an internal representation such as leading_null_blocks_by_group plus real tail hit blocks, preserve the existing num_new_computed_tokens and PrefixCacheStats.record semantics, and expand to concrete null blocks only at the boundary where req_to_blocks must be updated, or teach allocate_new_computed_blocks to prepend the counted null span directly. Existing prefix-cache-disabled and skip_reading_prefix_cache exits remain unchanged; tests in tests/v1/core/test_prefix_caching.py should keep asserting identical final block IDs for sliding-window and hybrid cases while adding a long-context case that verifies the compact path avoids building an O(num_prompt_blocks) null list.

**Novelty rationale.**

The existing deep research proposals change what storage is queried (tiered KV), what positions can be reused (non-prefix chunks), or memoize cache-hit boundaries across sessions. Agent A memoizes lookup results across requests within a scheduler step. This proposal does not add a new cache tier, broaden hit semantics, or memoize find_longest_cache_hit results across requests or turns; it keeps the same hit decision and targets the separate overhead of representing skipped local-attention prefixes as thousands of explicit null-block objects.

---
