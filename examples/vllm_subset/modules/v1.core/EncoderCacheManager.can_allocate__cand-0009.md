# EncoderCacheManager.can_allocate

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/encoder_cache_manager.py`](vllm/v1/core/encoder_cache_manager.py) (lines 119–178)
- **Symbol:** `EncoderCacheManager.can_allocate`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0009`

## Description
Checks whether a multimodal encoder output fits in the encoder cache and, if needed, evicts unreferenced cached entries from the freeable OrderedDict until enough slots are available.

## Current approach
Capacity check plus strict oldest-freeable eviction via freeable.popitem(last=False). The policy tracks only unreferenced insertion order, not recency of reuse, frequency, size efficiency, or expected near-future reuse.

## Estimated impact explanation
Better eviction can avoid recomputing expensive vision embeddings, improving multimodal TTFT. Impact is medium because it applies only when encoder cache pressure and reuse coexist.

## Evolve rationale
The code construct is the while num_embeds > self.num_free_slots eviction loop over self.freeable. Repeated images and tool-media in multimodal agentic workloads make encoder-cache hit rate a direct TTFT lever. Headroom includes LRU-on-hit, LFU, size-aware eviction, or protected/probationary queues while preserving the cached/freeable/freed accounting interface. Correctness oracle: tests/v1/core/test_encoder_cache_manager.py covers hit, free, eviction, and capacity invariants.

## Deep research proposals

### 1. Add TTL-based pinning to encoder cache freeable eviction for multi-turn agentic media reuse
- **Finding:** `find-0002` — *Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live*
- **Source URL:** <https://arxiv.org/abs/2511.02230>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify the eviction loop in EncoderCacheManager.can_allocate (vllm/v1/core/encoder_cache_manager.py:119-178) so that entries in self.freeable carry a per-entry time-to-live (or deadline) and are skipped during freeable.popitem-style eviction until their TTL expires. Concretely: when an encoder output transitions from cached/referenced to freeable (e.g., after its dependent request finishes a turn and pauses for a tool call), tag the freeable entry with a TTL derived from (a) the estimated re-encode cost of the multimodal input (proxy: num_embeds and modality) and (b) current cache pressure / queueing impact (e.g., share of capacity that is freeable vs. cached). The while num_embeds > self.num_free_slots loop should iterate self.freeable in insertion order but prefer entries whose TTL has expired before evicting unexpired ones; only fall back to evicting unexpired entries when no expired ones remain (preserving the existing capacity invariant and the cached/freeable/freed accounting interface). TTLs decrement with wall-clock or step time and entries become eligible for normal FIFO eviction once expired. Existing tests in tests/v1/core/test_encoder_cache_manager.py continue to pin down the hit/free/eviction/capacity invariants; new tests cover that an unexpired freeable entry survives an allocation that fits without it, and that an expired entry is evicted as before.

**Proposal rationale.**

The candidate's current policy is strict FIFO over unreferenced encoder outputs, which discards multimodal embeddings that are likely to be reused across short tool-call pauses in multi-turn agentic workloads — exactly the TTFT pathology the caller objective targets. The Continuum CacheTTL finding contributes a concrete, transferable mechanism: assign a TTL derived from reload/recompute cost and queueing delay, and use it to selectively protect cache entries during eviction without permanently occupying memory. Mapping KV-cache pinning to encoder-cache pinning is direct (both are recompute-vs-memory tradeoffs), and re-encoding vision embeddings is typically more expensive per byte than refilling KV, so the cost-aware TTL signal applies cleanly. This addresses the gap the candidate's evolve_rationale calls out (no recency/reuse/cost signal in eviction) while preserving the freeable/cached/freed interface and existing correctness oracle.

---

### 2. Workflow-aware encoder cache eviction using agent-step metadata
- **Finding:** `find-0003` — *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows*
- **Source URL:** <https://arxiv.org/abs/2507.07400>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify EncoderCacheManager.can_allocate (vllm/v1/core/encoder_cache_manager.py:119-178) to replace the strict oldest-freeable eviction (self.freeable.popitem(last=False)) with a workflow-aware priority policy inspired by KVFlow's Agent Step Graph. Augment the freeable OrderedDict with optional per-entry metadata describing temporal proximity to future activation (e.g., a steps-to-next-use score supplied by the scheduler/request when available). When eviction is required to satisfy num_embeds > self.num_free_slots, evict entries in descending steps-to-next-use order (i.e., least-soon-needed first), falling back to the existing FIFO-of-freeable behavior when no metadata is available. The cached/freeable/freed accounting interface and free()/get_cached_input_ids() semantics remain unchanged; only the selection key inside the eviction loop changes. Optionally expose a hook so callers (multi-agent orchestrators) can annotate multimodal inputs with expected reuse step distance at allocate/cache time, defaulting to +inf so unannotated entries behave as today and tests/v1/core/test_encoder_cache_manager.py invariants are preserved.

**Proposal rationale.**

The candidate's gap is exactly the one KVFlow targets in the KV-cache domain: a recency-only/insertion-order eviction policy that ignores knowable near-future reuse. In multi-turn agentic multimodal workloads (the stated caller objective), the same images or tool-media are often reused at predictable later agent steps, so a steps-to-execution-style score can protect entries whose recomputation cost (vision encoder forward pass) directly drives media TTFT. Transferring KVFlow's idea — bias eviction by temporal proximity to future activation rather than pure LRU/FIFO — addresses the freeable.popitem(last=False) limitation while keeping the existing accounting interface and capacity invariants intact, and it degrades gracefully to current behavior when no workflow metadata is supplied.

---

### 3. Adopt LRU-on-hit eviction for the encoder cache freeable queue
- **Finding:** `find-0004` — *Efficiently Programming Large Language Models using SGLang*
- **Source URL:** <https://arxiv.org/html/2312.07104v1>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify EncoderCacheManager.can_allocate (vllm/v1/core/encoder_cache_manager.py:119-178) and its companion hit/free paths to evict by least-recently-used reuse rather than oldest insertion order. Concretely: when an existing cached encoder output is reused (cache hit on get_cached_input_ids / has_cache), move its entry to the most-recently-used end of self.freeable (e.g. freeable.move_to_end(key, last=True)) if it is currently unreferenced, and continue inserting freshly freed entries at the MRU end on free(). The eviction loop `while num_embeds > self.num_free_slots: self.freeable.popitem(last=False)` then naturally drops the true LRU entry, mirroring SGLang RadixAttention's LRU policy over reusable cache state. Preserve the existing cached / freeable / freed bookkeeping and capacity invariants so tests/v1/core/test_encoder_cache_manager.py continue to hold; the only behavioral change is recency tracking on reuse.

**Proposal rationale.**

The current policy evicts strictly by insertion order of unreferenced entries, which is blind to reuse recency. SGLang's RadixAttention explicitly maintains an LRU cache of reusable KV state and reports prefix-hit gains in multi-call agent/chat workloads — the same access pattern (repeated images, tool-media) that drives encoder-cache reuse here. Transferring just the LRU-on-hit recency signal to the freeable OrderedDict closes the recency gap called out in the evolve rationale, is a minimal change to the eviction loop, and directly targets media TTFT by avoiding recomputation of recently-reused vision embeddings.

---

### 4. Priority/TTL-aware eviction in EncoderCacheManager.can_allocate
- **Finding:** `find-0008` — *Introducing New KV Cache Reuse Optimizations in NVIDIA TensorRT-LLM*
- **Source URL:** <https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the strict oldest-first FIFO eviction in vllm/v1/core/encoder_cache_manager.py:173-177 (the `while num_embeds > self.num_free_slots: self.freeable.popitem(last=False)` loop) with a priority-aware eviction policy inspired by TensorRT-LLM's priority-based KV eviction. Concretely: (1) attach a small per-entry retention hint when a multimodal input is admitted to the encoder cache - e.g., (priority_class, expiration_step_or_ttl, last_use_step) - derived from cheap signals available at admission time: whether the mm_hash appears in the request's system/role prefix, whether the same mm_hash has already been seen (reuse counter), the multimodal modality, and the request's session/conversation id for multi-turn agentic flows. (2) Replace `self.freeable: OrderedDict[str, int]` with a structure that stores those hints alongside the embedding count (still keyed by mm_hash so the cached/freeable/freed accounting interface is preserved). (3) In the eviction loop, instead of `popitem(last=False)`, pick the next victim by ordering freeable entries by (lowest priority_class, expired-TTL first, then oldest last_use_step), so that low-priority/expired entries are evicted before high-priority ones (e.g., system-prompt media, recently-reused tool media, active-session media) regardless of insertion order. (4) Update `get_cached_input_ids` / `try_reuse_cached` paths to bump last_use_step on cache hits (LRU-on-hit), and let the same priority comparator cover the LRU case as a tiebreaker. (5) Keep behavior equivalent to today when no hints are provided (default priority class, no TTL) so tests/v1/core/test_encoder_cache_manager.py hit/free/eviction/capacity invariants continue to hold. The existing `freed` list, `num_free_slots`, and `num_freeable_slots` accounting are unchanged.

**Proposal rationale.**

The candidate's gap is exactly what the finding addresses: today's eviction sees only insertion order in `freeable` and is blind to recency-of-reuse, expected near-future reuse, and per-entry importance. TensorRT-LLM's priority-based KV eviction shows that letting the producer of a cached entry attach a retention priority and duration is a low-overhead way to preserve the entries most likely to be reused. In a multi-turn agentic workload, encoder embeddings for system-prompt images, persistent agent-role media, and recently-referenced tool media are the high-value entries; one-off media from a single tool call are the cheap victims. Mapping that hierarchy onto the existing freeable structure - without changing the cached/freeable/freed contract - directly attacks media TTFT (avoid recomputing expensive vision embeddings on the next turn) and indirectly TPOT (less encoder recomputation contention with decode), which matches the caller's stated objective.

---

### 5. Scheduler-aware encoder cache eviction using upcoming-request mm_hash lookahead
- **Finding:** `find-0009` — *AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving*
- **Source URL:** <https://arxiv.org/html/2403.19708v2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the strict FIFO eviction in EncoderCacheManager.can_allocate (vllm/v1/core/encoder_cache_manager.py:119-178) with a scheduler-aware policy analogous to AttentionStore's scheduler-driven fetch/eviction. Concretely, when the while num_embeds > self.num_free_slots loop needs to free space, do not blindly call self.freeable.popitem(last=False); instead consult a lookahead set of mm_hash values that appear in waiting/queued requests (passed in or queryable from the scheduler that already owns EncoderCacheManager). Partition self.freeable entries into (a) entries whose mm_hash is referenced by an upcoming request in the near horizon — protected, evicted last and only if necessary — and (b) entries with no upcoming reference — evicted first, oldest-first within that group. Preserve the existing cached/freeable/freed accounting and free()/free_encoder_input() interface so test_encoder_cache_manager.py invariants for hit, free, eviction, and capacity continue to hold. Optionally, mirror AttentionStore's tiering by demoting evicted-but-soon-needed entries to a small host-side staging dict (CPU tensor copy) instead of fully discarding, with an async copy-back path used by the scheduler when the protected entry is actually requested again. Keep the policy switchable via a config flag so the FIFO behavior remains the default fallback.

**Proposal rationale.**

The candidate's gap is that freeable.popitem(last=False) ignores any signal about near-future reuse, which is exactly the lever AttentionStore exploits for KV caches: it uses scheduler hints to decide what to evict and what to keep hot. In vLLM's multi-turn agentic workload the scheduler already knows which mm_hash values are pending in queued requests (the same hashes drive cache hit/miss in EncoderCacheManager.has_cache and get_cached_input_ids), so feeding that information into the eviction loop is a direct, low-risk transfer of the finding's core idea. It targets the candidate's stated lever — encoder cache hit rate as a media-TTFT driver under repeated images/tool-media — without changing the cache's external accounting, and the optional host-tier demotion mirrors AttentionStore's hierarchy to recover hits that would otherwise force re-running the vision encoder.

---

### 6. SLO/cost-aware encoder cache eviction inspired by Mooncake's KVCache-centric scheduler
- **Finding:** `find-0011` — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*
- **Source URL:** <https://arxiv.org/abs/2407.00079>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify EncoderCacheManager.can_allocate (vllm/v1/core/encoder_cache_manager.py:119-178) so that the eviction loop over self.freeable no longer pops strictly the oldest unreferenced entry via freeable.popitem(last=False). Instead, score each freeable entry by an SLO/cost-aware key analogous to Mooncake's KVCache-centric scheduler, then evict the lowest-value entry first until num_embeds <= self.num_free_slots. The score should combine: (a) recomputation cost proxy (entry's num_encoder_tokens, since larger vision embeddings dominate TTFT when recomputed), (b) recency of last reuse (track a last_used counter updated on cache hits in EncoderCacheManager.get_cached_input_ids / has_cache; today freeable only tracks insertion order), and (c) a coarse reuse-likelihood signal derivable from the existing cached/freeable/freed accounting (e.g., entries that have already been hit at least once before being freed are protected over single-use entries, similar to a 2Q/protected-vs-probationary split). Keep the public interface (cached, freeable, freed dicts; can_allocate / allocate / free / get_cached_input_ids signatures) unchanged so tests/v1/core/test_encoder_cache_manager.py continues to enforce hit/free/eviction/capacity invariants. The eviction policy switch should be selected by a single config knob with the current FIFO behavior preserved as the default, and the cost-aware path activated for multimodal agentic workloads where repeated images and tool-media exhibit reuse.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out LRU-on-hit, LFU, size-aware, and protected/probationary queues as headroom over the current FIFO-on-freeable policy, and identifies multimodal-agentic reuse as the workload where encoder-cache hit rate becomes a direct TTFT lever. Mooncake's central contribution — making cache scheduling decisions aware of cost and SLO risk rather than only local free-block count — maps directly onto this gap: the encoder cache today evicts by insertion order regardless of how expensive the cached embedding was to produce or how likely it is to be reused. Treating recomputation cost (num_encoder_tokens) and reuse history as first-class eviction signals is the encoder-cache analogue of Mooncake's KVCache-centric, SLO-aware admission/eviction, and addresses exactly the TTFT objective stated in the caller context for multi-turn agentic workloads.

---

### 7. Adopt LRU-on-hit eviction with optional CPU-side spillover for the encoder cache
- **Finding:** `find-0014` — *Embedding Cache*
- **Source URL:** <https://docs.dynamo.nvidia.com/dynamo/user-guides/multimodal/embedding-cache>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Change the eviction policy in EncoderCacheManager.can_allocate (vllm/v1/core/encoder_cache_manager.py:119-178) so that the freeable OrderedDict is ordered by recency of last use rather than insertion order. Concretely: on every cache hit (when an existing entry is referenced again), move the entry to the MRU end of freeable when it is later released back to freeable, so that the existing `freeable.popitem(last=False)` call evicts the true LRU entry instead of the oldest-inserted freeable entry. As an optional extension informed by Dynamo's design, allow evicted entries to be demoted to a bounded CPU-side host tier rather than dropped outright; on a subsequent request for the same (request_id, input_id) media hash, the manager would rehydrate the embedding from host memory (via a copy analogous to Dynamo's NIXL transfer) instead of forcing the encoder worker to recompute it. The cached/freeable/freed bookkeeping interface and capacity invariants (covered by tests/v1/core/test_encoder_cache_manager.py) are preserved; only the ordering discipline of freeable changes, plus an opt-in host-tier shadow map keyed by media hash.

**Proposal rationale.**

The candidate's current FIFO-over-insertion-order eviction discards entries without regard to reuse recency, which is exactly the gap LRU addresses. Dynamo's embedding cache demonstrates that vision encoder outputs are a high-value reuse target in multimodal/agentic workloads, where the same images and tool-media reappear across turns; an LRU-on-hit ordering for freeable directly converts repeated-media references into avoided encoder recompute, lowering media TTFT. The host-tier spillover idea from Dynamo further extends the cache's effective capacity beyond GPU budget without changing the GPU-side allocation accounting, which is well-aligned with a multi-turn agentic objective and complements (rather than replaces) the existing freeable/freed flow in can_allocate.

---

## Agent proposals

### 1. Adopt ARC (Adaptive Replacement Cache) with ghost-list learning for encoder cache eviction
- **Agent:** claude

**Detailed description.**

Replace the single self.freeable OrderedDict in EncoderCacheManager (vllm/v1/core/encoder_cache_manager.py:119-178) and the strict `freeable.popitem(last=False)` eviction in can_allocate with a four-queue ARC structure that self-tunes between recency and frequency without any caller-supplied hints. Concretely: (1) Maintain T1 (freeable entries observed exactly once) and T2 (freeable entries observed 2+ times), each as an OrderedDict keyed by mm_hash. Their union replaces today's self.freeable; the cached/freed dicts and num_free_slots accounting are unchanged so test_encoder_cache_manager.py invariants still hold. (2) Maintain two bounded ghost lists B1 and B2 holding ONLY the mm_hash strings (no embeddings, trivial memory) of entries recently evicted from T1 and T2 respectively, capped at the cache's slot capacity. (3) Maintain an adaptive parameter p ∈ [0, capacity] that controls the target split between T1 and T2. On each cache hit observed via get_cached_input_ids/has_cache that lands on a freeable entry, promote it from T1→T2 (or refresh its position at T2's MRU end if already in T2). On a fresh allocation whose mm_hash hits B1, increase p by max(1, |B2|/|B1|) — recency tier is being under-served. On a hit to B2, decrease p by max(1, |B1|/|B2|) — frequency tier is being under-served. (4) In the eviction loop inside can_allocate, replace the unconditional popitem with: if |T1| > 0 and (|T1| > p or (mm_hash being admitted is in B2 and |T1| ≥ p)), evict the LRU end of T1 and push its hash to B1; otherwise evict the LRU end of T2 and push its hash to B2. Trim B1/B2 from their LRU ends to keep their combined size ≤ capacity. (5) Default to today's behavior under cold start: until any ghost-list hit occurs, T1 grows alone and behavior reduces to LRU-over-insertion order, preserving baseline correctness. Existing tests pin down hit/free/eviction/capacity invariants; new tests verify the T1↔T2 promotion on reuse, ghost-list-driven adjustment of p, and that capacity is never exceeded.

**Novelty rationale.**

None of the seven listed deep_research_proposals propose a self-tuning policy that learns from misses to recently-evicted entries. find-0002 (TTL), find-0003 (workflow metadata), find-0008 (priority hints), find-0009 (scheduler lookahead) all rely on externally supplied signals; find-0004 and find-0014 are pure LRU-on-hit with no frequency tier; find-0011 (Mooncake) blends recency+cost+reuse-history into a static scoring function and explicitly only mentions a '2Q-like' protected/probationary split. ARC's distinguishing mechanisms — the two ghost lists (B1, B2) tracking only mm_hash strings of recently-evicted entries, and the adaptive parameter p that shifts capacity between recency and frequency tiers based on which ghost list gets re-referenced — are absent from every listed proposal. This makes the cache self-tune to whichever signal (recency vs. frequency) actually predicts reuse in the current multi-turn agentic workload, with zero caller-side instrumentation and zero scheduler coupling.

---

### 2. Make freeable eviction occurrence-aware for repeated media
- **Agent:** codex

**Detailed description.**

Change EncoderCacheManager so can_allocate never evicts a hash that is still needed by a later occurrence of the same multimodal item in an active request. Today self.cached is keyed as mm_hash -> set[request_id], while free_encoder_input is called per input_id; if one request contains the same mm_hash at multiple positions, freeing the first occurrence can make the entry freeable even though the same request will need it again. Replace the request-id set with per-occurrence refs, e.g. dict[str, set[tuple[str, int]]], or an equivalent per-request counter initialized from the current and future mm_features with the same identifier when allocate/check_and_update_cache acquires the cache entry. free_encoder_input should remove only that occurrence reference and add the hash to self.freeable only when no occurrence refs remain. The eviction loop in can_allocate can then stay policy-pluggable, but it will pop only truly unreferenced hashes. Add a regression in tests/v1/core/test_encoder_cache_manager.py with one request containing two input_ids sharing an identifier at different positions: after freeing the first input, a pressure allocation must not evict that hash; after freeing the second, it becomes freeable and is evictable.

**Novelty rationale.**

The listed deep_research_proposals and Claude's ARC proposal all optimize which already-freeable victim to choose, using TTLs, workflow or scheduler hints, LRU/LFU-style scoring, priority, host spillover, or ghost-list adaptation. This proposal is orthogonal: it changes the accounting that decides whether an entry is freeable at all, fixing a per-request duplicate-media case where request-id-level refs are too coarse. It does not introduce a new eviction ranking policy and can compose with any of the proposed policies.

---
