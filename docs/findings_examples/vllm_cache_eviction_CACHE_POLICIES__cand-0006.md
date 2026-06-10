# _CACHE_POLICIES

[← v1.kv_offload](../v1.kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/manager.py`](vllm/v1/kv_offload/cpu/manager.py) (lines 19–22)
- **Symbol:** `_CACHE_POLICIES`
- **Kind:** plugin_seam
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0006`

## Description
Registers cache replacement policies available to CPUOffloadingManager by mapping external policy names to CachePolicy implementations.

## Current approach
The dict maps "lru" to LRUCachePolicy in vllm/v1/kv_offload/cpu/policies/lru.py and "arc" to ARCCachePolicy in vllm/v1/kv_offload/cpu/policies/arc.py. Both implement the CachePolicy interface in vllm/v1/kv_offload/cpu/policies/base.py. The runtime selector is kv_connector_extra_config["eviction_policy"], read by CPUOffloadingSpec in vllm/v1/kv_offload/cpu/spec.py before constructing CPUOffloadingManager.

## Estimated impact explanation
Replacement policy controls CPU-cache hit rate, which is the main workload-level signal for avoiding recompute or remote misses; a better policy can reduce end-to-end latency more than micro-optimizing a fixed policy.

## Evolve rationale
This is a real plugin seam with a stable interface: get, insert, remove, touch, and evict. New policies such as SLRU, 2Q, S3-FIFO, TinyLFU admission plus LRU eviction, or weighted ARC variants can be added as sibling implementations plus one dict entry. Correctness oracle: CachePolicy.evict's atomic contract in vllm/v1/kv_offload/cpu/policies/base.py and the existing LRU/ARC manager tests cover lookup, touch, prepare_store, complete_store, ref-count protection, and eviction ordering invariants.

## Deep research proposals

### 1. Add a future-aware CachePolicy that biases eviction by predicted next-use distance
- **Finding:** `find-0002` — *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows*
- **Source URL:** <https://arxiv.org/abs/2507.07400>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new CachePolicy implementation (e.g., FutureAwareCachePolicy in vllm/v1/kv_offload/cpu/policies/future_aware.py) and register it in the _CACHE_POLICIES dict at vllm/v1/kv_offload/cpu/manager.py:19-22 (e.g., key "future_aware" or "kvflow"). The policy implements the existing CachePolicy interface from vllm/v1/kv_offload/cpu/policies/base.py (get, insert, remove, touch, evict) but maintains, per cached block/prefix, a cheap reuse-distance estimate derived from observed access patterns (e.g., per-block EWMA of inter-access gap, or per-prefix-stem reuse counter updated on touch/get). On evict(), instead of strict recency (LRU) or recency+frequency balance (ARC), it returns the candidate with the largest predicted next-use distance among entries whose ref-count is zero, preserving the atomic eviction contract and ref-count protection invariants already covered by the LRU/ARC manager tests. Selection of the new policy is wired exclusively through the existing kv_connector_extra_config["eviction_policy"] string read by CPUOffloadingSpec in vllm/v1/kv_offload/cpu/spec.py; no other call sites change. Prefetching is explicitly out of scope for this seam (the CachePolicy interface has no load hook), so this proposal targets only the eviction-bias half of KVFlow's idea.

**Proposal rationale.**

The candidate is a plugin seam whose stable contract is exactly the eviction decision, and the finding's transferable contribution is a concrete signal (predicted future reuse) that can replace the recency/frequency proxy used by LRU and ARC. For multi-agent or repeated-prompt workloads where the same prefixes recur on a roughly periodic schedule, recency-only signals systematically evict soon-to-be-reused entries; a future-aware score directly addresses that gap while reusing the existing interface, tests, and selector, so the change is bounded and the correctness oracle already exists.

---

### 2. Add learned conversation-continuation policy to CPU offload _CACHE_POLICIES
- **Finding:** `find-0004` — *Learned Prefix Caching for Efficient LLM Inference*
- **Source URL:** <https://openreview.net/forum?id=Vj48eXaQDM>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Implement a new CachePolicy subclass (e.g., LearnedPrefixCachePolicy) under vllm/v1/kv_offload/cpu/policies/ that combines a lightweight reuse-likelihood predictor with recency for eviction decisions, then register it in the _CACHE_POLICIES dict at vllm/v1/kv_offload/cpu/manager.py:19-22 (e.g., "learned" -> LearnedPrefixCachePolicy). The policy keeps the existing CachePolicy contract from vllm/v1/kv_offload/cpu/policies/base.py: get/insert/remove/touch/evict with the atomic evict semantics and the protected ref-counted block guard. Internally, on touch/insert it would update a per-block score derived from a small predictor over recent OffloadKey context (e.g., conversation/turn metadata or block-hash history already available to the offload manager), and evict() would pick the n lowest-scoring unprotected blocks (score = w*recency + (1-w)*predicted_reuse_prob), maintaining the same atomic all-or-nothing return shape as LRU/ARC. Selection remains driven by kv_connector_extra_config["eviction_policy"] in vllm/v1/kv_offload/cpu/spec.py, so no plumbing changes are needed beyond the new sibling file plus one dict entry. Existing manager tests covering lookup, touch, prepare_store, complete_store, ref-count protection, and atomic eviction ordering serve as the correctness oracle; add a focused test that the predictor scoring path still respects the protected set and atomicity.

**Proposal rationale.**

The candidate is explicitly a plugin seam for cache replacement policies, and the finding contributes a concrete, transferable eviction idea — score cached prefixes by predicted reuse probability and combine with recency — that targets exactly the metric (host-cache hit rate on multi-turn chat) the candidate's evolve_rationale calls out as the dominant latency lever. It slots in as a sibling of LRU/ARC behind the same CachePolicy interface with no changes to the manager's atomicity or ref-count invariants, addressing the gap that pure recency (LRU) and frequency-recency (ARC) ignore semantic signals available in conversational workloads. Conservative scope (one new policy file + one dict entry, opt-in via existing config key) makes the risk small relative to the potential hit-rate improvement.

---

### 3. Add a TinyLFU-admission cache policy alongside LRU/ARC
- **Finding:** `find-0005` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://arxiv.org/abs/1512.00727>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new sibling policy module (e.g., vllm/v1/kv_offload/cpu/policies/tinylfu.py) implementing CachePolicy from vllm/v1/kv_offload/cpu/policies/base.py, and register it in the _CACHE_POLICIES dict at vllm/v1/kv_offload/cpu/manager.py:19-22 (e.g., "tinylfu": TinyLFUCachePolicy). The new class wraps an LRU recency structure (reusing the same doubly-linked-list/dict shape as LRUCachePolicy in vllm/v1/kv_offload/cpu/policies/lru.py) for replacement, and adds a TinyLFU admission filter: a small Count-Min Sketch of OffloadKey frequencies plus a doorkeeper Bloom filter, periodically aged via the W-TinyLFU reset/halving step. On every get/touch, increment the sketch for that key. On insert when the cache is full, compare the candidate's estimated frequency against the LRU eviction victim's estimate and only admit (insert + evict) when the candidate's frequency is at least the victim's; otherwise skip the insert so prepare_store treats it as a single-use block. The evict() contract from vllm/v1/kv_offload/cpu/policies/base.py is preserved: TinyLFUCachePolicy.evict still returns exactly n victims atomically by delegating to its internal LRU ordering. Sketch width/depth and aging interval are configured via kv_connector_extra_config in vllm/v1/kv_offload/cpu/spec.py, with conservative defaults sized as a small fraction of cache_capacity so memory overhead is sub-percent. Existing manager tests for lookup/touch/prepare_store/complete_store/ref-count protection continue to apply unchanged; admission-specific tests cover one-hit-wonder rejection and frequency-aging behavior.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out "TinyLFU admission plus LRU eviction" as a target policy, and find-0005 supplies the concrete mechanism: replace exact reuse counters with an approximate frequency sketch so admission decisions are cheap on the hot path. CPU offload caches in LLM serving see many one-hit blocks (long-tail prompts, sampling-divergent suffixes) that pollute an LRU/ARC cache and evict reusable hot blocks; TinyLFU's admission filter directly targets that pollution, which is exactly the workload-level hit-rate lever the candidate's estimated_impact identifies. The CachePolicy interface (get/insert/touch/evict) already exposes the right hooks for an admission policy: frequency updates on get/touch, an admission decision on insert, and atomic n-victim eviction via LRU. Adding it as a sibling implementation plus one _CACHE_POLICIES entry is consistent with the existing plugin seam and does not perturb LRU or ARC code paths.

---

### 4. Add SIEVE as a third CachePolicy implementation behind the _CACHE_POLICIES seam
- **Finding:** `find-0006` — *SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for Web Caches*
- **Source URL:** <https://yazhuozhang.com/assets/publication/nsdi24-sieve.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Implement a new SIEVECachePolicy in vllm/v1/kv_offload/cpu/policies/sieve.py that conforms to the CachePolicy interface defined in vllm/v1/kv_offload/cpu/policies/base.py (get, insert, remove, touch, evict, plus the prepare_store/complete_store ref-count protocol used by LRU/ARC). The implementation should follow the SIEVE design from the paper: a single FIFO-ordered queue of entries augmented with a per-entry 'visited' bit and a moving 'hand' pointer. On hit, touch() sets the visited bit in place rather than relinking the entry, eliminating per-hit list mutations (the lazy-promotion property the paper highlights). On evict(), the hand walks backward from its current position, clearing visited bits until it finds an unvisited, non-pinned (ref-count safe) entry, which is then evicted; this preserves the atomic eviction contract documented on CachePolicy.evict. Then register the new policy at vllm/v1/kv_offload/cpu/manager.py:19-22 by adding a single line to _CACHE_POLICIES (e.g. "sieve": SIEVECachePolicy), so it becomes selectable via kv_connector_extra_config["eviction_policy"] in CPUOffloadingSpec (vllm/v1/kv_offload/cpu/spec.py) with no other call-site changes. Reuse the existing LRU/ARC manager test suite as the correctness oracle for the shared invariants (lookup, touch, prepare_store/complete_store, ref-count protection, eviction ordering against pinned entries), and add SIEVE-specific tests covering the visited-bit/hand semantics.

**Proposal rationale.**

The candidate is explicitly a plugin seam designed to admit new CachePolicy implementations with one dict entry, and the evolve_rationale lists SIEVE-class admission/eviction policies as exactly the kind of sibling addition it is meant to support. The finding contributes a concrete, transferable algorithm (lazy promotion via a visited bit + hand pointer) whose central claim - lower per-hit mutation cost than LRU at comparable or better hit rates on skewed workloads - directly targets the caller's stated objective of reducing hot-path latency on common workloads, since touch() is invoked on every CPU-offload cache hit. SIEVE's interface fits the existing CachePolicy contract without changes to manager.py, spec.py, or the wider connector, so the change is bounded and the existing eviction-ordering and ref-count tests serve as a correctness oracle. This is a new, distinct policy rather than a restatement of LRU or ARC, so it meaningfully extends the seam rather than being topically adjacent.

---

## Agent proposals

### 1. Add S3-FIFO as a third CachePolicy with quick-demotion small/main/ghost queues
- **Agent:** claude

**Detailed description.**

Implement S3FIFOCachePolicy in a new vllm/v1/kv_offload/cpu/policies/s3fifo.py conforming to the CachePolicy interface in vllm/v1/kv_offload/cpu/policies/base.py (get, insert, remove, touch, evict, plus the prepare_store/complete_store ref-count protocol used by LRU and ARC), and register it at vllm/v1/kv_offload/cpu/manager.py:19-22 with one new entry (e.g., "s3fifo": S3FIFOCachePolicy). The implementation follows the S3-FIFO design (Yang et al., SOSP 2024): three FIFO data structures - a Small queue sized at roughly 10% of cache_capacity, a Main queue holding the remaining ~90%, and a Ghost queue of evicted-key fingerprints sized like Main. Each cached entry carries a small saturating counter (0..3). On insert, if the key is in Ghost it enters Main with counter 0; otherwise it enters Small with counter 0. touch() simply increments the per-entry counter (saturating at 3) without relinking - this is the lazy-promotion property that keeps hot-path mutation O(1) and avoids the per-hit list splicing that LRU/ARC perform. evict() reclaims n victims atomically by repeatedly: (a) if Small is over its quota, pop its head; if counter > 0 promote it to Main with counter reset to 0, else evict it and add its fingerprint to Ghost; (b) otherwise pop Main's head and re-enqueue it with counter decremented while counter > 0, evicting only when counter reaches 0. Pinned (ref-count > 0) entries are skipped over and re-enqueued, preserving the protected-block invariant; the n-victim atomic contract on CachePolicy.evict is honored by buffering victims and returning them as a unit. Ghost stores only key fingerprints (no value bytes), with Bloom-filter-style truncation so memory overhead stays sub-percent of cache_capacity. The Small/Main split ratio and Ghost size are read from kv_connector_extra_config in vllm/v1/kv_offload/cpu/spec.py (with sane defaults), reusing the same selector path as LRU/ARC. Existing LRU/ARC manager tests (lookup, touch, prepare_store, complete_store, ref-count protection, eviction ordering) serve as the correctness oracle; add S3-FIFO-specific tests for the quick-demotion property (one-hit-wonders evicted from Small without polluting Main) and Ghost-promotion (re-inserted-after-evict goes to Main, not Small).

**Novelty rationale.**

S3-FIFO is explicitly named in the candidate's evolve_rationale alongside SLRU and 2Q but is not the subject of any listed deep_research_proposal. It is structurally distinct from the four existing proposals: (1) find-0002 (future-aware) bases eviction on a learned reuse-distance estimate, whereas S3-FIFO uses a small saturating counter and quick demotion with no prediction; (2) find-0004 (learned conversation policy) wires a predictor over OffloadKey context, while S3-FIFO is a pure FIFO-and-counter algorithm with no learned signal; (3) find-0005 (TinyLFU) uses a Count-Min Sketch + Bloom doorkeeper as an admission filter on top of LRU, whereas S3-FIFO has no sketch and no admission filter - it gets one-hit-wonder rejection from the Small queue's quick-demotion behavior plus the Ghost queue, an architecturally different mechanism; (4) find-0006 (SIEVE) is a single-queue + visited-bit + hand-pointer design, while S3-FIFO is a three-queue (Small/Main/Ghost) design with multi-bit counters and a Ghost-driven promotion path that SIEVE has no analog of. The Ghost-queue mechanism in particular - using fingerprints of recently evicted keys to detect that a key was wrongly demoted and route it directly to Main on re-insert - is unique to this proposal among the listed alternatives, and it directly targets the KV-offload-specific failure mode where a prefix briefly stops being touched, gets evicted, then is needed again moments later.

---

### 2. Add a segmented-LRU CachePolicy with probationary and protected queues
- **Agent:** codex

**Detailed description.**

Implement an SLRUCachePolicy sibling under vllm/v1/kv_offload/cpu/policies/slru.py and register it in _CACHE_POLICIES at vllm/v1/kv_offload/cpu/manager.py:19-22, for example "slru": SLRUCachePolicy. The policy keeps two LRU-ordered segments: a probationary segment for newly inserted OffloadKey entries and a protected segment for entries that are touched after admission. insert() places new blocks in probationary; touch()/get() promotes probationary hits to protected and refreshes protected hits in place by moving them to the protected MRU end. If protected exceeds its configured target share, demote its LRU entry back to probationary rather than evicting it immediately. evict(n) selects victims from the probationary LRU tail first, falling back to the protected LRU tail only when needed, while skipping ref-count-protected entries and preserving the existing atomic all-or-nothing CachePolicy.evict contract. The protected/probationary split should default to a conservative fixed ratio such as 80/20 and be optionally configurable through the same kv_connector_extra_config path used by CPUOffloadingSpec for eviction_policy. Reuse the existing LRU/ARC manager tests for lookup, touch, prepare_store, complete_store, pinned-block protection, and atomic eviction; add SLRU-specific tests for first-hit promotion, protected overflow demotion, scan resistance, and fallback eviction when probationary entries are pinned.

**Novelty rationale.**

This is not covered by the listed deep_research_proposals: future-aware and learned policies use prediction or semantic reuse scoring; TinyLFU adds approximate frequency admission with sketches; SIEVE uses a single FIFO queue with visited bits and a hand pointer. It is also distinct from Agent A's S3-FIFO proposal: S3-FIFO uses Small/Main/Ghost FIFO queues, saturating counters, and ghost-driven re-entry, while SLRU uses only two exact LRU segments and promotion/demotion between probationary and protected queues. The actionable change is a simpler non-learned policy that targets the common LRU failure mode of scan pollution without the sketch overhead of TinyLFU or the ghost-queue mechanics of S3-FIFO.

---
