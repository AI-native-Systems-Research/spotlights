# LRUCachePolicy.evict

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/policies/lru.py`](vllm/v1/kv_offload/cpu/policies/lru.py) (lines 56–78)
- **Symbol:** `LRUCachePolicy.evict`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0003`

## Description
LRU batch eviction scans evictable blocks from the oldest end, skipping protected keys until n victims are collected.

## Current approach
Uses pure oldest-first victim selection over evictable_blocks, with only the protected-key filter. It returns None atomically if fewer than n non-protected candidates are available.

## Estimated impact explanation
Agentic multi-turn workloads often reuse prompt/prefix blocks across turns, and pure recency can evict those too aggressively. Better victim scoring can improve primary-tier hit rate and reduce promotion stalls that affect median TTFT.

## Evolve rationale
LRU is a fixed victim heuristic that can be replaced or augmented with light frequency, ghost-list, age-decay, or request-locality signals while preserving the CachePolicy API. Correctness oracle: existing cache-policy tests, capacity bounds, protected-key exclusion, and all evicted blocks having ref_cnt == 0.

## Deep research proposals

### 1. Priority-aware LRU eviction using agent-supplied retention hints
- **Finding:** `find-vllm_v1_kv_offload-0002` — *Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo*
- **Source URL:** <https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend LRUCachePolicy so evict() at vllm/v1/kv_offload/cpu/policies/lru.py:56-78 no longer scans evictable_blocks in pure oldest-first order. Introduce an optional per-key retention tier (e.g. low / normal / pinned) that the policy captures when insert() and touch() run: touch(keys, req_context) already receives ReqContext, so parse retention hints out of req_context.kv_transfer_params (and/or ReqContext._state populated in on_new_request) and record the tier alongside the key — either by keeping one OrderedDict per tier or by storing the tier on BlockStatus and grouping the scan. In evict(n, protected), walk tiers from lowest-value to highest-value, and within each tier keep the current oldest-first LRU order; still skip keys in protected, still assert block.ref_cnt == 0, still return None atomically (no state mutation) when fewer than n non-protected candidates exist across all tiers combined, and still delete from evictable_blocks and blocks only after n victims are collected. 'Pinned' keys behave like an implicit protected set for eviction while remaining evictable if pressure demands (configurable). Wire a small OffloadingSpec extra_config flag (e.g. retention_hints_enabled) so behavior is opt-in and the existing cache-policy tests, capacity bounds, and protected-key exclusion invariants continue to pass unchanged when the flag is off.

**Proposal rationale.**

The candidate's own evolve_rationale explicitly calls out request-locality signals as a substitute for pure recency, and the finding supplies a concrete transferable mechanism from Dynamo: let the agent harness declare which blocks are high-value for upcoming turns/tool calls and evict lower-priority blocks first. The signal channel already exists (ReqContext.kv_transfer_params is threaded into touch()), and CachePolicy is documented to co-own organization and replacement, so adding a tiered scan preserves the API and atomic semantics of evict(). The gap it addresses is specific to the caller objective — multi-turn agentic workloads reuse prompt/prefix blocks across turns; pure oldest-first evicts them under pressure, forcing storage-tier promotions that inflate median TTFT and stall decode (TPOT). Priority-aware victim selection keeps those blocks resident on the primary CPU tier without changing capacity bounds or the protected-key contract.

---

### 2. Add S3-FIFO cache policy as an alternative to LRUCachePolicy.evict for multi-turn prefix protection
- **Finding:** `find-vllm_v1_kv_offload-0004` — *FIFO queues are all you need for cache eviction*
- **Source URL:** <https://s3fifo.com/blog/2023/08/01/fifo-queues-are-all-you-need-for-cache-eviction/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new `S3FIFOCachePolicy` implementing the existing `CachePolicy` API (vllm/v1/kv_offload/cpu/policies/base.py) as a sibling of `LRUCachePolicy` at vllm/v1/kv_offload/cpu/policies/lru.py:12-91 and `ARCCachePolicy`. Register it in `CachePolicyFactory` (vllm/v1/kv_offload/cpu/policies/factory.py:85-90) under name `s3fifo`. Internally, replace the single `evictable_blocks: OrderedDict` used for eviction (lru.py:24, 56-78) with three FIFOs: a small probationary FIFO `S` (~10% of `cache_capacity`), a main FIFO `M` (~90% of `cache_capacity`), and a ghost FIFO `G` sized to `cache_capacity` that stores only keys (no `BlockStatus`). Add a tiny saturating 2-bit hit counter to `BlockStatus`-adjacent metadata (a parallel `dict[OffloadKey, int]` keyed by key, capped at 3). `insert` places new keys into `S` unless the key is in `G`, in which case it goes directly into `M` (ghost hit ⇒ likely reused prefix). `touch` increments the counter (saturating). `mark_evictable`/`mark_non_evictable` keep membership in `S`/`M` in sync. `evict(n, protected)` mirrors the current atomicity contract at lru.py:56-78 — build a candidate list first, only mutate on success, return `None` if fewer than `n` non-protected candidates exist — but selects victims by: pop the oldest from `S`; if its counter > 0, promote it into `M` and reset the counter; if 0, evict it and record the key in `G` (evicting oldest from `G` if full). When `S` is empty, pop from `M`; if counter > 0, decrement and re-enqueue at the tail of `M`; if 0, evict. `protected` keys are skipped (re-enqueued at the tail of their current FIFO) exactly as LRU does today. `clear` resets all three FIFOs and the counter dict. The existing correctness oracle (capacity bounds, protected-key exclusion, evicted blocks having `ref_cnt == 0`) is preserved.

**Proposal rationale.**

The candidate's `evolve_rationale` explicitly calls out `ghost-list` and `light frequency` signals as directions to explore beyond pure-recency LRU, and its `estimated_impact_explanation` identifies the exact failure mode S3-FIFO was designed for: hot multi-turn prefixes being evicted under scan-like pressure from one-hit blocks. S3-FIFO's small probationary FIFO quickly demotes one-hit blocks before they can pollute the main region, while the ghost FIFO detects returning multi-turn prefixes and promotes them into the main FIFO on the next store — directly targeting `primary-tier hit rate` and `promotion stalls that affect median TTFT` cited by the caller. Compared to ARC (already in-tree), S3-FIFO uses only append/pop-front FIFO operations and 2-bit counters, so its per-`evict` cost stays comparable to the current `OrderedDict` scan at lru.py:63-71 while adding no cross-list adaptive state. The `CachePolicy` interface and factory already support pluggable policies, so the change is additive and gated behind the `s3fifo` policy name — LRU remains the default.

---

### 3. Add TinyLFU frequency-sketch admission gate to CPU-tier LRU policy
- **Finding:** `find-vllm_v1_kv_offload-0005` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://paperity.org/p/377179711/tinylfu-a-highly-efficient-cache-admission-policy>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Augment LRUCachePolicy in vllm/v1/kv_offload/cpu/policies/lru.py with an approximate frequency sketch (e.g., a Count-Min Sketch paired with a small doorkeeper Bloom filter, as in TinyLFU) that tracks recent access frequency per OffloadKey. Update calls at the natural admission and use points already exposed by CachePolicy: increment the sketch on insert() and on touch()/load-completion so it captures both new arrivals and reuse. Change the eviction path at lines 56-78 so that when the LRU end is selected for eviction to make room for a new admission, the incoming block's estimated frequency is compared against the frequency of the oldest non-protected evictable candidate; if the incoming block's frequency is lower, admission is rejected (or the LRU victim is retained) instead of unconditionally evicting the LRU tail. Preserve the existing atomicity guarantee: evict still returns None if n non-protected candidates cannot be found, still asserts ref_cnt == 0 on victims, and still respects the protected set. Age the sketch periodically (halve counters after W accesses) to bound memory and preserve the recency signal that the current LRU order encodes. The CachePolicy API surface is unchanged; only LRUCachePolicy internals and its evict/insert semantics gain the frequency-aware gate.

**Proposal rationale.**

The candidate's own evolve_rationale calls out "light frequency" and ghost-list signals as the intended axis of improvement over pure oldest-first LRU, and the caller context is a multi-turn agentic workload where the same prompt/prefix blocks recur across turns. Pure recency evicts these popular prefixes whenever a burst of one-off blocks sweeps past them, causing repeated re-promotions from the CPU tier that add to TTFT. TinyLFU directly addresses this: its frequency sketch is O(1) per access, memory-cheap, and the paper's explicit contribution is deciding "whether it is worth admitting the new item" by comparing frequencies — exactly the gap between pure LRU victim selection and workload-aware admission. The change is scoped to a single policy class behind an existing abstract interface, so the correctness oracle listed on the candidate (capacity bounds, protected-key exclusion, ref_cnt == 0 on victims, atomic None-on-failure) remains verifiable with the current cache-policy tests.

---

## Agent proposals

### 1. Add cross-request prefix-aware eviction using PrefixCachingMetrics signal for shared-prefix protection
- **Agent:** claude

**Detailed description.**

Modify LRUCachePolicy.evict at vllm/v1/kv_offload/cpu/policies/lru.py:56-78 to consult a lightweight cross-request prefix-sharing counter when selecting victims, instead of pure oldest-first scanning of evictable_blocks. Extend LRUCachePolicy with a `shared_prefix_count: dict[OffloadKey, int]` populated in `insert()` and `touch()`: each time a key is inserted/touched, increment a counter that represents how many distinct request_ids (from ReqContext.request_id, already threaded into touch()) have referenced this block over a bounded sliding window (e.g., a small ring buffer of last-seen request_ids per key, capped at 4 slots). Modify evict() to make two passes over evictable_blocks in oldest-first order: first pass collects up to n victims with shared_prefix_count <= 1 (single-request blocks — safe to evict, typically decode tail); second pass, only if the first pass came up short, collects the remainder from shared_prefix_count >= 2 blocks in oldest-first order. Preserve every existing invariant: skip keys in `protected`, assert `block.ref_cnt == 0` on selected victims, return None atomically without mutating state when fewer than n non-protected candidates exist across both passes, and delete from `evictable_blocks` and `blocks` only after all n victims are gathered. On `clear()`, also clear `shared_prefix_count` and the ring buffers. Gate the two-pass behavior behind an OffloadingSpec `extra_config` flag (e.g., `cross_request_prefix_bias=true`) so the default behavior — and existing cache-policy tests — remain byte-identical when the flag is off.

**Novelty rationale.**

The three listed deep_research_proposals all use *within-key* signals: (1) explicit agent-supplied retention tiers passed via kv_transfer_params, (2) S3-FIFO's per-key hit counter and ghost list based on that key's own access pattern, and (3) TinyLFU's per-key frequency sketch. None of them use the *cross-request* fan-out of a block — how many distinct request_ids have touched it — as the eviction signal. This proposal keys off request_id (already available on ReqContext) rather than access frequency or explicit hints: a block referenced by 3 different requests is a shared prompt/system-prefix block regardless of its recency or absolute hit count, whereas a block touched 20 times by a single long-running request has high frequency but zero cross-request reuse value. This distinguishes shared prefixes (the exact multi-turn agentic reuse pattern in the caller context) from single-request hot spots — a discrimination none of the existing proposals make, since frequency counters, ghost hits, and retention tiers all fire equally for both cases.

---

### 2. Cap per-request decode-tail victims before evicting prompt-sized blocks
- **Agent:** codex

**Detailed description.**

Augment `LRUCachePolicy.evict` in `vllm/v1/kv_offload/cpu/policies/lru.py:56-78` with an optional request-local fairness guard that prevents one active request's long decode tail from forcing wholesale eviction of older prompt/prefix blocks. Track lightweight per-key origin metadata when `insert()` runs, using the available request context or block key structure to record the owning `request_id` and whether the block was produced after the request's initial prefill/prefix phase. In `evict(n, protected)`, keep the current atomic collect-then-delete behavior, but select victims in oldest-first buckets: first gather unprotected evictable decode-tail blocks from requests whose resident evictable count exceeds a configurable per-request soft cap; then fall back to normal LRU for the remaining victims. This keeps bursty single-request generation from flooding the CPU tier with low-reuse continuation blocks while preserving the protected-key exclusion, `ref_cnt == 0` assertion, and `None` return when fewer than `n` candidates exist. Gate it behind an `OffloadingSpec.extra_config` flag such as `decode_tail_eviction_bias` so existing LRU behavior remains the default.

**Novelty rationale.**

The existing deep-research proposals prioritize explicit retention tiers, S3-FIFO ghost/frequency behavior, or TinyLFU admission frequency; agent A prioritizes blocks touched by multiple distinct request IDs. This proposal instead uses per-request cache footprint and phase-of-block information to bias eviction against overrepresented decode-tail blocks from a single request. It is not a frequency signal, ghost-list signal, explicit hint, or cross-request fan-out metric: a block can be single-request and low-frequency but still protected from this rule if it belongs to the prompt/prefix portion, while a large stream of same-request continuation blocks becomes the preferred victim set because it is unlikely to improve median TTFT and can hurt median TPOT by displacing reusable prefixes.

---
