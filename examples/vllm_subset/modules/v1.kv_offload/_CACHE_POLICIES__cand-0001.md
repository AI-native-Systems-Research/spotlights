# _CACHE_POLICIES

[← v1.kv_offload](../v1.kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/manager.py`](vllm/v1/kv_offload/cpu/manager.py) (lines 19–22)
- **Symbol:** `_CACHE_POLICIES`
- **Kind:** plugin_seam
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0001`

## Description
Registration table mapping the runtime eviction-policy string to the CachePolicy implementation used by CPUOffloadingManager.

## Current approach
The registry exposes "lru" -> LRUCachePolicy in vllm/v1/kv_offload/cpu/policies/lru.py and "arc" -> ARCCachePolicy in vllm/v1/kv_offload/cpu/policies/arc.py. Both implement CachePolicy in vllm/v1/kv_offload/cpu/policies/base.py with get, insert, remove, touch, and evict. The runtime selector is kv_connector_extra_config["eviction_policy"], read in vllm/v1/kv_offload/cpu/spec.py and passed to CPUOffloadingManager(cache_policy=...).

## Estimated impact explanation
Eviction policy directly controls offloaded KV hit rate. In multi-turn agentic workloads, better handling of frequent shared prefixes and long reuse distances can reduce recomputed prefill, moving TTFT on cache-warm requests and reducing avoidable CPU-GPU traffic.

## Evolve rationale
A replacement policy is a contained extension: add a CachePolicy implementation and register it in this dict. The policy contract has concrete oracles in CachePolicy.evict's atomic exact-n-or-None semantics, ref_cnt protection, prepare_store/complete_store behavior, lookup/touch ordering, and existing manager tests for LRU and ARC. This is a good target for S3-FIFO, TinyLFU, SLRU, or other admission/replacement strategies benchmarked against prefix-cache hit rate on multi-turn traces.

## Deep research proposals

### 1. Add S3-FIFO CachePolicy with probationary FIFO admission to CPU offload registry
- **Finding:** `find-0008` — *FIFO Queues are All You Need for Cache Eviction*
- **Source URL:** <https://jasony.me/publication/sosp23-s3fifo.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Implement a new S3FIFOCachePolicy under vllm/v1/kv_offload/cpu/policies/s3fifo.py that conforms to the CachePolicy contract in vllm/v1/kv_offload/cpu/policies/base.py (get, insert, remove, touch, evict with atomic exact-n-or-None semantics and ref_cnt protection), and register it in the _CACHE_POLICIES dict at vllm/v1/kv_offload/cpu/manager.py:19-22 as "s3fifo" -> S3FIFOCachePolicy so it can be selected via kv_connector_extra_config["eviction_policy"] read in vllm/v1/kv_offload/cpu/spec.py. Internally maintain three FIFO structures as described in the paper: a small probationary FIFO (S, ~10% of capacity) that admits all newly inserted blocks; a main FIFO (M, ~90%) that holds blocks promoted from S after observing a second access; and a ghost FIFO (G) of recently evicted keys from S, used so that re-insertions for keys seen in G skip S and enter M directly. touch() should set/increment a small saturating frequency counter (0..3) on the entry without moving it between queues. evict() should scan from the head of S first, demoting entries with frequency==0 (recording their hash in G) and reinserting entries with frequency>0 into M with frequency reset; only blocks not protected by ref_cnt and not in prepare_store/complete_store-pending state are eligible, matching the existing LRU/ARC oracles. Reuse the existing manager tests for LRU/ARC as the conformance harness and add S3-FIFO-specific tests covering one-hit-wonder filtering, ghost-promotion, and the atomic exact-n eviction semantics. No changes to the CPUOffloadingManager itself are needed beyond the registry entry.

**Proposal rationale.**

The candidate is explicitly a plugin seam that already exposes "lru" and "arc"; its evolve_rationale calls out S3-FIFO as a high-value target for the CPU offload eviction policy. S3-FIFO's central claim — that a small probationary FIFO filters out most one-hit objects before they pollute the main cache — directly addresses the multi-turn agentic workload listed in the caller context, where each turn brings many transient KV blocks alongside a stable set of reusable shared prefixes; LRU mixes both, and ARC's ghost-list machinery is heavier than needed. By admitting only twice-touched blocks into M and demoting one-hit blocks quickly from S, the policy is expected to retain reusable prefix blocks longer, raising offload hit rate, reducing prefill recomputation on cache-warm requests, and thereby improving TTFT and median TPOT — the stated objective. The CachePolicy contract (atomic exact-n-or-None evict, ref_cnt protection, prepare_store/complete_store handling) gives concrete oracles for a correct implementation, so the change is contained to a new policy file plus one line in _CACHE_POLICIES.

---

### 2. Add TinyLFU admission policy as a new CachePolicy in _CACHE_POLICIES
- **Finding:** `find-0009` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://arxiv.org/pdf/1512.00727>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Implement a new CachePolicy subclass (e.g., TinyLFUCachePolicy in vllm/v1/kv_offload/cpu/policies/tinylfu.py) that wraps an LRU/SLRU main store with a TinyLFU admission filter, and register it in vllm/v1/kv_offload/cpu/manager.py:19-22 as "tinylfu" -> TinyLFUCachePolicy. The admission filter maintains an approximate recent-frequency sketch (Count-Min Sketch with periodic aging / doorkeeper bloom filter, sized to the offload capacity in blocks) keyed by the same block identity used by get/insert/touch in CachePolicy (vllm/v1/kv_offload/cpu/policies/base.py). On insert, the sketch is incremented for the candidate; the policy only admits the new block to the main store if its estimated frequency exceeds that of the chosen eviction victim, otherwise the new block is rejected (insert returns without committing storage and evict is not invoked). The implementation must preserve the existing CachePolicy contract: atomic exact-n-or-None evict semantics, ref_cnt-protected blocks remain ineligible for eviction, prepare_store/complete_store ordering is unchanged, and lookup/touch update both the recency structure and the frequency sketch. The runtime selector path in vllm/v1/kv_offload/cpu/spec.py (kv_connector_extra_config["eviction_policy"] -> CPUOffloadingManager(cache_policy=...)) is reused unchanged. Existing manager tests for LRU/ARC are mirrored for the new policy, plus targeted tests covering admission rejection (a low-frequency new block should not displace a high-frequency victim) and sketch aging behavior.

**Proposal rationale.**

The candidate is explicitly a plugin seam whose evolve_rationale calls out TinyLFU as a target, and the finding provides the exact mechanism: compare a newcomer to its eviction victim by approximate recent frequency before admitting. The caller objective (reduce TTFT/TPOT on a multi-turn agentic workload) and the workload hint of skewed repeated contexts are precisely the regime where TinyLFU's frequency-aware admission outperforms recency-only LRU and is competitive with ARC, by suppressing one-shot stores that would otherwise pollute the offload cache and evict warm prefix blocks. Because admission decisions are local to insert/evict and respect the existing CachePolicy oracles (atomic evict, ref_cnt protection, prepare_store/complete_store), the change is contained to a single new file plus one registry line, matching the seam's intended extension shape.

---

### 3. Add GreedyDual-Size cache policy for cost/size-aware KV block eviction
- **Finding:** `find-0010` — *GreedyDual-Size Algorithm*
- **Source URL:** <https://www.usenix.org/legacy/publications/library/proceedings/usits97/full_papers/cao/cao_html/node8.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Implement a new CachePolicy subclass `GDSCachePolicy` in `vllm/v1/kv_offload/cpu/policies/gds.py` (alongside `lru.py` and `arc.py`) and register it in the `_CACHE_POLICIES` table at `vllm/v1/kv_offload/cpu/manager.py:19-22` as `"gds" -> GDSCachePolicy`, selectable via `kv_connector_extra_config["eviction_policy"]` (already read in `vllm/v1/kv_offload/cpu/spec.py`). The policy maintains a per-block priority H = cost / size, plus a global inflation value L (running minimum of evicted H). On access (touch/get) and on insert, set `H_b = L + cost_b / size_b`; `evict(n)` removes the n entries with the smallest H whose `ref_cnt == 0`, and updates L to the largest evicted H, preserving `CachePolicy.evict`'s atomic exact-n-or-None semantics and `ref_cnt` protection from `vllm/v1/kv_offload/cpu/policies/base.py`. For the KV-offload setting, `size_b` is the block's bytes (block_size * dtype * num_layers * heads), and `cost_b` is an estimate of the work saved by keeping the block — initially the GPU prefill recompute time for that block (proportional to block_size and prefix depth) optionally combined with the CPU<->GPU transfer time. A simple first cut uses `cost_b = alpha * block_size * prefix_depth_b + beta * transfer_bytes_b`, with alpha/beta as tunables exposed through `kv_connector_extra_config`. Reuse the priority-queue / OrderedDict scaffolding from `LRUCachePolicy` but key on H instead of recency; reuse the existing manager tests for LRU/ARC as oracles for `prepare_store`/`complete_store`/`lookup`/`touch` ordering and add new tests that verify (a) blocks with higher cost-per-byte survive longer than equally-recent low-cost blocks and (b) the L-inflation prevents starvation of newly inserted entries.

**Proposal rationale.**

The candidate is explicitly a plugin seam for replacement strategies, and `evolve_rationale` calls out admission/replacement strategies as good fits with concrete oracles already in place. GreedyDual-Size adds a dimension neither LRU nor ARC capture: it weighs each block by the miss cost it would incur if evicted, normalized by the bytes it occupies. In a multi-turn agentic workload the value of a cached KV block is highly non-uniform — deep shared-prefix blocks save much more TTFT on recompute than tail blocks, and transfer-heavy blocks waste more PCIe bandwidth when re-fetched — so cost/size-aware eviction directly targets the stated objective (reduce TTFT and median TPOT) by preferentially retaining blocks with the highest savings-per-byte. The finding's exact mechanism (H = cost/size, with the L-inflation trick to combine recency and cost) is small, well-specified, and maps cleanly onto the existing CachePolicy contract without changing the manager interface.

---

### 4. Add SIEVE eviction policy to CPU offload _CACHE_POLICIES registry
- **Finding:** `find-0011` — *SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for Web Caches*
- **Source URL:** <https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Implement a new SIEVECachePolicy that conforms to the CachePolicy contract in vllm/v1/kv_offload/cpu/policies/base.py (get, insert, remove, touch, evict, prepare_store/complete_store, ref_cnt-respecting eviction with atomic exact-n-or-None semantics) and register it in vllm/v1/kv_offload/cpu/manager.py:19-22 _CACHE_POLICIES as "sieve" -> SIEVECachePolicy, alongside existing "lru" and "arc" entries. The implementation follows the SIEVE algorithm: maintain a FIFO queue of cached entries plus a per-entry visited bit; on hit, set visited=True and do no list reordering (this maps to a no-op or trivial touch(), unlike LRU's MRU promotion); on eviction, sweep a hand pointer backwards through the queue, clearing visited bits for entries marked True and evicting the first entry encountered with visited=False (skipping ref_cnt>0 entries to honor in-flight prepare_store/complete_store, matching how LRU/ARC respect pinning). New entries are inserted at the head of the queue with visited=False. The selector path is unchanged: kv_connector_extra_config["eviction_policy"] = "sieve" is read in vllm/v1/kv_offload/cpu/spec.py and forwarded to CPUOffloadingManager(cache_policy=...). Place the implementation at vllm/v1/kv_offload/cpu/policies/sieve.py and mirror the unit-test structure used for LRUCachePolicy and ARCCachePolicy (covering atomic evict semantics, ref_cnt protection, lookup/touch ordering, and prepare_store/complete_store interactions). Validate hit-rate on multi-turn agentic prefix-reuse traces against the existing "lru" and "arc" baselines.

**Proposal rationale.**

The candidate is explicitly a plugin seam for replacement policies, and its evolve_rationale calls out admission/replacement strategies as the intended extension axis. SIEVE is a concrete, well-characterized policy that fits the seam exactly: it is a drop-in CachePolicy with the same five-method contract, and its central claim — comparable or better miss ratios than LRU with no per-hit list manipulation — directly targets two constraints that matter here. First, on the multi-turn agentic workload in the caller context, SIEVE's queue+visited-bit structure avoids the LRU pathology where a long burst of unique tokens evicts a recently reused shared prefix, which is the exact reuse pattern that drives offloaded-KV hit rate and recomputed-prefill TTFT. Second, SIEVE's no-promotion-on-hit property reduces per-touch() overhead on the hot lookup path inside CPUOffloadingManager, which lowers the marginal cost of growing the offload index — relevant when CPU offload sizes get large. The finding contributes a transferable, named algorithm with a concrete data-structure recipe (FIFO + visited bit + hand pointer), not a generic "try a better policy" suggestion, so it adds information beyond the candidate's current LRU/ARC baselines.

---

## Agent proposals

### 1. Add prefix-chain-aware CachePolicy that evicts orphans/leaves first to avoid breaking shared KV prefix chains
- **Agent:** claude

**Detailed description.**

Implement a new PrefixChainCachePolicy in vllm/v1/kv_offload/cpu/policies/prefix_chain.py that conforms to the CachePolicy contract in vllm/v1/kv_offload/cpu/policies/base.py and register it in the _CACHE_POLICIES dict at vllm/v1/kv_offload/cpu/manager.py:19-22 as "prefix_chain" -> PrefixChainCachePolicy, selectable via kv_connector_extra_config["eviction_policy"] (read in vllm/v1/kv_offload/cpu/spec.py). The policy exploits a property that none of LRU/ARC/S3-FIFO/TinyLFU/GDS/SIEVE leverages: touch() and prepare_store() are called by CPUOffloadingManager with an ordered Collection[OffloadKey] that is the *prefix-hash chain* of a request (consecutive entries are parent->child in the vLLM block-hash chain, restricted to a single group_idx). The policy records, on every insert/touch, edges (parent_key -> child_key) and (child_key -> parent_key) for consecutive same-group entries (using get_offload_block_hash / get_offload_group_idx from vllm/v1/kv_offload/base.py to scope per group). Internally it keeps an LRU-ordered list as the recency backbone, plus a `cached_children_count[key]` map. A node is classified as: LEAF (count==0), INTERIOR (count>0), or ORPHAN (its parent is recorded in the chain index but not currently in cache). evict(n, protected) selects victims with priority: (1) ORPHANs (already wasted slots — their data is unreachable through prefix lookup once an ancestor is gone), (2) LEAVES in LRU order, (3) INTERIORs only as last resort, and only after decrementing/promoting their cached children's status. The selection respects the existing oracles: ref_cnt>0 blocks are skipped, atomic exact-n-or-None semantics are preserved (build the candidate list, return None if fewer than n candidates exist after applying protected/ref_cnt filters; only commit removals after the full set is found), and prepare_store/complete_store ordering is unchanged. On remove(), update the parent's cached_children_count and re-classify formerly-interior parents as leaves if they now have zero cached children, and re-classify the children of the removed node as orphans. Reuse the existing LRU/ARC manager tests as a conformance harness; add new tests covering: (a) interior shared-prefix block survives across many turns when its leaf siblings are evicted; (b) a chain break (interior eviction by force) causes its descendants to be the next eviction targets; (c) atomic exact-n-or-None when fewer than n eligible non-protected candidates exist; (d) per-group chain isolation so two different KV groups with colliding sub-hashes are not falsely linked.

**Novelty rationale.**

All four existing deep_research_proposals (S3-FIFO, TinyLFU, GreedyDual-Size, SIEVE) are domain-agnostic generic cache replacement algorithms whose decisions depend only on per-key access statistics (recency, frequency, size, cost). None of them models the directed prefix-chain dependency between KV blocks, which is the *defining* structural property of the offloaded data here: vLLM's prefix lookup walks a hash chain and stops at the first miss, so a block whose ancestor is missing is effectively dead weight in cache. This proposal is the only one that turns that structure into an eviction signal — preferring "orphan" and "leaf" blocks over deep interior shared-prefix blocks — and it does so by reading the ordering already implicit in CachePolicy.touch / CPUOffloadingManager.prepare_store calls, with no change to the manager interface or to OffloadKey. GDS (find-0010) uses a generic cost/size scalar without any notion of dependencies between blocks; this policy's contribution is *structural*, not scalar, and is orthogonal (could be combined with any of the four).

---

### 2. Add cross-KV-group stripe-aware CachePolicy that evicts partial group stripes first
- **Agent:** codex

**Detailed description.**

Implement a new KVGroupStripeCachePolicy in vllm/v1/kv_offload/cpu/policies/stripe.py and register it in _CACHE_POLICIES at vllm/v1/kv_offload/cpu/manager.py:19-22 as "stripe" -> KVGroupStripeCachePolicy, selectable through kv_connector_extra_config["eviction_policy"]. The policy should conform to CachePolicy and use get_offload_block_hash/get_offload_group_idx from vllm/v1/kv_offload/base.py to track, for each logical block hash, which KV group indices are cached. Keep an LRU OrderedDict as the fallback order, but have evict(n, protected) prefer keys whose block_hash has become a partial cross-group stripe: i.e. the policy has previously observed that block_hash in multiple group_idx values, but only a subset of those group siblings are currently cached. This matters because OffloadingConnectorScheduler tightens loadable hit tokens across KV groups; cached depth in one group often cannot improve the request hit unless the matching block exists in the other relevant groups too. Build eviction candidates atomically using virtual stripe counts, skip protected keys and ref_cnt != 0 blocks, return None without mutation if fewer than n victims are available, and after selecting one key from a complete stripe as a last-resort victim, prefer its now-partial siblings for subsequent victims in the same evict(n) call to avoid leaving low-value half-stripes behind. Reuse existing LRU/ARC manager tests as conformance coverage and add stripe-specific tests with two group_idx values sharing the same block_hash: incomplete stripes evict before complete stripes, a one-group burst does not destroy balanced prefix depth across groups, multi-key eviction cleans up siblings after breaking a stripe, and single-group workloads fall back to ordinary LRU behavior.

**Novelty rationale.**

The listed deep_research_proposals are generic replacement/admission algorithms based on recency, frequency, cost/size, or visited bits, and none models the fact that vLLM offload keys are block_hash plus KV group index. Agent A's PrefixChainCachePolicy uses parent-child dependencies within one group and only scopes by group_idx to avoid false links; it does not reason about cross-group completeness for the same block_hash. This proposal targets a different structural inefficiency: in multi-KV-group models, a cached block in only one group can be less useful because scheduler lookup is constrained by the minimum hit depth across groups. Preserving complete group stripes and evicting partial stripes first is therefore orthogonal to prefix-chain-aware eviction and to the four generic policies.

---
