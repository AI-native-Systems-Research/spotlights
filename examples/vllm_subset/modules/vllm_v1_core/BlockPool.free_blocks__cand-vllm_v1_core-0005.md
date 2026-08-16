# BlockPool.free_blocks

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/block_pool.py`](vllm/v1/core/block_pool.py) (lines 719–742)
- **Symbol:** `BlockPool.free_blocks`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0005`

## Description
Returns blocks whose ref_cnt reaches zero to the free queue and chooses their eviction ordering based on whether they still have prefix-cache hashes.

## Current approach
Loops through ordered blocks, decrements ref_cnt, classifies zero-ref non-null blocks into blocks_with_hash and blocks_without_hash lists, then prepends hashless blocks and appends hashed blocks to preserve the current LRU/hash-priority policy.

## Estimated impact explanation
The loop is small, but the ordering policy affects future prefix-cache hit rate. For multi-turn agents with overlapping histories, better retention ordering can improve TTFT on follow-up turns.

## Evolve rationale
This is both a per-call hot path and the cache-retention ordering policy that shapes future prefix-cache hit rate. Headroom exists in alternative retention policies, hash-length-weighted placement, LFU-style eviction, and lower-allocation batching of classification. Correctness oracle: free/allocate round-trip tests plus linked-list invariants that num_free_blocks matches the reachable queue and no referenced block is returned to the free list.

## Deep research proposals

### 1. Workflow-aware free ordering: prioritize retaining soon-needed agent-prefix blocks in free_blocks
- **Finding:** `find-vllm_v1_core-0001` — *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows*
- **Source URL:** <https://www.alphaxiv.org/abs/2507.07400>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend BlockPool.free_blocks (vllm/v1/core/block_pool.py:719-742) to consult an optional workflow-priority signal when placing zero-ref blocks into the free queue. Concretely: (1) allow requests/sessions to carry a lightweight 'steps-to-execution' (STE) hint (defaulting to +inf/unknown, so single-turn behavior is unchanged); (2) propagate that hint onto KVCacheBlock at allocation/touch time (e.g. min-STE across the referencing requests, refreshed via the existing touch path); (3) in free_blocks, split zero-ref hashed blocks into a 'soon-needed' bucket (small STE / known agent-shared prefix) and a 'dynamic-suffix' bucket (large or unknown STE), then order the tail of the free queue so the dynamic-suffix bucket is evicted first and the soon-needed bucket sits farthest from the eviction tail. Hashless blocks continue to prepend as today. The classification stays O(n) over the freed batch and reuses the existing prepend_n/append_n primitives; the only new state is a per-block STE attribute updated in the same loops that already visit these blocks. Fall back to the current LRU/hash-priority policy whenever no STE hints are present.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'alternative retention policies' and impact on 'multi-turn agents with overlapping histories', and the caller objective is median TTFT/TPOT on a multi-turn agentic workload. KVFlow's core claim is exactly that plain LRU evicts paused-agent prefixes that will be reused within a few steps, and that a steps-to-execution priority meaningfully improves multi-agent TTFT. free_blocks is the single choke point where retention order is decided in v1 core, and it already partitions blocks into two buckets fed to prepend_n/append_n, so a workflow-priority bucket is a minimal, transferable extension of the existing policy rather than a rewrite. Correctness invariants highlighted in evolve_rationale (num_free_blocks matches reachable queue, no referenced block returned) are preserved because ref_cnt gating and queue primitives are unchanged.

---

### 2. Add priority-tier retention ordering to BlockPool.free_blocks via an auxiliary priority queue
- **Finding:** `find-vllm_v1_core-0003` — *[RFC]: Context-Aware KV-Cache Retention API (Prioritized Evictions)*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/37003>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend BlockPool.free_blocks (vllm/v1/core/block_pool.py:719-742) to consult a per-block retention priority (annotated via a new token-range retention directive on the request/KVCacheBlock) when returning zero-ref blocks to the reclaim structures. Concretely: (1) keep the current LRU free_block_queue as the default reclaim path for un-annotated blocks; (2) introduce a companion min-heap / priority queue keyed by (priority, TTL_deadline, recency) for blocks that carry a retention annotation; (3) in the classification loop currently splitting into blocks_with_hash / blocks_without_hash, add a third bucket for annotated blocks that gets pushed into the priority structure instead of the LRU tail, while hashless and unannotated hashed blocks continue to be prepended/appended to free_block_queue as today; (4) update the allocator's eviction picker to drain the LRU queue first and only fall back to the priority queue when LRU is exhausted (or TTL has expired), preserving the invariant that num_free_blocks equals the reachable union of both structures and that no ref_cnt>0 block enters either. free_blocks itself remains O(n) in the batch: one classification pass, then bulk prepend_n / append_n / heappush.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out alternative retention policies and hash-length-weighted placement as headroom, and free_blocks is the exact chokepoint where zero-ref blocks are placed into the reclaim ordering that shapes future prefix-cache hit rate. The finding contributes a concrete, transferable mechanism (two-structure evictor: LRU for unprioritized + priority queue for annotated ranges, with TTL) that plugs into this hook without hard-pinning. For the caller's multi-turn agentic workload targeting median TTFT/TPOT, protecting shared system/tool prefixes across tool-call pauses is precisely the scenario where LRU under-retains high-value prefixes; a priority tier at free-time is a minimal, local extension of the current classification that addresses that gap while preserving the free/allocate round-trip and linked-list invariants named in the candidate's correctness oracle.

---

### 3. Score-based eviction ordering in free_blocks using reuse-likelihood and compute-savings-per-block
- **Finding:** `find-vllm_v1_core-0004` — *Marconi: Prefix Caching for the Era of Hybrid LLMs*
- **Source URL:** <https://huggingface.co/papers/2411.19379>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the two-bucket (with-hash vs without-hash) placement in BlockPool.free_blocks at vllm/v1/core/block_pool.py:719-742 with a small score-based insertion that ranks freed hashed blocks by an estimated value density before returning them to free_block_queue. Concretely: keep the existing loop that decrements ref_cnt and separates hashless blocks (still prepended for eviction-first), but for the blocks_with_hash bucket compute a lightweight score per block that combines (a) a reuse-likelihood proxy already available on the block (e.g. block_hash present, prefix depth / position within the sequence, whether the block belongs to a system-prompt-style prefix that has been touched by multiple requests, or a hit-count field on KVCacheBlock updated in touch()) with (b) a compute-savings-per-memory proxy (block token length is uniform, so this reduces to hash-length / prefix-depth: deeper prefix blocks save more recomputation per byte retained). Blocks with higher score are appended further from the eviction head (kept longer), lower-score hashed blocks are appended near the head. Implementation stays O(n) per call by bucketing into a small fixed number of priority tiers (e.g. 2-3) rather than sorting, so the hot-path allocation profile is unchanged. Preserve current invariants: only zero-ref non-null blocks enter the queue, hashless blocks still evict first, and num_free_blocks accounting is untouched. The scoring hook should be a single private method so the policy can be swapped or disabled via the existing enable_caching path without changing callers.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out headroom in alternative retention policies and hash-length-weighted placement, and the caller context is a multi-turn agentic workload where TTFT on follow-up turns is dominated by prefix-cache hit rate. Marconi's contribution is precisely a retention policy that scores blocks by predicted reuse and compute-savings-per-memory rather than pure recency, which is what the current with-hash/without-hash split approximates only crudely (it treats all hashed blocks as equivalently valuable and orders them by free-time). Applying that scoring idea inside free_blocks is a targeted, transferable change: it operates on the exact code path where retention order is decided, keeps the LRU/hashless-first invariant as a fallback tier, and directly addresses the retention gap the candidate identifies. Marconi's framing is for hybrid attention/Mamba where checkpoint size varies, but the underlying principle -- rank freed cache entries by expected future savings per unit of retained memory -- carries over to uniform-sized attention KV blocks by using prefix depth as the compute-savings proxy, which is information already reachable from the block/request state.

---

### 4. Bias free_blocks eviction ordering with a learned continuation-probability score
- **Finding:** `find-vllm_v1_core-0005` — *Learned Prefix Caching for Efficient LLM Inference*
- **Source URL:** <https://papers.neurips.cc/paper_files/paper/2025/hash/414f642a1ea9350006669774cba9bcd4-Abstract-Conference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/core/block_pool.py at BlockPool.free_blocks (lines 719-742), extend the current two-bucket placement policy (blocks_without_hash prepended to head, blocks_with_hash appended to tail) with a third, LPC-style signal: a lightweight predicted continuation score for the owning session/request. Concretely, attach a small, cached scalar `p_continue` to each hashed block at free time — computed once per request when free_blocks is called, from cheap conversational-content features already available in the request (e.g., token count, presence of a trailing user turn, elapsed idle time since last access from the metrics_collector's last-access timestamp) fed through a tiny predictor (logistic regression / small MLP, or even a hand-tuned linear scoring rule as a first version). Split blocks_with_hash into two sub-buckets: `hashed_likely_continue` (p_continue >= threshold) and `hashed_likely_done` (p_continue < threshold). Order the final append as: prepend_n(blocks_without_hash), append_n(hashed_likely_done), append_n(hashed_likely_continue) — so likely-continuing sessions sit deepest in the LRU tail and survive longer under eviction pressure, while inactive-but-hashed blocks that LRU would otherwise protect get demoted. Keep the existing invariants (num_free_blocks accounting, no ref_cnt>0 block enqueued, null-block exclusion) untouched; the change is purely ordering within the already-classified hashed bucket. Guard the whole predictor behind a config flag (default off) so LRU-only behavior remains the fallback, and expose the threshold and feature weights as tunables. Correctness is preserved because the set of blocks freed and the queue-membership invariants are unchanged; only the relative order within the hashed tail differs.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'alternative retention policies' and notes that ordering shapes future prefix-cache hit rate — and the caller context targets median TTFT on a multi-turn agentic workload, which is exactly the regime LPC studies. The finding's core observation is that pure LRU misclassifies paused-but-continuing sessions as low-value, evicting KV blocks a follow-up turn would have reused; a lightweight conversational-content + last-access-time predictor biases retention toward likely-continuing sessions. free_blocks is the single choke point where per-block eviction priority is materialized into the free queue, making it the minimal-blast-radius insertion point for such a predictor. The change is transferable (the paper's feature set is small and inference-time-cheap), addresses a concrete gap (LRU's inactivity blind spot) rather than restating the current approach, and directly targets follow-up-turn TTFT via improved prefix-cache retention.

---

### 5. Radix-tree-informed eviction ordering in BlockPool.free_blocks
- **Finding:** `find-vllm_v1_core-0006` — *RadixAttention*
- **Source URL:** <https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify BlockPool.free_blocks (vllm/v1/core/block_pool.py:719-742) so that, when appending hashed blocks back to the free queue, ordering is informed by a lightweight prefix-relationship index rather than raw insertion order. Concretely: maintain (alongside the existing hash-based cached_block_hash_to_block map) a compact trie/radix structure over prefix-cache hash chains that tracks, per hashed block, how many other currently-cached descendants share it as a prefix (i.e. its subtree fan-out). Within the classified `blocks_with_hash` list produced by the loop at lines 730-738, sort/partition so blocks with higher descendant fan-out (prefix roots shared across multiple cached continuations) are appended nearer the tail of the free queue and thus evicted last, while leaf-only hashed blocks are appended nearer the head. `blocks_without_hash` handling stays unchanged (still prepended). The trie need only be updated on cache_full_blocks/_maybe_evict_cached_block paths and consulted here in a single O(len(ordered_blocks)) pass; no change to the invariant that referenced blocks never enter the free list.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out `alternative retention policies` and `hash-length-weighted placement` as headroom areas, and its impact story is future prefix-cache hit rate — exactly what RadixAttention optimizes by making shared-prefix structure first-class rather than implicit in a flat hash map. The finding's specific suggestion (`adapt a trie-like secondary index or prefix-match ordering`) maps directly onto the ordering decision at lines 740-742, which today treats every hashed block identically in FIFO append order. In the multi-turn agentic workload named in the caller context, later turns re-hit early-turn prefix blocks; a fan-out-weighted eviction order preserves those shared roots preferentially, plausibly improving hit rate and thus median TTFT. The change is confined to `free_blocks` plus a small auxiliary index, preserves the current linked-list invariants and correctness oracle (free/allocate round-trip, num_free_blocks reachability, no referenced block freed), and does not require replacing the hash cache — matching the finding's `even without replacing the hash cache` framing.

---

### 6. Add W-TinyLFU frequency-aware admission to free_blocks placement
- **Finding:** `find-vllm_v1_core-0009` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://doczz.net/doc/8550019/tinylfu--a-highly-efficient-cache-admission-policy>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Augment `BlockPool.free_blocks` (vllm/v1/core/block_pool.py:719-742) so that when a block with a prefix hash becomes reclaimable (ref_cnt reaches 0 and block_hash is not None), its placement into `free_block_queue` is guided by an approximate frequency estimate rather than a pure LRU/hash-priority split. Maintain a compact frequency sketch (e.g., a Count-Min sketch with a doorkeeper bloom filter, sized proportionally to num_gpu_blocks) keyed by block_hash; increment the counter on cache hits (updated wherever `touch` runs on a cached block) and periodically age the sketch by halving all counters after every N accesses to bound frequency drift, matching W-TinyLFU. In `free_blocks`, split hashed reclaimable blocks into two sub-lists using the sketch: 'hot' blocks (estimated frequency above a threshold, or above the frequency of the current LRU victim at the head of the free queue) are appended (kept longer for prefix-cache reuse), while 'cold' hashed blocks whose estimated frequency is below the victim's are placed nearer the eviction end — effectively an admission decision that lets a new hashed block enter the retained set only when it is expected to increase the hit ratio. Hashless blocks (`block_hash is None and enable_caching`) continue to be prepended for immediate eviction, preserving the current invariant. Gate the whole mechanism behind an off-by-default flag so LRU behavior is preserved when disabled, and keep the sketch update O(1) so the per-call cost of `free_blocks` remains constant.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'alternative retention policies' and 'LFU-style eviction' as headroom, and TinyLFU is the canonical way to add frequency awareness without the memory cost of exact LFU. For the stated multi-turn agentic workload, hot shared prefixes (system prompts, tool schemas, common turn preambles) are re-hit across many requests, while one-off user turns produce hashed blocks that will likely never be reused; the current policy treats both classes of hashed blocks identically and can let a burst of one-off blocks push hot prefixes out of the free-queue tail. TinyLFU's sketch-plus-doorkeeper admission — as the finding's quoted architecture describes — directly targets that failure mode by admitting a candidate into the retained region only when its estimated frequency exceeds the victim's, which plausibly increases prefix-cache hit rate and thereby reduces median TTFT on follow-up turns (the caller's stated objective). The change is localized to the ordering decision already made in `free_blocks` plus a small counter update on the existing `touch` path, so it composes cleanly with the LRU/hash-priority policy the correctness oracle (free/allocate round-trip and free-list invariants) already covers.

---

### 7. Apply SIEVE-style lazy promotion and quick demotion to free_blocks eviction ordering
- **Finding:** `find-vllm_v1_core-0010` — *SIEVE: Cache eviction can be simple, effective, and scalable*
- **Source URL:** <https://www.usenix.org/publications/loginonline/sieve-cache-eviction-can-be-simple-effective-and-scalable>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/core/block_pool.py:719-742 (BlockPool.free_blocks), replace the current two-bucket LRU/hash-priority classification with a SIEVE-inspired retention policy. Concretely: (1) add a lightweight `visited` bit on KVCacheBlock (set in `touch` on cache hits — lazy promotion — instead of any queue mutation there), and (2) in `free_blocks`, when returning zero-ref blocks to `free_block_queue`, keep a moving eviction hand: blocks whose `visited` bit is set get the bit cleared and are reinserted at the tail (survive one pass), while blocks whose `visited` bit is unset are placed near the hand for quick demotion. Blocks without a hash (which can never match APC) remain the highest-priority eviction candidates and continue to be prepended. This preserves the existing correctness invariants (num_free_blocks, no ref_cnt>0 block enters the free queue) while changing only ordering, and reduces per-hit queue mutation in `touch` (which currently removes-from-free-queue on every hit — a per-call cost on the hot path).

**Proposal rationale.**

The candidate explicitly calls out headroom in alternative retention policies and notes that ordering affects future prefix-cache hit rate for multi-turn agents with overlapping histories — exactly the workload where SIEVE outperforms plain LRU by resisting pollution from one-hit ephemeral prompts. SIEVE's lazy promotion also aligns with the hot-path concern: today `touch` unlinks-and-relinks blocks on every prefix-cache hit, whereas SIEVE only flips a bit on hit and defers work to eviction time, directly reducing per-hit allocation/pointer churn. The finding's two mechanisms (single visited bit + moving hand) map cleanly onto the existing block metadata and doubly-linked free queue without new data structures, so it is a concrete transferable change rather than a topical restatement.

---

## Agent proposals

### 1. Order free_blocks by observed per-block reuse-interval EMA to approximate Belady
- **Agent:** claude

**Detailed description.**

Modify BlockPool.free_blocks (vllm/v1/core/block_pool.py:719-742) to order the `blocks_with_hash` tail using an empirically observed per-block reuse-interval signal rather than pure free-time FIFO. Concretely: (1) add a small always-on scalar `reuse_gap_ema_ns` (plus `last_access_ns`) to KVCacheBlock — updated in `touch()` alongside the existing `metrics_collector.on_block_accessed(block)` call, using an EMA over the interval `(now - last_access_ns)` on each hit; this leverages exactly the signal the KVCacheMetricsCollector already computes as `reuse_gaps_seconds` (vllm/v1/core/kv_cache_metrics.py:39-43) but promotes it out of the 1%-sampled diagnostic path into an always-on eviction feature on hashed blocks only, so the extra state is O(num_gpu_blocks) scalars and the update is O(1) per hit. (2) In the existing classification loop at lines 730-738, when a block enters `blocks_with_hash`, compute a Belady-style priority `expected_next_hit_delay = max(0, reuse_gap_ema_ns - (now - last_access_ns))`; blocks that are 'overdue' relative to their own historical cadence get a low score (evict sooner), blocks whose cadence says a hit is imminent get a high score (evict later). (3) Bucket the hashed blocks into 2-3 fixed priority tiers by score (keeping the pass O(n) — no sort), then emit `append_n(low_tier)` before `append_n(mid_tier)` before `append_n(high_tier)` so imminently-reused blocks sit farthest from the eviction head. Hashless blocks continue to prepend, and the metrics_collector sampling stays as-is for observability. Correctness invariants (num_free_blocks accounting, no ref_cnt>0 block enqueued, null-block exclusion) are untouched — only ordering within the already-classified hashed bucket changes. Gate behind a config flag defaulting off so pure LRU is the fallback.

**Novelty rationale.**

None of the seven listed deep_research_proposals use observed per-block inter-access-interval history as the ordering signal. STE (find-0001) uses an extrinsic caller-supplied steps-to-execution hint; the priority-tier proposal (find-0003) uses annotated retention priority with TTL, not observed intervals; Marconi-style scoring (find-0004) uses prefix depth as a compute-savings-per-memory proxy and reuse-likelihood proxies like block_hash presence / hit-count, not the *distribution* of gaps between hits; LPC (find-0005) predicts session-level continuation from conversational features (token count, trailing user turn, request-level idle time), not per-block reuse cadence; the radix proposal (find-0006) is structural (subtree fan-out), not temporal; W-TinyLFU (find-0009) tracks admission by frequency count in a sketch, which discards timing entirely and cannot distinguish a block hit 10 times in the last second from one hit 10 times over the last hour; SIEVE (find-0010) uses a single 0/1 visited bit with no interval information. This proposal is a Belady-approximating temporal-cadence ordering built on a signal (per-block reuse gaps) that the codebase already computes for observability (KVCacheMetricsCollector.get_reuse_gaps_seconds) but does not currently feed into the eviction ordering — a distinct mechanism from frequency counting, visited bits, structural fan-out, extrinsic hints, and session-level continuation prediction.

---

### 2. Batch free_blocks with allocation-free linked-list splicing
- **Agent:** codex

**Detailed description.**

Refactor `BlockPool.free_blocks` in `vllm/v1/core/block_pool.py:719-742` to avoid allocating and then iterating two temporary Python lists on every free path. Instead, classify zero-ref non-null blocks into two local linked chains as the loop decrements `ref_cnt`: a hashless chain for `block.block_hash is None and self.enable_caching`, and a cached/reusable chain for the existing append path. Add small `FreeKVCacheBlockQueue` helpers that splice a prebuilt `(head, tail, count)` chain at the front or back of the queue, updating `prev_free_block`, `next_free_block`, and `num_free_blocks` once per chain. Preserve the exact current policy: hashless blocks are still prepended and hashed/caching-disabled blocks are still appended in the same relative order. Add focused linked-list invariant tests that cover empty input, all hashless, all hashed, mixed batches, null blocks, and blocks whose decremented `ref_cnt` remains positive.

**Novelty rationale.**

The existing deep_research_proposals all change eviction-retention semantics using workflow hints, priority/TTL tiers, score models, continuation prediction, radix fan-out, TinyLFU frequency, or SIEVE visited bits. Agent A changes ordering using observed reuse-interval EMA. This proposal deliberately does not introduce a new retention signal or reorder hashed blocks at all; it keeps the current eviction policy byte-for-byte equivalent at the behavioral level and targets the separate hot-path overhead noted in the candidate rationale: per-call list allocation and duplicate traversal before queue insertion. That makes it novel relative to both the policy-focused proposals and Agent A's temporal-cadence proposal.

---
