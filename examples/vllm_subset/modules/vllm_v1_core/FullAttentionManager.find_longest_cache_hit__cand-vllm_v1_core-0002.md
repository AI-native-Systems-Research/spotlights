# FullAttentionManager.find_longest_cache_hit

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 682–777)
- **Symbol:** `FullAttentionManager.find_longest_cache_hit`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_core-0002`

## Description
Full-attention prefix-cache hit lookup using a full-block forward scan plus optional fine-grained interior-boundary probe for partial hash hits.

## Current approach
Runs a Python loop over candidate block hashes, calls block_pool.get_cached_block once per block, stops on the first chained-hash miss, appends cached blocks into per-group lists, scans partial boundaries high-to-low in fine-grained mode, then applies EAGLE and alignment trimming.

## Estimated impact explanation
Called for most models on every admission. Long multi-turn prompts require hundreds of block-hash probes; reducing per-probe Python overhead lowers median TTFT.

## Evolve rationale
Default attention cache-hit inner loop for most attention-only models. Headroom in batched hash lookups against BlockHashToBlockMap, materialized bounded hash views, native/Cythonized scanning, or prefix sketches that avoid repeated dict probes on misses. Correctness oracle: output hit_length is alignment-valid, returned blocks match cached_block_hash_to_block for all hit positions, and removing a cached block shrinks the hit in tests/v1/core cache-manager suites.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Galloping + binary-search hit boundary in Phase-1 full-block scan (exploit chained-hash monotonicity)
- **Agent:** claude

**Detailed description.**

Replace the linear left-to-right probe loop in FullAttentionManager.find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:731-739) with an exponential (galloping) search followed by binary search to locate the first miss, then materialize the confirmed hit prefix. The chained-hash invariant already documented at line 731-732 ("A missing block implies every later block misses too") makes hit/miss a monotone predicate over block index, so bisection is valid. Concrete change: (1) let N = max_length // block_size and treat `full_block_hashes` as random-accessible (use BlockHashListWithBlockSize / list slicing; if the input is a non-indexable iterable, first materialize the bounded window via itertools.islice into a list — bounded by N, which is already the current loop bound). (2) Gallop: probe indices 1, 2, 4, 8, ... via block_pool.get_cached_block(full_block_hashes[i-1], kv_cache_group_ids) until the first miss at index `hi` or `hi >= N`. Track the last confirmed hit position `lo`. (3) Binary-search in [lo, hi) for the largest `k` such that position `k` is a hit. Cache probe results in a dict keyed by index to avoid re-probing. (4) Materialize the hit prefix by iterating positions 0..k left-to-right: if a probe result was cached, reuse it; otherwise call get_cached_block and append into per-group `computed_blocks` lists. Under the monotonicity invariant every such position is a hit, so no additional break check is needed. (5) Set `hit_length = k * block_size` and continue to the unchanged Phase-2 fine-grained interior probe (lines 741-762), EAGLE handling (line 768), and alignment trim (lines 770-777). Phase 2 is left alone because interior-boundary hashes are not chained and cannot be bisected. Complexity drops from O(hit_len / block_size) probes to O(log(hit_len / block_size)) probes plus a single O(hit_len / block_size) list append (dict lookup avoided on the appends via the cache from step 3). For N=256 blocks with a full hit this is ~9 probes + 256 dict `get` reuses instead of 256 probes; for a miss-at-position-k it is ~2*log2(k) probes instead of k. Correctness oracle from the candidate description holds: hit_length remains a multiple of alignment_tokens after the unchanged trim; returned blocks are exactly the cache entries at positions [0, k); removing any cached block at position j <= k moves the first miss to j and shrinks the hit accordingly. No changes to BlockHashToBlockMap, no new indices, no memory overhead beyond the O(log N) probe cache.

**Novelty rationale.**

The candidate lists zero deep_research_proposals, but the evolve_rationale enumerates four directions to watch out for: batched hash lookups against BlockHashToBlockMap, materialized bounded hash views, native/Cythonized scanning, and prefix sketches to avoid repeated dict probes on misses. This proposal is orthogonal to all four. It does not batch lookups, add a new index/sketch, materialize a new view beyond what the current code already produces via resolve_block_hashes, or move work to native code. Instead it exploits the chained-hash monotonicity invariant already documented in the code to reduce the number of probes asymptotically from O(hit_len) to O(log hit_len). Batched lookups still probe every position; sketches speed up misses but not hits; native scanning reduces per-probe cost but keeps the linear count. Galloping+bisect is the only direction listed here that changes the probe-count complexity class on the hot path — full-hit multi-turn admissions — which directly targets the caller's median-TTFT objective.

---

### 2. Memoize exact prefix-cache lookup results within a scheduler step
- **Agent:** codex

**Detailed description.**

Add a tiny scheduler-step-scoped memoization layer around `FullAttentionManager.find_longest_cache_hit` in `vllm/v1/core/single_type_kv_cache_manager.py:682-777`. Key it by the bounded full-block hash window actually eligible under `max_length`, the relevant fine-grained boundary hashes when fine-grained lookup is enabled, `kv_cache_group_ids`, `block_size`, `max_length`, and the EAGLE/alignment-relevant flags. On a hit, return shallow-copied per-group block lists plus the cached `hit_length`, so callers cannot mutate the memoized lists. Scope the memo to one scheduling iteration, or clear it on any block-pool mutation, so eviction/allocation cannot make returned block references stale. This targets multi-turn agentic batches where the same prefix candidate can be checked repeatedly during admission, preemption, or retries, avoiding both the full-block scan and the optional interior-boundary scan for exact repeated lookups.

**Novelty rationale.**

There are no deep_research_proposals listed for this candidate. This is also distinct from Claude's proposal: Claude changes the search strategy inside one lookup using chained-hash monotonicity, while this proposal avoids re-executing identical lookups during the same safe lifetime. It does not require galloping or binary search and remains compatible with either the current linear scan or Claude's proposed scan.

---
