# BlockPool.get_new_blocks

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/block_pool.py`](vllm/v1/core/block_pool.py) (lines 322–352)
- **Symbol:** `BlockPool.get_new_blocks`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0011`

## Description
Allocates num_blocks free KV blocks from FreeKVCacheBlockQueue, evicts cached metadata when caching is enabled, increments ref_cnt, and notifies the metrics collector.

## Current approach
Calls popleft_n(num_blocks), then runs one per-block Python loop specialized for caching-enabled or caching-disabled mode. Each block handles eviction, ref_cnt mutation, assertions, and optional metrics callbacks individually.

## Estimated impact explanation
Allocation overhead scales with blocks per admitted prefill chunk and with preemption churn. Improvements reduce TTFT under bursty admission and modestly reduce TPOT when decode allocates lookahead blocks.

## Evolve rationale
The optimization unit is the allocation loop immediately behind every allocate_slots call. Headroom includes specializing away metrics checks when absent, batching metrics callbacks, reducing per-block eviction map work, and adding slab-style allocation paths for large prefill chunks. Correctness oracles include tests/v1/core/test_prefix_caching.py and tests/v1/core/test_single_type_kv_cache_manager.py, which exercise free-queue, cache eviction, and allocation invariants.

## Deep research proposals

### 1. Add priority/TTL-aware eviction to get_new_blocks for agentic prefixes
- **Finding:** `find-0008` — *Introducing New KV Cache Reuse Optimizations in NVIDIA TensorRT-LLM*
- **Source URL:** <https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend BlockPool.get_new_blocks (vllm/v1/core/block_pool.py:322-352) and its supporting free-queue/eviction path so that the per-block eviction step (_maybe_evict_cached_block, line 341) consults retention-priority and TTL hints attached to cached blocks rather than always evicting the LRU candidate produced by free_block_queue.popleft_n. Concretely: (1) carry an optional priority/retention tag on KVCacheBlock (e.g., system-prompt, agent-role, recent-multimodal-prefix, active-session) populated when blocks are cached via cache_full_blocks; (2) change get_new_blocks to pull a candidate batch from popleft_n and, when enable_caching is true, skip-and-requeue any candidate whose priority tag or remaining TTL forbids eviction, falling back to a lower-priority block; (3) keep the fast path (no priorities set) behaviorally identical to today's loop so the metrics callbacks and ref_cnt updates remain branch-predictable. Tag assignment can be driven from request metadata already flowing through cache_full_blocks (lora_request, multimodal hashes, request type) so no new external API is required initially. Validate against tests/v1/core/test_prefix_caching.py and tests/v1/core/test_single_type_kv_cache_manager.py to preserve allocation/eviction invariants.

**Proposal rationale.**

get_new_blocks is exactly the chokepoint where blocks are repurposed and their cached hashes are dropped; today that decision is pure LRU via free_block_queue order. The finding's TensorRT-LLM priority-based eviction model maps cleanly onto this path because the candidate already iterates each allocated block through _maybe_evict_cached_block, giving a natural place to consult retention priority before resetting the hash. For the caller's multi-turn agentic workload, system-prompt and agent-role prefixes are reused across turns; retaining them under bursty admission directly attacks media TTFT (cache hit on prefill) and median TPOT (fewer recomputations on preemption churn), which is precisely the impact lever evolve_rationale identifies. The change is transferable rather than topically adjacent: the finding contributes a concrete signal (retention priority + duration) that is missing in the current LRU-only eviction and that the existing per-block loop can act on with bounded code change.

---

## Agent proposals

### 1. Bind a specialized get_new_blocks variant at construction and batch metrics + cache-map updates
- **Agent:** claude

**Detailed description.**

Replace the dual-mode per-block loop in BlockPool.get_new_blocks (vllm/v1/core/block_pool.py:322-352) with a closed-over, branch-free implementation chosen once at BlockPool.__init__. At construction, set self._allocate_blocks to one of four pre-bound functions selected from the (enable_caching, metrics_collector is not None) cross-product, so the per-block `if self.enable_caching` and `if self.metrics_collector` checks (lines 339, 344, 350) disappear from the hot path. Inside the cached variant, restructure _maybe_evict_cached_block (lines 354-379) into a bulk helper that: (1) iterates `ret` once to collect (block, block_hash) tuples for blocks where block_hash is not None; (2) issues a single `dict.pop`-style batch delete against `cached_block_hash_to_block` (e.g., via a list comprehension that calls pop, or by inlining the dict access without the intermediate function call); (3) calls `block.reset_hash()` and ref_cnt mutation on each block. Also add a batched metrics surface — `KVCacheMetricsCollector.on_blocks_allocated(blocks)` and `on_blocks_evicted(blocks)` in vllm/v1/core/kv_cache_metrics.py — that does the random.random() sample-rate filter once per call (e.g., generates a numpy bernoulli mask with `np.random.random(len(blocks)) < self.sample_rate`) instead of per-block, and bulk-creates BlockMetricsState entries with a single `time.monotonic_ns()` read. Keep `assert block.ref_cnt == 0` only when `__debug__` is true (it's already a no-op under -O, but document this). Net effect: large prefill chunks (num_blocks in the hundreds) collapse from N Python-level method calls to a handful, which is the slab-style allocation path the candidate's evolve_rationale explicitly flags. Validate against tests/v1/core/test_prefix_caching.py and tests/v1/core/test_single_type_kv_cache_manager.py for invariant preservation, and add a microbenchmark that times get_new_blocks(256) with caching on/off and metrics on/off to confirm the speedup.

**Novelty rationale.**

The single existing deep_research_proposal (find-0008) changes *which* cached block is selected for eviction by adding a priority/TTL signal — i.e., it modifies allocation policy but keeps the per-block Python loop intact. This proposal is orthogonal and complementary: it does not change which block is chosen, it changes the per-block dispatch cost by (a) eliminating the per-iteration branch on enable_caching/metrics_collector through __init__-time specialization, (b) batching the cached_block_hash_to_block dict updates and metrics callbacks instead of paying one method call per block, and (c) restructuring the metrics collector to amortize random sampling and time reads across the whole batch. The evolve_rationale explicitly calls out 'specializing away metrics checks when absent, batching metrics callbacks, reducing per-block eviction map work, and adding slab-style allocation paths' — all four levers — and the existing proposal addresses none of them.

---

### 2. Prefer uncached free blocks before evicting prefix-cache blocks
- **Agent:** codex

**Detailed description.**

Change BlockPool.get_new_blocks (vllm/v1/core/block_pool.py:322-352) so allocation first consumes free blocks whose block_hash is None, and only falls back to cached free blocks when no clean blocks remain. Implement this by extending the free-list bookkeeping with a clean tier and a cached-evictable tier, or an equivalent helper such as popleft_clean_then_cached_n(num_blocks). free_blocks should enqueue ref_cnt-zero unhashed blocks into the clean tier and hashed blocks into the cached tier; touch should remove cached hits from the tier they occupy; get_num_free_blocks should return the combined count. In get_new_blocks, blocks drawn from the clean tier can skip _maybe_evict_cached_block entirely, while cached-tier blocks keep the existing eviction/reset/event path. Add a test that frees a cached full prefix block, then frees an uncached partial/decode block behind it, and verifies a later allocation consumes the uncached block while the cached hash remains available.

**Novelty rationale.**

The existing deep_research proposal changes eviction policy among cached blocks by adding priority/TTL metadata. This proposal avoids eviction altogether when a non-reusable free block is available, without adding request-level retention tags. Agent A's proposal focuses on reducing per-block Python overhead through specialization and batching while preserving the current candidate order from popleft_n; this proposal changes the allocation order to preserve prefix-cache residency under mixed cached/uncached free lists, which is a distinct hit-rate and TTFT lever.

---
