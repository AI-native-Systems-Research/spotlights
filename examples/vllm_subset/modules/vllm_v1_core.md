# vllm/v1/core

[← All modules](../index.md)

## Module
- **Path:** `vllm/v1/core`
- **Description:** Scheduler and KV-cache manager: block allocation, prefix caching, and step scheduling.
- **Depends on:** _(none)_
- **Main files:**
  - `vllm/v1/core/kv_cache_manager.py` — Block-level KV cache manager with prefix caching
  - `vllm/v1/core/kv_cache_coordinator.py` — Coordinates KV blocks across layers/adapters
  - `vllm/v1/core/block_pool.py` — Physical KV block pool
- **Run status:** DEGRADED
- **Findings:** 13
- **Issues:** 2

## Candidates

| Candidate | Impact | Deep research proposals |
|---|---|---:|
| [`BlockPool.free_blocks`](vllm_v1_core/BlockPool.free_blocks__cand-vllm_v1_core-0005.md) | medium | 7 |
| [`Scheduler.schedule`](vllm_v1_core/Scheduler.schedule__cand-vllm_v1_core-0009.md) | high | 4 |
| [`MambaManager.reachable_block_mask`](vllm_v1_core/MambaManager.reachable_block_mask__cand-vllm_v1_core-0022.md) | medium | 3 |
| [`SlidingWindowManager.reachable_block_mask`](vllm_v1_core/SlidingWindowManager.reachable_block_mask__cand-vllm_v1_core-0021.md) | medium | 2 |
| [`KVCacheManager.allocate_slots`](vllm_v1_core/KVCacheManager.allocate_slots__cand-vllm_v1_core-0012.md) | high | 1 |
| [`HybridKVCacheCoordinator.find_longest_cache_hit`](vllm_v1_core/HybridKVCacheCoordinator.find_longest_cache_hit__cand-vllm_v1_core-0001.md) | high | 0 |
| [`FullAttentionManager.find_longest_cache_hit`](vllm_v1_core/FullAttentionManager.find_longest_cache_hit__cand-vllm_v1_core-0002.md) | high | 0 |
| [`SlidingWindowManager.find_longest_cache_hit`](vllm_v1_core/SlidingWindowManager.find_longest_cache_hit__cand-vllm_v1_core-0003.md) | medium | 0 |
| [`BlockPool.get_new_blocks`](vllm_v1_core/BlockPool.get_new_blocks__cand-vllm_v1_core-0004.md) | high | 0 |
| [`BlockHashToBlockMap and BlockPool.get_cached_block`](vllm_v1_core/BlockHashToBlockMap_and_BlockPool.get_cached_block__cand-vllm_v1_core-0006.md) | high | 0 |
| [`get_request_block_hasher.request_block_hasher`](vllm_v1_core/get_request_block_hasher.request_block_hasher__cand-vllm_v1_core-0007.md) | high | 0 |
| [`FreeKVCacheBlockQueue`](vllm_v1_core/FreeKVCacheBlockQueue__cand-vllm_v1_core-0008.md) | medium | 0 |
| [`Scheduler._make_cached_request_data`](vllm_v1_core/Scheduler._make_cached_request_data__cand-vllm_v1_core-0010.md) | medium | 0 |
| [`Scheduler._mamba_block_aligned_split`](vllm_v1_core/Scheduler._mamba_block_aligned_split__cand-vllm_v1_core-0011.md) | medium | 0 |
| [`SingleTypeKVCacheManager.get_num_blocks_to_allocate`](vllm_v1_core/SingleTypeKVCacheManager.get_num_blocks_to_allocate__cand-vllm_v1_core-0013.md) | medium | 0 |
| [`MambaManager.allocate_new_blocks`](vllm_v1_core/MambaManager.allocate_new_blocks__cand-vllm_v1_core-0014.md) | medium | 0 |
| [`KVCacheCoordinator.get_num_blocks_to_allocate`](vllm_v1_core/KVCacheCoordinator.get_num_blocks_to_allocate__cand-vllm_v1_core-0015.md) | low | 0 |
| [`BlockHashListWithBlockSize`](vllm_v1_core/BlockHashListWithBlockSize__cand-vllm_v1_core-0016.md) | medium | 0 |
| [`MambaManager.find_longest_cache_hit`](vllm_v1_core/MambaManager.find_longest_cache_hit__cand-vllm_v1_core-0019.md) | medium | 0 |
| [`BlockPool.cache_full_blocks`](vllm_v1_core/BlockPool.cache_full_blocks__cand-vllm_v1_core-0020.md) | medium | 0 |
| [`ChunkedLocalAttentionManager.find_longest_cache_hit`](vllm_v1_core/ChunkedLocalAttentionManager.find_longest_cache_hit__cand-vllm_v1_core-0023.md) | medium | 0 |

## Findings (full list)

1. **KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows**
   - Source type: paper
   - URL: <https://www.alphaxiv.org/abs/2507.07400>
   - Technique: Use workflow-aware eviction and prefetching: model an agent workflow as an Agent Step Graph, assign each agent/cache node a steps-to-execution priority, retain soon-needed shared prefixes, evict dynamic suffixes first, and prefetch upcoming KV blocks from CPU to GPU. This directly targets multi-turn agentic TTFT regressions caused by LRU evicting paused sessions before reuse.
   - Evidence: Quote: "KVFlow adopts a workflow-aware eviction strategy that prioritizes evicting KV caches belonging to agents with large steps-to-execution." Pointer: abstract/design overview, lines describing runtime optimization and Section 3.1.
2. **[RFC]: Cache-affinity-aware request ordering for the V1 scheduler**
   - Source type: issue
   - URL: <https://github.com/vllm-project/vllm/issues/42185>
   - Technique: Reorder the waiting queue by cached-prefix length before scheduling, with priority-class preservation, score bucketing, sticky handling for preempted requests, and a starvation deadline. This could lower median TTFT by running cache-warm multi-turn continuations while their prefixes are still resident.
   - Evidence: Quote: "reorders the V1 waiting queue by cached-prefix length before each scheduling iteration." Pointer: Summary and Proposed approach, issue lines 161-186.
3. **[RFC]: Context-Aware KV-Cache Retention API (Prioritized Evictions)**
   - Source type: issue
   - URL: <https://github.com/vllm-project/vllm/issues/37003>
   - Technique: Add token-range retention directives with priority and TTL, backed by a two-structure evictor: the current LRU free queue for unprioritized blocks plus a priority queue for annotated blocks. This lets an orchestrator protect shared agent prefixes during tool-call pauses without hard-pinning all blocks.
   - Evidence: Quote: "A token-range retention directive that lets the orchestrator annotate requests with per-range eviction priorities." Pointer: Proposed Change, issue lines 176-202.
4. **Marconi: Prefix Caching for the Era of Hybrid LLMs**
   - Source type: paper
   - URL: <https://huggingface.co/papers/2411.19379>
   - Technique: Adopt admission and eviction scoring based on predicted reuse scenarios and compute-savings-per-memory-footprint rather than recency alone. This is especially relevant to hybrid attention/Mamba cache retention, where sparse recurrent-state checkpoints can be large and not equally valuable.
   - Evidence: Quote: "novel admission and eviction policies" assess "reuse likelihood" and "compute savings" relative to memory. Pointer: abstract, lines 72-75.
5. **Learned Prefix Caching for Efficient LLM Inference**
   - Source type: paper
   - URL: <https://papers.neurips.cc/paper_files/paper/2025/hash/414f642a1ea9350006669774cba9bcd4-Abstract-Conference.html>
   - Technique: Use lightweight conversational-content features plus last-access timestamps to predict which conversations are likely to continue, then bias prefix-cache eviction accordingly. This could improve retention for paused multi-turn sessions where LRU mistakes inactivity for low future value.
   - Evidence: Quote: "LPC leverages conversational content analysis to provide predictive guidance for eviction." Pointer: NeurIPS 2025 abstract, lines 7-9.
6. **RadixAttention**
   - Source type: docs
   - URL: <https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention>
   - Technique: Maintain a compressed radix tree over token prefixes with node splitting, reference counts along matched paths, page alignment, and scheduler integration that prioritizes longer prefix matches. Even without replacing the hash cache, the module could adapt a trie-like secondary index or prefix-match ordering to reduce repeated hash probes and improve cache-aware admission.
   - Evidence: Quote: "RadixAttention combines two key innovations: Radix Tree Data Structure" and attention over shared tree structure. Pointer: What is RadixAttention, lines 122-126; scheduler integration lines 386-405.
7. **CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**
   - Source type: paper
   - URL: <https://axi.lims.ac.uk/paper/2405.16444>
   - Technique: Support non-prefix chunk reuse by combining precomputed KV caches and selectively recomputing only a small token subset to correct cross-attention dependencies, with KV retrieval pipelined alongside recomputation. The transferable idea is a request-only fragmented reuse path that avoids exact-prefix cache pollution while reducing prefill work for RAG and agent prompts with reordered context.
   - Evidence: Quote: "selectively recomputes the KV values of a small subset of tokens" to update reused KV caches. Pointer: abstract, lines 16-20.
8. **[RFC]: Semantic KV Cache Reuse Interface**
   - Source type: issue
   - URL: <https://github.com/vllm-project/vllm/issues/44223>
   - Technique: Introduce an explicit external KV cache-commit policy, such as EXACT_COMMIT versus REQUEST_ONLY, so externally loaded or approximate donor KV can be used for the current request without being inserted into the exact prefix-cache map. This would make scheduler allocation paths safer and more flexible for semantic or connector-provided reuse.
   - Evidence: Quote: "The minimal generic extension is a connector-controlled cache policy for external tokens." Pointer: Required Upstream Gap: Cache-Commit Policy, lines 768-794.
9. **TinyLFU: A Highly Efficient Cache Admission Policy**
   - Source type: paper
   - URL: <https://doczz.net/doc/8550019/tinylfu--a-highly-efficient-cache-admission-policy>
   - Technique: Add approximate frequency-based cache admission on top of the existing eviction policy, using a compact sketch/doorkeeper and optionally W-TinyLFU’s small recency window. This could prevent one-off or low-reuse KV blocks from displacing hot shared prefixes under memory pressure.
   - Evidence: Quote: "TinyLFU decides if replacing the cache victim with the new item is expected to increase the hit-ratio." Pointer: TinyLFU Architecture, lines 242-246.
10. **SIEVE: Cache eviction can be simple, effective, and scalable**
   - Source type: blog
   - URL: <https://www.usenix.org/publications/loginonline/sieve-cache-eviction-can-be-simple-effective-and-scalable>
   - Technique: Use lazy promotion and quick demotion with a single visited bit and moving eviction hand, keeping popular objects in place while quickly removing one-hit entries. For a KV block pool, this suggests reducing per-hit queue mutation while improving resistance to cache pollution from ephemeral prompts.
   - Evidence: Quote: "an efficient cache eviction algorithm should use lazy promotion and quick demotion." Pointer: eviction algorithm discussion, lines 42-49.
11. **SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills**
   - Source type: paper
   - URL: <https://arxiv.gg/abs/2308.16369>
   - Technique: Use equal-sized chunked prefills and decode-maximal batching: construct each iteration from one bounded prefill chunk plus as many decodes as possible. The scheduler already has chunking machinery, but Sarathi’s uniform-work and decode-maximal policy is a concrete tuning target for reducing TPOT spikes while preserving TTFT.
   - Evidence: Quote: "decode-maximal batching, which constructs a batch using a single prefill chunk and populates the remaining slots with decodes." Pointer: abstract, lines 23-27.
12. **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**
   - Source type: paper
   - URL: <https://www.alphaxiv.org/abs/2401.09670>
   - Technique: Disaggregate prefill and decode scheduling so each phase has separate resource allocation, queues, and optimization targets, with KV handoff between phases. Within a single-engine scheduler, the adaptable idea is to isolate cold prefill/connector work from hot decode paths and gate admission by phase-specific TTFT/TPOT pressure.
   - Evidence: Quote: "Disaggregating prefill and decoding naturally resolves the interference between the two phases." Pointer: design overview, lines 58-63.
13. **Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching**
   - Source type: paper
   - URL: <https://ojs.aaai.org/index.php/AAAI/article/view/39224>
   - Technique: Expose a scheduler-to-runtime prefetch plan for upcoming KV blocks so idle memory bandwidth can prefetch them into GPU L2 before attention reads. While the kernel work is outside the module, the scheduler and KV manager can provide stable upcoming block IDs and access order to enable this latency-hiding path.
   - Evidence: Quote: "proactively prefetches required KV Cache into GPU L2 cache" during active computation windows. Pointer: abstract, lines 32-35.

## Issues

### module_deep_research
- **[warning, recoverable]** codex: Local ripgrep was unavailable in the sandbox; file inspection used sed instead.
- **[warning, recoverable]** codex: The Ray prefix-aware routing documentation result was found in search, but opening it returned HTTP 429, so it was not used as a finding source.
