# vllm/v1/kv_offload

[← All modules](../index.md)

## Module
- **Path:** `vllm/v1/kv_offload`
- **Description:** Multi-tier KV-offload framework with pluggable CPU/host tiers.
- **Depends on:** _(none)_
- **Main files:**
  - `vllm/v1/kv_offload/base.py` — Offload backend interface
  - `vllm/v1/kv_offload/factory.py` — Backend factory
- **Run status:** DEGRADED
- **Findings:** 12
- **Issues:** 1

## Candidates

| Candidate | Impact | Deep research proposals |
|---|---|---:|
| [`TieringOffloadingManager._initiate_promotion`](vllm_v1_kv_offload/TieringOffloadingManager._initiate_promotion__cand-vllm_v1_kv_offload-0009.md) | high | 9 |
| [`SecondaryTierFactory built-in registrations`](vllm_v1_kv_offload/SecondaryTierFactory_built-in_registrations__cand-vllm_v1_kv_offload-0014.md) | high | 6 |
| [`ObjectStoreSecondaryTierManager._submit_transfer/_poll_active_transfers`](vllm_v1_kv_offload/ObjectStoreSecondaryTierManager._submit_transfer__poll_active_transfers__cand-vllm_v1_kv_offload-0017.md) | medium | 6 |
| [`LRUCachePolicy.evict`](vllm_v1_kv_offload/LRUCachePolicy.evict__cand-vllm_v1_kv_offload-0003.md) | medium | 3 |
| [`TieringOffloadingManager.lookup`](vllm_v1_kv_offload/TieringOffloadingManager.lookup__cand-vllm_v1_kv_offload-0008.md) | high | 3 |
| [`CachePolicyFactory built-in registrations`](vllm_v1_kv_offload/CachePolicyFactory_built-in_registrations__cand-vllm_v1_kv_offload-0013.md) | high | 3 |
| [`FileSystemTierManager.submit_store/submit_load`](vllm_v1_kv_offload/FileSystemTierManager.submit_store_submit_load__cand-vllm_v1_kv_offload-0015.md) | medium | 3 |
| [`ARCCachePolicy.evict`](vllm_v1_kv_offload/ARCCachePolicy.evict__cand-vllm_v1_kv_offload-0001.md) | high | 2 |
| [`DualQueueThreadPool._worker`](vllm_v1_kv_offload/DualQueueThreadPool._worker__cand-vllm_v1_kv_offload-0011.md) | medium | 2 |
| [`CPUOffloadingManager.prepare_store`](vllm_v1_kv_offload/CPUOffloadingManager.prepare_store__cand-vllm_v1_kv_offload-0012.md) | medium | 2 |
| [`ARCCachePolicy.touch`](vllm_v1_kv_offload/ARCCachePolicy.touch__cand-vllm_v1_kv_offload-0002.md) | medium | 1 |
| [`SingleDirectionOffloadingHandler.transfer_async`](vllm_v1_kv_offload/SingleDirectionOffloadingHandler.transfer_async__cand-vllm_v1_kv_offload-0006.md) | high | 1 |
| [`_select_swap_blocks_fn`](vllm_v1_kv_offload/_select_swap_blocks_fn__cand-vllm_v1_kv_offload-0004.md) | high | 0 |
| [`NUM_SMS/THRESHOLD_BYTES/MIN_N`](vllm_v1_kv_offload/NUM_SMS_THRESHOLD_BYTES_MIN_N__cand-vllm_v1_kv_offload-0005.md) | medium | 0 |
| [`compute_sub_block_ptrs`](vllm_v1_kv_offload/compute_sub_block_ptrs__cand-vllm_v1_kv_offload-0007.md) | medium | 0 |
| [`AsyncLookupManager._worker`](vllm_v1_kv_offload/AsyncLookupManager._worker__cand-vllm_v1_kv_offload-0010.md) | medium | 0 |
| [`batch_store_block/batch_load_block`](vllm_v1_kv_offload/batch_store_block_batch_load_block__cand-vllm_v1_kv_offload-0016.md) | medium | 0 |
| [`ServerRole.add_stored_blocks/on_fetch`](vllm_v1_kv_offload/ServerRole.add_stored_blocks_on_fetch__cand-vllm_v1_kv_offload-0018.md) | high | 0 |
| [`ServerRole.serve_external_requests/_process_inbound_lookup`](vllm_v1_kv_offload/ServerRole.serve_external_requests__process_inbound_lookup__cand-vllm_v1_kv_offload-0019.md) | medium | 0 |

## Findings (full list)

1. **Serving Agentic Workloads at Scale with vLLM x Mooncake**
   - Source type: blog
   - URL: <https://vllm.ai/blog/2026-05-06-mooncake-store>
   - Technique: Adopt a shared distributed KV cache pool for multi-turn agentic workloads, with scheduler-side block lookup and worker-side asynchronous data movement. This directly targets median TTFT by preserving prefixes across turns and workers instead of relying only on per-instance CPU or disk tiers.
   - Evidence: Quote: "The cached prefix ... is reused turn after turn" and "all RDMA operations run on a dedicated background I/O thread." Pointer: Agentic workloads, Design highlights, lines 37-39 and 74-79.
2. **Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo**
   - Source type: blog
   - URL: <https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/>
   - Technique: Expose prefetch and retention controls so an agent harness can tell the cache which blocks will be needed after tool calls, which blocks to pin, and which to evict first. The target module could use request metadata to admit, promote, or retain blocks by workflow value rather than pure recency.
   - Evidence: Quote: "bring these blocks from storage to GPU ahead of the next request" and "lower-priority blocks are evicted first." Pointer: KV cache management, lines 128-136.
3. **[RFC]: Progressive KV Cache CPU Onloading**
   - Source type: issue
   - URL: <https://github.com/vllm-project/vllm/issues/33526>
   - Technique: Split large CPU-to-GPU reloads into progressive batches, with smaller early batches for the first part of a request and larger later batches to balance overhead. This can reduce head-of-line blocking when a short request shares a prefix with a longer onloading request, improving TTFT without changing store semantics.
   - Evidence: Quote: "Instead of a single block transfer request per inference request, we can split up our transfers into multiple." Pointer: Proposed Change, lines 176-185.
4. **FIFO queues are all you need for cache eviction**
   - Source type: blog
   - URL: <https://s3fifo.com/blog/2023/08/01/fifo-queues-are-all-you-need-for-cache-eviction/>
   - Technique: Add an S3-FIFO-style cache policy using a small probationary FIFO, main FIFO, ghost FIFO, and tiny per-object counters to quickly demote one-hit blocks. This is a concrete replacement or sibling policy for LRU/ARC that may protect hot multi-turn prefixes from cache pollution while keeping eviction metadata simple.
   - Evidence: Quote: "S3-FIFO uses three FIFO queues: a small FIFO queue (S), a main FIFO queue (M), and a ghost FIFO queue (G)." Pointer: S3-FIFO algorithm, lines 141-144.
5. **TinyLFU: A Highly Efficient Cache Admission Policy**
   - Source type: paper
   - URL: <https://paperity.org/p/377179711/tinylfu-a-highly-efficient-cache-admission-policy>
   - Technique: Use an approximate frequency sketch as an admission gate: compare an incoming block's recent frequency with the victim's and reject low-value candidates. This could improve CPU-tier store decisions and reduce eviction churn for repeated agent prefixes under capacity pressure.
   - Evidence: Quote: "decides, based on the recent access history, whether it is worth admitting the new item." Pointer: Abstract preview, lines 23-28.
6. **Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving**
   - Source type: paper
   - URL: <https://arxiv.org/abs/2605.03375>
   - Technique: For SSD secondary tiers, use larger KV-cache object abstractions, asynchronous GPU direct object I/O, and slack-aware scheduling to avoid fragmented tiny random I/O and GPU stalls. The design suggests replacing per-block filesystem tasks with chunked object transfers and timing-aware load scheduling.
   - Evidence: Quote: "GPU-native object abstraction that enables bulk KV cache transfers and management." Pointer: Abstract, lines 16-20.
7. **GPUDirect Storage Overview Guide**
   - Source type: docs
   - URL: <https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html>
   - Technique: Use batched asynchronous file I/O and CUDA stream ordering for storage-backed KV loads and stores. This maps directly to filesystem tier submission and chunking: group adjacent block I/Os to amortize fixed per-submission cost and coordinate storage completion with GPU work.
   - Evidence: Quote: "batching reduces the overhead by amortizing that fixed overhead across the transactions in the batch." Pointer: Sections 1.2.2.4-1.2.2.5, lines 220-229.
8. **Enhancing Distributed Inference Performance with the NVIDIA Inference Transfer Library**
   - Source type: blog
   - URL: <https://developer.nvidia.com/blog/?p=113426>
   - Technique: Model transfers as nonblocking descriptor-list operations over pluggable memory and storage backends, with metadata registration and polling. This supports improving object-store and P2P tiers by batching descriptors, reusing prepared handles, and choosing the backend path per tier without changing higher-level semantics.
   - Evidence: Quote: "The user describes any memory or storage through a list of descriptors." Pointer: NIXL design, lines 78-88.
9. **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**
   - Source type: paper
   - URL: <https://arxiv.org/abs/2310.07240>
   - Technique: Compress offloaded KV tensors with a custom tensor encoder and adapt compression level to available bandwidth. This could reduce CPU-storage or remote-tier transfer volume and improve TTFT when promotions are bandwidth-bound, especially for long contexts.
   - Evidence: Quote: "CacheGen adapts the compression level of different parts of a KV cache to cope with changes in available bandwidth." Pointer: Abstract, lines 15-17.
10. **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**
   - Source type: paper
   - URL: <https://www.alphaxiv.org/abs/2406.19707>
   - Technique: Prefetch only essential KV entries by speculating next-layer attention with a lightweight rehearsal, and evict infrequently selected entries from the CPU pool. While more model-aware than the current block-level module, the transferable idea is promotion admission based on predicted usefulness instead of blindly recalling every available block.
   - Evidence: Quote: "prefetch only the essential KV cache entries (without fetching them all)." Pointer: Abstract and design summary, lines 21-40.
11. **ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs**
   - Source type: paper
   - URL: <https://www.usenix.org/conference/osdi26/presentation/liu-guangda>
   - Technique: Exploit predictable future KV use to prefetch during computation and overlap recall with another kernel's work. The target module could adapt this as a tier-promotion scheduler that starts secondary-to-primary loads before demand blocks the request, reducing exposed TTFT and TPOT stalls.
   - Evidence: Quote: "ECHO enables lossless intra-query prefetching for decoding and inter-query prefetching for prefill." Pointer: USENIX abstract, lines 7-10.
12. **Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching**
   - Source type: paper
   - URL: <https://ojs.aaai.org/index.php/AAAI/article/view/39224>
   - Technique: Schedule KV prefetches into idle memory-bandwidth windows and overlap them with compute. Although the paper targets GPU L2/HBM, the same latency-hiding principle can guide CPU/GPU offload path timing and chunk sizes to reduce TPOT impact during decode.
   - Evidence: Quote: "strategically scheduling idle memory bandwidth during active computation windows." Pointer: Abstract, lines 32-35.

## Issues

### module_deep_research
- **[warning, recoverable]** codex: Some searched pages failed to fetch directly, including the SGLang HiCache GitHub page and one Dynamo docs URL; equivalent reachable sources were used where possible.
