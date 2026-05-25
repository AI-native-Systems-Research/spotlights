# distributed.kv_transfer

[← All modules](../index.md)

## Module
- **Path:** `vllm/distributed/kv_transfer`
- **Description:** Disaggregated prefill/decode KV-cache transport between engine instances.
- **Depends on:** _(none)_
- **Main files:**
  - `vllm/distributed/kv_transfer/kv_transfer_state.py` — KV transfer state machine.
- **Run status:** DEGRADED
- **Findings:** 22
- **Issues:** 1

## Candidates

| Candidate | Impact | Deep research proposals |
|---|---|---:|
| [`KVConnectorFactory connector registration block`](distributed.kv_transfer/KVConnectorFactory_connector_registration_block__cand-0001.md) | high | 22 |
| [`NixlConnectorScheduler.get_num_new_matched_tokens`](distributed.kv_transfer/NixlConnectorScheduler.get_num_new_matched_tokens__cand-0002.md) | high | 7 |
| [`AsyncOperationManager._handle_save_task / _handle_load_task`](distributed.kv_transfer/AsyncOperationManager._handle_save_task____handle_load_task__cand-0011.md) | medium | 6 |
| [`NixlConnectorWorker.sync_recved_kv_to_device / save_kv_to_host`](distributed.kv_transfer/NixlConnectorWorker.sync_recved_kv_to_device___save_kv_to_host__cand-0004.md) | medium | 4 |
| [`P2pNcclEngine.send / P2pNcclEngine.recv`](distributed.kv_transfer/P2pNcclEngine.send___P2pNcclEngine.recv__cand-0006.md) | high | 4 |
| [`TensorMemoryPool allocate/free/store/load staging path`](distributed.kv_transfer/TensorMemoryPool_allocate_free_store_load_staging_path__cand-0007.md) | medium | 3 |
| [`OffloadingConnectorScheduler._lookup`](distributed.kv_transfer/OffloadingConnectorScheduler._lookup__cand-0008.md) | high | 3 |
| [`MoRIIOConnectorWorker handshake-ready spin loops`](distributed.kv_transfer/MoRIIOConnectorWorker_handshake-ready_spin_loops__cand-0013.md) | medium | 3 |
| [`MooncakeConnectorWorker._build_transfer_params`](distributed.kv_transfer/MooncakeConnectorWorker._build_transfer_params__cand-0010.md) | high | 2 |
| [`MoRIIOWriter._write_worker_loop / _process_deferred_tasks`](distributed.kv_transfer/MoRIIOWriter._write_worker_loop____process_deferred_tasks__cand-0012.md) | medium | 2 |
| [`HF3FSKVConnector.get_num_new_matched_tokens`](distributed.kv_transfer/HF3FSKVConnector.get_num_new_matched_tokens__cand-0014.md) | high | 2 |
| [`NixlConnectorWorker post-receive KV post-processing`](distributed.kv_transfer/NixlConnectorWorker_post-receive_KV_post-processing__cand-0005.md) | medium | 1 |
| [`NixlConnectorWorker._pop_done_transfers`](distributed.kv_transfer/NixlConnectorWorker._pop_done_transfers__cand-0003.md) | medium | 0 |
| [`OffloadingConnectorScheduler._build_store_jobs`](distributed.kv_transfer/OffloadingConnectorScheduler._build_store_jobs__cand-0009.md) | medium | 0 |

## Findings (full list)

1. **[PD Disaggregation] Avoid Transferring Prefix Cache’s KVCache of Decode Node**
   - Source type: pr
   - URL: <https://github.com/sgl-project/sglang/pull/7990>
   - Technique: Teach the transfer path to subtract any prefix KV cache already present on the decode node, then transfer only the remaining delta. This directly targets multi-turn agentic workloads where repeated conversation history can otherwise be retransferred on every turn, reducing TTFT and transfer bandwidth.
   - Evidence: Quote: "This PR prevents the transfer of KVCache that already exists on the decode node (s) from the prefill node (s)." Pointer: PR motivation, lines 4-7.
2. **[Roadmap] Prefill-Decode Disaggregation Roadmap (2026 Q2)**
   - Source type: issue
   - URL: <https://github.com/sgl-project/sglang/issues/21703>
   - Technique: Add a delta-KV transfer mode where decode fetches the shared prefix from a hierarchical cache and prefill sends only newly computed KV. This is especially aligned with multi-turn agentic traffic, where most turns share long prefixes and only append small deltas.
   - Evidence: Quote: "For agentic scenarios, decode can now use radix cache to reuse shared prefixes and request only the delta KV from prefill instead of transferring the full prefix on every turn." Pointer: issue section "Prefill send delta KVCache while Decode fetch prefix from Hicache for agentic use cases", line 6.
3. **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**
   - Source type: paper
   - URL: <https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf>
   - Technique: Compress transferred KV tensors with a KV-specific encoder and adapt compression level per chunk based on available bandwidth. This could reduce KV transport time on the TTFT path when remote prefill or remote cache hits fetch large contexts.
   - Evidence: Quote: "CacheGen uses a custom tensor encoder, leveraging KV cache’s distributional properties to encode a KV cache into more compact bitstream representations with negligible decoding overhead, to save bandwidth usage." Pointer: abstract, lines 1-3.
4. **[PD] optimize kv cache transfer directly using batch transfer**
   - Source type: pr
   - URL: <https://github.com/sgl-project/sglang/pull/9149>
   - Technique: Batch all layer-transfer descriptors into a single transfer operation instead of issuing per-layer executor work. This reduces launch/control overhead in KV transfer and can lower TTFT for large models with many layers.
   - Evidence: Quote: "Pack all layers' transfer parameters to one single batch. Call batch transfer interface directly instead of using multiple executors to fully unleash the potential of batch transfer." Pointer: PR motivation/modifications, line 1.
5. **Disaggregated Serving — TensorRT LLM**
   - Source type: docs
   - URL: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc4/features/disagg-serving.html>
   - Technique: Overlap KV cache transmission for one request with computation for other independent requests. A transfer scheduler that keeps forward passes moving while sends/receives complete can reduce median TPOT impact from KV movement and smooth TTFT under load.
   - Evidence: Quote: "TensorRT LLM overlaps the KV cache transmission with computation for multiple independent requests. While one request is sending or receiving its KV cache blocks, other requests can proceed with computation." Pointer: "Overlap Optimization" search excerpt.
6. **[Roadmap] Prefill-Decode Disaggregation Roadmap (2026 Q2)**
   - Source type: issue
   - URL: <https://github.com/sgl-project/sglang/issues/21703>
   - Technique: Use a GPU staging buffer with a dynamic ring allocator to gather many small heterogeneous-TP KV head slices into larger contiguous transfers, then scatter on receive. This can improve transfer efficiency where many small RDMA operations dominate TTFT.
   - Evidence: Quote: "This feature will gather the KV heads at prefill, then send the kvcache to decode by uniting them as a large piece through a ring-based staging buffer." Pointer: issue section "GPU staging buffer for accelerating heterogeneous TP KV transfer", lines 3-4.
7. **Prefill-decode disaggregation — Ray 2.55.1**
   - Source type: docs
   - URL: <https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html>
   - Technique: Pre-warm KV transfer connectors so the first real request does not pay handshake setup latency. This is directly applicable to remote prefill/decode connectors that currently establish remote metadata, agents, or transport handles lazily.
   - Evidence: Quote: "KV transfer connectors (such as NIXL) require a handshake between each prefill and decode replica that happens eagerly upon the first request." Pointer: "Pre-warming the connector" search excerpt.
8. **Disaggregated Serving | NVIDIA Dynamo Documentation**
   - Source type: docs
   - URL: <https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving>
   - Technique: Run prefill and KV transfer as non-blocking background work so decode-side computation and other forward passes can proceed while transfer is in flight. This suggests extending transfer metadata/state to support earlier decode admission and finer-grained readiness rather than waiting for a synchronous prefill handoff.
   - Evidence: Quote: "The KV transfer is non-blocking, allowing GPU forward passes to continue serving other requests during the transfer." Pointer: design doc "Efficient KV Transfer", lines 2-3.
9. **Mooncake/docs/source/design/mooncake-store.md at main · kvcache-ai/Mooncake**
   - Source type: docs
   - URL: <https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/mooncake-store.md>
   - Technique: Adopt distributed-cache transport features such as multi-replica object placement, zero-copy transfers, multi-NIC RDMA pooling, and striping for large KV objects. These methods can reduce hot-spotting and improve transfer bandwidth for shared prefixes in multi-turn workloads.
   - Evidence: Quote: "Mooncake Store supports storing multiple data replicas for the same object, effectively alleviating hotspots in access pressure." Pointer: design doc key features, lines 3-5.
10. **LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference**
   - Source type: paper
   - URL: <https://arxiv.org/abs/2510.09665>
   - Technique: Use batched KV data movement, compute/I/O pipelining, and a first-class cache-control API spanning GPU, CPU, storage, and network tiers. This could make KV transfer less connector-specific and better able to overlap movement with execution for lower TTFT.
   - Evidence: Quote: "LMCACHE's high performance and wide adoption stem from the following contributions: (1) highly optimized KV cache data movement powered by batched data movement operations, compute and I/O pipelining." Pointer: abstract, lines 2-3.
11. **HiSparse: Hierarchical Sparse Attention - SGLang Documentation**
   - Source type: docs
   - URL: <https://docs.sglang.io/docs/advanced_features/hisparse_guide>
   - Technique: For sparse-attention models, transfer remote prefill KV directly into the decode host pool, keeping only a small hot KV buffer on GPU and swapping top-k entries on demand. This reduces transient GPU memory pressure during transfer and can increase decode concurrency, improving TPOT under long-context load.
   - Evidence: Quote: "In PD disaggregation mode, the prefill instance transfers KV cache directly into the decode instance’s host pool via RDMA, bypassing the GPU entirely on the decode side." Pointer: "PD Disaggregation Integration (Direct-to-Host)", lines 4-5.
12. **MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool**
   - Source type: paper
   - URL: <https://arxiv.org/abs/2406.17565>
   - Technique: Introduce an elastic distributed memory pool and global prompt-tree locality policy for KV cache reuse across disaggregated serving instances. A KV transfer layer could use this idea to coordinate remote-cache lookups and placement rather than treating each P/D transfer as a one-off point-to-point copy.
   - Evidence: Quote: "MemServe introduces MemPool, an elastic memory pool managing distributed memory and KV caches across serving instances." Pointer: arXiv abstract, lines 1-2.
13. **Router Guide | NVIDIA Dynamo Documentation**
   - Source type: docs
   - URL: <https://docs.dynamo.nvidia.com/dynamo/user-guides/kv-cache-aware-routing>
   - Technique: Make routing and transfer decisions from a cost model that combines active decode load, new prefill work, and KV overlap. The transfer module could expose overlap, inflight transfer, and block-availability signals so callers route multi-turn requests to endpoints that avoid redundant transfer and recompute.
   - Evidence: Quote: "It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks), using KV cache overlap to minimize redundant computation." Pointer: overview, lines 0-4.
14. **Cache-aware prefill–decode disaggregation (CPD) for up to 40% faster long-context LLM serving**
   - Source type: blog
   - URL: <https://www.together.ai/blog/cache-aware-disaggregated-inference>
   - Technique: Separate warm and cold prefill paths by cache-hit rate so cache-reusable requests do not queue behind large cold prefills. KV transfer could contribute by reporting cache hit/miss and transfer-cost classes that let the caller prioritize warm delta transfers for lower median TTFT.
   - Evidence: Quote: "CPD, a serving architecture that purposely separates cold and warm workloads by cache hit rate, resulting in fast context reuse." Pointer: blog introduction, lines 1-6.
15. **Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot**
   - Source type: paper
   - URL: <https://www.usenix.org/system/files/fast25-qin.pdf>
   - Technique: Treat KV cache as a first-class distributed resource backed by GPU-cluster CPU, DRAM, SSD, and NIC capacity, with a KVCache-centric scheduler. This suggests broadening point-to-point KV transfer into a tier-aware cache fabric that can trade storage/transfer for less prefill recompute in long multi-turn contexts.
   - Evidence: Quote: "This platform features a KVCache-centric disaggregated architecture that not only separates prefill and decoding clusters but also efficiently utilizes the underexploited CPU, DRAM, SSD and NIC resources of the GPU cluster to establish a disaggregated KV-Cache." Pointer: abstract, lines 1-2.
16. **CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**
   - Source type: paper
   - URL: <https://www.microsoft.com/en-us/research/wp-content/uploads/2024/09/eurosys25-final999.pdf>
   - Technique: Reuse KV caches for non-prefix chunks and selectively recompute a small subset of tokens to repair cross-chunk context effects, pipelining recompute with KV retrieval. KV transfer could support chunk-addressed cache fetches plus recompute markers, improving TTFT for agentic/RAG prompts assembled from recurring pieces.
   - Evidence: Quote: "CacheBlend ... reuses the pre-computed KV caches, regardless prefix or not, and selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache." Pointer: abstract, lines 1-3.
17. **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**
   - Source type: paper
   - URL: <https://proceedings.mlr.press/v235/liu24bz.html>
   - Technique: Apply asymmetric KV-cache quantization: per-channel for keys and per-token for values. As an optional wire/storage format for KV transfer, this can reduce bytes moved and improve TTFT when transfer bandwidth is the bottleneck.
   - Evidence: Quote: "the key cache should be quantized per-channel ... In contrast, the value cache should be quantized per-token." Pointer: PMLR abstract, lines 2-4.
18. **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**
   - Source type: paper
   - URL: <https://www.usenix.org/system/files/osdi24-lee.pdf>
   - Technique: Use a minimal rehearsal to predict important KV entries for the next layer and prefetch only those entries instead of fetching all KV. The idea can inform remote/offloaded decode-side KV transfer paths that need to reduce per-token transfer work and median TPOT for long contexts.
   - Evidence: Quote: "This allows us to prefetch only the essential KV cache entries (without fetching them all), thereby mitigating the fetch overhead from the host memory." Pointer: abstract, lines 2-3.
19. **[STORE] feat: Frequency admission + LRU lock optimization for local hot cache**
   - Source type: pr
   - URL: <https://github.com/kvcache-ai/Mooncake/pull/1596>
   - Technique: Protect a local hot KV cache from one-shot pollution with Count-Min Sketch frequency admission, and reduce read-path contention with shared locks plus deferred LRU touches. This is applicable to any KV lookup buffer or local remote-cache tier used to accelerate repeated agent prefixes.
   - Evidence: Quote: "keys are only promoted into the hot cache when their access count reaches the admission threshold (default K=2)." Pointer: PR description, lines 1-6.
20. **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**
   - Source type: paper
   - URL: <https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf>
   - Technique: Use bandwidth-aware placement for separated prefill and decode phases to minimize the communication overhead introduced by KV transfer. The KV transfer module could expose measured transfer costs and endpoint bandwidth to help caller-side placement avoid TTFT regressions.
   - Evidence: Quote: "DistServe also places the two phases according to the serving cluster’s bandwidth to minimize the communication caused by disaggregation." Pointer: abstract, lines 3-5.
21. **Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter**
   - Source type: paper
   - URL: <https://arxiv.org/html/2604.15039v1>
   - Technique: Selectively offload only suitable long-context prefills to remote compute-dense clusters using bandwidth-aware scheduling and cache-aware placement. This suggests adding admission/backpressure policies to KV transfer so remote prefill is used when transfer cost is expected to beat local recompute.
   - Evidence: Quote: "PrfaaS combines model-side KV efficiency with system-side selective offloading, bandwidth-aware scheduling, and cache-aware request placement." Pointer: abstract, lines 2-4.
22. **SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills**
   - Source type: paper
   - URL: <https://www.microsoft.com/en-us/research/publication/sarathi-efficient-llm-inference-by-piggybacking-decodes-with-chunked-prefills/>
   - Technique: Split large prefills into chunks and build schedules around uniform chunks. KV transfer could adapt this into chunk-level KV streaming, allowing decode-side work or subsequent transfers to start before an entire long prompt's KV has been materialized and copied.
   - Evidence: Quote: "SARATHI employs chunked-prefills, which splits a prefill request into equal sized chunks, and decode-maximal batching." Pointer: Microsoft Research abstract, lines 1-3.

## Issues

### proposal_from_finding_creator
- **[error, recoverable]** agent failure (candidate_id=cand-0008, finding_id=find-0007): claude timed out after 600.0s
