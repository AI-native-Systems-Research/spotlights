# Sorted candidates — run-4b5859bcc6cc3ad1 (spotlights-out)

**Objective:** reduce the median TTFT and median TPOT (Time Per Output Token)
**Ranked:** 173 candidates · **Method:** listwise sub-agent judge · 5 shards + 1 merge round
**Source:** `spotlights-out/result.json`

## Ranking summary

| # | Candidate | Module | Symbol | Impact | Score | Rationale |
|---|-----------|--------|--------|--------|-------|-----------|
| 1 | [`cand-vllm_v1_attention-0002`](../modules/vllm_v1_attention/FlashAttentionMetadataBuilder.build__cand-vllm_v1_attention-0002.md) | vllm/v1/attention | `FlashAttentionMetadataBuilder.build` | high | 95 | FlashAttentionMetadataBuilder.build sits on the per-step metadata path for every decode token and every prefill; cascade-plan caching, coalesced H2D staging, and pipelined AOT scheduler-metadata compound directly into b… |
| 2 | [`cand-vllm_v1_worker-0001`](../modules/vllm_v1_worker/GPUModelRunner._prepare_inputs__cand-vllm_v1_worker-0001.md) | vllm/v1/worker | `GPUModelRunner._prepare_inputs` | high | 94 | GPUModelRunner._prepare_inputs is the largest steady per-step host cost on the legacy runner; the pure-decode fast path (skip np.repeat/cumsum/index_select when every request has one token) and H2D coalescing directly r… |
| 3 | [`cand-vllm_v1_worker-0006`](../modules/vllm_v1_worker/GPUModelRunner._bookkeeping_sync__cand-vllm_v1_worker-0006.md) | vllm/v1/worker | `GPUModelRunner._bookkeeping_sync` | high | 93 | _bookkeeping_sync's _to_list(sampled_token_ids) D2H is the largest post-sample host roundtrip on every decode step; pipelining the sync behind the next step or landing sampled tokens in pinned host buffers is a direct T… |
| 4 | [`cand-vllm_v1_worker-0027`](../modules/vllm_v1_worker/AsyncOutput.get_output__cand-vllm_v1_worker-0027.md) | vllm/v1/worker | `AsyncOutput.get_output` | high | 92 | AsyncOutput.get_output holds copy_event.synchronize plus tolist/per-row Python work that blocks every async decode step; eliminating the D2H stall lowers median TPOT and the fixes are unusually concrete (pinned/UVA land… |
| 5 | [`cand-vllm_v1_attention-0004`](../modules/vllm_v1_attention/FlashInferMetadataBuilder.build__cand-vllm_v1_attention-0004.md) | vllm/v1/attention | `FlashInferMetadataBuilder.build` | high | 91 | FlashInferMetadataBuilder.build is the per-step planning gate for FlashInfer users; plan caching keyed by CUDA-graph batch descriptor, coalesced H2D, and Triton-fused metadata prep are all directly on the TPOT critical … |
| 6 | [`cand-vllm_v1_core-0009`](../modules/vllm_v1_core/Scheduler.schedule__cand-vllm_v1_core-0009.md) | vllm/v1/core | `Scheduler.schedule` | high | 90 | Scheduler.schedule is on the CPU critical path of every decode step and every admission; at agentic concurrency the Python scheduler floor can dominate CPU-side TPOT, and prefix-length-ordered admission plus multi-turn-… |
| 7 | [`cand-vllm_v1_attention-0008`](../modules/vllm_v1_attention/unified_attention_launch-parameter_selection__cand-vllm_v1_attention-0008.md) | vllm/v1/attention | `unified_attention launch-parameter selection` | high | 89 | Triton unified_attention launch parameters govern the dominant compute kernel for both prefill (TTFT) and decode (TPOT); per-shape autotuned buckets and occupancy-aware split-KV directly shift the median kernel time. Sl… |
| 8 | [`cand-vllm_v1_executor-0001`](../modules/vllm_v1_executor/MultiprocExecutor.collective_rpc__cand-vllm_v1_executor-0001.md) | vllm/v1/executor | `MultiprocExecutor.collective_rpc` | high | 88 | MultiprocExecutor.collective_rpc fires per execute_model/sample_tokens step; cached cloudpickle payloads, precompiled RPC descriptors, and a static Ray Compiled Graph for the steady-state control plane remove per-token … |
| 9 | [`cand-vllm_v1_worker-0023`](../modules/vllm_v1_worker/GPUModelRunner.prepare_inputs__cand-vllm_v1_worker-0023.md) | vllm/v1/worker | `GPUModelRunner.prepare_inputs` | high | 87 | GPUModelRunner.prepare_inputs is the new runner's central per-step CPU/GPU preparation; coalescing H2D copies, removing seq_lens_cpu host sync, and precomputing metadata in a scheduler-side thread overlapped with model … |
| 10 | [`cand-vllm_v1_worker-0003`](../modules/vllm_v1_worker/GPUModelRunner._prepare_input_ids__cand-vllm_v1_worker-0003.md) | vllm/v1/worker | `GPUModelRunner._prepare_input_ids` | high | 86 | _prepare_input_ids sits on the async decode fast path with fixed per-step Python-list construction and four micro-copies; single-token multi-turn agent decode is exactly where this fixed overhead is most visible in medi… |
| 11 | [`cand-vllm_v1_worker-0004`](../modules/vllm_v1_worker/GPUModelRunner._calc_spec_decode_metadata__cand-vllm_v1_worker-0004.md) | vllm/v1/worker | `GPUModelRunner._calc_spec_decode_metadata` | high | 84 | _calc_spec_decode_metadata builds spec-decode indices via five separate H2D copies plus a .tolist() sync every step; spec decode is a core TPOT lever, but demoted slightly vs shard because gains are conditional on the w… |
| 12 | [`cand-vllm_v1_engine-0001`](../modules/vllm_v1_engine/DPLBAsyncMPClient.get_core_engine_for_request__cand-vllm_v1_engine-0001.md) | vllm/v1/engine | `DPLBAsyncMPClient.get_core_engine_for_request` | high | 83 | DPLB routing gates admission and per-engine KV pressure; prefix-affinity + P2C changes can materially lower median TTFT under multi-turn agentic bursts. Demoted below universal per-step levers because the gain requires … |
| 13 | [`cand-vllm_v1_attention-0001`](../modules/vllm_v1_attention/use_cascade_attention__cand-vllm_v1_attention-0001.md) | vllm/v1/attention | `use_cascade_attention` | high | 82 | use_cascade_attention routing is a fixed-gate branch between numerically equivalent attention kernels; long shared conversation prefixes are exactly the case where flipping to cascade cuts TTFT-dominant prefill cost, an… |
| 14 | [`cand-vllm_v1_engine-0005`](../modules/vllm_v1_engine/OutputProcessor.process_outputs__cand-vllm_v1_engine-0005.md) | vllm/v1/engine | `OutputProcessor.process_outputs` | high | 81 | OutputProcessor.process_outputs is the only full-batch Python loop over every emitted token; branch specialization plus batched DecodeStream.step directly cut frontend CPU per token, translating into lower median TPOT u… |
| 15 | [`cand-vllm_v1_core-0006`](../modules/vllm_v1_core/BlockHashToBlockMap_and_BlockPool.get_cached_block__cand-vllm_v1_core-0006.md) | vllm/v1/core | `BlockHashToBlockMap and BlockPool.get_cached_block` | high | 80 | BlockPool.get_cached_block runs on every prefix-cache probe, often hundreds per admission for long agent histories; split-map single-group fast path and precomputed group keys compress prefix-lookup wall time directly o… |
| 16 | [`cand-vllm_distributed_kv_transfer-0011`](../modules/vllm_distributed_kv_transfer/HF3FSKVConnector._generate_block_hashes__cand-vllm_distributed_kv_transfer-0011.md) | vllm/distributed/kv_transfer | `HF3FSKVConnector._generate_block_hashes` | high | 79 | HF3FSKVConnector._generate_block_hashes scales with prompt length on the TTFT-critical prefix lookup path; replacing stringified-list MD5 with XXH3 and caching chains cuts per-turn hashing when long multi-turn history r… |
| 17 | [`cand-vllm_v1_kv_offload-0009`](../modules/vllm_v1_kv_offload/TieringOffloadingManager._initiate_promotion__cand-vllm_v1_kv_offload-0009.md) | vllm/v1/kv_offload | `TieringOffloadingManager._initiate_promotion` | high | 78 | TieringOffloadingManager._initiate_promotion controls how fast secondary hits become primary hits; batched prepare_write plus TinyLFU/S3-FIFO admission gates directly move TTFT on cache-hit turns and TPOT during promoti… |
| 18 | [`cand-vllm_v1_worker-0019`](../modules/vllm_v1_worker/UBatchWrapper._capture_ubatches_and__run_ubatches__cand-vllm_v1_worker-0019.md) | vllm/v1/worker | `UBatchWrapper._capture_ubatches and _run_ubatches` | high | 77 | UBatchWrapper's capture/run implements DBO to overlap mixed prefill/decode; CUDA graph reuse and persistent worker threads cut per-forward overhead and improve overlap for exactly the non-uniform agent batches this work… |
| 19 | [`cand-vllm_v1_worker-0011`](../modules/vllm_v1_worker/BlockTable_and_MultiGroupBlockTable_row_mutation_methods__cand-vllm_v1_worker-0011.md) | vllm/v1/worker | `BlockTable and MultiGroupBlockTable row mutation methods` | high | 76 | BlockTable per-group row mutation runs O(active changes x KV groups) every step; batching row updates and copying only dirty rows removes repeated Python calls for hybrid/multi-group models. Visible on both TTFT admissi… |
| 20 | [`cand-vllm_distributed_kv_transfer-0005`](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorScheduler.__init___transfer_policy_defaults__cand-vllm_distributed_kv_transfer-0005.md) | vllm/distributed/kv_transfer | `NixlBaseConnectorScheduler.__init__ transfer policy defaults` | high | 74 | NIXL pull-vs-recompute defaults determine per-turn TTFT on every remote-prefill hit; an EWMA-adjusted kv_recompute_threshold is well-founded and multi-turn agent prompts hit this decision constantly. Demoted because the… |
| 21 | [`cand-vllm_v1_engine-0006`](../modules/vllm_v1_engine/BaseIncrementalDetokenizer.update__cand-vllm_v1_engine-0006.md) | vllm/v1/engine | `BaseIncrementalDetokenizer.update` | high | 73 | BaseIncrementalDetokenizer.update is per-token frontend CPU work; batched DecodeStream, no-stop fast path, and Aho-Corasick stop matching cut Python cost on the TPOT path, but detokenization is usually a small fraction … |
| 22 | [`cand-vllm_v1_core-0001`](../modules/vllm_v1_core/HybridKVCacheCoordinator.find_longest_cache_hit__cand-vllm_v1_core-0001.md) | vllm/v1/core | `HybridKVCacheCoordinator.find_longest_cache_hit` | high | 72 | HybridKVCacheCoordinator.find_longest_cache_hit is the central hybrid prefix-cache admission loop; reordering attention groups and length-only reconciliation cut TTFT for multi-turn resubmits. Impact narrower than full-… |
| 23 | [`cand-vllm_v1_core-0007`](../modules/vllm_v1_core/get_request_block_hasher.request_block_hasher__cand-vllm_v1_core-0007.md) | vllm/v1/core | `get_request_block_hasher.request_block_hasher` | high | 70 | request_block_hasher scans hundreds/thousands of blocks synchronously before admission; LRU over completed block hash tuples and lifted extra-key invariants shave TTFT on long-context turns. Solid but scoped to admissio… |
| 24 | [`cand-vllm_distributed_kv_transfer-0006`](../modules/vllm_distributed_kv_transfer/MultiConnector.get_num_new_matched_tokens__cand-vllm_distributed_kv_transfer-0006.md) | vllm/distributed/kv_transfer | `MultiConnector.get_num_new_matched_tokens` | high | 68 | MultiConnector.get_num_new_matched_tokens selects where turn-2 KV loads from across tiers; a scored router replacing first-positive-wins can meaningfully move TTFT when GPU/CPU/remote/object connectors coexist. Conditio… |
| 25 | [`cand-vllm_v1_executor-0013`](../modules/vllm_v1_executor/RayWorkerWrapper.execute_model_ray__cand-vllm_v1_executor-0013.md) | vllm/v1/executor | `RayWorkerWrapper.execute_model_ray` | high | 66 | execute_model_ray is the per-worker body of the Ray steady-state DAG; overlapping AsyncModelRunnerOutput D2H with the next channel return and eliminating per-step Python branching cut host overhead every decode. Gated o… |
| 26 | [`cand-vllm_v1_worker-0010`](../modules/vllm_v1_worker/_compute_slot_mapping_kernel__cand-vllm_v1_worker-0010.md) | vllm/v1/worker | `_compute_slot_mapping_kernel` | high | 64 | _compute_slot_mapping_kernel's fixed 1024-lane per-request tile wastes lanes on one-token decodes; a 2D token-tile grid or one-token fast path shrinks kernel time on decode-heavy agent turns. Kernel micro-optimization o… |
| 27 | [`cand-vllm_v1_worker-0024`](../modules/vllm_v1_worker/BlockTables.gather_block_tables_and_compute_slot_mappings__cand-vllm_v1_worker-0024.md) | vllm/v1/worker | `BlockTables.gather_block_tables and compute_slot_mappings` | high | 62 | BlockTables.gather_block_tables + compute_slot_mappings run every step in the new runner with a fixed 1024 tile; decode-fused path mapping one request per lane cuts TPOT for one-token-per-request agent batches. Same cla… |
| 28 | [`cand-vllm_v1_worker-0002`](../modules/vllm_v1_worker/GPUModelRunner._update_states__cand-vllm_v1_worker-0002.md) | vllm/v1/worker | `GPUModelRunner._update_states` | high | 60 | _update_states runs before every legacy model step with Python cost scaling with active-request churn — matches multi-turn agent traffic. MRV2-style decoupled state and fused slot-swap admission reduce the per-step floo… |
| 29 | [`cand-vllm_v1_core-0002`](../modules/vllm_v1_core/FullAttentionManager.find_longest_cache_hit__cand-vllm_v1_core-0002.md) | vllm/v1/core | `FullAttentionManager.find_longest_cache_hit` | high | 58 | FullAttentionManager.find_longest_cache_hit runs on every admission; galloping search over chained-hash monotonicity plus per-step memoization cuts TTFT for the agentic workload. Gains are concentrated in admission, not… |
| 30 | [`cand-vllm_distributed_kv_transfer-0001`](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorWorker._pop_done_transfers__cand-vllm_distributed_kv_transfer-0001.md) | vllm/distributed/kv_transfer | `NixlBaseConnectorWorker._pop_done_transfers` | high | 56 | NixlBaseConnectorWorker._pop_done_transfers is polled every decode step; notification-driven completion collapses O(in_flight * handles) probes into a single wait. Demoted vs shard because effect only binds when NIXL di… |
| 31 | [`cand-vllm_v1_core-0012`](../modules/vllm_v1_core/KVCacheManager.allocate_slots__cand-vllm_v1_core-0012.md) | vllm/v1/core | `KVCacheManager.allocate_slots` | high | 54 | KVCacheManager.allocate_slots is on the scheduler path for every capacity-needing request; removing the duplicate coordinator.get_num_blocks_to_allocate and reordering skipped-block removal shave scheduler overhead. Mod… |
| 32 | [`cand-vllm_v1_executor-0004`](../modules/vllm_v1_executor/WorkerProc.enqueue_output_handle_output_async_output_busy_loop__cand-vllm_v1_executor-0004.md) | vllm/v1/executor | `WorkerProc.enqueue_output/handle_output/async_output_busy_loop` | high | 52 | WorkerProc async output handling is the host/device sync boundary for token materialization; overlapping D2H with next RPC and batch-draining the queue shorten token-visible latency. Overlaps significantly with the Asyn… |
| 33 | [`cand-vllm_v1_executor-0012`](../modules/vllm_v1_executor/RayDistributedExecutor._compiled_ray_dag__cand-vllm_v1_executor-0012.md) | vllm/v1/executor | `RayDistributedExecutor._compiled_ray_dag` | high | 50 | Ray compiled-DAG topology and per-edge transport selection govern steady-state PP/TP comm overhead; TTFT gains from eager CGraph warmup are real but one-shot per model. Multi-node Ray deployment is a narrow slice, so ra… |
| 34 | [`cand-vllm_v1_worker-0028`](../modules/vllm_v1_worker/GPUModelRunner.add_requests_and_update_requests__cand-vllm_v1_worker-0028.md) | vllm/v1/worker | `GPUModelRunner.add_requests and update_requests` | high | 48 | add_requests/update_requests loops are hit on new agent turns; vectorizing state deltas via scatter cuts TTFT admission overhead. High rf_count is 0 and gains are Python-loop micro-savings, which are dwarfed by the H2D/… |
| 35 | [`cand-vllm_distributed_kv_transfer-0002`](../modules/vllm_distributed_kv_transfer/NixlPullConnectorWorker._read_blocks_for_req__cand-vllm_distributed_kv_transfer-0002.md) | vllm/distributed/kv_transfer | `NixlPullConnectorWorker._read_blocks_for_req` | high | 46 | NixlPullConnectorWorker._read_blocks_for_req batches per-rank READ posts on the pre-resume path; multi-turn reuse hits this on every remote-prefill cache hit but only on heterogeneous-TP NIXL disaggregated deployments, … |
| 36 | [`cand-vllm_v1_core-0004`](../modules/vllm_v1_core/BlockPool.get_new_blocks__cand-vllm_v1_core-0004.md) | vllm/v1/core | `BlockPool.get_new_blocks` | high | 44 | BlockPool.get_new_blocks funnels every slot allocation; fast-path eviction skip and batched BlockRemoved events accumulate under concurrency, but per-block savings are small in absolute terms compared to per-step host s… |
| 37 | [`cand-vllm_v1_kv_offload-0008`](../modules/vllm_v1_kv_offload/TieringOffloadingManager.lookup__cand-vllm_v1_kv_offload-0008.md) | vllm/v1/kv_offload | `TieringOffloadingManager.lookup` | high | 42 | TieringOffloadingManager.lookup gates promotion scheduling; batching secondary probes cuts synchronous lookup delay when traffic hits non-primary tiers. Demoted because reach is bounded to configurations with tiered off… |
| 38 | [`cand-vllm_v1_executor-0007`](../modules/vllm_v1_executor/RayDistributedExecutor._execute_dag__cand-vllm_v1_executor-0007.md) | vllm/v1/executor | `RayDistributedExecutor._execute_dag` | high | 40 | RayDistributedExecutor._execute_dag is the driver-side per-token blocking wait for Ray compiled-DAG; overlapping detach with ray.wait cuts TPOT but Ray backend + connector mode is a narrow slice, and effects duplicate e… |
| 39 | [`cand-vllm_multimodal-0001`](../modules/vllm_multimodal/MultiModalHasher.serialize_item__cand-vllm_multimodal-0001.md) | vllm/multimodal | `MultiModalHasher.serialize_item` | high | 38 | MultiModalHasher.serialize_item hashing dominates TTFT for images/CUDA tensors before hit detection; GPU-side hashing and pooled pinned buffers are clean fixes but the workload is described as agentic without multimodal… |
| 40 | [`cand-vllm_v1_engine-0003`](../modules/vllm_v1_engine/EngineCore.step_with_batch_queue__cand-vllm_v1_engine-0003.md) | vllm/v1/engine | `EngineCore.step_with_batch_queue` | medium | 35 | step_with_batch_queue owns PP drain-vs-fill policy; adaptive drain moves TTFT (first-output) and TPOT (bubble) — but only when PP batch queueing is enabled. Shard already flagged it medium impact; ranked lowest because … |
| 41 | [`cand-vllm_v1_kv_offload-0018`](../modules/vllm_v1_kv_offload/ServerRole.add_stored_blocks_on_fetch__cand-vllm_v1_kv_offload-0018.md) | vllm/v1/kv_offload | `ServerRole.add_stored_blocks/on_fetch` | high | 74 | P2P ServerRole matching gates how quickly peer-held KV reaches consumers on secondary hits; priority-aware first-block fast-path can replace prefill with peer fetch, directly moving TTFT for disaggregated agentic traffi… |
| 42 | [`cand-vllm_v1_worker-0029`](../modules/vllm_v1_worker/rejection_sample__cand-vllm_v1_worker-0029.md) | vllm/v1/worker | `rejection_sample` | high | 68 | rejection_sample is the spec-decode inner sampling kernel; persistent scratch tensors, autotuned tiling, and a greedy fast path directly reduce per-step launch/traffic overhead when spec decode is on, which is a common … |
| 43 | [`cand-vllm_v1_attention-0011`](../modules/vllm_v1_attention/split_decodes_and_prefills__cand-vllm_v1_attention-0011.md) | vllm/v1/attention | `split_decodes_and_prefills` | medium | 70 | split_decodes_and_prefills is called every step by multiple metadata builders and does redundant CPU tensor materialization; numpy searchsorted on cumulative query_start_loc removes fixed overhead each step, compounding… |
| 44 | [`cand-vllm_distributed_kv_transfer-0003`](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorWorker._handshake_initiation_executor____nixl_handshake__cand-vllm_distributed_kv_transfer-0003.md) | vllm/distributed/kv_transfer | `NixlBaseConnectorWorker._handshake_initiation_executor / _nixl_handshake` | high | 74 | NIXL handshake gates the first remote KV read and pays remote_pp_size * target_tp_rank sequential RTTs; caching validated metadata and pipelining removes first-hit TTFT spikes recurring in disaggregated multi-turn servi… |
| 45 | [`cand-vllm_v1_attention-0006`](../modules/vllm_v1_attention/fast_plan_decode__cand-vllm_v1_attention-0006.md) | vllm/v1/attention | `fast_plan_decode` | medium | 75 | fast_plan_decode is invoked every token on FlashInfer CUDA-graph decode; short-circuiting on unchanged batch descriptors and coalescing H2D copies removes steady-state per-step overhead that shows up in median TPOT. |
| 46 | [`cand-vllm_v1_kv_offload-0001`](../modules/vllm_v1_kv_offload/ARCCachePolicy.evict__cand-vllm_v1_kv_offload-0001.md) | vllm/v1/kv_offload | `ARCCachePolicy.evict` | high | 72 | ARC evict policy controls recency/frequency balance for the primary KV tier; better victim selection raises hit rate for repeated agentic prefixes and cuts promotion stalls that show up as median TTFT/TPOT. |
| 47 | [`cand-vllm_multimodal-0004`](../modules/vllm_multimodal/MultiModalBatchedField._reduce_data__cand-vllm_multimodal-0004.md) | vllm/multimodal | `MultiModalBatchedField._reduce_data` | high | 66 | MultiModalBatchedField._reduce_data allocates pinned buffers and stacks tensors on every multimodal prefill; pooling pinned outputs and eliding stack copies removes real host work from the TTFT path, gated by multimodal… |
| 48 | [`cand-vllm_distributed_kv_transfer-0007`](../modules/vllm_distributed_kv_transfer/OffloadingScheduler._lookup_complete_chunks__cand-vllm_distributed_kv_transfer-0007.md) | vllm/distributed/kv_transfer | `OffloadingScheduler._lookup_complete_chunks` | high | 67 | OffloadingScheduler._lookup_complete_chunks runs on every cache-eligible admission and long shared prefixes exercise it repeatedly; a step-level OffloadKey cache and probe reordering cut backend calls that gate TTFT for… |
| 49 | [`cand-vllm_v1_worker-0015`](../modules/vllm_v1_worker/DP_synchronization_post-processing_helpers__cand-vllm_v1_worker-0015.md) | vllm/v1/worker | `DP synchronization post-processing helpers` | high | 73 | DP synchronization is a per-step latency floor when serving is scaled across ranks; batching scalar reads into a single host transfer and replacing the SUM all-reduce with MIN/MAX collectives reduces the CPU-visible tai… |
| 50 | [`cand-vllm_v1_worker-0030`](../modules/vllm_v1_worker/DefaultModelState.prepare_attn__cand-vllm_v1_worker-0030.md) | vllm/v1/worker | `DefaultModelState.prepare_attn` | medium | 72 | prepare_attn runs every new-runner forward with .item() host syncs and per-step Python metadata walks; eliminating these directly reduces steady-state median TPOT. |
| 51 | [`cand-vllm_v1_kv_offload-0013`](../modules/vllm_v1_kv_offload/CachePolicyFactory_built-in_registrations__cand-vllm_v1_kv_offload-0013.md) | vllm/v1/kv_offload | `CachePolicyFactory built-in registrations` | high | 70 | CachePolicyFactory adds workload-tuned sibling policies (S3-FIFO, TinyLFU, SIEVE) that can raise primary hit rate for shared agentic prefixes; high leverage but indirect and gated on config selection to realize gains. |
| 52 | [`cand-vllm_v1_worker-0005`](../modules/vllm_v1_worker/GPUModelRunner._build_attention_metadata__cand-vllm_v1_worker-0005.md) | vllm/v1/worker | `GPUModelRunner._build_attention_metadata` | medium | 65 | _build_attention_metadata runs every step and repeats group-invariant work; making seq_lens_cpu optional and persisting FlashInfer-style planning shaves steady TPOT overhead. Impact bounded by the fact that this is CPU … |
| 53 | [`cand-vllm_v1_attention-0019`](../modules/vllm_v1_attention/FlashInferMetadataBuilder_fixed_split_policy__cand-vllm_v1_attention-0019.md) | vllm/v1/attention | `FlashInferMetadataBuilder fixed split policy` | medium | 64 | FlashInferMetadataBuilder fixed split constants gate FlashInfer kernel partitioning for prefill and decode; occupancy-aware or page-aligned split policies improve latency across TPOT and TTFT, but the impact is continge… |
| 54 | [`cand-vllm_v1_engine-0004`](../modules/vllm_v1_engine/AsyncLLM._run_output_handler.output_handler__cand-vllm_v1_engine-0004.md) | vllm/v1/engine | `AsyncLLM._run_output_handler.output_handler` | high | 68 | AsyncLLM.output_handler processes every token delivered to clients; skipping empty-tail sleeps, batching aborts, and adapting yield chunking to queue pressure reduces frontend per-token overhead — most visible on short … |
| 55 | [`cand-vllm_v1_worker-0009`](../modules/vllm_v1_worker/GPUModelRunner._determine_batch_execution_and_padding__cand-vllm_v1_worker-0009.md) | vllm/v1/worker | `GPUModelRunner._determine_batch_execution_and_padding` | medium | 70 | _determine_batch_execution_and_padding is per-step in DP-enabled deployments with a D2H scalar sync and redundant dispatch resolution; memoization/pinned-host removes recurring TPOT overhead when DP agent traffic keeps … |
| 56 | [`cand-vllm_v1_kv_offload-0014`](../modules/vllm_v1_kv_offload/SecondaryTierFactory_built-in_registrations__cand-vllm_v1_kv_offload-0014.md) | vllm/v1/kv_offload | `SecondaryTierFactory built-in registrations` | high | 62 | SecondaryTierFactory is a broad extension surface with the highest rf_count (6), but the proposals describe registering new tier implementations rather than optimizing the hot path; benefit to median TTFT depends on whi… |
| 57 | [`cand-vllm_v1_kv_offload-0006`](../modules/vllm_v1_kv_offload/SingleDirectionOffloadingHandler.transfer_async__cand-vllm_v1_kv_offload-0006.md) | vllm/v1/kv_offload | `SingleDirectionOffloadingHandler.transfer_async` | high | 65 | SingleDirectionOffloadingHandler.transfer_async runs per promoted block group on cache hits; precomputing descriptor templates and splitting large onloads directly reduces TTFT when offloaded prefixes are reused, which … |
| 58 | [`cand-vllm_v1_worker-0007`](../modules/vllm_v1_worker/GPUModelRunner.execute_model__cand-vllm_v1_worker-0007.md) | vllm/v1/worker | `GPUModelRunner.execute_model` | medium | 68 | execute_model runs every step and hosts several small per-step Python costs; caching static backend properties and skipping seq-lens materialization gives modest but repeated TPOT wins in agentic decode. |
| 59 | [`cand-vllm_v1_attention-0003`](../modules/vllm_v1_attention/FlashAttentionImpl._forward_with_dcp__cand-vllm_v1_attention-0003.md) | vllm/v1/attention | `FlashAttentionImpl._forward_with_dcp` | medium | 66 | DCP forward runs in every attention layer when enabled, so occupancy-aware split and all-gather/attention overlap can move both TTFT and TPOT — but only when DCP is active, which is not implied by the agentic workload h… |
| 60 | [`cand-vllm_v1_attention-0005`](../modules/vllm_v1_attention/FlashInferMetadataBuilder._compute_flashinfer_kv_metadata__cand-vllm_v1_attention-0005.md) | vllm/v1/attention | `FlashInferMetadataBuilder._compute_flashinfer_kv_metadata` | medium | 68 | FlashInfer metadata computation runs per step for decode/prefill/cascade with two H2D copies plus CPU cumsum; coalescing to a single pinned staging buffer reduces per-step overhead, directly moving decode-heavy median T… |
| 61 | [`cand-vllm_v1_kv_offload-0003`](../modules/vllm_v1_kv_offload/LRUCachePolicy.evict__cand-vllm_v1_kv_offload-0003.md) | vllm/v1/kv_offload | `LRUCachePolicy.evict` | medium | 66 | Pure-recency LRU can prematurely evict shared agentic prefixes; prefix-aware or frequency-augmented eviction raises primary hit rate, reducing promotion stalls that hit TTFT/TPOT. |
| 62 | [`cand-vllm_multimodal-0006`](../modules/vllm_multimodal/_can_batch_mm_items__batch_mm_items_group_and_batch_mm_items_group_and_batch_mm_kwargs__cand-vllm_multimodal-0006.md) | vllm/multimodal | `_can_batch_mm_items/_batch_mm_items/group_and_batch_mm_items/group_and_batch_mm_kwargs` | high | 62 | group_and_batch_mm_items runs on every multimodal prefill and its Python overhead scales with items per batch. Single-pass streaming with cached compatibility signatures shaves TTFT for multi-item multimodal turns; ceil… |
| 63 | [`cand-vllm_v1_kv_offload-0004`](../modules/vllm_v1_kv_offload/_select_swap_blocks_fn__cand-vllm_v1_kv_offload-0004.md) | vllm/v1/kv_offload | `_select_swap_blocks_fn` | high | 60 | CPU-to-GPU swap dispatch selects between DMA and Triton; batch-size-aware dispatch and hybrid paths for mixed page-size groups raise transfer bandwidth on the promotion critical path, cutting TTFT stall on CPU-tier hits… |
| 64 | [`cand-vllm_v1_worker-0017`](../modules/vllm_v1_worker/EncoderCudaGraphManager_budget_generation_and_lookup__cand-vllm_v1_worker-0017.md) | vllm/v1/worker | `EncoderCudaGraphManager budget generation and lookup` | high | 62 | Encoder CUDA-graph budgets gate multimodal TTFT on screenshot/image turns common in agentic workloads; workload-adaptive budget ladders reduce eager fallback and padding waste — impact contingent on multimodal traffic m… |
| 65 | [`cand-vllm_v1_attention-0022`](../modules/vllm_v1_attention/resolve_seq_and_query_len___find_seq_idx__cand-vllm_v1_attention-0022.md) | vllm/v1/attention | `resolve_seq_and_query_len / find_seq_idx` | medium | 65 | find_seq_idx binary search runs per Triton attention program plus reduce_segments; precomputing q-block-to-sequence metadata removes repeated per-program work on decode-heavy agentic batches, chipping at TPOT. |
| 66 | [`cand-vllm_v1_engine-0010`](../modules/vllm_v1_engine/DPEngineCoreProc._should_throttle_prefills__cand-vllm_v1_engine-0010.md) | vllm/v1/engine | `DPEngineCoreProc._should_throttle_prefills` | medium | 60 | _should_throttle_prefills gates DP prefill admission; load-aware or progress-informed policies can move TTFT under bursty agentic arrivals. Effect confined to DP deployments and bounded by the existing cadence being rea… |
| 67 | [`cand-vllm_distributed_kv_transfer-0021`](../modules/vllm_distributed_kv_transfer/LookupKeyClient.lookup__cand-vllm_distributed_kv_transfer-0021.md) | vllm/distributed/kv_transfer | `LookupKeyClient.lookup` | medium | 64 | LookupKeyClient.lookup with REQ+single-thread serializes prefix lookups behind one lane; DEALER multiplexing or step-batched RPC directly parallelizes TTFT-critical lookups under concurrent turns. |
| 68 | [`cand-vllm_v1_core-0008`](../modules/vllm_v1_core/FreeKVCacheBlockQueue__cand-vllm_v1_core-0008.md) | vllm/v1/core | `FreeKVCacheBlockQueue` | medium | 57 | FreeKVCacheBlockQueue is touched on every allocate/free/touch and already exists to reduce Python overhead; further reductions (intrusive sidecar arrays, batched touch removal) only bound scheduler overhead, so effect o… |
| 69 | [`cand-vllm_v1_attention-0009`](../modules/vllm_v1_attention/TritonAttentionMetadataBuilder_2D_3D_tuning_constants__cand-vllm_v1_attention-0009.md) | vllm/v1/attention | `TritonAttentionMetadataBuilder 2D/3D tuning constants` | medium | 62 | Triton 2D/3D switch threshold + segment count govern SM utilization for small-batch decode common in agent serving; occupancy-aware tuning is a concrete TPOT lever but scoped to Triton backend and small batches. |
| 70 | [`cand-vllm_v1_engine-0015`](../modules/vllm_v1_engine/EngineCore.step__cand-vllm_v1_engine-0015.md) | vllm/v1/engine | `EngineCore.step` | medium | 60 | EngineCore.step is the non-pipeline scheduling boundary paid once per decode iteration; overlapping abort/sampling work with GPU execute reduces per-step host overhead, but model execution dominates so gains are moderat… |
| 71 | [`cand-vllm_distributed_kv_transfer-0012`](../modules/vllm_distributed_kv_transfer/MooncakeConnectorWorker._build_transfer_params__cand-vllm_distributed_kv_transfer-0012.md) | vllm/distributed/kv_transfer | `MooncakeConnectorWorker._build_transfer_params` | high | 62 | Mooncake transfer descriptor coalescing reduces RDMA initiation on the turn-2 KV fetch path; TTFT gain is real for P/D reuse but confined to Mooncake deployments. |
| 72 | [`cand-vllm_distributed_kv_transfer-0015`](../modules/vllm_distributed_kv_transfer/OffloadingScheduler.update_state_after_alloc_load-job_construction__cand-vllm_distributed_kv_transfer-0015.md) | vllm/distributed/kv_transfer | `OffloadingScheduler.update_state_after_alloc load-job construction` | high | 58 | OffloadingScheduler.update_state_after_alloc load-job construction sits between allocator and worker for every offloaded prefix hit before resume. Vectorizing group_blocks walks trims TTFT on the multi-turn cache-hit pa… |
| 73 | [`cand-vllm_v1_engine-0009`](../modules/vllm_v1_engine/EngineCoreProc._process_engine_step__cand-vllm_v1_engine-0009.md) | vllm/v1/engine | `EngineCoreProc._process_engine_step` | medium | 55 | EngineCoreProc._process_engine_step's 1 ms fixed sleep bounds how quickly WAITING_FOR_REMOTE_KVS requests are reconsidered; event-driven wakeups shave that floor from TTFT for remote-KV/prefix-cache paths, useful for di… |
| 74 | [`cand-vllm_distributed_kv_transfer-0013`](../modules/vllm_distributed_kv_transfer/MooncakeConnectorWorker.__init___sender_pool_sizing__cand-vllm_distributed_kv_transfer-0013.md) | vllm/distributed/kv_transfer | `MooncakeConnectorWorker.__init__ sender pool sizing` | high | 60 | Mooncake sender pool sizing controls producer send concurrency for remote KV; adaptive sizing under many concurrent turn-2 fetches can reduce TTFT tails but is scoped to Mooncake-backed disaggregated deployments. |
| 75 | [`cand-vllm_distributed_kv_transfer-0019`](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorWorker.sync_recved_kv_to_device___save_kv_to_host__cand-vllm_distributed_kv_transfer-0019.md) | vllm/distributed/kv_transfer | `NixlBaseConnectorWorker.sync_recved_kv_to_device / save_kv_to_host` | medium | 56 | sync_recved_kv_to_device sits directly before request resume for host-buffer NIXL loads, so batching copy_blocks and moving to a dedicated stream can move TTFT — but only when the host-buffer mode is configured. |
| 76 | [`cand-vllm_v1_attention-0025`](../modules/vllm_v1_attention/Triton_MLA_decode_split_policy__cand-vllm_v1_attention-0025.md) | vllm/v1/attention | `Triton MLA decode split policy` | medium | 55 | Triton MLA num_kv_splits policy trades parallelism vs reduction overhead every decode step for MLA models; occupancy/seq-len-aware splits improve median TPOT for MLA agent serving, but scoped to the MLA backend. |
| 77 | [`cand-vllm_v1_core-0021`](../modules/vllm_v1_core/SlidingWindowManager.reachable_block_mask__cand-vllm_v1_core-0021.md) | vllm/v1/core | `SlidingWindowManager.reachable_block_mask` | medium | 53 | SlidingWindowManager.reachable_block_mask affects retention policy and therefore future-turn hit rate; better tail scoring lowers follow-up-turn TTFT but the effect is amortized across turns and mediated by downstream e… |
| 78 | [`cand-vllm_v1_engine-0002`](../modules/vllm_v1_engine/DPEngineCoreProc._has_global_unfinished_reqs__cand-vllm_v1_engine-0002.md) | vllm/v1/engine | `DPEngineCoreProc._has_global_unfinished_reqs` | medium | 58 | The 32-step DP finish-sync cadence can add up to 31 steps of wave-end wait; adaptive or event-driven sync trims tail TTFT/TPOT gaps for short bursty agentic waves, but median gain depends on wave shape. |
| 79 | [`cand-vllm_distributed_kv_transfer-0020`](../modules/vllm_distributed_kv_transfer/AsyncOperationManager._handle_load_task__cand-vllm_distributed_kv_transfer-0020.md) | vllm/distributed/kv_transfer | `AsyncOperationManager._handle_load_task` | medium | 58 | HF3FS load-task blocking stream sync gates prefix-hit resumption; polling with events and pre-registered pinned staging reduces turn-2 TTFT for HF3FS-backed reuse but only when that backend is configured. |
| 80 | [`cand-vllm_v1_core-0010`](../modules/vllm_v1_core/Scheduler._make_cached_request_data__cand-vllm_v1_core-0010.md) | vllm/v1/core | `Scheduler._make_cached_request_data` | medium | 55 | _make_cached_request_data runs once per non-empty scheduler step and scales with active requests; pre-sizing buffers and skipping empty KV updates lowers the TPOT floor at high concurrency but per-request savings are mo… |
| 81 | [`cand-vllm_v1_engine-0018`](../modules/vllm_v1_engine/EngineCoreProc.process_output_sockets__cand-vllm_v1_engine-0018.md) | vllm/v1/engine | `EngineCoreProc.process_output_sockets` | medium | 56 | process_output_sockets is on the observable token transport path; coalescing and buffer reclamation trim frontend overhead touching TPOT, but the effect is smaller than in-runner work. |
| 82 | [`cand-vllm_v1_engine-0007`](../modules/vllm_v1_engine/check_stop_strings__cand-vllm_v1_engine-0007.md) | vllm/v1/engine | `check_stop_strings` | medium | 50 | check_stop_strings runs after every decoded chunk; agentic clients with many tool/chat delimiters pay O(num_stops * suffix_window). Aho-Corasick/first-char prefilter cuts frontend CPU on TPOT, but savings are modest per… |
| 83 | [`cand-vllm_v1_executor-0014`](../modules/vllm_v1_executor/AsyncOutputFuture.result_UniProcExecutor.collective_rpc__cand-vllm_v1_executor-0014.md) | vllm/v1/executor | `AsyncOutputFuture.result/UniProcExecutor.collective_rpc` | medium | 50 | UniProcExecutor collective_rpc/AsyncOutputFuture.result is per-step but only for single-process serving; removing list wrapping and blocking get_output reduces TPOT overhead, though multi-turn agentic deployments typica… |
| 84 | [`cand-vllm_distributed_kv_transfer-0009`](../modules/vllm_distributed_kv_transfer/kv_cache_scatter_kernel___kv_cache_gather_kernel_and_wrappers__cand-vllm_distributed_kv_transfer-0009.md) | vllm/distributed/kv_transfer | `kv_cache_scatter_kernel / kv_cache_gather_kernel and wrappers` | medium | 50 | HF3FS gather/scatter kernel autotuning and launch fusion reduce per-block promotion latency; only relevant to the HF3FS reuse path, which caps applicability to that workload configuration. |
| 85 | [`cand-vllm_v1_engine-0011`](../modules/vllm_v1_engine/DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task__cand-vllm_v1_engine-0011.md) | vllm/v1/engine | `DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task` | medium | 57 | DP load-balancer stats freshness/smoothness affects DPLBAsyncMPClient routing; EWMA/hysteresis improvements reduce misroutes and tail TTFT in DP agentic bursts but the median effect is smaller. |
| 86 | [`cand-vllm_v1_worker-0012`](../modules/vllm_v1_worker/preprocess_mamba__cand-vllm_v1_worker-0012.md) | vllm/v1/worker | `preprocess_mamba` | high | 48 | preprocess_mamba is per-step and vectorization removes O(num_reqs x layer-groups) Python work, but the impact is gated by whether the served model is a hybrid Mamba/SSM architecture; strong within that regime, narrow ac… |
| 87 | [`cand-vllm_v1_engine-0016`](../modules/vllm_v1_engine/InputProcessor.process_inputs__cand-vllm_v1_engine-0016.md) | vllm/v1/engine | `InputProcessor.process_inputs` | medium | 55 | InputProcessor.process_inputs is per-request TTFT overhead for every new turn; fast-pathing pre-rendered EngineInput and deferring multimodal materialization is concrete but touches only new-turn admission wall time. |
| 88 | [`cand-vllm_v1_engine-0017`](../modules/vllm_v1_engine/RequestOutputCollector.put_get_nowait_get__cand-vllm_v1_engine-0017.md) | vllm/v1/engine | `RequestOutputCollector.put/get_nowait/get` | medium | 48 | RequestOutputCollector is the AsyncLLM streaming handoff for every token batch; a lazy Future replacing asyncio.Event trims frontend event churn under many concurrent short generations. Improves frontend CPU but only in… |
| 89 | [`cand-vllm_v1_worker-0026`](../modules/vllm_v1_worker/sync_cudagraph_and_dp_padding__cand-vllm_v1_worker-0026.md) | vllm/v1/worker | `sync_cudagraph_and_dp_padding` | medium | 48 | sync_cudagraph_and_dp_padding adds per-step CPU coordination in DP serving; packed collectives and idle-rank skipping trim scalar sync but the CPU-side floor is smaller than the sibling DP sync helpers in _run_ar. |
| 90 | [`cand-vllm_v1_core-0020`](../modules/vllm_v1_core/BlockPool.cache_full_blocks__cand-vllm_v1_core-0020.md) | vllm/v1/core | `BlockPool.cache_full_blocks` | medium | 55 | cache_full_blocks per-block insertion runs when finalized tokens are cached and shapes future prefix hit availability; fusing hash-insert with events reduces scheduler overhead and improves reuse-driven TTFT. |
| 91 | [`cand-vllm_v1_kv_offload-0015`](../modules/vllm_v1_kv_offload/FileSystemTierManager.submit_store_submit_load__cand-vllm_v1_kv_offload-0015.md) | vllm/v1/kv_offload | `FileSystemTierManager.submit_store/submit_load` | medium | 46 | FileSystemTierManager chunking and coalescing improve promotion latency on FS-backed reuse and are on the TTFT path when spill goes to disk; the benefit is real but only for deployments that actually use the FS tier, an… |
| 92 | [`cand-vllm_v1_attention-0023`](../modules/vllm_v1_attention/FlexAttentionMetadata._build_block_mask_direct__cand-vllm_v1_attention-0023.md) | vllm/v1/attention | `FlexAttentionMetadata._build_block_mask_direct` | medium | 52 | FlexAttention _build_block_mask_direct can dominate small-batch decode metadata cost; TPOT gain is meaningful but limited to the Flex backend rather than the default agentic serving stack. |
| 93 | [`cand-vllm_v1_executor-0008`](../modules/vllm_v1_executor/FutureWrapper.result__cand-vllm_v1_executor-0008.md) | vllm/v1/executor | `FutureWrapper.result` | medium | 46 | FutureWrapper.result sits on the async Ray decode completion path with a sequential detach loop; ray.wait-based overlap helps TPOT under connectors/multi-output but non-connector single-output cases have only one detach. |
| 94 | [`cand-vllm_v1_executor-0015`](../modules/vllm_v1_executor/RayDistributedExecutor._init_workers_ray__cand-vllm_v1_executor-0015.md) | vllm/v1/executor | `RayDistributedExecutor._init_workers_ray` | high | 52 | Ray placement/rank layout is initialization code, but topology-aware TP intra-node / PP cross-node choices reduce cross-node transport per token step; steady-state TPOT gain is real but bounded to multi-node PP/TP setup… |
| 95 | [`cand-vllm_v1_engine-0012`](../modules/vllm_v1_engine/RequestState.make_request_output__cand-vllm_v1_engine-0012.md) | vllm/v1/engine | `RequestState.make_request_output` | medium | 46 | make_request_output is allocated per request per output-processing step; hoisting FINAL_ONLY gating and deferring text decoding lowers frontend TPOT for many short concurrent generations, but constant-factor gains are b… |
| 96 | [`cand-vllm_v1_executor-0006`](../modules/vllm_v1_executor/detach_zero_copy_from_model_runner_output__cand-vllm_v1_executor-0006.md) | vllm/v1/executor | `detach_zero_copy_from_model_runner_output` | medium | 46 | detach_zero_copy_from_model_runner_output is on the Ray output critical path but only bites when logprobs or routed_experts are requested; conditional agentic gains for Ray backends make this a mid-tier TPOT lever. |
| 97 | [`cand-vllm_v1_attention-0012`](../modules/vllm_v1_attention/make_local_attention_virtual_batches__cand-vllm_v1_attention-0012.md) | vllm/v1/attention | `make_local_attention_virtual_batches` | medium | 50 | make_local_attention_virtual_batches runs per step for local-attention Gemma models; coalescing four H2D uploads cuts CPU/H2D overhead but the win is scoped to that model family. |
| 98 | [`cand-vllm_v1_core-0016`](../modules/vllm_v1_core/BlockHashListWithBlockSize__cand-vllm_v1_core-0016.md) | vllm/v1/core | `BlockHashListWithBlockSize` | medium | 45 | BlockHashListWithBlockSize view overhead only bites hybrid models with mismatched block sizes; a strided-slice backing removes per-access attribute walks on prefix-lookup hot code, modestly reducing median TTFT for that… |
| 99 | [`cand-vllm_v1_kv_offload-0017`](../modules/vllm_v1_kv_offload/ObjectStoreSecondaryTierManager._submit_transfer__poll_active_transfers__cand-vllm_v1_kv_offload-0017.md) | vllm/v1/kv_offload | `ObjectStoreSecondaryTierManager._submit_transfer/_poll_active_transfers` | medium | 50 | Object-store transfer submission/polling reduces secondary promotion delay before TTFT; helpful for cross-turn reuse but bounded by intrinsic network/object-store latency, indirect effect on median. |
| 100 | [`cand-vllm_multimodal-0019`](../modules/vllm_multimodal/MultiModalBudget._get_max_items__cand-vllm_multimodal-0019.md) | vllm/multimodal | `MultiModalBudget._get_max_items` | medium | 45 | MultiModalBudget._get_max_items sets encoder-work admission ceilings; cache-aware and p95-driven sizing can reduce queueing-induced TTFT but only in multimodal-heavy mixes that the hint does not explicitly require. |
| 101 | [`cand-vllm_v1_engine-0014`](../modules/vllm_v1_engine/DPCoordinator.run_polling_publish_loop__cand-vllm_v1_engine-0014.md) | vllm/v1/engine | `DPCoordinator.run polling/publish loop` | medium | 44 | DPCoordinator publish cadence controls DP routing snapshot freshness; adaptive/event-driven publication reduces stale-load misrouting TTFT in bursty DP serving, but only when data-parallel coordination is enabled. |
| 102 | [`cand-vllm_v1_core-0013`](../modules/vllm_v1_core/SingleTypeKVCacheManager.get_num_blocks_to_allocate__cand-vllm_v1_core-0013.md) | vllm/v1/core | `SingleTypeKVCacheManager.get_num_blocks_to_allocate` | medium | 48 | get_num_blocks_to_allocate can be called twice per waiting request in full_sequence_must_fit; memoizing the evictable scan trims admission overhead on TTFT, but gains are modest per turn. |
| 103 | [`cand-vllm_v1_worker-0025`](../modules/vllm_v1_worker/StagedWriteTensor_and_FusedStagedWriter_apply_path__cand-vllm_v1_worker-0025.md) | vllm/v1/worker | `StagedWriteTensor and FusedStagedWriter apply path` | medium | 46 | StagedWriteTensor/FusedStagedWriter apply path scales with request churn and KV groups; persistent pinned/UVA ring buffers eliminate list rebuilds and a sync H2D. Real gains on admission/resume churn, though not on stea… |
| 104 | [`cand-vllm_v1_attention-0021`](../modules/vllm_v1_attention/AttentionMetadataBuilder._init_reorder_batch_threshold__cand-vllm_v1_attention-0021.md) | vllm/v1/attention | `AttentionMetadataBuilder._init_reorder_batch_threshold` | medium | 48 | Reorder-batch threshold routes short-extend/spec tokens between decode and prefill kernels; better routing keeps efficient decode kernels active, but effect is second-order relative to kernel-level tunables. |
| 105 | [`cand-vllm_v1_worker-0020`](../modules/vllm_v1_worker/maybe_create_ubatch_slices__cand-vllm_v1_worker-0020.md) | vllm/v1/worker | `maybe_create_ubatch_slices` | medium | 44 | maybe_create_ubatch_slices governs DBO split points; compute-cost-weighted splitting improves overlap on mixed prefill/decode agent batches but the benefit is second-order and DBO must be enabled. |
| 106 | [`cand-vllm_v1_worker-0013`](../modules/vllm_v1_worker/postprocess_mamba_fused_kernel__cand-vllm_v1_worker-0013.md) | vllm/v1/worker | `postprocess_mamba_fused_kernel` | high | 42 | postprocess_mamba_fused_kernel tiling improvements shave hybrid Mamba spec-decode TPOT via reduced state-copy time; net effect is scoped to hybrid Mamba plus speculative decoding, so impact per-run is real but audience … |
| 107 | [`cand-vllm_v1_attention-0010`](../modules/vllm_v1_attention/reorder_batch_to_split_decodes_and_prefills__cand-vllm_v1_attention-0010.md) | vllm/v1/attention | `reorder_batch_to_split_decodes_and_prefills` | medium | 45 | reorder_batch_to_split_decodes_and_prefills runs every scheduler step; two-pointer minimum-swap and no-reorder fast paths reduce per-step overhead when agent traffic alternates decode and short extends. Small per-step s… |
| 108 | [`cand-vllm_v1_core-0023`](../modules/vllm_v1_core/ChunkedLocalAttentionManager.find_longest_cache_hit__cand-vllm_v1_core-0023.md) | vllm/v1/core | `ChunkedLocalAttentionManager.find_longest_cache_hit` | medium | 46 | ChunkedLocalAttentionManager.find_longest_cache_hit avoids large null-list materialization on admission; TTFT-relevant but scoped to chunked-local models. |
| 109 | [`cand-vllm_v1_kv_offload-0012`](../modules/vllm_v1_kv_offload/CPUOffloadingManager.prepare_store__cand-vllm_v1_kv_offload-0012.md) | vllm/v1/kv_offload | `CPUOffloadingManager.prepare_store` | medium | 45 | CPU prepare_store admission improves future primary hit rate and reduces all-or-nothing store failures; complementary to eviction policy but effect on median TTFT/TPOT is indirect and multi-turn-cache-dependent. |
| 110 | [`cand-vllm_v1_attention-0020`](../modules/vllm_v1_attention/compute_mm_prefix_range_tensor__cand-vllm_v1_attention-0020.md) | vllm/v1/attention | `compute_mm_prefix_range_tensor` | medium | 43 | compute_mm_prefix_range_tensor is per-step multimodal prefill work with padded H2D uploads; a jagged CSR layout and pinned staging cut CPU list work on multimodal turns only. |
| 111 | [`cand-vllm_distributed_kv_transfer-0014`](../modules/vllm_distributed_kv_transfer/HF3FSKVConnector._gather_or_scatter_kv_caches__cand-vllm_distributed_kv_transfer-0014.md) | vllm/distributed/kv_transfer | `HF3FSKVConnector._gather_or_scatter_kv_caches` | medium | 44 | HF3FS gather/scatter batching reduces per-block kernel launches on turn-2 fetch; helpful for HF3FS deployments but the win depends on many small blocks and doesn't touch the common non-HF3FS path. |
| 112 | [`cand-vllm_v1_attention-0015`](../modules/vllm_v1_attention/_make_mm_prefix_mask_mod__cand-vllm_v1_attention-0015.md) | vllm/v1/attention | `_make_mm_prefix_mask_mod` | medium | 44 | _make_mm_prefix_mask_mod runs inside FA4 prefill blocks for multimodal prompts; precomputing full/partial KV block metadata skips per-element masking on fully-included blocks, helping image-heavy TTFT. Scope narrow (FA4… |
| 113 | [`cand-vllm_distributed_kv_transfer-0018`](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorWorker._compute_desc_ids__cand-vllm_distributed_kv_transfer-0018.md) | vllm/distributed/kv_transfer | `NixlBaseConnectorWorker._compute_desc_ids` | medium | 40 | _compute_desc_ids pays per remote-prefill transfer setup; fused numpy or contiguous-range paths cut Python allocation, but the cost is a small fraction of NIXL transfer time and only shows up for HMA/multi-region setups… |
| 114 | [`cand-vllm_v1_core-0005`](../modules/vllm_v1_core/BlockPool.free_blocks__cand-vllm_v1_core-0005.md) | vllm/v1/core | `BlockPool.free_blocks` | medium | 44 | free_blocks retention ordering affects future prefix-cache hit rate; better retention helps follow-up-turn TTFT, but the loop itself is small and the mechanism is one policy signal among many governing hit rate. |
| 115 | [`cand-vllm_v1_worker-0008`](../modules/vllm_v1_worker/GPUModelRunner._calc_mrope_positions__cand-vllm_v1_worker-0008.md) | vllm/v1/worker | `GPUModelRunner._calc_mrope_positions` | medium | 41 | _calc_mrope_positions is a per-request per-step loop on M-RoPE multimodal models; vectorizing helps multimodal TPOT/TTFT only when those models are being served. |
| 116 | [`cand-vllm_v1_core-0022`](../modules/vllm_v1_core/MambaManager.reachable_block_mask__cand-vllm_v1_core-0022.md) | vllm/v1/core | `MambaManager.reachable_block_mask` | medium | 42 | Mamba retention_interval affects long-context reuse for Mamba models; retention improvements matter but are scoped to Mamba/hybrid deployments rather than the default agentic serving stack. |
| 117 | [`cand-vllm_distributed_kv_transfer-0004`](../modules/vllm_distributed_kv_transfer/_PUSH_WRITER_POLL_INTERVAL_MS__cand-vllm_distributed_kv_transfer-0004.md) | vllm/distributed/kv_transfer | `_PUSH_WRITER_POLL_INTERVAL_MS` | medium | 42 | _PUSH_WRITER_POLL_INTERVAL_MS is 1 ms and adds a bounded per-turn wait in push-mode NIXL; notification-driven wakeups remove that floor for turn-2 TTFT, but the absolute magnitude is small. |
| 118 | [`cand-vllm_v1_worker-0031`](../modules/vllm_v1_worker/MambaHybridModelState.preprocess_state__cand-vllm_v1_worker-0031.md) | vllm/v1/worker | `MambaHybridModelState.preprocess_state` | medium | 39 | MambaHybridModelState.preprocess_state amortization and fused-precopy fusion save decode-step launches, but the benefit is again limited to hybrid Mamba workloads and largely overlaps with the earlier mamba_utils candid… |
| 119 | [`cand-vllm_v1_executor-0002`](../modules/vllm_v1_executor/FutureWrapper.result__cand-vllm_v1_executor-0002.md) | vllm/v1/executor | `FutureWrapper.result` | medium | 42 | FutureWrapper.result drain loop only shows up for callers using non_block futures; polling all rank queues in one pass helps async decode sync but scope is narrower than collective_rpc. |
| 120 | [`cand-vllm_v1_attention-0013`](../modules/vllm_v1_attention/fill_mm_prefix_query_ranges__cand-vllm_v1_attention-0013.md) | vllm/v1/attention | `fill_mm_prefix_query_ranges` | medium | 40 | fill_mm_prefix_query_ranges runs on multimodal prefill/extend steps in the Triton backend; caching resolved spans avoids the O(num_actual_tokens) fill, but the workload trigger is narrow. |
| 121 | [`cand-vllm_multimodal-0017`](../modules/vllm_multimodal/PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host__cand-vllm_multimodal-0017.md) | vllm/multimodal | `PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host` | medium | 37 | PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host lowers NVDEC decode+D2H latency via pinned buffer pooling and stream pipelining, but only benefits video-ingesting multi-turn workloads and is off-path for text-onl… |
| 122 | [`cand-vllm_v1_core-0003`](../modules/vllm_v1_core/SlidingWindowManager.find_longest_cache_hit__cand-vllm_v1_core-0003.md) | vllm/v1/core | `SlidingWindowManager.find_longest_cache_hit` | medium | 42 | SlidingWindowManager.find_longest_cache_hit reverse-scan is O(max_num_blocks) with a known asymptotic win to O(max_num_blocks/K + K); helps SWA admission TTFT but confined to SWA/SWA-hybrid models. |
| 123 | [`cand-vllm_v1_kv_offload-0010`](../modules/vllm_v1_kv_offload/AsyncLookupManager._worker__cand-vllm_v1_kv_offload-0010.md) | vllm/v1/kv_offload | `AsyncLookupManager._worker` | medium | 40 | AsyncLookupManager batching/dedup shortens the retry window before promotion, tightening TTFT tail; median gain is bounded by tier backend latency and depends on secondary-hit frequency. |
| 124 | [`cand-vllm_v1_executor-0016`](../modules/vllm_v1_executor/RayExecutorV2._init_executor__cand-vllm_v1_executor-0016.md) | vllm/v1/executor | `RayExecutorV2._init_executor` | medium | 40 | RayExecutorV2._init_executor is mostly startup code affecting TTFT only during bring-up; the MessageQueue locality does bleed into TPOT for multi-node MQ deployments, but single-node runs see little movement. |
| 125 | [`cand-vllm_multimodal-0002`](../modules/vllm_multimodal/MultiModalHasher.iter_item_to_bytes_hash_kwargs__cand-vllm_multimodal-0002.md) | vllm/multimodal | `MultiModalHasher.iter_item_to_bytes/hash_kwargs` | medium | 38 | MultiModalHasher runs on cache-key construction; batching hasher updates and iterative traversal shave overhead but savings are far smaller than avoiding tensor copies and only visible on cache hits. |
| 126 | [`cand-vllm_multimodal-0018`](../modules/vllm_multimodal/_VIDEO_LOADER_REGISTRY.register__opencv____cand-vllm_multimodal-0018.md) | vllm/multimodal | `@VIDEO_LOADER_REGISTRY.register("opencv")` | high | 36 | Alternative VIDEO_LOADER_REGISTRY backends can cut per-clip encoder FLOPs by sampling fewer frames, potentially large TTFT wins for long-clip video prompts, but the objective's workload hint doesn't call out video inges… |
| 127 | [`cand-vllm_distributed_kv_transfer-0008`](../modules/vllm_distributed_kv_transfer/OffloadingScheduler._build_store_jobs__cand-vllm_distributed_kv_transfer-0008.md) | vllm/distributed/kv_transfer | `OffloadingScheduler._build_store_jobs` | medium | 38 | OffloadingScheduler._build_store_jobs adds per-step CPU cost that scales with active requests; batching prepare_store trims scheduler overhead but the win is small vs runner-side hot paths. |
| 128 | [`cand-vllm_multimodal-0020`](../modules/vllm_multimodal/BaseMultiModalField.reduce_data__cand-vllm_multimodal-0020.md) | vllm/multimodal | `BaseMultiModalField.reduce_data` | medium | 40 | BaseMultiModalField.reduce_data does repeated list/set/traversal work per field per grouped prefill; a fused single-pass path is a clean CPU win but small in absolute TTFT terms and multimodal-only. |
| 129 | [`cand-vllm_multimodal-0022`](../modules/vllm_multimodal/MultiModalKwargsItems.from_hf_inputs__cand-vllm_multimodal-0022.md) | vllm/multimodal | `MultiModalKwargsItems.from_hf_inputs` | medium | 37 | from_hf_inputs churns Python objects per modality item; transposed assembly reduces cache-miss TTFT for multimodal prompts but is narrow and constant-factor. |
| 130 | [`cand-vllm_v1_engine-0013`](../modules/vllm_v1_engine/LogprobsProcessor._update_sample_logprobs__cand-vllm_v1_engine-0013.md) | vllm/v1/engine | `LogprobsProcessor._update_sample_logprobs` | medium | 38 | LogprobsProcessor per-token conversion cost is real but only paid when logprobs are enabled; TPOT gain is conditional on request-level flags rather than baseline agentic traffic. |
| 131 | [`cand-vllm_distributed_kv_transfer-0016`](../modules/vllm_distributed_kv_transfer/MoRIIOConnectorWorker._read_blocks__cand-vllm_distributed_kv_transfer-0016.md) | vllm/distributed/kv_transfer | `MoRIIOConnectorWorker._read_blocks` | medium | 34 | MoRIIOConnectorWorker._read_blocks batching helps remote-prefill TTFT on MoRIIO READ mode; impact is real but the connector is niche, and NIXL-based paths dominate most disaggregated deployments. |
| 132 | [`cand-vllm_v1_worker-0016`](../modules/vllm_v1_worker/KVBlockZeroer__cand-vllm_v1_worker-0016.md) | vllm/v1/worker | `KVBlockZeroer` | medium | 40 | KVBlockZeroer segment-aware tiling shaves wasted programs during new-turn allocation; helps TTFT/tail when page sizes vary across cache segments, but allocation-time cost isn't a dominant term for steady serving. |
| 133 | [`cand-vllm_v1_worker-0022`](../modules/vllm_v1_worker/copy_kv_cache_blocks_inplace__cand-vllm_v1_worker-0022.md) | vllm/v1/worker | `copy_kv_cache_blocks_inplace` | medium | 36 | copy_kv_cache_blocks_inplace fusion saves launches during promotion/copy, but the copy path isn't every step and the raw cost is small relative to input prep and attention. |
| 134 | [`cand-vllm_v1_kv_offload-0011`](../modules/vllm_v1_kv_offload/DualQueueThreadPool._worker__cand-vllm_v1_kv_offload-0011.md) | vllm/v1/kv_offload | `DualQueueThreadPool._worker` | medium | 36 | DualQueueThreadPool._worker governs FS-tier read/write balance; slack-aware scheduling helps load-burst TTFT but the FS bandwidth ceiling bounds gains and the path is optional. |
| 135 | [`cand-vllm_v1_kv_offload-0007`](../modules/vllm_v1_kv_offload/compute_sub_block_ptrs__cand-vllm_v1_kv_offload-0007.md) | vllm/v1/kv_offload | `compute_sub_block_ptrs` | medium | 35 | compute_sub_block_ptrs eliminates small per-call materialization inside transfer submission; ceiling bounded by surrounding scheduling and memcpy cost, so effect on TTFT is minor. |
| 136 | [`cand-vllm_v1_executor-0003`](../modules/vllm_v1_executor/WorkerProc.worker_busy_loop__cand-vllm_v1_executor-0003.md) | vllm/v1/executor | `WorkerProc.worker_busy_loop` | medium | 34 | worker_busy_loop dispatch overhead is real per step but small relative to model execution and queue transport; specializing the dispatch gives a thin TPOT gain concentrated in short-generation regimes. |
| 137 | [`cand-vllm_v1_kv_offload-0002`](../modules/vllm_v1_kv_offload/ARCCachePolicy.touch__cand-vllm_v1_kv_offload-0002.md) | vllm/v1/kv_offload | `ARCCachePolicy.touch` | medium | 33 | ARCCachePolicy.touch tuning influences future eviction targets and therefore primary-cache hit rate, but the connection to median TTFT/TPOT is indirect and capped by downstream eviction decisions and steady-state hit ra… |
| 138 | [`cand-vllm_distributed_kv_transfer-0017`](../modules/vllm_distributed_kv_transfer/MooncakeStoreWorker.lookup__cand-vllm_distributed_kv_transfer-0017.md) | vllm/distributed/kv_transfer | `MooncakeStoreWorker.lookup` | medium | 40 | MooncakeStoreWorker.lookup adds Python key expansion cost on every prefix-hit lookup; skipping redundant find_longest_cache_hit passes trims TTFT for Mooncake-backed multi-turn prompts, but backend-specific and bounded … |
| 139 | [`cand-vllm_multimodal-0003`](../modules/vllm_multimodal/nested_tensors_equal__cand-vllm_multimodal-0003.md) | vllm/multimodal | `nested_tensors_equal` | medium | 36 | nested_tensors_equal is called during multimodal batching; storage-identity short-circuits avoid CUDA-syncing torch.equal calls, but effect is confined to prefill batching with repeated shared fields. |
| 140 | [`cand-vllm_v1_worker-0021`](../modules/vllm_v1_worker/WorkspaceManager._ensure_workspace_size__cand-vllm_v1_worker-0021.md) | vllm/v1/worker | `WorkspaceManager._ensure_workspace_size` | medium | 33 | WorkspaceManager growth policy is a tail-latency lever (avoiding empty_cache reallocation stalls on long tool-response prefills), not a median TPOT driver in steady state. |
| 141 | [`cand-vllm_distributed_kv_transfer-0010`](../modules/vllm_distributed_kv_transfer/KVConnectorFactory.register_connector_registrations__cand-vllm_distributed_kv_transfer-0010.md) | vllm/distributed/kv_transfer | `KVConnectorFactory.register_connector registrations` | medium | 40 | The KVConnectorFactory registry itself is cold; rf_count=9 reflects proposal breadth but the seam is a plumbing point, not a hot loop. Direct TTFT/TPOT movement depends entirely on which connector variant ships behind i… |
| 142 | [`cand-vllm_multimodal-0026`](../modules/vllm_multimodal/PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec__cand-vllm_multimodal-0026.md) | vllm/multimodal | `PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec` | medium | 32 | Removing per-request temp-file staging in PyNvVideoCodec avoidance is a clear TTFT win for repeated video ingestion, but video is not the described workload, so the score reflects narrow applicability. |
| 143 | [`cand-vllm_v1_worker-0014`](../modules/vllm_v1_worker/stage_postprocess_inputs_to_gpu__cand-vllm_v1_worker-0014.md) | vllm/v1/worker | `stage_postprocess_inputs_to_gpu` | medium | 32 | Mamba postprocess H2D coalescing helps hybrid spec-decode workloads; TPOT gain is real but confined to that model/decoding path. |
| 144 | [`cand-vllm_v1_executor-0011`](../modules/vllm_v1_executor/Executor.get_class__cand-vllm_v1_executor-0011.md) | vllm/v1/executor | `Executor.get_class` | medium | 34 | Executor.get_class is a selector, not a hot path; it enables alternative executor topologies rather than performing the optimization itself, so impact is indirect and gated on downstream implementations. |
| 145 | [`cand-vllm_multimodal-0005`](../modules/vllm_multimodal/MultiModalFlatField._reduce_data__cand-vllm_multimodal-0005.md) | vllm/multimodal | `MultiModalFlatField._reduce_data` | medium | 30 | MultiModalFlatField._reduce_data reduces per-item slice-assign overhead for variable-length audio/video features; TTFT gain only for multimodal requests, no effect on text-only agent turns. |
| 146 | [`cand-vllm_multimodal-0024`](../modules/vllm_multimodal/PyNvVideoCodecVideoBackendMixin._configure_decoder_slots__borrow_decoder_slot__cand-vllm_multimodal-0024.md) | vllm/multimodal | `PyNvVideoCodecVideoBackendMixin._configure_decoder_slots/_borrow_decoder_slot` | medium | 38 | PyNvVideoCodec decoder-slot admission (LIFO wait loop, cond.notify) affects concurrent video prefill overlap; FIFO/priority queues and per-device sharding help TTFT tails but only when NVDEC contention is real, narrow s… |
| 147 | [`cand-vllm_v1_attention-0024`](../modules/vllm_v1_attention/get_kernel_options__cand-vllm_v1_attention-0024.md) | vllm/v1/attention | `get_kernel_options` | medium | 30 | FlexAttention BLOCK_M/BLOCK_N tuning can shift compiled kernel occupancy for TTFT prefills and repeated decode, but the effect is scoped to Flex backend and depends on device/shape. |
| 148 | [`cand-vllm_v1_worker-0018`](../modules/vllm_v1_worker/EncoderCudaGraphManager._execute_local__cand-vllm_v1_worker-0018.md) | vllm/v1/worker | `EncoderCudaGraphManager._execute_local` | medium | 30 | EncoderCudaGraphManager packing moves near-budget encoder batches from eager to cudagraph, cutting multimodal TTFT; scoped to mixed-modality agent turns, so limited relevance to the described text-heavy multi-turn workl… |
| 149 | [`cand-vllm_v1_core-0011`](../modules/vllm_v1_core/Scheduler._mamba_block_aligned_split__cand-vllm_v1_core-0011.md) | vllm/v1/core | `Scheduler._mamba_block_aligned_split` | medium | 28 | Mamba block-aligned split is scoped strictly to hybrid-Mamba align-mode models; potentially meaningful TTFT/TPOT for those workloads but a small slice of agentic serving overall. |
| 150 | [`cand-vllm_v1_core-0019`](../modules/vllm_v1_core/MambaManager.find_longest_cache_hit__cand-vllm_v1_core-0019.md) | vllm/v1/core | `MambaManager.find_longest_cache_hit` | medium | 32 | MambaManager.find_longest_cache_hit accelerates admission cache-hit lookup but only for Mamba/hybrid models; agentic workloads on the mainstream attention family will not exercise this path. |
| 151 | [`cand-vllm_multimodal-0010`](../modules/vllm_multimodal/find_split_point__cand-vllm_multimodal-0010.md) | vllm/multimodal | `find_split_point` | medium | 26 | Vectorizing find_split_point cuts audio preprocessing wall time on TTFT, but only for voice-agent/ASR workloads and not the described general multi-turn agentic path. |
| 152 | [`cand-vllm_v1_engine-0008`](../modules/vllm_v1_engine/LogprobsProcessor._update_prompt_logprobs__cand-vllm_v1_engine-0008.md) | vllm/v1/engine | `LogprobsProcessor._update_prompt_logprobs` | medium | 36 | LogprobsProcessor._update_prompt_logprobs is a per-position Python loop paid only when prompt_logprobs is enabled; gated feature limits reach, and per-token savings are small even when active. |
| 153 | [`cand-vllm_v1_worker-0032`](../modules/vllm_v1_worker/RopeState.prepare_positions_and__prepare_rope_positions_kernel__cand-vllm_v1_worker-0032.md) | vllm/v1/worker | `RopeState.prepare_positions and _prepare_rope_positions_kernel` | medium | 29 | RopeState.prepare_positions removes wasted lanes on one-token decode for M-RoPE/XD-RoPE; only helps multimodal RoPE workloads and gains are small relative to the full decode step. |
| 154 | [`cand-vllm_multimodal-0023`](../modules/vllm_multimodal/BaseMultiModalReceiverCache.get_and_update_features__cand-vllm_multimodal-0023.md) | vllm/multimodal | `BaseMultiModalReceiverCache.get_and_update_features` | medium | 26 | Encoder-cache get_and_update_features deduplication reduces Python overhead on multimodal ingress; effect on TTFT only for multimodal-heavy agent batches, small per-call work. |
| 155 | [`cand-vllm_v1_kv_offload-0005`](../modules/vllm_v1_kv_offload/NUM_SMS_THRESHOLD_BYTES_MIN_N__cand-vllm_v1_kv_offload-0005.md) | vllm/v1/kv_offload | `NUM_SMS/THRESHOLD_BYTES/MIN_N` | medium | 30 | Tuning NUM_SMS/THRESHOLD_BYTES/MIN_N adjusts CPU-to-GPU Triton swap crossover on non-H100 links; strictly a tuning knob for the promotion path chosen by _select_swap_blocks_fn. |
| 156 | [`cand-vllm_multimodal-0008`](../modules/vllm_multimodal/MultiModalCache.get_leaf_size_get_item_size__cand-vllm_multimodal-0008.md) | vllm/multimodal | `MultiModalCache.get_leaf_size/get_item_size` | medium | 24 | MultiModalCache size accounting runs on cache inserts/evictions; touches TTFT for new-media cache misses but only in multimodal workloads and with small per-operation savings. |
| 157 | [`cand-vllm_multimodal-0021`](../modules/vllm_multimodal/MultiModalGPUMemoryPool.acquire__release__cand-vllm_multimodal-0021.md) | vllm/multimodal | `MultiModalGPUMemoryPool.acquire/_release` | medium | 27 | MultiModalGPUMemoryPool admission scheduling cuts thundering-herd wakeups for concurrent video ingress, improving tail more than median and only when video traffic is bursty; small median TTFT effect for the described w… |
| 158 | [`cand-vllm_v1_kv_offload-0016`](../modules/vllm_v1_kv_offload/batch_store_block_batch_load_block__cand-vllm_v1_kv_offload-0016.md) | vllm/v1/kv_offload | `batch_store_block/batch_load_block` | medium | 36 | batch_store_block/batch_load_block for FS-tier offload sees Python fallback loops parallelized with a ThreadPoolExecutor. Only activates when reused prefixes spill past CPU primary memory, and even then FS bandwidth is … |
| 159 | [`cand-vllm_multimodal-0027`](../modules/vllm_multimodal/DeepStreamVideoBackendMixin.decode_indices__cand-vllm_multimodal-0027.md) | vllm/multimodal | `DeepStreamVideoBackendMixin.decode_indices` | medium | 28 | DeepStream decode_indices affects video TTFT on deployments using that backend; pinned buffers and stream-aware handles reduce blocking D2H but the trigger workload is very narrow. |
| 160 | [`cand-vllm_multimodal-0007`](../modules/vllm_multimodal/PlaceholderRange.embeds_cumsum_extract_embeds_range__cand-vllm_multimodal-0007.md) | vllm/multimodal | `PlaceholderRange.embeds_cumsum/extract_embeds_range` | medium | 24 | PlaceholderRange cumsum/extract avoids small tensor ops on placeholder masks; only exercised by multi-image prompts, and each call is already tiny. |
| 161 | [`cand-vllm_multimodal-0009`](../modules/vllm_multimodal/ShmObjectStoreSenderCache.get_and_update_item_remove_dangling_items__cand-vllm_multimodal-0009.md) | vllm/multimodal | `ShmObjectStoreSenderCache.get_and_update_item/remove_dangling_items` | medium | 34 | ShmObjectStoreSenderCache.remove_dangling_items O(N) prune bursts create TTFT outliers under long-running churn; incremental cursor-based pruning smooths tails rather than shifts the median much. |
| 162 | [`cand-vllm_multimodal-0011`](../modules/vllm_multimodal/resample_audio_pyav__cand-vllm_multimodal-0011.md) | vllm/multimodal | `resample_audio_pyav` | medium | 25 | resample_audio_pyav single-pass multi-channel resample cuts audio ingress CPU, but audio isn't the target workload and per-call setup is a small fraction of overall TTFT. |
| 163 | [`cand-vllm_multimodal-0025`](../modules/vllm_multimodal/reserve_mm_ipc_gpu_memory__cand-vllm_multimodal-0025.md) | vllm/multimodal | `reserve_mm_ipc_gpu_memory` | medium | 22 | mm_ipc_gpu_memory reservation trades KV headroom vs decode concurrency for video deployments; effect on TTFT/TPOT is real but tied to video-heavy configurations, not the stated workload. |
| 164 | [`cand-vllm_multimodal-0012`](../modules/vllm_multimodal/OpenCVVideoBackendMixin._read_frames_with_recovery__cand-vllm_multimodal-0012.md) | vllm/multimodal | `OpenCVVideoBackendMixin._read_frames_with_recovery` | medium | 26 | OpenCV frame-read recovery loop is preprocessing overhead confined to sampled video prompts; keyframe seek reduces CPU cost but does not touch the mainline agentic decode/prefill critical path. |
| 165 | [`cand-vllm_multimodal-0014`](../modules/vllm_multimodal/PyAVVideoBackendMixin.decode_frames__cand-vllm_multimodal-0014.md) | vllm/multimodal | `PyAVVideoBackendMixin.decode_frames` | medium | 22 | PyAV video decode seek/GOP coalescing reduces preprocessing wall time for video ingress; TTFT gain only for video-heavy requests, off the multi-turn text agent hot path. |
| 166 | [`cand-vllm_multimodal-0015`](../modules/vllm_multimodal/PYNVVIDEOCODEC___module-level_constants__cand-vllm_multimodal-0015.md) | vllm/multimodal | `PYNVVIDEOCODEC_* module-level constants` | medium | 20 | PyNvVideoCodec reservation constants affect NVDEC slot counts and KV headroom for video bursts; scope is narrow (NVDEC deployments) and it is a policy-constant tune with weak evidence. |
| 167 | [`cand-vllm_v1_core-0014`](../modules/vllm_v1_core/MambaManager.allocate_new_blocks__cand-vllm_v1_core-0014.md) | vllm/v1/core | `MambaManager.allocate_new_blocks` | medium | 32 | MambaManager.allocate_new_blocks reuse policy affects TPOT/follow-up TTFT for Mamba align-mode models only; recycling speculative blocks and lazy allocation help, but model-scope narrow. |
| 168 | [`cand-vllm_v1_kv_offload-0019`](../modules/vllm_v1_kv_offload/ServerRole.serve_external_requests__process_inbound_lookup__cand-vllm_v1_kv_offload-0019.md) | vllm/v1/kv_offload | `ServerRole.serve_external_requests/_process_inbound_lookup` | medium | 24 | P2P ServerRole lookup batching helps peer-cache-hit TTFT via earlier LookupResp emission, but the effect is bounded by scheduler-step cadence and only matters for symmetric P2P deployments, which are uncommon. |
| 169 | [`cand-vllm_v1_attention-0014`](../modules/vllm_v1_attention/get_dcp_local_seq_lens__cand-vllm_v1_attention-0014.md) | vllm/v1/attention | `get_dcp_local_seq_lens` | low | 22 | get_dcp_local_seq_lens is per-DCP-metadata-build scalar arithmetic and small allocation; caching rank_offsets is real but per-call cost is tiny and DCP must be enabled, so impact on median TTFT/TPOT is minimal. |
| 170 | [`cand-vllm_multimodal-0016`](../modules/vllm_multimodal/DeepStreamVideoBackendMixin._get_pool__cand-vllm_multimodal-0016.md) | vllm/multimodal | `DeepStreamVideoBackendMixin._get_pool` | low | 18 | DeepStream pool sizing only affects DeepStream video deployments; narrow scope and unrelated to the multi-turn text agentic hot path implied by the workload hint. |
| 171 | [`cand-vllm_v1_attention-0016`](../modules/vllm_v1_attention/FlashInferMetadataBuilder._get_workspace_buffer__cand-vllm_v1_attention-0016.md) | vllm/v1/attention | `FlashInferMetadataBuilder._get_workspace_buffer` | low | 18 | FlashInfer workspace sizing mainly touches TTFT outliers on long prefills rather than median latency; self-labeled low impact and no supporting research-finding proposals. |
| 172 | [`cand-vllm_v1_core-0015`](../modules/vllm_v1_core/KVCacheCoordinator.get_num_blocks_to_allocate__cand-vllm_v1_core-0015.md) | vllm/v1/core | `KVCacheCoordinator.get_num_blocks_to_allocate` | low | 24 | KVCacheCoordinator.get_num_blocks_to_allocate is called twice per waiting request in admission but absolute cost is tiny; useful as part of broader admission-loop tuning, not a standalone TTFT mover. |
| 173 | [`cand-vllm_v1_executor-0009`](../modules/vllm_v1_executor/RayDistributedExecutor.collective_rpc__cand-vllm_v1_executor-0009.md) | vllm/v1/executor | `RayDistributedExecutor.collective_rpc` | low | 20 | RayDistributedExecutor.collective_rpc is off the steady-state decode path (Ray uses compiled DAG); improvements only touch startup, warmup, and occasional control RPCs, so median TTFT/TPOT movement is minimal by design. |

## Candidate details

### 1. `cand-vllm_v1_attention-0002` — FlashAttentionMetadataBuilder.build (score 95, impact: high)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flash_attn.py:479` (`FlashAttentionMetadataBuilder.build`, method)
- **Description:** Builds per-step FlashAttention metadata, including AOT scheduler metadata, DCP lengths, cascade tensors, multimodal prefix ranges, and R-SWA buffers.
- **Current approach:** Runs a sequential Python-side build each step. The nested schedule() closure can recompute scheduler_metadata for repeated shapes; cascade paths allocate cu_prefix_query_lens and prefix_kv_lens tensors; multimodal and R-SWA paths perform staging-buffer copies into device buffers.
- **Why this impact / rank:** FlashAttentionMetadataBuilder.build sits on the per-step metadata path for every decode token and every prefill; cascade-plan caching, coalesced H2D staging, and pipelined AOT scheduler-metadata compound directly into both median TPOT and TTFT. Strongest evidence base (rf_count 4, 5 concrete proposals) and the widest reach of any attention-metadata card. — For multi-turn decode this runs every model step and directly contributes to median TPOT; for prefills it also contributes to TTFT. Reducing Python work, allocation, and H2D preparation compounds across generated tokens.
- **Proposals (5):**
  - Make cascade dispatch shape-aware and cache prefix scheduler metadata in FlashAttentionMetadataBuilder.build _(source: research_finding)_
  - Cache cascade planning artifacts across steps in FlashAttentionMetadataBuilder.build _(source: research_finding)_
  - Coalesce per-step H2D metadata copies and eliminate cascade tiny-tensor allocations in FlashAttentionMetadataBuilder.build _(source: research_finding)_
  - Cache CUDA-graph-compatible plans keyed by stable batch descriptors in FlashAttentionMetadataBuilder.build _(source: research_finding)_
  - Pipeline AOT scheduler_metadata across steps via a helper-thread build so build() returns before get_scheduler_metadata finishes _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0002](../modules/vllm_v1_attention/FlashAttentionMetadataBuilder.build__cand-vllm_v1_attention-0002.md)

### 2. `cand-vllm_v1_worker-0001` — GPUModelRunner._prepare_inputs (score 94, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_model_runner.py:2001` (`GPUModelRunner._prepare_inputs`, method)
- **Description:** Main legacy runner per-step CPU-to-GPU input preparation for request indices, positions, token indices, query_start_loc, optimistic sequence lengths, discard masks, slot mapping, and spec-decode counts.
- **Current approach:** Uses several NumPy/Torch CPU passes, torch.index_select, repeated CpuGpuBuffer.copy_to_gpu calls, a per-request prompt-embeds loop, a req_ids list walk for num_tokens, and an event synchronize before staging num_accepted_tokens.
- **Why this impact / rank:** GPUModelRunner._prepare_inputs is the largest steady per-step host cost on the legacy runner; the pure-decode fast path (skip np.repeat/cumsum/index_select when every request has one token) and H2D coalescing directly reduce median TPOT for low-batch multi-turn agent decode. — It is one of the largest steady per-step host costs in the runner; packing small H2D copies and removing per-request Python work directly targets median TPOT, especially in low-batch multi-turn agent decode.
- **Proposals (5):**
  - Amortize _prepare_inputs via worker-local multi-step decode with on-GPU input advancement _(source: research_finding)_
  - Defer seq_lens host materialization in _prepare_inputs to enable async spec-decode overlap _(source: research_finding)_
  - Adopt MRV2 persistent-state pattern: gather per-step inputs on GPU from GPU-resident state _(source: research_finding)_
  - Remove the per-step accepted-tokens sync from _prepare_inputs by deferring num_accepted_tokens materialization one step _(source: research_finding)_
  - Add a pure-decode fast path in _prepare_inputs that skips np.repeat, cumsum, and index_select when every request has num_scheduled_tokens==1 _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0001](../modules/vllm_v1_worker/GPUModelRunner._prepare_inputs__cand-vllm_v1_worker-0001.md)

### 3. `cand-vllm_v1_worker-0006` — GPUModelRunner._bookkeeping_sync (score 93, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_model_runner.py:3763` (`GPUModelRunner._bookkeeping_sync`, method)
- **Description:** Synchronous legacy post-sampling bookkeeping that consumes sampled token IDs, discards invalid rows, updates token_ids_cpu/is_token_ids/num_tokens_no_spec, and extends request output tokens.
- **Current approach:** Converts sampled_token_ids to Python lists, then loops over sampled requests to perform CPU tensor slices, dict lookups, and list extensions.
- **Why this impact / rank:** _bookkeeping_sync's _to_list(sampled_token_ids) D2H is the largest post-sample host roundtrip on every decode step; pipelining the sync behind the next step or landing sampled tokens in pinned host buffers is a direct TPOT lever with concrete proposals. — This is the largest post-sample host roundtrip in synchronous legacy scheduling; deferring or coalescing CPU conversion reduces median TPOT and improves overlap after decode.
- **Proposals (5):**
  - Batch multiple decode steps before running _bookkeeping_sync to amortize its host work _(source: research_finding)_
  - Adopt MRV2 persistent-state/per-step decoupling to eliminate _bookkeeping_sync host roundtrip _(source: research_finding)_
  - Pipeline sync bookkeeping behind the next step to eliminate the _to_list D2H stall _(source: research_finding)_
  - Issue sampled_token_ids D2H immediately after sampling and vectorize the per-request CPU scatter with NumPy fancy indexing _(source: agent_knowledge)_
  - Fast-path bookkeeping for empty or fully discarded sampled rows _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0006](../modules/vllm_v1_worker/GPUModelRunner._bookkeeping_sync__cand-vllm_v1_worker-0006.md)

### 4. `cand-vllm_v1_worker-0027` — AsyncOutput.get_output (score 92, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/async_utils.py:32` (`AsyncOutput.get_output`, region)
- **Description:** New GPU runner asynchronous output staging and final materialization of sampled tokens, logprobs, NaN counts, routed experts, and EP fault status.
- **Current approach:** Records a copy-stream event, synchronizes in get_output, converts sampled_token_ids and num_sampled_tokens NumPy arrays to Python lists, trims each row in a Python loop, and may call .item() plus mask.cpu().tolist() for fault reporting.
- **Why this impact / rank:** AsyncOutput.get_output holds copy_event.synchronize plus tolist/per-row Python work that blocks every async decode step; eliminating the D2H stall lowers median TPOT and the fixes are unusually concrete (pinned/UVA landing, lazy view, batched materialization). — This is the new runner counterpart to the legacy post-sample host roundtrip; reducing blocking conversion and Python row work lowers median TPOT in async multi-turn decode.
- **Proposals (3):**
  - Batch AsyncOutput materialization across a multi-step decode window _(source: research_finding)_
  - Land sampler outputs directly in pinned/UVA host buffers to eliminate per-step D2H copy and synchronize _(source: agent_knowledge)_
  - Introduce a lazy ModelRunnerOutput token view to defer Python list materialization _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0027](../modules/vllm_v1_worker/AsyncOutput.get_output__cand-vllm_v1_worker-0027.md)

### 5. `cand-vllm_v1_attention-0004` — FlashInferMetadataBuilder.build (score 91, impact: high)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flashinfer.py:1119` (`FlashInferMetadataBuilder.build`, method)
- **Description:** Builds FlashInfer per-step metadata, splits decode and prefill regions, computes required paged-KV metadata, and plans native FlashInfer, TRTLLM prefill, XQA, trtllm-gen decode, or cascade paths.
- **Current approach:** A monolithic dispatch recomputes path decisions every step and invokes wrapper plan calls for native FlashInfer paths even when batch structure repeats. It conditionally materializes CPU sequence lengths and prepares H2D metadata copies for paged-KV indices when native paths need them.
- **Why this impact / rank:** FlashInferMetadataBuilder.build is the per-step planning gate for FlashInfer users; plan caching keyed by CUDA-graph batch descriptor, coalesced H2D, and Triton-fused metadata prep are all directly on the TPOT critical path with the shard's highest evidence (rf_count 5, seven proposals). — This method runs once per step for the FlashInfer backend. Reducing plan overhead and transfer preparation moves median TPOT in stable decode batches and TTFT for first-step prefill-heavy agentic turns.
- **Proposals (7):**
  - Shape-aware, multi-level cascade dispatch in FlashInferMetadataBuilder.build _(source: research_finding)_
  - Cache FlashInfer plan artifacts across steps keyed by stable batch/page shape _(source: research_finding)_
  - Replace fixed split-KV constants with an occupancy-aware policy in FlashInferMetadataBuilder.build _(source: research_finding)_
  - Batch small paged-KV metadata H2D copies into one pinned staging transfer in FlashInferMetadataBuilder.build _(source: research_finding)_
  - Key FlashInfer plan cache by CUDA-graph batch descriptor to skip redundant planning on stable decode batches _(source: research_finding)_
  - Fuse per-step FlashInfer metadata prep into a single Triton kernel to eliminate CPU cumsum and multiple H2D copies _(source: agent_knowledge)_
  - Materialize paged-KV metadata only for native FlashInfer request ranges _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0004](../modules/vllm_v1_attention/FlashInferMetadataBuilder.build__cand-vllm_v1_attention-0004.md)

### 6. `cand-vllm_v1_core-0009` — Scheduler.schedule (score 90, impact: high)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/sched/scheduler.py:426` (`Scheduler.schedule`, method)
- **Description:** Main scheduler step that schedules running requests, admits waiting requests, handles preemption, prefix-cache lookup, connector loads, encoder budgets, Mamba alignment, and SchedulerOutput construction.
- **Current approach:** Large Python method with separate running and waiting loops, many per-request branches, inline admission/preemption bookkeeping, and a PRIORITY preemption path that uses max(self.running) plus self.running.index on every allocation failure.
- **Why this impact / rank:** Scheduler.schedule is on the CPU critical path of every decode step and every admission; at agentic concurrency the Python scheduler floor can dominate CPU-side TPOT, and prefix-length-ordered admission plus multi-turn-aware eviction also help TTFT for follow-up turns. — Every TPOT measurement includes this method. At high concurrency, Python scheduler overhead can dominate the CPU side of a decode step, so per-request reductions move median TPOT directly.
- **Proposals (5):**
  - Reorder waiting queue by cached-prefix length before waiting loop in Scheduler.schedule _(source: research_finding)_
  - Bias prefix-cache eviction using conversational-continuation signals for multi-turn scheduling _(source: research_finding)_
  - Adopt Sarathi decode-maximal batching in Scheduler.schedule _(source: research_finding)_
  - Emit upcoming KV block prefetch plan from Scheduler.schedule for L2 prefetching _(source: research_finding)_
  - Maintain a priority-ordered index over self.running for O(log n) PRIORITY preemption victim selection in Scheduler.schedule _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0009](../modules/vllm_v1_core/Scheduler.schedule__cand-vllm_v1_core-0009.md)

### 7. `cand-vllm_v1_attention-0008` — unified_attention launch-parameter selection (score 89, impact: high)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/ops/triton_unified_attention.py:784` (`unified_attention launch-parameter selection`, region)
- **Description:** Selects tile sizes, BLOCK_M/BLOCK_Q, tensor-descriptor gates, prefill/decode launch shape, special large-head tuning, and 2D versus 3D unified-attention launch mode.
- **Current approach:** Uses chained heuristics: _get_tile_size returns 32 for Gemma3 or prefill, 16 for bf16/fp16 decode, and 32 for fp8 decode; BLOCK_M is 16 or next_power_of_2(num_queries_per_kv); an SM100 head_size==256 branch sets BLOCK_M=32, TILE_SIZE_PREFILL=128, 8 warps, and 2 stages; use_3d is disabled for prefills, large batches, batch invariance, or missing segment buffers.
- **Why this impact / rank:** Triton unified_attention launch parameters govern the dominant compute kernel for both prefill (TTFT) and decode (TPOT); per-shape autotuned buckets and occupancy-aware split-KV directly shift the median kernel time. Slightly under host-CPU wins because low-batch decode is often memory-bound on scheduling, not compute. — These launch choices directly control occupancy, split-K work, and memory traffic. Better per-shape settings move TTFT for prefills and median TPOT for repeated decode in multi-turn agentic serving.
- **Proposals (4):**
  - Adaptive Flash-Decoding split-KV: choose NUM_SEGMENTS_PER_SEQ and 3D gate from seqlen and occupancy _(source: research_finding)_
  - Gate 3D split-KV decode via an occupancy-aware policy over SMs, batch, and KV heads _(source: research_finding)_
  - Extend 3D split-KV to long chunked prefills to attack TTFT, not just TPOT _(source: agent_knowledge)_
  - Add per-shape autotuned launch-parameter buckets for 2D attention _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0008](../modules/vllm_v1_attention/unified_attention_launch-parameter_selection__cand-vllm_v1_attention-0008.md)

### 8. `cand-vllm_v1_executor-0001` — MultiprocExecutor.collective_rpc (score 88, impact: high)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/multiproc_executor.py:365` (`MultiprocExecutor.collective_rpc`, method)
- **Description:** Per-step multiprocessing control-plane RPC for execute_model, sample_tokens, and related worker calls; broadcasts work, selects response queues, and returns a FutureWrapper that drains worker replies.
- **Current approach:** Each call computes a deadline, normalizes kwargs, allocates either a partial KVOutputAggregator.aggregate or identity lambda, serializes callable methods with cloudpickle, enqueues one tuple on rpc_broadcast_mq, builds a per-call get_response closure, and wraps it in FutureWrapper. Multi-rank responses are drained sequentially with a fresh remaining-time calculation before each MessageQueue.deque…
- **Why this impact / rank:** MultiprocExecutor.collective_rpc fires per execute_model/sample_tokens step; cached cloudpickle payloads, precompiled RPC descriptors, and a static Ray Compiled Graph for the steady-state control plane remove per-token Python allocation overhead visible in median TPOT. — This is directly on the per-token executor path for multiprocessing and RayExecutorV2. Reducing Python allocation, serialization, and response-drain overhead can lower median TPOT and per-step jitter in multi-turn agentic workloads with many short decode steps.
- **Proposals (4):**
  - Precompile a static Ray Compiled Graph for the steady-state execute_model/sample_tokens control plane _(source: research_finding)_
  - Batch response fan-in with a zmq_poller wait-all across per-rank MessageQueues _(source: research_finding)_
  - Cache cloudpickle payloads for repeated callable RPCs in MultiprocExecutor.collective_rpc _(source: research_finding)_
  - Precompile per-hot-method RPC descriptors to eliminate per-step Python allocation on both leader and worker _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0001](../modules/vllm_v1_executor/MultiprocExecutor.collective_rpc__cand-vllm_v1_executor-0001.md)

### 9. `cand-vllm_v1_worker-0023` — GPUModelRunner.prepare_inputs (score 87, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/model_runner.py:971` (`GPUModelRunner.prepare_inputs`, method)
- **Description:** New GPU runner per-step input preparation for request ordering, idx mappings, scheduled-token arrays, query_start_loc, prefill inputs, positions, seq_lens, logits indices, and InputBatch assembly.
- **Current approach:** Sorts scheduler request IDs in Python, builds multiple NumPy arrays from dict iterators, performs separate H2D copies for idx_mapping and query_start_loc/cu_num_logits, then launches separate kernels for prefill token copy, positions/seq_lens, and sampled/draft token combination.
- **Why this impact / rank:** GPUModelRunner.prepare_inputs is the new runner's central per-step CPU/GPU preparation; coalescing H2D copies, removing seq_lens_cpu host sync, and precomputing metadata in a scheduler-side thread overlapped with model forward directly moves median TPOT and TTFT. — This is the new runner's central per-step CPU/GPU prep path; coalescing scheduler arrays and reducing H2D/kernel count directly lowers median TPOT and TTFT in multi-turn agent decode.
- **Proposals (4):**
  - Eliminate host-blocking seq_lens_cpu dependency in prepare_inputs to enable async spec-decode overlap _(source: research_finding)_
  - Coalesce per-step H2D transfers and move CPU idx-gathers onto GPU in prepare_inputs _(source: research_finding)_
  - Precompute per-step prepare_inputs metadata in a scheduler-side worker thread overlapped with model forward _(source: agent_knowledge)_
  - Emit decode-first ordered scheduler arrays to remove per-step sorting in prepare_inputs _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0023](../modules/vllm_v1_worker/GPUModelRunner.prepare_inputs__cand-vllm_v1_worker-0023.md)

### 10. `cand-vllm_v1_worker-0003` — GPUModelRunner._prepare_input_ids (score 86, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_model_runner.py:1825` (`GPUModelRunner._prepare_input_ids`, method)
- **Description:** Legacy async-scheduling path that scatters previously sampled and drafted tokens into input_ids.gpu for the next step.
- **Current approach:** Walks num_reqs in Python, builds four Python index lists, calls cu_num_tokens[cur_index].item() on a NumPy scalar, materializes multiple pinned torch.tensor objects, and launches separate scatter operations for sampled and draft tokens.
- **Why this impact / rank:** _prepare_input_ids sits on the async decode fast path with fixed per-step Python-list construction and four micro-copies; single-token multi-turn agent decode is exactly where this fixed overhead is most visible in median TPOT, and the fused-scatter kernel is a well-defined replacement. — Async scheduling exists to lower TPOT; four micro-copies plus Python list construction per step are fixed overhead that is most visible in single-token multi-turn agent decode.
- **Proposals (3):**
  - Replace async-path index-list construction and dual scatter with a fused CUDA kernel for next-step input_ids updates _(source: research_finding)_
  - Drop per-step int32 cast and intermediate gather-materializations by using index_copy_ with pre-cast draft buffer _(source: agent_knowledge)_
  - Copy only CPU-origin input_ids ranges before async overwrite _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0003](../modules/vllm_v1_worker/GPUModelRunner._prepare_input_ids__cand-vllm_v1_worker-0003.md)

### 11. `cand-vllm_v1_worker-0004` — GPUModelRunner._calc_spec_decode_metadata (score 84, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_model_runner.py:2891` (`GPUModelRunner._calc_spec_decode_metadata`, method)
- **Description:** Constructs cumulative draft/sample counts and logits index tensors for legacy speculative decoding.
- **Current approach:** Computes all index arrays on CPU with NumPy, then issues five independent async_tensor_h2d transfers for cu_num_draft_tokens, cu_num_sampled_tokens, logits_indices, target_logits_indices, and bonus_logits_indices; also returns num_draft_tokens as a Python list.
- **Why this impact / rank:** _calc_spec_decode_metadata builds spec-decode indices via five separate H2D copies plus a .tolist() sync every step; spec decode is a core TPOT lever, but demoted slightly vs shard because gains are conditional on the workload using speculative decoding. — Spec decode is a primary TPOT lever for agent serving; reducing per-step launch and copy overhead in its metadata path directly improves median TPOT for speculative workloads.
- **Proposals (3):**
  - Replace CPU-side spec-decode index computation and 5 H2D copies with a fused GPU kernel _(source: research_finding)_
  - Build spec-decode logits index tensors on-device to remove five H2D transfers and the num_draft_tokens tolist() sync _(source: research_finding)_
  - Amortize spec-decode metadata construction with a persistent pinned staging buffer plus one packed H2D and a torch.compile'd device kernel … _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0004](../modules/vllm_v1_worker/GPUModelRunner._calc_spec_decode_metadata__cand-vllm_v1_worker-0004.md)

### 12. `cand-vllm_v1_engine-0001` — DPLBAsyncMPClient.get_core_engine_for_request (score 83, impact: high)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/core_client.py:1471` (`DPLBAsyncMPClient.get_core_engine_for_request`, method)
- **Description:** Per-request data-parallel load balancer that selects one engine core from the locally managed DP engines.
- **Current approach:** Scans every engine on each request, scoring each as max(client_count * local_inflight, waiting + running) plus waiting * 6.0 * max(0, kv_cache_usage - 0.5) when there are queued requests. Ties are broken by a rotating scan start, and the stats snapshot can be stale between coordinator updates.
- **Why this impact / rank:** DPLB routing gates admission and per-engine KV pressure; prefix-affinity + P2C changes can materially lower median TTFT under multi-turn agentic bursts. Demoted below universal per-step levers because the gain requires DP deployment and stale-snapshot conditions to bind. — Routing choice directly controls per-engine queue depth and KV pressure; under multi-turn agentic bursts, avoiding stale or overloaded cores can lower median TTFT without changing model kernels.
- **Proposals (7):**
  - Adopt Power-of-Two Choices with weighted least-request for DP admission scoring _(source: research_finding)_
  - Add prefix-reuse-aware multiplicative routing to DPLB engine selection _(source: research_finding)_
  - Add prefix-aware three-tier DP routing (affinity → cache-util → P2C) in DPLBAsyncMPClient _(source: research_finding)_
  - Score DP engines on separate prefill and decode token dimensions _(source: research_finding)_
  - Smooth per-engine load with short/long EWMA divergence to detect queueing trend between coordinator updates _(source: research_finding)_
  - Make DP admission request-size aware via a projected-KV-headroom penalty _(source: agent_knowledge)_
  - Add deterministic request-affinity stickiness with bounded spillover _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0001](../modules/vllm_v1_engine/DPLBAsyncMPClient.get_core_engine_for_request__cand-vllm_v1_engine-0001.md)

### 13. `cand-vllm_v1_attention-0001` — use_cascade_attention (score 82, impact: high)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flash_attn.py:1586` (`use_cascade_attention`, function)
- **Description:** Decides whether FlashAttention cascade attention is used for a batch with a shared prefix.
- **Current approach:** Uses fixed gates for common_prefix_len < 256 and num_reqs < 8, then compares a rough CTA-count model with hardcoded 128-token Q and KV tile assumptions. The source explicitly marks the two gates as TODOs to tune.
- **Why this impact / rank:** use_cascade_attention routing is a fixed-gate branch between numerically equivalent attention kernels; long shared conversation prefixes are exactly the case where flipping to cascade cuts TTFT-dominant prefill cost, and correctness is easy to certify by output equality. — Multi-turn agentic batches often share long conversation prefixes. Better routing can reduce median TTFT by using cascade when shared-prefix work dominates and avoiding cascade overhead when it does not.
- **Proposals (4):**
  - Refine cascade-vs-FlashDecoding routing using Flash-Decoding's split-KV parallelization model _(source: research_finding)_
  - Replace fixed cascade gates with an occupancy-aware routing policy _(source: research_finding)_
  - Route cascade vs Flash-Decoding by HBM-bytes and L2 residency of the shared prefix _(source: agent_knowledge)_
  - Make cascade routing aware of non-decode query lengths _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0001](../modules/vllm_v1_attention/use_cascade_attention__cand-vllm_v1_attention-0001.md)

### 14. `cand-vllm_v1_engine-0005` — OutputProcessor.process_outputs (score 81, impact: high)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/output_processor.py:589` (`OutputProcessor.process_outputs`, method)
- **Description:** Main per-batch EngineCoreOutput loop that updates stats, detokenizes, processes logprobs, builds RequestOutputs, enqueues them, and frees finished requests.
- **Current approach:** Runs a sequential Python loop over engine_core_outputs. Each iteration performs request-state dict lookup, stats updates, detokenizer.update, logprobs processing, RequestOutput construction, queue insertion, finish handling, and optional tracing.
- **Why this impact / rank:** OutputProcessor.process_outputs is the only full-batch Python loop over every emitted token; branch specialization plus batched DecodeStream.step directly cut frontend CPU per token, translating into lower median TPOT under sustained agentic streaming. — This loop touches every emitted token before the caller can observe it; constant-factor reductions compound directly into lower frontend CPU cost and median TPOT.
- **Proposals (3):**
  - Batch DecodeStream.step over multi-token engine outputs in fast detokenizer path _(source: research_finding)_
  - Partition engine_core_outputs by branch and hoist bound-method locals in process_outputs _(source: agent_knowledge)_
  - Coalesce queued DELTA outputs before allocating RequestOutput _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0005](../modules/vllm_v1_engine/OutputProcessor.process_outputs__cand-vllm_v1_engine-0005.md)

### 15. `cand-vllm_v1_core-0006` — BlockHashToBlockMap and BlockPool.get_cached_block (score 80, impact: high)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/block_pool.py:33` (`BlockHashToBlockMap and BlockPool.get_cached_block`, region)
- **Description:** Prefix-cache hash lookup stack: maps hash/group keys to cached KV blocks and resolves a requested hash across one or more KV cache group ids.
- **Current approach:** BlockPool.get_cached_block allocates a result list, builds a BlockHashWithGroupId per group, and calls BlockHashToBlockMap.get_one_block. The map wraps a Python dict whose values are either a single KVCacheBlock or a dict of duplicate blocks, with isinstance dispatch in get_one_block, contain, insert, and pop.
- **Why this impact / rank:** BlockPool.get_cached_block runs on every prefix-cache probe, often hundreds per admission for long agent histories; split-map single-group fast path and precomputed group keys compress prefix-lookup wall time directly on TTFT. Micro-optimization though, so gains are constant-factor. — Every cache-hit probe performs at least one lookup here, often hundreds per admission for long agent histories. Micro-optimizations reduce prefix-lookup wall time and median TTFT.
- **Proposals (2):**
  - Split-map + single-group fast path for prefix-cache probes _(source: agent_knowledge)_
  - Add a direct representative-block index for hash probes _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0006](../modules/vllm_v1_core/BlockHashToBlockMap_and_BlockPool.get_cached_block__cand-vllm_v1_core-0006.md)

### 16. `cand-vllm_distributed_kv_transfer-0011` — HF3FSKVConnector._generate_block_hashes (score 79, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:955` (`HF3FSKVConnector._generate_block_hashes`, method)
- **Description:** Builds HF3FS block hash chains by looping over block-sized token slices and computing one prefix hash per full block.
- **Current approach:** Pure Python loop over range(0, len(token_ids), self._block_size). Each iteration slices token_ids, calls _compute_prefix_hash, conditionally appends the hash, and advances previous_hash sequentially; lookup, save, and load paths repeat this work.
- **Why this impact / rank:** HF3FSKVConnector._generate_block_hashes scales with prompt length on the TTFT-critical prefix lookup path; replacing stringified-list MD5 with XXH3 and caching chains cuts per-turn hashing when long multi-turn history reuse is present. Gated on HF3FS being in the path. — Hash generation is on the TTFT-critical prefix lookup path and scales with prompt length, which is common in multi-turn agentic conversations with long shared history.
- **Proposals (4):**
  - Replace stringified-list MD5 with xxHash XXH3/XXH128 over binary token buffers _(source: research_finding)_
  - Adopt APC-style faster block hash and cached chain in HF3FSKVConnector._generate_block_hashes _(source: research_finding)_
  - Bound _generate_block_hashes work in get_num_new_matched_tokens with an exponential-probe existence check _(source: agent_knowledge)_
  - Stop hashing the same prefix twice by passing matched block hashes through scheduler metadata _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0011](../modules/vllm_distributed_kv_transfer/HF3FSKVConnector._generate_block_hashes__cand-vllm_distributed_kv_transfer-0011.md)

### 17. `cand-vllm_v1_kv_offload-0009` — TieringOffloadingManager._initiate_promotion (score 78, impact: high)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/manager.py:410` (`TieringOffloadingManager._initiate_promotion`, method)
- **Description:** Reserves primary-tier space for a secondary-to-primary promotion and accumulates pending load submissions by tier and request.
- **Current approach:** Each secondary hit calls primary_tier.prepare_write([key], req_context) immediately for a single key, then appends returned keys and block IDs into a per-tier/per-request PendingPromotion batch flushed later.
- **Why this impact / rank:** TieringOffloadingManager._initiate_promotion controls how fast secondary hits become primary hits; batched prepare_write plus TinyLFU/S3-FIFO admission gates directly move TTFT on cache-hit turns and TPOT during promotion waits. High evidence (rf_count 9) but conditional on tiered offload being enabled. — Promotion admission and allocation determine how quickly secondary hits become primary hits and whether the primary tier thrashes. This directly affects TTFT on cache-hit turns and TPOT when decode waits for promoted blocks.
- **Proposals (11):**
  - Prioritize promotions by agentic workflow value, not just recency _(source: research_finding)_
  - Split per-request promotion batch into progressive sub-batches to reduce HOL blocking _(source: research_finding)_
  - Gate secondary-to-primary promotions with an S3-FIFO probationary + ghost queue _(source: research_finding)_
  - Add a TinyLFU admission gate before promoting secondary blocks to the primary tier _(source: research_finding)_
  - Batch primary-tier allocation for promotions instead of per-key prepare_write _(source: research_finding)_
  - Batch prepare_write allocations per step to amortize per-key primary-tier admission overhead _(source: research_finding)_
  - Gate secondary-to-primary promotion on predicted usefulness instead of promoting every hit _(source: research_finding)_
  - Speculative prefetch promotion to hide secondary-to-primary load latency _(source: research_finding)_
  - Schedule promotion transfers into decode idle-bandwidth windows with batched prepare_write _(source: research_finding)_
  - Coalesce cross-request duplicate promotions within a step via a shared in-flight promotion table _(source: agent_knowledge)_
  - Release reserved promotion slots when requests cancel before load completion _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0009](../modules/vllm_v1_kv_offload/TieringOffloadingManager._initiate_promotion__cand-vllm_v1_kv_offload-0009.md)

### 18. `cand-vllm_v1_worker-0019` — UBatchWrapper._capture_ubatches and _run_ubatches (score 77, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_ubatch_wrapper.py:212` (`UBatchWrapper._capture_ubatches and _run_ubatches`, region)
- **Description:** Threaded CUDA graph capture and runtime execution for legacy DBO microbatches.
- **Current approach:** Starts one Python thread per ubatch, synchronizes with a Barrier and threading.Event handoff, and serializes completion through thread joins and sorted result concatenation.
- **Why this impact / rank:** UBatchWrapper's capture/run implements DBO to overlap mixed prefill/decode; CUDA graph reuse and persistent worker threads cut per-forward overhead and improve overlap for exactly the non-uniform agent batches this workload produces. Gated on DBO being enabled. — DBO exists to lower TPOT for mixed prefill/decode batches; reducing scheduler overhead or adapting the handoff policy improves overlap and median TPOT for non-uniform agent batches.
- **Proposals (3):**
  - Reuse ubatch CUDA graphs across token buckets via cudaGraphExecUpdate _(source: research_finding)_
  - Reuse persistent ubatch worker threads instead of spawning per forward pass _(source: agent_knowledge)_
  - Reuse ubatch synchronization contexts across eager forwards _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0019](../modules/vllm_v1_worker/UBatchWrapper._capture_ubatches_and__run_ubatches__cand-vllm_v1_worker-0019.md)

### 19. `cand-vllm_v1_worker-0011` — BlockTable and MultiGroupBlockTable row mutation methods (score 76, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/block_table.py:138` (`BlockTable and MultiGroupBlockTable row mutation methods`, region)
- **Description:** CPU-side block-table row append/add/clear/move/swap operations used while applying scheduler block allocations to one or more KV-cache groups.
- **Current approach:** BlockTable.append_row performs per-row NumPy writes and hybrid block expansion; MultiGroupBlockTable fans row operations out through Python loops across KV-cache groups.
- **Why this impact / rank:** BlockTable per-group row mutation runs O(active changes x KV groups) every step; batching row updates and copying only dirty rows removes repeated Python calls for hybrid/multi-group models. Visible on both TTFT admission and TPOT decode, but a scoped constant-factor win. — Request churn makes this O(active changes x KV groups) host work visible in TTFT and TPOT; batching removes repeated Python calls for hybrid and multi-group models.
- **Proposals (3):**
  - Batch block-table row mutations across requests and KV groups (MRV2-style persistent state) _(source: research_finding)_
  - GPU-resident block-table applier: fused Triton kernel replaces per-step H2D row copies _(source: agent_knowledge)_
  - Copy only dirty block-table rows during commit _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0011](../modules/vllm_v1_worker/BlockTable_and_MultiGroupBlockTable_row_mutation_methods__cand-vllm_v1_worker-0011.md)

### 20. `cand-vllm_distributed_kv_transfer-0005` — NixlBaseConnectorScheduler.__init__ transfer policy defaults (score 74, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:70` (`NixlBaseConnectorScheduler.__init__ transfer policy defaults`, config_block)
- **Description:** Initializes NIXL policy defaults for KV lease duration, heartbeat interval, recompute threshold, bidirectional transfer enablement, and decoder KV block TTL.
- **Current approach:** Static extra-config defaults are applied once at scheduler construction. kv_recompute_threshold is later used as a fixed token-count cutoff in pull_scheduler.py to choose remote pull versus local recompute, independent of observed RTT, bandwidth, queue depth, or prefill throughput.
- **Why this impact / rank:** NIXL pull-vs-recompute defaults determine per-turn TTFT on every remote-prefill hit; an EWMA-adjusted kv_recompute_threshold is well-founded and multi-turn agent prompts hit this decision constantly. Demoted because the effect only lands for disaggregated NIXL deployments. — A wrong pull-vs-recompute decision directly adds TTFT on every candidate remote-prefill hit; short turn-2 agentic prompts make this a frequent per-turn decision.
- **Proposals (5):**
  - Replace static kv_recompute_threshold with a Dynamo-style cost-scored pull-vs-recompute decision _(source: research_finding)_
  - Replace static kv_recompute_threshold with a CacheBlend-style dynamic transfer-vs-recompute policy in NIXL scheduler _(source: research_finding)_
  - Learn kv_recompute_threshold from recorded NIXL transfer stats _(source: research_finding)_
  - Adapt decoder_kv_blocks_ttl and heartbeat cadence per remote engine from observed inter-turn reuse gaps _(source: agent_knowledge)_
  - Make bidirectional KV transfer opt-in per engine after capability handshake _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0005](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorScheduler.__init___transfer_policy_defaults__cand-vllm_distributed_kv_transfer-0005.md)

### 21. `cand-vllm_v1_engine-0006` — BaseIncrementalDetokenizer.update (score 73, impact: high)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/detokenizer.py:96` (`BaseIncrementalDetokenizer.update`, method)
- **Description:** Per-request incremental detokenization and stop-check driver for newly generated token ids.
- **Current approach:** Loops in Python over new_token_ids, appends each id, calls decode_next per token, concatenates into output_text, adjusts min_tokens stop-check offset, then calls check_stop_strings once for the newly appended text.
- **Why this impact / rank:** BaseIncrementalDetokenizer.update is per-token frontend CPU work; batched DecodeStream, no-stop fast path, and Aho-Corasick stop matching cut Python cost on the TPOT path, but detokenization is usually a small fraction of TPOT compared to sampler/H2D syncs. — For agentic workloads with many short turns, detokenization is a repeated frontend CPU cost; reducing per-token Python decoding and string concatenation can lower median TPOT.
- **Proposals (4):**
  - Use per-request Aho-Corasick automaton for incremental stop-string matching _(source: research_finding)_
  - Batch DecodeStream over new_token_ids in FastIncrementalDetokenizer for speculative multi-token steps _(source: research_finding)_
  - Amortize output_text growth via chunk buffer with bounded stop-check tail _(source: agent_knowledge)_
  - Add a no-stop fast path for requests without stop strings _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0006](../modules/vllm_v1_engine/BaseIncrementalDetokenizer.update__cand-vllm_v1_engine-0006.md)

### 22. `cand-vllm_v1_core-0001` — HybridKVCacheCoordinator.find_longest_cache_hit (score 72, impact: high)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/kv_cache_coordinator.py:685` (`HybridKVCacheCoordinator.find_longest_cache_hit`, method)
- **Description:** Fixed-point reconciliation of prefix-cache hit length across heterogeneous KV cache groups including full attention, sliding-window attention, chunked-local attention, and Mamba.
- **Current approach:** Iterates over sorted attention_groups, invokes each group's classmethod finder, tracks EAGLE verification in a Python set, memoizes the downward-closed full-attention hit, and repeats until curr_hit_length stops shrinking. The simple hybrid case of one full-attention group plus one other group exits after one pass.
- **Why this impact / rank:** HybridKVCacheCoordinator.find_longest_cache_hit is the central hybrid prefix-cache admission loop; reordering attention groups and length-only reconciliation cut TTFT for multi-turn resubmits. Impact narrower than full-attention path because it only binds for hybrid models. — Hybrid prefix-cache lookup sits on TTFT for newly admitted, resumed, and preempted requests. Multi-turn agent workloads repeatedly submit long shared prefixes, so reducing reconciliation probes directly reduces median TTFT.
- **Proposals (2):**
  - Reorder attention groups by empirical shrink probability and skip groups that already accepted the current candidate length _(source: agent_knowledge)_
  - Split hybrid lookup into length-only reconciliation and one final block materialization pass _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0001](../modules/vllm_v1_core/HybridKVCacheCoordinator.find_longest_cache_hit__cand-vllm_v1_core-0001.md)

### 23. `cand-vllm_v1_core-0007` — get_request_block_hasher.request_block_hasher (score 70, impact: high)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/kv_cache_utils.py:685` (`get_request_block_hasher.request_block_hasher`, function)
- **Description:** Request-local closure that computes chained block hashes for newly completed hash-size token blocks.
- **Current approach:** Runs a pure-Python while loop over new full blocks, computes multimodal/LoRA/embedding extra keys, slices token_ids per block, calls hash_block_tokens, appends the result, and threads the parent hash forward.
- **Why this impact / rank:** request_block_hasher scans hundreds/thousands of blocks synchronously before admission; LRU over completed block hash tuples and lifted extra-key invariants shave TTFT on long-context turns. Solid but scoped to admission of new long prompts, and hashing is typically cheaper than IO. — Agent prompts can contain tens of thousands of tokens, producing hundreds or thousands of hashes before scheduling. Reducing this CPU loop directly lowers median TTFT for long-context turns.
- **Proposals (2):**
  - Precompute extra-keys fast path and lift per-request invariants out of the block loop _(source: agent_knowledge)_
  - Add an LRU cache for completed block hash tuples _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0007](../modules/vllm_v1_core/get_request_block_hasher.request_block_hasher__cand-vllm_v1_core-0007.md)

### 24. `cand-vllm_distributed_kv_transfer-0006` — MultiConnector.get_num_new_matched_tokens (score 68, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:385` (`MultiConnector.get_num_new_matched_tokens`, method)
- **Description:** Routes prefix-hit lookup across an ordered connector list and pins the request to the first connector that reports tokens > 0.
- **Current approach:** First-positive-wins policy is enforced by the to_return[0] == 0 guard. The method still queries every connector to detect pending async lookups, but does not compare hit length, expected load latency, async behavior, backpressure, or tier speed before choosing a connector.
- **Why this impact / rank:** MultiConnector.get_num_new_matched_tokens selects where turn-2 KV loads from across tiers; a scored router replacing first-positive-wins can meaningfully move TTFT when GPU/CPU/remote/object connectors coexist. Conditional on multi-connector deployment. — The chosen connector determines where turn-2 KV is loaded from, often the dominant TTFT factor when multiple cache tiers are configured.
- **Proposals (4):**
  - Replace first-positive-wins with a scored connector router using overlap + projected load cost _(source: research_finding)_
  - Score connectors by predicted load latency instead of first-positive-wins _(source: research_finding)_
  - Speculative parallel prefetch from top-K connectors with first-complete-wins arbitration _(source: agent_knowledge)_
  - Require compatible hit spans before accepting a connector match _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0006](../modules/vllm_distributed_kv_transfer/MultiConnector.get_num_new_matched_tokens__cand-vllm_distributed_kv_transfer-0006.md)

### 25. `cand-vllm_v1_executor-0013` — RayWorkerWrapper.execute_model_ray (score 66, impact: high)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/ray_utils.py:125` (`RayWorkerWrapper.execute_model_ray`, method)
- **Description:** Worker-side Ray compiled-DAG node body: sets the CUDA device if needed, unpacks scheduler/intermediate inputs, runs model_runner.execute_model, prepares PP intermediate handoff, and materializes final async outputs before returning through Ray.
- **Current approach:** Every DAG invocation calls setup_device_if_necessary, branches on tuple length, calls worker.model_runner.execute_model, checks _is_intermediate_tensors, may loop over scheduled_new_reqs to clear mm_features for PP transfer, calls AsyncModelRunnerOutput.get_output for final execute_model output, checks _is_last_rank, and may call sample_tokens plus another get_output when execute_model returned N…
- **Why this impact / rank:** execute_model_ray is the per-worker body of the Ray steady-state DAG; overlapping AsyncModelRunnerOutput D2H with the next channel return and eliminating per-step Python branching cut host overhead every decode. Gated on Ray backend, capping the blast radius. — This is the per-worker body of the Ray steady-state execution graph. Removing host/device sync or Python overhead here affects every token step across all Ray workers, so it can reduce median TPOT in multi-turn workloads that use the Ray backend.
- **Proposals (5):**
  - Restructure PP handoff return path in execute_model_ray to enable Ray CG GPU communication overlap _(source: research_finding)_
  - Overlap AsyncModelRunnerOutput D2H materialization with next Ray DAG step via pinned buffers on a side stream _(source: research_finding)_
  - Defer AsyncModelRunnerOutput materialization in Ray DAG node to overlap D2H with channel return _(source: research_finding)_
  - Specialize execute_model_ray into role-bound closures at DAG compile time to eliminate per-step Python branching _(source: agent_knowledge)_
  - Return a PP-transfer view of SchedulerOutput instead of mutating mm_features per step _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0013](../modules/vllm_v1_executor/RayWorkerWrapper.execute_model_ray__cand-vllm_v1_executor-0013.md)

### 26. `cand-vllm_v1_worker-0010` — _compute_slot_mapping_kernel (score 64, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/block_table.py:380` (`_compute_slot_mapping_kernel`, kernel)
- **Description:** Legacy Triton kernel that maps scheduled token positions to KV-cache slot IDs and pads unused CUDA graph slots.
- **Current approach:** Launches one program per request plus one padder program, with a fixed BLOCK_SIZE=1024 vector per request tile regardless of actual decode length.
- **Why this impact / rank:** _compute_slot_mapping_kernel's fixed 1024-lane per-request tile wastes lanes on one-token decodes; a 2D token-tile grid or one-token fast path shrinks kernel time on decode-heavy agent turns. Kernel micro-optimization on a lightweight kernel, so absolute savings are modest. — Slot mapping runs every step per cache group; decode-heavy agent workloads often schedule one token per request, so the current tiling wastes lanes and affects steady TPOT.
- **Proposals (2):**
  - Reshape slot-mapping kernel to a 2D token-tile grid with binary-search req lookup _(source: agent_knowledge)_
  - Add a one-token decode fast path without req lookup _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0010](../modules/vllm_v1_worker/_compute_slot_mapping_kernel__cand-vllm_v1_worker-0010.md)

### 27. `cand-vllm_v1_worker-0024` — BlockTables.gather_block_tables and compute_slot_mappings (score 62, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/block_table.py:136` (`BlockTables.gather_block_tables and compute_slot_mappings`, region)
- **Description:** New GPU runner block-table gather and slot-mapping preparation for attention metadata and KV cache writes.
- **Current approach:** Launches a gather kernel over num_kv_cache_groups x num_reqs_padded, then launches a slot-mapping kernel over num_groups x (num_reqs + 1), both with fixed 1024-element tiles and a dedicated padding program.
- **Why this impact / rank:** BlockTables.gather_block_tables + compute_slot_mappings run every step in the new runner with a fixed 1024 tile; decode-fused path mapping one request per lane cuts TPOT for one-token-per-request agent batches. Same class of win as _compute_slot_mapping_kernel but on the new runner path. — Block-table gather and slot mapping run every step in the new runner; decode-heavy agent batches with many one-token requests waste tile lanes, so adaptive tiling/fusion reduces median TPOT.
- **Proposals (2):**
  - Add a decode-fused path for compute_slot_mappings and gather_block_tables that maps one request per lane instead of one request per program _(source: agent_knowledge)_
  - Bound slot-mapping padding to the returned padded token count _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0024](../modules/vllm_v1_worker/BlockTables.gather_block_tables_and_compute_slot_mappings__cand-vllm_v1_worker-0024.md)

### 28. `cand-vllm_v1_worker-0002` — GPUModelRunner._update_states (score 60, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_model_runner.py:1233` (`GPUModelRunner._update_states`, method)
- **Description:** Applies scheduler deltas for finished, unscheduled, new, resumed, and scheduled requests to cached request state and the persistent legacy GPU input batch.
- **Current approach:** Uses nested Python loops over scheduler request groups, per-request dict lookups, list extensions for block_ids and output_token_ids, per-request CPU tensor slice writes, and per-request block_table.append_row/add_row calls before condensing the batch.
- **Why this impact / rank:** _update_states runs before every legacy model step with Python cost scaling with active-request churn — matches multi-turn agent traffic. MRV2-style decoupled state and fused slot-swap admission reduce the per-step floor, but many turns are cheap when churn is low. — Admission, preemption, and finish handling gate TTFT for new turns and affect churn-heavy decode steps; reducing Python O(active requests) work lowers median TTFT and TPOT under high request turnover.
- **Proposals (3):**
  - Adopt MRV2-style decoupled persistent/per-step state in _update_states _(source: research_finding)_
  - Fuse remove/condense/add churn into in-step slot-swap admission in _update_states _(source: agent_knowledge)_
  - Cache per-step scheduler membership to avoid repeated set/dict probes _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0002](../modules/vllm_v1_worker/GPUModelRunner._update_states__cand-vllm_v1_worker-0002.md)

### 29. `cand-vllm_v1_core-0002` — FullAttentionManager.find_longest_cache_hit (score 58, impact: high)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/single_type_kv_cache_manager.py:682` (`FullAttentionManager.find_longest_cache_hit`, method)
- **Description:** Full-attention prefix-cache hit lookup using a full-block forward scan plus optional fine-grained interior-boundary probe for partial hash hits.
- **Current approach:** Runs a Python loop over candidate block hashes, calls block_pool.get_cached_block once per block, stops on the first chained-hash miss, appends cached blocks into per-group lists, scans partial boundaries high-to-low in fine-grained mode, then applies EAGLE and alignment trimming.
- **Why this impact / rank:** FullAttentionManager.find_longest_cache_hit runs on every admission; galloping search over chained-hash monotonicity plus per-step memoization cuts TTFT for the agentic workload. Gains are concentrated in admission, not steady-state, so ranked below per-step levers. — Called for most models on every admission. Long multi-turn prompts require hundreds of block-hash probes; reducing per-probe Python overhead lowers median TTFT.
- **Proposals (2):**
  - Galloping + binary-search hit boundary in Phase-1 full-block scan (exploit chained-hash monotonicity) _(source: agent_knowledge)_
  - Memoize exact prefix-cache lookup results within a scheduler step _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0002](../modules/vllm_v1_core/FullAttentionManager.find_longest_cache_hit__cand-vllm_v1_core-0002.md)

### 30. `cand-vllm_distributed_kv_transfer-0001` — NixlBaseConnectorWorker._pop_done_transfers (score 56, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2210` (`NixlBaseConnectorWorker._pop_done_transfers`, method)
- **Description:** Called from get_finished() every engine step, scans all in-flight NIXL receive transfers and, in push mode, send transfers, probing one handle at a time.
- **Current approach:** Iterates dict[req_id -> list[handle]] in Python and calls nixl_wrapper.check_xfer_state(handle) for each handle. DONE handles synchronously fetch telemetry, record stats, and release the transfer handle inline on the engine thread.
- **Why this impact / rank:** NixlBaseConnectorWorker._pop_done_transfers is polled every decode step; notification-driven completion collapses O(in_flight * handles) probes into a single wait. Demoted vs shard because effect only binds when NIXL disaggregated transfers are active. — The loop runs on every decode step; polling latency directly affects TPOT, and delayed DONE detection for remote-prefill reads affects TTFT in concurrent multi-turn agentic workloads.
- **Proposals (3):**
  - Replace per-handle polling in _pop_done_transfers with NIXL notification-driven completion _(source: research_finding)_
  - Batch-harvest NIXL completions per poll to amortize FFI and telemetry costs _(source: research_finding)_
  - Short-circuit per-request handle probing with a resumable non-DONE cursor _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0001](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorWorker._pop_done_transfers__cand-vllm_distributed_kv_transfer-0001.md)

### 31. `cand-vllm_v1_core-0012` — KVCacheManager.allocate_slots (score 54, impact: high)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/kv_cache_manager.py:345` (`KVCacheManager.allocate_slots`, method)
- **Description:** Central KV slot allocation entry point for admitted or extending requests: fit checks, skipped-block removal, local/external prefix blocks, new block allocation, and cache insertion.
- **Current approach:** Computes local and total computed tokens, applies watermark and reserved-block gates, optionally performs a full-sequence fit pre-check that duplicates get_num_blocks_to_allocate work, removes skipped blocks, recomputes required blocks, allocates computed and new blocks, and caches finalized tokens.
- **Why this impact / rank:** KVCacheManager.allocate_slots is on the scheduler path for every capacity-needing request; removing the duplicate coordinator.get_num_blocks_to_allocate and reordering skipped-block removal shave scheduler overhead. Modest per-request Python savings only. — Every scheduled request that needs KV capacity goes through this method. Removing redundant admission work and reducing allocation bookkeeping lowers TTFT for prefills and TPOT for decode steps that allocate blocks.
- **Proposals (3):**
  - Add a commit-policy parameter to allocate_slots to gate exact-prefix caching of external/approximate KV _(source: research_finding)_
  - Reuse the full_sequence_must_fit sizing result to eliminate a duplicate coordinator.get_num_blocks_to_allocate call _(source: agent_knowledge)_
  - Move skipped-block removal before the full-sequence fit gate _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0012](../modules/vllm_v1_core/KVCacheManager.allocate_slots__cand-vllm_v1_core-0012.md)

### 32. `cand-vllm_v1_executor-0004` — WorkerProc.enqueue_output/handle_output/async_output_busy_loop (score 52, impact: high)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/multiproc_executor.py:961` (`WorkerProc.enqueue_output/handle_output/async_output_busy_loop`, region)
- **Description:** Worker output materialization and response enqueue path, including the async-output thread that drains queued AsyncModelRunnerOutput objects.
- **Current approach:** enqueue_output converts AsyncModelRunnerOutput by calling get_output inline, maps exceptions to FAILURE tuples and other outputs to SUCCESS tuples, then enqueues on worker_response_mq. handle_output either calls enqueue_output immediately or puts the object on async_output_queue. async_output_busy_loop sets the worker device once and then handles one queued output at a time with blocking queue.Qu…
- **Why this impact / rank:** WorkerProc async output handling is the host/device sync boundary for token materialization; overlapping D2H with next RPC and batch-draining the queue shorten token-visible latency. Overlaps significantly with the AsyncOutput.get_output candidate, so ranked to avoid double-counting. — This is the multiprocessing worker's main host/device synchronization point for model outputs. Reducing blocking get_output time or batching copies directly shortens token-produced to token-visible latency, moving median TPOT in async multi-turn workloads.
- **Proposals (3):**
  - Stage async model outputs into pinned host buffers on a non-default CUDA stream to overlap D2H copies with the next worker step _(source: research_finding)_
  - Batch-drain async_output_queue and launch get_output() with lookahead so per-rank sync cost amortizes across queued outputs _(source: agent_knowledge)_
  - Pipeline async output materialization separately from response-mq serialization _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0004](../modules/vllm_v1_executor/WorkerProc.enqueue_output_handle_output_async_output_busy_loop__cand-vllm_v1_executor-0004.md)

### 33. `cand-vllm_v1_executor-0012` — RayDistributedExecutor._compiled_ray_dag (score 50, impact: high)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/ray_executor.py:527` (`RayDistributedExecutor._compiled_ray_dag`, method)
- **Description:** Compile-time construction of the Ray compiled DAG used for steady-state model execution, including PP/TP topology, tensor transport policy, optional vLLM PP communicator registration, and Ray overlap flag.
- **Current approach:** The method validates Ray/cgraph support, sets RAY_CGRAPH_get_timeout default, builds a MultiOutputNode by chaining execute_model_ray.bind across pp_tp_workers, applies with_tensor_transport for non-shm intermediate PP edges, optionally registers RayPPCommunicator, then calls experimental_compile with _overlap_gpu_communication from VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM.
- **Why this impact / rank:** Ray compiled-DAG topology and per-edge transport selection govern steady-state PP/TP comm overhead; TTFT gains from eager CGraph warmup are real but one-shot per model. Multi-node Ray deployment is a narrow slice, so ranked below universal levers despite high shard score. — Ray compiled-DAG topology and transport choices directly affect per-step communication and synchronization in multi-node PP/TP serving. Better choices can reduce both median TTFT after graph creation and steady-state median TPOT for Ray-backed agentic workloads.
- **Proposals (7):**
  - Enable Ray CGraph GPU comm/compute overlap by default for PP DAGs and tune per topology _(source: research_finding)_
  - Topology-aware PP/TP rank placement and per-edge transport selection in Ray compiled DAG _(source: research_finding)_
  - Topology- and message-size-aware per-PP-edge transport selection in the Ray compiled DAG _(source: research_finding)_
  - Topology-aware per-PP-edge transport selection in _compiled_ray_dag _(source: research_finding)_
  - Set explicit _max_inflight_executions on Ray compiled DAG to bound async concurrency _(source: research_finding)_
  - Eagerly compile and warm up the Ray CGraph in _init_executor to hide compile latency from first-request TTFT _(source: agent_knowledge)_
  - Stop forwarding scheduler metadata through PP tensor handoff edges _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0012](../modules/vllm_v1_executor/RayDistributedExecutor._compiled_ray_dag__cand-vllm_v1_executor-0012.md)

### 34. `cand-vllm_v1_worker-0028` — GPUModelRunner.add_requests and update_requests (score 48, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/model_runner.py:879` (`GPUModelRunner.add_requests and update_requests`, region)
- **Description:** New GPU runner per-step request admission and cached-request state update path before input preparation.
- **Current approach:** Loops over scheduled_new_reqs and scheduled_cached_reqs, repeatedly removes/re-adds request state, performs dict lookups, stages request/model/sampler writes separately, appends block IDs per request, and updates CPU mirrors with per-request assignments plus a full np.minimum pass.
- **Why this impact / rank:** add_requests/update_requests loops are hit on new agent turns; vectorizing state deltas via scatter cuts TTFT admission overhead. High rf_count is 0 and gains are Python-loop micro-savings, which are dwarfed by the H2D/plan-cache wins above. — Admission and cached-request churn are common in multi-turn agent workloads; batching state deltas reduces TTFT for new turns and TPOT during high-turnover decode.
- **Proposals (2):**
  - Vectorize update_requests via a single scatter over pre-packed CPU arrays _(source: agent_knowledge)_
  - Preserve request slots for streaming-input re-admission _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0028](../modules/vllm_v1_worker/GPUModelRunner.add_requests_and_update_requests__cand-vllm_v1_worker-0028.md)

### 35. `cand-vllm_distributed_kv_transfer-0002` — NixlPullConnectorWorker._read_blocks_for_req (score 46, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:126` (`NixlPullConnectorWorker._read_blocks_for_req`, method)
- **Description:** For each remote pull, builds one ReadSpec per source rank and posts READ transfers serially before a request can resume.
- **Current approach:** ReadSpec construction rebuilds per-group local and remote block-id lists for every source rank, including empty groups. The method then loops over read_specs, selects the side handles, and calls _read_blocks one rank at a time, followed by a Python send_notif fan-out for pure MLA heterogeneous TP.
- **Why this impact / rank:** NixlPullConnectorWorker._read_blocks_for_req batches per-rank READ posts on the pre-resume path; multi-turn reuse hits this on every remote-prefill cache hit but only on heterogeneous-TP NIXL disaggregated deployments, sharply narrowing reach. — Turn-2 TTFT includes READ post latency plus transfer completion; agentic multi-turn reuse exercises this path on each remote-prefill cache hit.
- **Proposals (3):**
  - Batch and pipeline per-rank NIXL READ posts using non-blocking API and cached remote metadata _(source: research_finding)_
  - Batch cross-rank READ transfers and coalesce sparse scatter lists per request _(source: research_finding)_
  - Pre-allocate reusable NIXL xfer descriptor ring per (engine_id, tp_ratio) to eliminate make_prepped_xfer allocation on the pre-resume path _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0002](../modules/vllm_distributed_kv_transfer/NixlPullConnectorWorker._read_blocks_for_req__cand-vllm_distributed_kv_transfer-0002.md)

### 36. `cand-vllm_v1_core-0004` — BlockPool.get_new_blocks (score 44, impact: high)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/block_pool.py:647` (`BlockPool.get_new_blocks`, method)
- **Description:** Allocates fresh KV cache blocks from the free queue, evicts stale prefix-cache metadata when needed, increments ref counts, and records allocation metrics.
- **Current approach:** Calls FreeKVCacheBlockQueue.popleft_n, then loops per returned block. The caching branch calls _maybe_evict_cached_block, asserts ref_cnt is zero, increments it, and emits metrics; the no-cache branch duplicates the ref-count and metrics loop without eviction.
- **Why this impact / rank:** BlockPool.get_new_blocks funnels every slot allocation; fast-path eviction skip and batched BlockRemoved events accumulate under concurrency, but per-block savings are small in absolute terms compared to per-step host syncs. — Runs on the scheduler path for prefill chunks and decode steps that cross block boundaries. Per-block savings accumulate under high-concurrency agent workloads and reduce TPOT scheduler overhead.
- **Proposals (2):**
  - Fast-path skip eviction for never-cached blocks and batch KV BlockRemoved events by group in get_new_blocks _(source: agent_knowledge)_
  - Fuse free-queue pop with allocation initialization _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0004](../modules/vllm_v1_core/BlockPool.get_new_blocks__cand-vllm_v1_core-0004.md)

### 37. `cand-vllm_v1_kv_offload-0008` — TieringOffloadingManager.lookup (score 42, impact: high)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/manager.py:311` (`TieringOffloadingManager.lookup`, method)
- **Description:** Per-block tiered lookup checks the primary tier, then scans secondary tiers and initiates promotion on the first secondary hit.
- **Current approach:** Processes finished jobs once per step, probes the primary tier, then iterates secondary_tiers in configuration order with load-tier filters. It short-circuits on the first HIT and tracks any RETRY, with no hit-rate-aware ordering or batched secondary probing.
- **Why this impact / rank:** TieringOffloadingManager.lookup gates promotion scheduling; batching secondary probes cuts synchronous lookup delay when traffic hits non-primary tiers. Demoted because reach is bounded to configurations with tiered offload enabled AND traffic missing primary. — Secondary lookup latency contributes directly to TTFT before promotions can be scheduled. Reordering or batching probes can reduce synchronous lookup delay for multi-turn agentic traffic that frequently hits non-primary tiers.
- **Proposals (5):**
  - Batch secondary-tier lookups per request and offload probing to a background I/O path _(source: research_finding)_
  - Batch SSD-tier lookups into object-granularity probes with async promotions _(source: research_finding)_
  - Add anticipatory secondary-to-primary promotion to hide lookup-path latency _(source: research_finding)_
  - Adaptive tier-order permutation via online hit-rate bandit in lookup() _(source: agent_knowledge)_
  - Add retry-state memoization to avoid repeated secondary rescans _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0008](../modules/vllm_v1_kv_offload/TieringOffloadingManager.lookup__cand-vllm_v1_kv_offload-0008.md)

### 38. `cand-vllm_v1_executor-0007` — RayDistributedExecutor._execute_dag (score 40, impact: high)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/ray_executor.py:434` (`RayDistributedExecutor._execute_dag`, method)
- **Description:** Per-step Ray compiled-DAG execution boundary: submits scheduler output to the compiled DAG, blocks or returns a FutureWrapper, detaches Ray SHM views, and aggregates multi-worker outputs when a KV connector is present.
- **Current approach:** The method lazily compiles the DAG, calls self.forward_dag.execute, then either blocks on refs[0].get() and detaches one output, returns FutureWrapper(refs[0]), or ray.gets all refs for connector mode, detaches outputs sequentially, and calls kv_output_aggregator.aggregate.
- **Why this impact / rank:** RayDistributedExecutor._execute_dag is the driver-side per-token blocking wait for Ray compiled-DAG; overlapping detach with ray.wait cuts TPOT but Ray backend + connector mode is a narrow slice, and effects duplicate execute_model_ray-side wins above. — This method is the driver-side per-token wait point for the Ray compiled-DAG backend. Any overlap or reduction in get-detach-aggregate wall time directly reduces median TPOT, especially in multi-node or connector-enabled Ray deployments.
- **Proposals (3):**
  - Use ray.wait to overlap detach and KV aggregation with worker readiness in connector mode _(source: research_finding)_
  - Set explicit _max_inflight_executions on the compiled DAG to bound nonblocking step queueing _(source: research_finding)_
  - Detach only the output_rank output in connector-mode blocking path _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0007](../modules/vllm_v1_executor/RayDistributedExecutor._execute_dag__cand-vllm_v1_executor-0007.md)

### 39. `cand-vllm_multimodal-0001` — MultiModalHasher.serialize_item (score 38, impact: high)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/hasher.py:53` (`MultiModalHasher.serialize_item`, method)
- **Description:** Serializes arbitrary multimodal hash inputs, including PIL images, MediaWithBytes wrappers, torch tensors, numpy arrays, primitives, and pickle fallback values, for processor cache keys.
- **Current approach:** Uses explicit isinstance dispatch. PIL Image values are converted through np.asarray; torch.Tensor values always call .cpu() before numpy conversion; bfloat16 tensors are made contiguous and viewed as uint8; non-contiguous numpy arrays copy through .tobytes(). MediaWithBytes image/video wrappers may use original_bytes after EXIF ImageID checks and size heuristics.
- **Why this impact / rank:** MultiModalHasher.serialize_item hashing dominates TTFT for images/CUDA tensors before hit detection; GPU-side hashing and pooled pinned buffers are clean fixes but the workload is described as agentic without multimodal signal, so gain is conditional. — Large images, video frame arrays, and CUDA tensors make hashing dominated by memory copies. Reducing those copies moves median TTFT because the hash is paid even when the processor cache later hits.
- **Proposals (4):**
  - Stage CUDA-tensor D2H copies in hasher through a pooled pinned buffer with non_blocking + explicit sync _(source: research_finding)_
  - Coalesce metadata and enable BLAKE3 parallel updates for large tensor byte spans _(source: research_finding)_
  - Memoize hex digests on Python object identity via a weakref-keyed LRU with a data-integrity tag _(source: agent_knowledge)_
  - Serialize PIL images from raw image bytes instead of allocating an ndarray _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0001](../modules/vllm_multimodal/MultiModalHasher.serialize_item__cand-vllm_multimodal-0001.md)

### 40. `cand-vllm_v1_engine-0003` — EngineCore.step_with_batch_queue (score 35, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/core.py:621` (`EngineCore.step_with_batch_queue`, method)
- **Description:** Pipeline-parallel batch-queue step that decides when to enqueue new work, when to drain the oldest future, and when to defer sampling for structured outputs.
- **Current approach:** Prioritizes filling the batch queue before blocking for model outputs, appends non-deferred work immediately, pops the oldest queued future when needed, and handles deferred sampling at the end of the pop path. The local comment notes this fixed policy favors TTFT over TPOT/throughput.
- **Why this impact / rank:** step_with_batch_queue owns PP drain-vs-fill policy; adaptive drain moves TTFT (first-output) and TPOT (bubble) — but only when PP batch queueing is enabled. Shard already flagged it medium impact; ranked lowest because activation surface is narrow for this workload. — For pipeline-parallel serving, this policy determines bubble size and how quickly first outputs are drained, so it can move median TTFT and TPOT whenever the batch queue is enabled.
- **Proposals (5):**
  - Slack-guided adaptive drain-vs-fill policy in step_with_batch_queue _(source: research_finding)_
  - Adopt stall-free chunked-prefill + decode-maximal batching in step_with_batch_queue _(source: research_finding)_
  - Latency-budgeted adaptive drain policy for step_with_batch_queue _(source: research_finding)_
  - Opportunistic zero-cost drain of ready futures before enqueue in step_with_batch_queue _(source: agent_knowledge)_
  - Prioritize deferred sampling completions as latency-critical queue entries _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0003](../modules/vllm_v1_engine/EngineCore.step_with_batch_queue__cand-vllm_v1_engine-0003.md)

### 41. `cand-vllm_v1_kv_offload-0018` — ServerRole.add_stored_blocks/on_fetch (score 74, impact: high)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/p2p/session/server.py:305` (`ServerRole.add_stored_blocks/on_fetch`, region)
- **Description:** P2P server role matches locally stored blocks with peer fetch demand and submits transfers when supply and demand meet.
- **Current approach:** Maintains per-round available and demanded dicts keyed by OffloadKey. Stored blocks and FetchMsg demand are matched immediately; symmetric lookup-supplied rounds fail unmatched demand immediately, while PD rounds park demand until store supply arrives.
- **Why this impact / rank:** P2P ServerRole matching gates how quickly peer-held KV reaches consumers on secondary hits; priority-aware first-block fast-path can replace prefill with peer fetch, directly moving TTFT for disaggregated agentic traffic. — P2P secondary hits can replace local prefill for multi-turn or disaggregated agentic traffic. Matching and batching decisions determine how quickly peer-held KV reaches the consumer, so they can directly move median TTFT and reduce TPOT stalls when decode waits for fetched blocks.
- **Proposals (2):**
  - Priority-aware first-block fast-path in add_stored_blocks/on_fetch to cut decode-blocking TPOT _(source: agent_knowledge)_
  - Coalesce per-peer matched rounds into a single transfer submission burst _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0018](../modules/vllm_v1_kv_offload/ServerRole.add_stored_blocks_on_fetch__cand-vllm_v1_kv_offload-0018.md)

### 42. `cand-vllm_v1_worker-0029` — rejection_sample (score 68, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:922` (`rejection_sample`, function)
- **Description:** New GPU runner speculative-decoding rejection sampler pipeline that computes target/draft statistics, verifies draft tokens, resamples rejected/bonus tokens, and inserts outputs.
- **Current approach:** Allocates several per-call temporary tensors, uses fixed VOCAB_BLOCK_SIZE=8192 and RESAMPLE_BLOCK_SIZE=1024, and launches separate kernels for local logits stats, optional block-verification residuals, rejection, resampling, and insertion.
- **Why this impact / rank:** rejection_sample is the spec-decode inner sampling kernel; persistent scratch tensors, autotuned tiling, and a greedy fast path directly reduce per-step launch/traffic overhead when spec decode is on, which is a common agent TPOT lever. — Spec decode is a core TPOT lever for agent serving; reducing temporary traffic, launch count, or poorly matched vocab-block tiling improves median TPOT for speculative workloads.
- **Proposals (2):**
  - Persist and autotune rejection-sampler scratch tensors + block sizes across calls _(source: agent_knowledge)_
  - Add a greedy-only rejection sampler fast path _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0029](../modules/vllm_v1_worker/rejection_sample__cand-vllm_v1_worker-0029.md)

### 43. `cand-vllm_v1_attention-0011` — split_decodes_and_prefills (score 70, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/utils.py:635` (`split_decodes_and_prefills`, function)
- **Description:** Finds the decode/prefill boundary in an already reordered batch and returns request and token counts for each region.
- **Current approach:** Uses CPU tensor diffs, scalar .item() reads, torch.any, and argmax().item() to find the first prefill. Data is already CPU-side, so the main cost is repeated tensor/Python scalar work rather than GPU synchronization.
- **Why this impact / rank:** split_decodes_and_prefills is called every step by multiple metadata builders and does redundant CPU tensor materialization; numpy searchsorted on cumulative query_start_loc removes fixed overhead each step, compounding into median TPOT on stable-shape decode batches. — Called by multiple metadata builders every step. Reducing repeated CPU tensor overhead improves median TPOT when decode shapes repeat across many agentic turns.
- **Proposals (2):**
  - Use numpy searchsorted on cumulative query_start_loc to find decode/prefill boundary without materializing query_lens _(source: agent_knowledge)_
  - Cache split counts upstream when the batch is reordered _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0011](../modules/vllm_v1_attention/split_decodes_and_prefills__cand-vllm_v1_attention-0011.md)

### 44. `cand-vllm_distributed_kv_transfer-0003` — NixlBaseConnectorWorker._handshake_initiation_executor / _nixl_handshake (score 74, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:487` (`NixlBaseConnectorWorker._handshake_initiation_executor / _nixl_handshake`, region)
- **Description:** NIXL handshake initiation uses one background worker, and each handshake job sequentially queries every remote PP/rank pair through a single ZMQ REQ socket.
- **Current approach:** ThreadPoolExecutor(max_workers=1) is hard-coded because NIXL thread safety is uncertain. _nixl_handshake iterates itertools.product(range(remote_pp_size), p_remote_ranks), sends one metadata request, blocks for recv_multipart with a 5s timeout, decodes compatibility and agent metadata, and registers the remote agent before the next query.
- **Why this impact / rank:** NIXL handshake gates the first remote KV read and pays remote_pp_size * target_tp_rank sequential RTTs; caching validated metadata and pipelining removes first-hit TTFT spikes recurring in disaggregated multi-turn serving. — The handshake gates the first remote KV read for an engine, contributing directly to first-hit TTFT and recurring when remote pools churn in disaggregated multi-turn serving.
- **Proposals (3):**
  - Pipeline NIXL handshake requests via DEALER/ROUTER for concurrent metadata exchange _(source: research_finding)_
  - Cache validated NIXL handshake metadata to skip repeat remote-engine handshakes _(source: agent_knowledge)_
  - Add a batched NIXL metadata handshake request _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0003](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorWorker._handshake_initiation_executor____nixl_handshake__cand-vllm_distributed_kv_transfer-0003.md)

### 45. `cand-vllm_v1_attention-0006` — fast_plan_decode (score 75, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flashinfer.py:2296` (`fast_plan_decode`, function)
- **Description:** Provides a CUDA-graph-aware fast path for FlashInfer BatchDecodeWithPagedKVCacheWrapper planning.
- **Current approach:** Uses a first-call boolean to warm the wrapper with plan(), then delegates subsequent CUDA-graph calls to flashinfer.decode.fast_decode_plan with supplied indptr and last_page_len CPU buffers. It does not short-circuit identical metadata or coalesce repeated host metadata preparation at the vLLM layer.
- **Why this impact / rank:** fast_plan_decode is invoked every token on FlashInfer CUDA-graph decode; short-circuiting on unchanged batch descriptors and coalescing H2D copies removes steady-state per-step overhead that shows up in median TPOT. — Steady-state FlashInfer CUDA-graph decode calls this every token. Skipping unchanged plans or reducing host metadata traffic cuts per-step overhead and improves median TPOT.
- **Proposals (5):**
  - Cache and reuse fast_decode_plan auxiliary structures across identical decode steps _(source: research_finding)_
  - Coalesce indptr and last_page_len into a single pinned H2D copy per fast_plan_decode call _(source: research_finding)_
  - Short-circuit fast_plan_decode on unchanged batch descriptor for CUDA-graph decode replays _(source: research_finding)_
  - Overlap fast_plan_decode on a side CUDA stream with the previous decode step's kernels _(source: agent_knowledge)_
  - Add a page-rollover-only fast path for last_page_len-only updates _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0006](../modules/vllm_v1_attention/fast_plan_decode__cand-vllm_v1_attention-0006.md)

### 46. `cand-vllm_v1_kv_offload-0001` — ARCCachePolicy.evict (score 72, impact: high)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/cpu/policies/arc.py:112` (`ARCCachePolicy.evict`, method)
- **Description:** ARC batch eviction selects victims from T1/T2, moves evicted keys to B1/B2, and bounds ghost-list sizes.
- **Current approach:** Uses monotonic T1/T2 iterators and chooses from T1 while virtual_t1_size >= int(target_t1_size), otherwise T2. Entries with ref_cnt > 0 or protected keys are skipped, and ghost lists are trimmed only after a successful batch eviction.
- **Why this impact / rank:** ARC evict policy controls recency/frequency balance for the primary KV tier; better victim selection raises hit rate for repeated agentic prefixes and cuts promotion stalls that show up as median TTFT/TPOT. — Primary-tier hit rate directly affects whether multi-turn agentic prompts reuse CPU KV blocks or pay a secondary-tier promotion. The ARC eviction split is one of the main controls for recency/frequency balance under shifting working sets, so it can move median TTFT and TPOT.
- **Proposals (4):**
  - Add workflow-priority-aware skip in ARC evict victim selection _(source: research_finding)_
  - Add a TinyLFU frequency-sketch admission gate around ARC eviction/insertion _(source: research_finding)_
  - Re-queue skipped ref-counted/protected blocks to MRU during ARC evict scan _(source: agent_knowledge)_
  - Add cross-partition fallback before ARC evict returns None _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0001](../modules/vllm_v1_kv_offload/ARCCachePolicy.evict__cand-vllm_v1_kv_offload-0001.md)

### 47. `cand-vllm_multimodal-0004` — MultiModalBatchedField._reduce_data (score 66, impact: high)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/inputs.py:513` (`MultiModalBatchedField._reduce_data`, method)
- **Description:** Reduces a list of per-item NestedTensors into a batched tensor via an unsqueeze fast path or torch.stack.
- **Current approach:** For one tensor it unsqueezes, then may call contiguous or pin_memory. For multiple same-shaped tensors it checks shapes in Python, allocates a fresh output tensor with torch.empty, and calls torch.stack(batch, out=out).
- **Why this impact / rank:** MultiModalBatchedField._reduce_data allocates pinned buffers and stacks tensors on every multimodal prefill; pooling pinned outputs and eliding stack copies removes real host work from the TTFT path, gated by multimodal traffic fraction. — Pixel values and embedding tensors pass through this path on multimodal prefills. Large host allocations and stack copies can dominate CPU prefill preparation, so reducing them can materially lower median TTFT.
- **Proposals (3):**
  - Pool pinned output buffers for MultiModalBatchedField._reduce_data batched stacks _(source: research_finding)_
  - Detect shared-storage contiguous slices to elide the torch.stack copy in _reduce_data _(source: agent_knowledge)_
  - Short-circuit stacking for explicit singleton repeated batches _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0004](../modules/vllm_multimodal/MultiModalBatchedField._reduce_data__cand-vllm_multimodal-0004.md)

### 48. `cand-vllm_distributed_kv_transfer-0007` — OffloadingScheduler._lookup_complete_chunks (score 67, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:697` (`OffloadingScheduler._lookup_complete_chunks`, method)
- **Description:** Core offloading prefix-hit lookup iterates KV groups, performs full-attention prefix and sliding-window suffix backend lookups, and may re-run groups when constraints tighten max_hit_size_tokens.
- **Current approach:** A Python convergence while loop repeatedly calls _maximal_prefix_lookup or _sliding_window_lookup one group at a time. Sliding-window and EAGLE constraints can force another pass over groups and another series of backend probes; eagle_verified tracks per-iteration state.
- **Why this impact / rank:** OffloadingScheduler._lookup_complete_chunks runs on every cache-eligible admission and long shared prefixes exercise it repeatedly; a step-level OffloadKey cache and probe reordering cut backend calls that gate TTFT for offloaded prefix hits in multi-turn workloads. — Scheduler lookup latency is on the TTFT path for every offloaded prefix hit; long shared prefixes in multi-turn agentic prompts repeatedly exercise it.
- **Proposals (2):**
  - Per-scheduler-step OffloadKey lookup cache shared across requests _(source: agent_knowledge)_
  - Probe most-restrictive KV groups first in prefix-hit convergence _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0007](../modules/vllm_distributed_kv_transfer/OffloadingScheduler._lookup_complete_chunks__cand-vllm_distributed_kv_transfer-0007.md)

### 49. `cand-vllm_v1_worker-0015` — DP synchronization post-processing helpers (score 73, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/dp_utils.py:36` (`DP synchronization post-processing helpers`, region)
- **Description:** Coordinates legacy DP ranks for per-step microbatching, DP padding, and synced cudagraph mode.
- **Current approach:** Builds a 4 x dp_size tensor, all-reduces it, then reads several scalar decisions through torch .item(), .max().item(), and .cpu() operations across helper functions.
- **Why this impact / rank:** DP synchronization is a per-step latency floor when serving is scaled across ranks; batching scalar reads into a single host transfer and replacing the SUM all-reduce with MIN/MAX collectives reduces the CPU-visible tail every decode step. — DP synchronization is a per-step latency floor in scaled serving; reducing scalar synchronization and post-processing improves TTFT and TPOT when agent traffic is distributed across ranks.
- **Proposals (3):**
  - Batch DP sync scalar materialization into a single host transfer _(source: research_finding)_
  - Replace one-hot SUM all-reduce with semantic MIN/MAX/BAND + conditional all-gather in `_run_ar` _(source: agent_knowledge)_
  - Reuse pinned DP sync scratch buffers in `_run_ar` _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0015](../modules/vllm_v1_worker/DP_synchronization_post-processing_helpers__cand-vllm_v1_worker-0015.md)

### 50. `cand-vllm_v1_worker-0030` — DefaultModelState.prepare_attn (score 72, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/model_states/default.py:125` (`DefaultModelState.prepare_attn`, method)
- **Description:** New GPU runner default attention metadata construction for decoder-only models across cudagraph modes.
- **Current approach:** Builds CPU query_start_loc views every step, reads max_query_len and max_seq_len through tensor .max().item(), optionally computes multimodal prefix ranges in Python, and then calls build_attn_metadata with per-step host values.
- **Why this impact / rank:** prepare_attn runs every new-runner forward with .item() host syncs and per-step Python metadata walks; eliminating these directly reduces steady-state median TPOT. — This runs every new-runner forward step, so removing scalar syncs and repeated CPU metadata work can reduce steady median TPOT, with extra TTFT benefit for multimodal prefix-LM turns.
- **Proposals (3):**
  - Eliminate host syncs in prepare_attn by making CPU seq/query metadata optional _(source: research_finding)_
  - Cache multimodal prefix ranges per request to skip redundant Python walks on decode steps _(source: agent_knowledge)_
  - Skip multimodal prefix metadata on pure decode steps _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0030](../modules/vllm_v1_worker/DefaultModelState.prepare_attn__cand-vllm_v1_worker-0030.md)

### 51. `cand-vllm_v1_kv_offload-0013` — CachePolicyFactory built-in registrations (score 70, impact: high)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/cpu/policies/factory.py:84` (`CachePolicyFactory built-in registrations`, plugin_seam)
- **Description:** Registration site for pluggable CPU cache eviction policies selected by cache-policy configuration.
- **Current approach:** The interface is CachePolicy in vllm/v1/kv_offload/cpu/policies/base.py; built-in implementations are LRUCachePolicy in vllm/v1/kv_offload/cpu/policies/lru.py and ARCCachePolicy in vllm/v1/kv_offload/cpu/policies/arc.py. Runtime selection uses the eviction_policy config key and optional cache_policy_module_path, resolved by CachePolicyFactory.get_cache_policy_cls.
- **Why this impact / rank:** CachePolicyFactory adds workload-tuned sibling policies (S3-FIFO, TinyLFU, SIEVE) that can raise primary hit rate for shared agentic prefixes; high leverage but indirect and gated on config selection to realize gains. — A workload-tuned primary-cache policy can materially improve hit rate for repeated agentic prefixes and reduce secondary promotions. Because callers select policies through config, this is a high-leverage extension point for TTFT and TPOT experiments.
- **Proposals (5):**
  - Add a workflow-priority CachePolicy that consumes per-request retention hints _(source: research_finding)_
  - Add S3-FIFO cache policy as a sibling to LRU/ARC in the CPU cache policy factory _(source: research_finding)_
  - Add TinyLFU admission-gated cache policy as a built-in sibling to LRU and ARC _(source: research_finding)_
  - Add SIEVE cache policy as a built-in sibling to LRU and ARC _(source: agent_knowledge)_
  - Add a size-aware GDSF cache policy for variable KV block cost _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0013](../modules/vllm_v1_kv_offload/CachePolicyFactory_built-in_registrations__cand-vllm_v1_kv_offload-0013.md)

### 52. `cand-vllm_v1_worker-0005` — GPUModelRunner._build_attention_metadata (score 65, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_model_runner.py:2325` (`GPUModelRunner._build_attention_metadata`, method)
- **Description:** Builds CommonAttentionMetadata and per-layer attention metadata across KV-cache groups and attention groups in the legacy runner.
- **Current approach:** Computes max_seq_len from a CPU tensor every step, walks multimodal-prefix ranges in Python, shallow-copies CommonAttentionMetadata per KV group, calls _get_encoder_seq_lens per group, and builds or updates metadata per attention group.
- **Why this impact / rank:** _build_attention_metadata runs every step and repeats group-invariant work; making seq_lens_cpu optional and persisting FlashInfer-style planning shaves steady TPOT overhead. Impact bounded by the fact that this is CPU bookkeeping rather than a GPU stall. — Attention metadata is built every step, so memoizing group-invariant pieces and precomputing host values reduces steady TPOT overhead, though the main cost is CPU bookkeeping rather than a mandatory GPU stall.
- **Proposals (4):**
  - Make seq_lens_cpu optional in _build_attention_metadata to unblock async spec-decode overlap _(source: research_finding)_
  - Persist FlashInfer-style auxiliary attention planning across steps in _build_attention_metadata _(source: research_finding)_
  - Precompute per-step group-invariant slices in _build_attention_metadata via a compiled numpy path _(source: agent_knowledge)_
  - Make padded block-table tails lazily maintained instead of filling them every step _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0005](../modules/vllm_v1_worker/GPUModelRunner._build_attention_metadata__cand-vllm_v1_worker-0005.md)

### 53. `cand-vllm_v1_attention-0019` — FlashInferMetadataBuilder fixed split policy (score 64, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flashinfer.py:645` (`FlashInferMetadataBuilder fixed split policy`, config_block)
- **Description:** Sets FlashInfer fixed split sizes and split-KV disabling policy under batch-invariant mode.
- **Current approach:** When VLLM_BATCH_INVARIANT is enabled, decode_fixed_split_size is hardcoded to 2048, prefill_fixed_split_size to 4096, and disable_split_kv to True; otherwise both split sizes are -1 and split-KV is enabled.
- **Why this impact / rank:** FlashInferMetadataBuilder fixed split constants gate FlashInfer kernel partitioning for prefill and decode; occupancy-aware or page-aligned split policies improve latency across TPOT and TTFT, but the impact is contingent on FlashInfer being the active backend. — These split settings affect FlashInfer planning and kernel partitioning for both TTFT and TPOT. Better values by model, page size, and GPU architecture can improve median latency while preserving outputs.
- **Proposals (3):**
  - Replace fixed split constants with occupancy-aware split policy for batch-invariant FlashInfer decode/prefill _(source: research_finding)_
  - Keep FlashInfer split-KV enabled under batch-invariant mode by pinning a page-size-aligned fixed split factor _(source: agent_knowledge)_
  - Short-circuit fixed splitting when configured context bounds cannot exceed one chunk _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0019](../modules/vllm_v1_attention/FlashInferMetadataBuilder_fixed_split_policy__cand-vllm_v1_attention-0019.md)

### 54. `cand-vllm_v1_engine-0004` — AsyncLLM._run_output_handler.output_handler (score 68, impact: high)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/async_llm.py:684` (`AsyncLLM._run_output_handler.output_handler`, region)
- **Description:** Async front-end output loop that reads EngineCoreOutputs, chunks them, processes each chunk, yields between chunks, dispatches aborts, and records stats.
- **Current approach:** Uses fixed env-configured VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, slices the output list, calls output_processor.process_outputs per slice, awaits asyncio.sleep(0) between chunks, and performs abort dispatch and logging in the same loop.
- **Why this impact / rank:** AsyncLLM.output_handler processes every token delivered to clients; skipping empty-tail sleeps, batching aborts, and adapting yield chunking to queue pressure reduces frontend per-token overhead — most visible on short agentic bursts. — Small token bursts are common in multi-turn agentic workloads; reducing unnecessary chunking/yield overhead or adapting it to queue pressure can lower median TPOT and sometimes TTFT on the frontend path.
- **Proposals (2):**
  - Coalesce and batch abort dispatch; skip empty-tail sleep to reduce per-token loop overhead _(source: agent_knowledge)_
  - Yield by elapsed processing budget instead of fixed chunk boundaries _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0004](../modules/vllm_v1_engine/AsyncLLM._run_output_handler.output_handler__cand-vllm_v1_engine-0004.md)

### 55. `cand-vllm_v1_worker-0009` — GPUModelRunner._determine_batch_execution_and_padding (score 70, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_model_runner.py:4025` (`GPUModelRunner._determine_batch_execution_and_padding`, method)
- **Description:** Selects legacy cudagraph mode, batch descriptor, DP microbatching, and DP padding each step.
- **Current approach:** Dispatches cudagraph selection, optionally coordinates across DP ranks, then re-dispatches after synced token counts; DP mode reads num_tokens_across_dp[dp_rank].item() when padding is active.
- **Why this impact / rank:** _determine_batch_execution_and_padding is per-step in DP-enabled deployments with a D2H scalar sync and redundant dispatch resolution; memoization/pinned-host removes recurring TPOT overhead when DP agent traffic keeps ranks active. — DP-enabled deployments pay this per step; removing redundant dispatch and D2H scalar reads improves median TPOT, especially when data-parallel agent traffic keeps all ranks active.
- **Proposals (2):**
  - Eliminate per-step D2H sync and redundant dispatch by using pinned host buffer for DP token counts and memoizing dispatch keys _(source: agent_knowledge)_
  - Drop CG padding when DP synchronization downgrades the step to eager _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0009](../modules/vllm_v1_worker/GPUModelRunner._determine_batch_execution_and_padding__cand-vllm_v1_worker-0009.md)

### 56. `cand-vllm_v1_kv_offload-0014` — SecondaryTierFactory built-in registrations (score 62, impact: high)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/factory.py:103` (`SecondaryTierFactory built-in registrations`, plugin_seam)
- **Description:** Registration site for pluggable secondary offload tiers selected from secondary_tiers configuration entries.
- **Current approach:** The interface is SecondaryTierManager in vllm/v1/kv_offload/tiering/base.py; built-in implementations include ExampleSecondaryTierManager in vllm/v1/kv_offload/tiering/example/manager.py, FileSystemTierManager in vllm/v1/kv_offload/tiering/fs/manager.py, P2PSecondaryTierManager in vllm/v1/kv_offload/tiering/p2p/manager.py, and ObjectStoreSecondaryTierManager in vllm/v1/kv_offload/tiering/obj/mana…
- **Why this impact / rank:** SecondaryTierFactory is a broad extension surface with the highest rf_count (6), but the proposals describe registering new tier implementations rather than optimizing the hot path; benefit to median TTFT depends on which tier ships and whether workloads spill past CPU primary. — Secondary-tier latency determines how expensive non-primary hits are. A faster or locality-aware tier implementation can reduce promotion wait time and improve median TTFT for multi-turn workloads whose reused KV blocks spill beyond the CPU primary tier.
- **Proposals (7):**
  - Register a Mooncake-style shared distributed KV pool as a secondary tier _(source: research_finding)_
  - Register a TinyLFU-admission secondary tier variant behind the factory seam _(source: research_finding)_
  - Register an SSD-object secondary tier optimized for bulk transfers and slack-aware scheduling _(source: research_finding)_
  - Register a GPUDirect Storage (GDS) secondary tier for direct, batched storage→GPU KV loads _(source: research_finding)_
  - Add a bandwidth-adaptive KV-compression secondary tier (CacheGen-style) _(source: research_finding)_
  - Register a usefulness-admission secondary tier variant that gates promotions by predicted reuse _(source: research_finding)_
  - Register a hierarchical composite tier that federates existing secondary tiers with fast/slow cascade _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0014](../modules/vllm_v1_kv_offload/SecondaryTierFactory_built-in_registrations__cand-vllm_v1_kv_offload-0014.md)

### 57. `cand-vllm_v1_kv_offload-0006` — SingleDirectionOffloadingHandler.transfer_async (score 65, impact: high)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/cpu/gpu_worker.py:240` (`SingleDirectionOffloadingHandler.transfer_async`, method)
- **Description:** Builds descriptor buffers for a batched transfer and submits the copy on a CUDA stream with event-based ordering.
- **Current approach:** Each call pops or allocates pinned descriptor buffers, converts slices to NumPy views, loops over groups and data refs, calls compute_sub_block_ptrs twice per data ref, records timing-enabled start/end events, and serializes transfers by waiting on the previous transfer's end event.
- **Why this impact / rank:** SingleDirectionOffloadingHandler.transfer_async runs per promoted block group on cache hits; precomputing descriptor templates and splitting large onloads directly reduces TTFT when offloaded prefixes are reused, which is exactly the multi-turn agentic pattern. — For cache hits, every promoted block group pays this CPU-side descriptor build and stream synchronization before GPU execution can continue. Reducing this overhead directly lowers median TTFT for small promotions and can improve TPOT when promotions occur during decode.
- **Proposals (2):**
  - Split large CPU→GPU onloads in transfer_async into progressive sub-batches _(source: research_finding)_
  - Precompute and cache per-(src_kv_caches, dst_kv_caches, block_size) descriptor templates to eliminate per-call compute_sub_block_ptrs work _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0006](../modules/vllm_v1_kv_offload/SingleDirectionOffloadingHandler.transfer_async__cand-vllm_v1_kv_offload-0006.md)

### 58. `cand-vllm_v1_worker-0007` — GPUModelRunner.execute_model (score 68, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_model_runner.py:4259` (`GPUModelRunner.execute_model`, method)
- **Description:** Legacy top-level per-step orchestration for state updates, input prep, cascade attention, cudagraph dispatch, Mamba preprocessing, slot mappings, attention metadata, forward, and logits computation.
- **Current approach:** Builds num_scheduled_tokens_np from scheduler dict lookups each step, recomputes has_separate_kv_update through nested generators, and copies scheduler dictionaries for ngram GPU mode.
- **Why this impact / rank:** execute_model runs every step and hosts several small per-step Python costs; caching static backend properties and skipping seq-lens materialization gives modest but repeated TPOT wins in agentic decode. — These costs are smaller than input prep or sampling sync, but they run every step and scale with active requests, so they can move median TPOT in agentic decode.
- **Proposals (5):**
  - Skip CPU seq-lens materialization in execute_model spec-decode path _(source: research_finding)_
  - Adopt MRV2-style stable-row batch state to cut per-step Python overhead in execute_model _(source: research_finding)_
  - Cache static attention-backend properties to eliminate per-step recomputation in execute_model _(source: research_finding)_
  - Early-exit cascade prefix computation when no KV group has a shared prefix _(source: agent_knowledge)_
  - Defer microbatching veto computation until DP coordination actually needs it _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0007](../modules/vllm_v1_worker/GPUModelRunner.execute_model__cand-vllm_v1_worker-0007.md)

### 59. `cand-vllm_v1_attention-0003` — FlashAttentionImpl._forward_with_dcp (score 66, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flash_attn.py:1177` (`FlashAttentionImpl._forward_with_dcp`, method)
- **Description:** Runs FlashAttention with Decode Context Parallelism by gathering query tensors, computing context attention, combining DCP states, computing new-token attention, and merging attention states.
- **Current approach:** Executes a fixed sequence of contiguous(), get_dcp_group().all_gather, workspace reservation, context flash_attn_varlen_func, dcp_combine, new-token flash_attn_varlen_func, and merge_attn_states. The FA2/FA4 context split is static and the local new-token kernel is not overlapped with the collective/context path.
- **Why this impact / rank:** DCP forward runs in every attention layer when enabled, so occupancy-aware split and all-gather/attention overlap can move both TTFT and TPOT — but only when DCP is active, which is not implied by the agentic workload hint. — For DCP-enabled serving this path runs in every attention layer and can move TTFT and TPOT through better overlap or split selection. Impact is medium because the workload hint is agentic but not necessarily DCP-enabled.
- **Proposals (6):**
  - Adaptive Flash-Decoding split policy for the DCP context attention call _(source: research_finding)_
  - Split new-token attention into overlap-friendly attention-state chunks and merge n-ary with DCP context state _(source: research_finding)_
  - Occupancy-aware split policy for DCP context/new-token attention _(source: research_finding)_
  - Replace DCP all-gather with Ulysses-style all-to-all head redistribution in _forward_with_dcp _(source: research_finding)_
  - Token-chunked pipelining of the DCP head-dim all_gather with the context flash_attn call _(source: agent_knowledge)_
  - Compact zero-context requests out of the DCP context attention call _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0003](../modules/vllm_v1_attention/FlashAttentionImpl._forward_with_dcp__cand-vllm_v1_attention-0003.md)

### 60. `cand-vllm_v1_attention-0005` — FlashInferMetadataBuilder._compute_flashinfer_kv_metadata (score 68, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flashinfer.py:1062` (`FlashInferMetadataBuilder._compute_flashinfer_kv_metadata`, method)
- **Description:** Computes FlashInfer paged-KV indptr, page indices, and last-page lengths for native FlashInfer paths.
- **Current approach:** Uses numpy cumsum on CPU, copies indptr through a second CPU buffer before H2D, launches _copy_page_indices_kernel with BLOCK_SIZE=1024, and computes last_page_len with numpy modulo/where followed by another H2D copy.
- **Why this impact / rank:** FlashInfer metadata computation runs per step for decode/prefill/cascade with two H2D copies plus CPU cumsum; coalescing to a single pinned staging buffer reduces per-step overhead, directly moving decode-heavy median TPOT. — Runs for native FlashInfer decode, prefill, and cascade metadata. Coalescing transfers or moving simple metadata computation to device reduces per-step CPU/H2D overhead, improving median TPOT for decode-heavy agentic workloads.
- **Proposals (3):**
  - Coalesce FlashInfer metadata H2D copies into a single pinned staging buffer _(source: research_finding)_
  - Fuse paged_kv_last_page_len computation into the per-request Triton page-indices kernel to remove one H2D copy and the CPU modulo/where work _(source: agent_knowledge)_
  - Bypass the page-indices copy kernel for single-request native FlashInfer metadata _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0005](../modules/vllm_v1_attention/FlashInferMetadataBuilder._compute_flashinfer_kv_metadata__cand-vllm_v1_attention-0005.md)

### 61. `cand-vllm_v1_kv_offload-0003` — LRUCachePolicy.evict (score 66, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/cpu/policies/lru.py:56` (`LRUCachePolicy.evict`, method)
- **Description:** LRU batch eviction scans evictable blocks from the oldest end, skipping protected keys until n victims are collected.
- **Current approach:** Uses pure oldest-first victim selection over evictable_blocks, with only the protected-key filter. It returns None atomically if fewer than n non-protected candidates are available.
- **Why this impact / rank:** Pure-recency LRU can prematurely evict shared agentic prefixes; prefix-aware or frequency-augmented eviction raises primary hit rate, reducing promotion stalls that hit TTFT/TPOT. — Agentic multi-turn workloads often reuse prompt/prefix blocks across turns, and pure recency can evict those too aggressively. Better victim scoring can improve primary-tier hit rate and reduce promotion stalls that affect median TTFT.
- **Proposals (5):**
  - Priority-aware LRU eviction using agent-supplied retention hints _(source: research_finding)_
  - Add S3-FIFO cache policy as an alternative to LRUCachePolicy.evict for multi-turn prefix protection _(source: research_finding)_
  - Add TinyLFU frequency-sketch admission gate to CPU-tier LRU policy _(source: research_finding)_
  - Add cross-request prefix-aware eviction using PrefixCachingMetrics signal for shared-prefix protection _(source: agent_knowledge)_
  - Cap per-request decode-tail victims before evicting prompt-sized blocks _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0003](../modules/vllm_v1_kv_offload/LRUCachePolicy.evict__cand-vllm_v1_kv_offload-0003.md)

### 62. `cand-vllm_multimodal-0006` — _can_batch_mm_items/_batch_mm_items/group_and_batch_mm_items/group_and_batch_mm_kwargs (score 62, impact: high)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/utils.py:170` (`_can_batch_mm_items/_batch_mm_items/group_and_batch_mm_items/group_and_batch_mm_kwargs`, region)
- **Description:** Groups consecutive multimodal kwargs by modality and field compatibility, then batches each compatible run into model kwargs.
- **Current approach:** group_and_batch_mm_kwargs materializes each modality group into a list. group_and_batch_mm_items scans adjacent pairs with _can_batch_mm_items, doing key checks, field type checks, dict lookups, and shared-field equality. When a group closes, _batch_mm_items walks the same items again to populate a defaultdict before reducing each field.
- **Why this impact / rank:** group_and_batch_mm_items runs on every multimodal prefill and its Python overhead scales with items per batch. Single-pass streaming with cached compatibility signatures shaves TTFT for multi-item multimodal turns; ceiling limited to that traffic slice. — Every multimodal prefill goes through this grouping path. Python overhead scales with batch size and number of multimodal items, so reducing duplicated scans moves TTFT for multi-turn agentic batches with many small items.
- **Proposals (2):**
  - Single-pass streaming group-and-batch with cached compatibility signatures _(source: agent_knowledge)_
  - Add a singleton-batch fast path before accumulating field lists _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0006](../modules/vllm_multimodal/_can_batch_mm_items__batch_mm_items_group_and_batch_mm_items_group_and_batch_mm_kwargs__cand-vllm_multimodal-0006.md)

### 63. `cand-vllm_v1_kv_offload-0004` — _select_swap_blocks_fn (score 60, impact: high)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/cpu/gpu_worker.py:35` (`_select_swap_blocks_fn`, function)
- **Description:** Selects the CPU/GPU block-transfer implementation: C++ DMA path or Triton batched-copy kernel.
- **Current approach:** Hard-coded dispatch: GPU-to-CPU always uses DMA; CPU-to-GPU uses Triton only when Triton is available, the platform is not XPU/ROCm, all page sizes are 8-byte-aligned, and max page size is below THRESHOLD_BYTES. Triton chunk size is min(next_power_of_2(max_page_size), 8192).
- **Why this impact / rank:** CPU-to-GPU swap dispatch selects between DMA and Triton; batch-size-aware dispatch and hybrid paths for mixed page-size groups raise transfer bandwidth on the promotion critical path, cutting TTFT stall on CPU-tier hits for agentic reuse. — CPU-to-GPU promotion runs on cache hits from the CPU tier. Better crossover selection can raise transfer bandwidth and reduce the promotion stall on the TTFT critical path for cache-hit-heavy agentic turns.
- **Proposals (2):**
  - Make CPU->GPU dispatch and Triton grid size batch-size aware at call time _(source: agent_knowledge)_
  - Use a hybrid CPU->GPU path for mixed page-size KV groups _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0004](../modules/vllm_v1_kv_offload/_select_swap_blocks_fn__cand-vllm_v1_kv_offload-0004.md)

### 64. `cand-vllm_v1_worker-0017` — EncoderCudaGraphManager budget generation and lookup (score 62, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/encoder_cudagraph.py:190` (`EncoderCudaGraphManager budget generation and lookup`, config_block)
- **Description:** Generates encoder CUDA graph token budgets and finds the smallest captured budget for a multimodal batch.
- **Current approach:** Uses a power-of-two budget ladder plus max_budget and performs a linear smallest-fitting lookup over captured budgets.
- **Why this impact / rank:** Encoder CUDA-graph budgets gate multimodal TTFT on screenshot/image turns common in agentic workloads; workload-adaptive budget ladders reduce eager fallback and padding waste — impact contingent on multimodal traffic mix. — Multimodal agent TTFT depends on encoder graph hits; denser or workload-adaptive budgets can reduce eager fallback and padding waste for screenshot/image turns.
- **Proposals (2):**
  - Add workload-adaptive encoder budget ladder driven by observed token histograms _(source: agent_knowledge)_
  - Add a padding-efficiency cutoff before replaying encoder graphs _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0017](../modules/vllm_v1_worker/EncoderCudaGraphManager_budget_generation_and_lookup__cand-vllm_v1_worker-0017.md)

### 65. `cand-vllm_v1_attention-0022` — resolve_seq_and_query_len / find_seq_idx (score 65, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/ops/triton_attention_helpers.py:44` (`resolve_seq_and_query_len / find_seq_idx`, function)
- **Description:** Maps each unified-attention program id to its sequence index and local query-block index inside Triton kernels.
- **Current approach:** Every program performs a binary search over query_start_len_ptr via find_seq_idx, then reloads prefix entries and sequence length to derive q_block_local_idx and per-sequence lengths.
- **Why this impact / rank:** find_seq_idx binary search runs per Triton attention program plus reduce_segments; precomputing q-block-to-sequence metadata removes repeated per-program work on decode-heavy agentic batches, chipping at TPOT. — This executes for every Triton attention program and again in reduce_segments. Removing repeated prefix searches can reduce kernel overhead and improve median TPOT for decode-heavy agentic batches.
- **Proposals (3):**
  - Batch-descriptor dispatch to a decode-specialized attention kernel that bypasses find_seq_idx _(source: research_finding)_
  - Precompute per-q-block metadata table in AttentionMetadata; kernel does a single indexed load _(source: agent_knowledge)_
  - Add a seq-major Triton launch path for low-waste mixed batches _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0022](../modules/vllm_v1_attention/resolve_seq_and_query_len___find_seq_idx__cand-vllm_v1_attention-0022.md)

### 66. `cand-vllm_v1_engine-0010` — DPEngineCoreProc._should_throttle_prefills (score 60, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/core.py:2086` (`DPEngineCoreProc._should_throttle_prefills`, method)
- **Description:** DP prefill admission throttle that decides whether new prefills may be scheduled on the current step.
- **Current approach:** Returns true whenever prefill_schedule_interval > 1 and step_counter is not divisible by that fixed interval; the cadence is deterministic and identical across DP ranks.
- **Why this impact / rank:** _should_throttle_prefills gates DP prefill admission; load-aware or progress-informed policies can move TTFT under bursty agentic arrivals. Effect confined to DP deployments and bounded by the existing cadence being reasonably close. — Prefill admission timing drives TTFT for new turns and interacts with decode throughput; better cadence can lower median TTFT under bursty agentic arrivals without overloading DP ranks.
- **Proposals (6):**
  - Urgency-based bypass of DP prefill throttle for waiting requests _(source: research_finding)_
  - Replace fixed-cadence prefill throttle with load-aware gate on prefill/decode token pressure _(source: research_finding)_
  - Replace fixed prefill cadence with progress-informed adaptive throttle _(source: research_finding)_
  - Adaptive DP prefill throttle via short/long EWMA divergence (Gradient2-style) _(source: research_finding)_
  - Stagger DP prefill throttle phase offset per rank to interleave prefill admission across ranks _(source: agent_knowledge)_
  - Skip DP prefill throttle when no decodes are running _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0010](../modules/vllm_v1_engine/DPEngineCoreProc._should_throttle_prefills__cand-vllm_v1_engine-0010.md)

### 67. `cand-vllm_distributed_kv_transfer-0021` — LookupKeyClient.lookup (score 64, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:1989` (`LookupKeyClient.lookup`, method)
- **Description:** MooncakeStore async prefix lookup client serializes lookup RPCs through one worker thread and one blocking ZMQ REQ socket.
- **Current approach:** LookupKeyClient creates ThreadPoolExecutor(max_workers=1). lookup() stores one Future per request; the worker calls _lookup(), which sends multipart frames and blocks on socket.recv(). With non_block=True, scheduler calls return None until the queued future completes.
- **Why this impact / rank:** LookupKeyClient.lookup with REQ+single-thread serializes prefix lookups behind one lane; DEALER multiplexing or step-batched RPC directly parallelizes TTFT-critical lookups under concurrent turns. — This is connector-specific, but for MooncakeStore deployments it gates get_num_new_matched_tokens; reducing lookup queueing lowers median TTFT under many concurrent agentic turns.
- **Proposals (3):**
  - Replace REQ+single-thread executor with a DEALER/ROUTER multiplexed lookup client _(source: research_finding)_
  - Replace REQ+single-thread serialization with DEALER-style async submit and completion polling in LookupKeyClient _(source: research_finding)_
  - Coalesce prefix lookups at each scheduler step into one batched RPC to LookupKeyServer _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0021](../modules/vllm_distributed_kv_transfer/LookupKeyClient.lookup__cand-vllm_distributed_kv_transfer-0021.md)

### 68. `cand-vllm_v1_core-0008` — FreeKVCacheBlockQueue (score 57, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/kv_cache_utils.py:184` (`FreeKVCacheBlockQueue`, region)
- **Description:** Intrusive doubly linked list used as the free KV cache block queue and eviction-order structure.
- **Current approach:** Maintains prev_free_block and next_free_block pointers directly on KVCacheBlock objects with fake head/tail sentinels. popleft, popleft_n, remove, append, append_n, prepend_n, and iteration are implemented with Python attribute writes and manual num_free_blocks accounting.
- **Why this impact / rank:** FreeKVCacheBlockQueue is touched on every allocate/free/touch and already exists to reduce Python overhead; further reductions (intrusive sidecar arrays, batched touch removal) only bound scheduler overhead, so effect on median TPOT is modest but present at high concurrency. — Touched on every block allocation, free, and cache-hit touch. The per-operation cost is bounded, but reducing it lowers scheduler overhead and TPOT at high concurrency.
- **Proposals (2):**
  - Replace pointer-per-KVCacheBlock intrusive list with sidecar integer arrays keyed by block_id _(source: agent_knowledge)_
  - Add batched removal for prefix-cache touch runs _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0008](../modules/vllm_v1_core/FreeKVCacheBlockQueue__cand-vllm_v1_core-0008.md)

### 69. `cand-vllm_v1_attention-0009` — TritonAttentionMetadataBuilder 2D/3D tuning constants (score 62, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/triton_attn.py:54` (`TritonAttentionMetadataBuilder 2D/3D tuning constants`, config_block)
- **Description:** Defines the TritonAttention 2D/3D switch threshold and allocates segmented softmax buffers for the 3D decode path.
- **Current approach:** MIN_LAUNCH_GRID_SIZE_2D=128 and NUM_PAR_SOFTMAX_SEGMENTS=16. seq_threshold_3D is 128 // num_heads_kv, optionally snapped to the nearest CUDA graph capture size, and buffers are allocated using that threshold and segment count.
- **Why this impact / rank:** Triton 2D/3D switch threshold + segment count govern SM utilization for small-batch decode common in agent serving; occupancy-aware tuning is a concrete TPOT lever but scoped to Triton backend and small batches. — Small-batch decode, common in agentic serving, is where the 2D/3D switch matters. Tuning the threshold and segment count can improve SM utilization and median TPOT.
- **Proposals (4):**
  - Adaptive Flash-Decoding split count for the Triton 3D decode path _(source: research_finding)_
  - Replace fixed 2D/3D switch threshold with occupancy-aware heuristic incorporating SM count _(source: research_finding)_
  - Per-CUDA-graph-capture-size 2D/3D dispatch table replacing single scalar seq_threshold_3D _(source: agent_knowledge)_
  - Add measured autotuning for Triton 2D/3D decode cutover _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0009](../modules/vllm_v1_attention/TritonAttentionMetadataBuilder_2D_3D_tuning_constants__cand-vllm_v1_attention-0009.md)

### 70. `cand-vllm_v1_engine-0015` — EngineCore.step (score 60, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/core.py:580` (`EngineCore.step`, method)
- **Description:** Default engine-core scheduler/model step that schedules work, overlaps grammar-mask construction with model execution, samples tokens, processes aborts, and updates the scheduler from model output.
- **Current approach:** Calls scheduler.schedule, launches execute_model(non_block=True), immediately computes get_grammar_bitmask, then blocks on future.result(); if the model runner did not sample internally, it calls sample_tokens, processes aborts, and calls scheduler.update_from_output.
- **Why this impact / rank:** EngineCore.step is the non-pipeline scheduling boundary paid once per decode iteration; overlapping abort/sampling work with GPU execute reduces per-step host overhead, but model execution dominates so gains are moderate. — Model execution dominates many steps, but this method is paid once per decode iteration; reducing host-side blocking or wasted per-step work can lower median TPOT and TTFT for short multi-turn agentic requests.
- **Proposals (3):**
  - Overlap abort processing with GPU execute in EngineCore.step using profiling-informed CPU placement _(source: research_finding)_
  - Speculatively pre-schedule the next step's CPU work during the current GPU execute window _(source: agent_knowledge)_
  - Run deferred sampling asynchronously and overlap its wait with post-forward CPU work _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0015](../modules/vllm_v1_engine/EngineCore.step__cand-vllm_v1_engine-0015.md)

### 71. `cand-vllm_distributed_kv_transfer-0012` — MooncakeConnectorWorker._build_transfer_params (score 62, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1424` (`MooncakeConnectorWorker._build_transfer_params`, method)
- **Description:** Builds Mooncake RDMA transfer descriptors for producer-to-consumer KV movement, grouping contiguous block runs and emitting either one descriptor per run or one descriptor per block.
- **Current approach:** Coalescing is all-or-nothing through _can_coalesce_block_transfers and only applies when source/destination offsets are zero and transfer lengths match full region block lengths. Otherwise descriptor creation falls back to per-block entries inside nested request/group/region loops.
- **Why this impact / rank:** Mooncake transfer descriptor coalescing reduces RDMA initiation on the turn-2 KV fetch path; TTFT gain is real for P/D reuse but confined to Mooncake deployments. — For Mooncake P/D reuse, descriptor setup and RDMA efficiency are on the turn-2 KV fetch path, so reducing descriptors can lower TTFT and improve producer throughput under parallel sessions.
- **Proposals (3):**
  - Relax coalescing to fuse partial-region contiguous runs into scatter-list batch descriptors _(source: research_finding)_
  - Fuse contiguously-registered regions (K/V halves, adjacent layers) into per-block-run cross-region descriptors _(source: agent_knowledge)_
  - Add a final adjacency merge over emitted transfer descriptors _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0012](../modules/vllm_distributed_kv_transfer/MooncakeConnectorWorker._build_transfer_params__cand-vllm_distributed_kv_transfer-0012.md)

### 72. `cand-vllm_distributed_kv_transfer-0015` — OffloadingScheduler.update_state_after_alloc load-job construction (score 58, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:993` (`OffloadingScheduler.update_state_after_alloc load-job construction`, method)
- **Description:** After external-token allocation, builds offloading load jobs by scanning groups and allocated blocks, collecting keys_to_load and destination block IDs, and registering TransferJob/TransferJobStatus entries.
- **Current approach:** Python loops over KV groups and group blocks repeatedly slice block lists, extend key and block arrays, update allocated-block sets, and call manager.prepare_load once for the request. Load job construction is per request rather than batched across the scheduler step.
- **Why this impact / rank:** OffloadingScheduler.update_state_after_alloc load-job construction sits between allocator and worker for every offloaded prefix hit before resume. Vectorizing group_blocks walks trims TTFT on the multi-turn cache-hit path, but gains are CPU-bookkeeping-sized. — Every offloaded hit that will be loaded asynchronously passes through this before it can resume; reducing this path moves TTFT for multi-turn requests with cache hits.
- **Proposals (2):**
  - Precompute per-group null-safe flag and fuse group_blocks walks into one pass _(source: agent_knowledge)_
  - Pass allocation deltas into load-job construction _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0015](../modules/vllm_distributed_kv_transfer/OffloadingScheduler.update_state_after_alloc_load-job_construction__cand-vllm_distributed_kv_transfer-0015.md)

### 73. `cand-vllm_v1_engine-0009` — EngineCoreProc._process_engine_step (score 55, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/core.py:1429` (`EngineCoreProc._process_engine_step`, method)
- **Description:** Engine-core busy-loop step wrapper that executes one scheduler/model step, emits outputs, calls post_step, and yields when work remains but no model execution occurred.
- **Current approach:** If model_executed is false while scheduler.has_requests() remains true, sleeps for a fixed 1 ms with time.sleep(0.001) to give KV-transfer/background threads GIL time.
- **Why this impact / rank:** EngineCoreProc._process_engine_step's 1 ms fixed sleep bounds how quickly WAITING_FOR_REMOTE_KVS requests are reconsidered; event-driven wakeups shave that floor from TTFT for remote-KV/prefix-cache paths, useful for disaggregated multi-turn workloads. — The 1 ms pause bounds how quickly KV-waiting requests are reconsidered; reducing unnecessary delay can improve median TTFT in multi-turn prefix-cache or remote-KV workloads.
- **Proposals (2):**
  - Replace fixed 1 ms sleep with event-driven wakeup from KV-transfer completion _(source: agent_knowledge)_
  - Gate the idle yield on explicit scheduler stall reasons _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0009](../modules/vllm_v1_engine/EngineCoreProc._process_engine_step__cand-vllm_v1_engine-0009.md)

### 74. `cand-vllm_distributed_kv_transfer-0013` — MooncakeConnectorWorker.__init__ sender pool sizing (score 60, impact: high)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:921` (`MooncakeConnectorWorker.__init__ sender pool sizing`, config_block)
- **Description:** Configures Mooncake producer-side sender concurrency with num_sender_workers defaulting to 10 and num_sender_tasks fixed at twice that count, then constructs the sender ThreadPoolExecutor.
- **Current approach:** ThreadPoolExecutor(max_workers=num_sender_workers) and a 2x async task surplus are static heuristics from kv_connector_extra_config, independent of TP/PP world size, request concurrency, RDMA queue depth, or observed queueing latency.
- **Why this impact / rank:** Mooncake sender pool sizing controls producer send concurrency for remote KV; adaptive sizing under many concurrent turn-2 fetches can reduce TTFT tails but is scoped to Mooncake-backed disaggregated deployments. — Producer-side send concurrency controls how quickly remote KV becomes available to decode nodes; many agentic sessions can queue turn-2 fetches concurrently.
- **Proposals (2):**
  - Adapt sender pool + async task budget from measured send latency and queue backlog _(source: agent_knowledge)_
  - Split Mooncake sender executors by transfer size class _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0013](../modules/vllm_distributed_kv_transfer/MooncakeConnectorWorker.__init___sender_pool_sizing__cand-vllm_distributed_kv_transfer-0013.md)

### 75. `cand-vllm_distributed_kv_transfer-0019` — NixlBaseConnectorWorker.sync_recved_kv_to_device / save_kv_to_host (score 56, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:1869` (`NixlBaseConnectorWorker.sync_recved_kv_to_device / save_kv_to_host`, region)
- **Description:** Host-buffer NIXL mode copies received KV from host buffers to device and saves device KV to host by issuing one blocking copy_blocks call per KV group.
- **Current approach:** sync_recved_kv_to_device loops over local_block_ids groups and calls copy_blocks for H2D after receive completion. save_kv_to_host loops requests and groups, converts logical IDs, then calls copy_blocks for D2H with an in-code blocking note. There is no coalescing across groups or stream/event handoff.
- **Why this impact / rank:** sync_recved_kv_to_device sits directly before request resume for host-buffer NIXL loads, so batching copy_blocks and moving to a dedicated stream can move TTFT — but only when the host-buffer mode is configured. — The path is conditional on host-buffer mode, but when enabled the copy sits directly before request resume for loads and before outbound transfer for saves, moving TTFT and sometimes TPOT.
- **Proposals (4):**
  - Coalesce host-buffer copy_blocks calls into a single batched scatter/gather transfer per direction _(source: research_finding)_
  - Fuse per-group KV copy_blocks into a single coalesced H2D/D2H launch _(source: research_finding)_
  - Move host-buffer copy_blocks onto a dedicated CUDA copy stream with event-based handoff _(source: agent_knowledge)_
  - Skip D2H copies for host blocks already known fresh _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0019](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorWorker.sync_recved_kv_to_device___save_kv_to_host__cand-vllm_distributed_kv_transfer-0019.md)

### 76. `cand-vllm_v1_attention-0025` — Triton MLA decode split policy (score 55, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/mla/triton_mla.py:35` (`Triton MLA decode split policy`, config_block)
- **Description:** Chooses the number of KV splits for Triton MLA decode and reserves matching workspace capacity.
- **Current approach:** Uses _MIN_WORK_PER_SPLIT=512 and _SPLIT_OCCUPANCY_MULTIPLIER=2; _compute_num_kv_splits rounds max_seq_len // 512 up to a power of two and caps it at sm_count * 2. Batch-invariant mode later forces one split in forward_mqa.
- **Why this impact / rank:** Triton MLA num_kv_splits policy trades parallelism vs reduction overhead every decode step for MLA models; occupancy/seq-len-aware splits improve median TPOT for MLA agent serving, but scoped to the MLA backend. — This split-KV policy directly trades parallelism against reduction overhead in Triton MLA decode. MLA agentic serving can be decode-heavy, so better split counts can move median TPOT while preserving outputs.
- **Proposals (4):**
  - Adapt Triton MLA num_kv_splits to per-decode batch and actual seq_len using Flash-Decoding occupancy heuristic _(source: research_finding)_
  - Make Triton MLA decode split policy occupancy-aware over batch and heads _(source: research_finding)_
  - Use intra-batch seq_len distribution (p90 of decode.seq_lens) instead of max_seq_len when choosing Triton MLA num_kv_splits _(source: agent_knowledge)_
  - Gate split-KV on actual decode token count to avoid splitting tiny steps _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0025](../modules/vllm_v1_attention/Triton_MLA_decode_split_policy__cand-vllm_v1_attention-0025.md)

### 77. `cand-vllm_v1_core-0021` — SlidingWindowManager.reachable_block_mask (score 53, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/single_type_kv_cache_manager.py:996` (`SlidingWindowManager.reachable_block_mask`, config_block)
- **Description:** Sliding-window sparse-retention mask policy that decides which newly full blocks are worth registering in the prefix cache under retention_interval, alignment, and EAGLE settings.
- **Current approach:** Computes the contiguous run length needed for a hit, applies an EAGLE shift, allocates a boolean mask, marks segment-boundary tails based on retention_interval, and explicitly preserves replay/shared-prefix reachable-boundary tails.
- **Why this impact / rank:** SlidingWindowManager.reachable_block_mask affects retention policy and therefore future-turn hit rate; better tail scoring lowers follow-up-turn TTFT but the effect is amortized across turns and mediated by downstream eviction, so it lands mid-pack. — The policy trades cache memory for future hit rate. For multi-turn agents with long overlapping prefixes, better SWA retention improves follow-up-turn cache hits and reduces median TTFT.
- **Proposals (4):**
  - Score sliding-window tail retention by predicted reuse × compute-saved / memory _(source: research_finding)_
  - Gate sliding-window mask tails with a TinyLFU frequency sketch to skip one-off boundaries _(source: research_finding)_
  - Build the sliding-window mask in closed form over tail intervals instead of a per-block Python loop _(source: agent_knowledge)_
  - Validate sparse retention intervals against scheduler alignment _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0021](../modules/vllm_v1_core/SlidingWindowManager.reachable_block_mask__cand-vllm_v1_core-0021.md)

### 78. `cand-vllm_v1_engine-0002` — DPEngineCoreProc._has_global_unfinished_reqs (score 58, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/core.py:2165` (`DPEngineCoreProc._has_global_unfinished_reqs`, method)
- **Description:** DP wave-completion detector that periodically synchronizes all ranks to decide whether the busy loop may pause.
- **Current approach:** Increments step_counter and performs ParallelConfig.sync_dp_state only when step_counter % 32 == 0; all other steps assume global unfinished work remains.
- **Why this impact / rank:** The 32-step DP finish-sync cadence can add up to 31 steps of wave-end wait; adaptive or event-driven sync trims tail TTFT/TPOT gaps for short bursty agentic waves, but median gain depends on wave shape. — The interval trades all-reduce overhead against wave-end latency; improving it can reduce median TPOT/TTFT gaps for short, bursty agentic waves that currently wait up to 31 extra steps to observe global idle.
- **Proposals (3):**
  - Replace fixed 32-step DP finish-sync cadence with EWMA-divergence adaptive trigger _(source: research_finding)_
  - Event-driven DP finish-sync triggered by local wave-boundary transitions _(source: agent_knowledge)_
  - Add a wall-clock latency budget for DP finish-sync _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0002](../modules/vllm_v1_engine/DPEngineCoreProc._has_global_unfinished_reqs__cand-vllm_v1_engine-0002.md)

### 79. `cand-vllm_distributed_kv_transfer-0020` — AsyncOperationManager._handle_load_task (score 58, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:354` (`AsyncOperationManager._handle_load_task`, method)
- **Description:** HF3FS load tasks fetch page locations, allocate device buffers, submit fixed-size batch_read calls, wait for every read future, scatter buffers to KV cache, and synchronize the load stream before completing the request future.
- **Current approach:** DEFAULT_MAX_IO_ENTRIES-sized chunks are submitted to a ThreadPoolExecutor, but the task then blocks while collecting every future.result(). After scatter, _load_stream.synchronize() blocks the worker thread before the request can be reported as done.
- **Why this impact / rank:** HF3FS load-task blocking stream sync gates prefix-hit resumption; polling with events and pre-registered pinned staging reduces turn-2 TTFT for HF3FS-backed reuse but only when that backend is configured. — HF3FS prefix hits cannot resume until this load future completes, so read batching and stream synchronization overhead directly affect turn-2 TTFT for HF3FS-backed reuse.
- **Proposals (3):**
  - Pipeline HF3FS batch reads with incremental completion polling and event-based load-stream sync _(source: research_finding)_
  - Poll-and-reap load completions instead of blocking per task on stream sync and future.result _(source: research_finding)_
  - Pre-register a per-worker pinned staging pool and issue HF3FS batch_read directly into it to remove buffer alloc + H2D staging on the load … _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0020](../modules/vllm_distributed_kv_transfer/AsyncOperationManager._handle_load_task__cand-vllm_distributed_kv_transfer-0020.md)

### 80. `cand-vllm_v1_core-0010` — Scheduler._make_cached_request_data (score 55, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/sched/scheduler.py:1405` (`Scheduler._make_cached_request_data`, method)
- **Description:** Builds CachedRequestData for scheduled running and resumed requests, including token slices, block IDs, computed counts, and output-token counts.
- **Current approach:** Iterates over itertools.chain(running_reqs, resumed_reqs), appends to several parallel lists, conditionally copies token IDs for PP/non-async, tracks resumed ids in a set, and calls get_block_ids for each request.
- **Why this impact / rank:** _make_cached_request_data runs once per non-empty scheduler step and scales with active requests; pre-sizing buffers and skipping empty KV updates lowers the TPOT floor at high concurrency but per-request savings are modest. — Cost scales with the number of scheduled requests every step. Reducing packing overhead lowers the TPOT floor at high concurrency.
- **Proposals (2):**
  - Pre-size list buffers and split PP/non-PP loops in _make_cached_request_data _(source: agent_knowledge)_
  - Avoid serializing empty KV block updates for cache-hit steps _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0010](../modules/vllm_v1_core/Scheduler._make_cached_request_data__cand-vllm_v1_core-0010.md)

### 81. `cand-vllm_v1_engine-0018` — EngineCoreProc.process_output_sockets (score 56, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/core.py:1737` (`EngineCoreProc.process_output_sockets`, method)
- **Description:** Engine-core output I/O thread that serializes EngineCoreOutputs and sends them to frontend or coordinator sockets with reusable buffers.
- **Current approach:** Blocks on output_queue.get, encodes each output with MsgpackEncoder, maintains a reuse buffer pool sized to len(sockets)+1, periodically retires pending zmq trackers, and sends one multipart message per EngineCoreOutputs using copy=False.
- **Why this impact / rank:** process_output_sockets is on the observable token transport path; coalescing and buffer reclamation trim frontend overhead touching TPOT, but the effect is smaller than in-runner work. — Output transport is on the observable token path; reducing serialization and socket bookkeeping overhead can lower median TPOT and avoid TTFT tails when many short agentic requests complete close together.
- **Proposals (4):**
  - Apply HWM-aware backpressure and coalesce coordinator stat messages on engine output sockets _(source: research_finding)_
  - Latency-budgeted coalescing of adjacent EngineCoreOutputs in the output I/O thread _(source: research_finding)_
  - Reclaim non-head-of-line buffers and fast-path small stat-only sends with copy=True _(source: agent_knowledge)_
  - Inline small EngineCoreOutput array frames on the output socket path _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0018](../modules/vllm_v1_engine/EngineCoreProc.process_output_sockets__cand-vllm_v1_engine-0018.md)

### 82. `cand-vllm_v1_engine-0007` — check_stop_strings (score 50, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/detokenizer.py:310` (`check_stop_strings`, function)
- **Description:** Stop-string matcher that finds the earliest stop string completing in the newly generated suffix and returns the truncation offset.
- **Current approach:** For each stop string, calls output_text.find with a negative start bound, then selects the match with the smallest end offset and stop-list-order tie behavior.
- **Why this impact / rank:** check_stop_strings runs after every decoded chunk; agentic clients with many tool/chat delimiters pay O(num_stops * suffix_window). Aho-Corasick/first-char prefilter cuts frontend CPU on TPOT, but savings are modest per token. — Agentic clients often use multiple chat/tool delimiters; reducing O(num_stops * suffix_window) scans lowers frontend CPU and median TPOT when stop strings are configured.
- **Proposals (3):**
  - Replace per-stop str.find loop with per-request Aho-Corasick automaton retaining state across decoded chunks _(source: research_finding)_
  - Stateless bounded-tail scan with first-char bucket + shingle prefilter for check_stop_strings _(source: agent_knowledge)_
  - Use a cached literal-regex matcher for multi-stop lists _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0007](../modules/vllm_v1_engine/check_stop_strings__cand-vllm_v1_engine-0007.md)

### 83. `cand-vllm_v1_executor-0014` — AsyncOutputFuture.result/UniProcExecutor.collective_rpc (score 50, impact: medium)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/uniproc_executor.py:26` (`AsyncOutputFuture.result/UniProcExecutor.collective_rpc`, region)
- **Description:** Single-process executor RPC and async-output materialization path used by UniProcExecutor execute_model, sample_tokens, and draft-token calls.
- **Current approach:** collective_rpc normalizes kwargs, calls run_method on the driver_worker immediately, calls AsyncModelRunnerOutput.get_output inline for blocking calls, and wraps results in a list unless single_value is requested. For non_block calls, run_method still executes immediately; AsyncModelRunnerOutput is wrapped in AsyncOutputFuture, whose result() later calls get_output and stores either the single ou…
- **Why this impact / rank:** UniProcExecutor collective_rpc/AsyncOutputFuture.result is per-step but only for single-process serving; removing list wrapping and blocking get_output reduces TPOT overhead, though multi-turn agentic deployments typically use multiproc/Ray, capping reach. — The path is paid for every token step on UniProcExecutor. It is medium because single-process deployments avoid interprocess/Ray overhead, but removing get_output blocking and Python wrapper churn can still lower median TPOT for short multi-turn decode workloads.
- **Proposals (2):**
  - Fast-path collective_rpc: cache execute_model/sample_tokens bound methods and elide list wrapping via a dedicated async_call path _(source: agent_knowledge)_
  - Release AsyncModelRunnerOutput after future materialization _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0014](../modules/vllm_v1_executor/AsyncOutputFuture.result_UniProcExecutor.collective_rpc__cand-vllm_v1_executor-0014.md)

### 84. `cand-vllm_distributed_kv_transfer-0009` — kv_cache_scatter_kernel / kv_cache_gather_kernel and wrappers (score 50, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py:9` (`kv_cache_scatter_kernel / kv_cache_gather_kernel and wrappers`, kernel)
- **Description:** Triton gather/scatter kernels move KV data between contiguous HF3FS buffers and paged KV cache storage, while wrappers construct token_indices tensors and launch one grid over layer and token.
- **Current approach:** Grid is (num_layers, num_tokens_in_block), BLOCK_SIZE is hard-coded to 128, num_warps/num_stages use defaults, and each wrapper builds token_indices as a CPU tensor then transfers it to device on every launch.
- **Why this impact / rank:** HF3FS gather/scatter kernel autotuning and launch fusion reduce per-block promotion latency; only relevant to the HF3FS reuse path, which caps applicability to that workload configuration. — For HF3FS-backed reuse, launch and hidden-size scan latency is paid per loaded or saved block; many small multi-turn chunks can make this visible in TTFT.
- **Proposals (4):**
  - Autotune BLOCK_SIZE, num_warps, and num_stages for HF3FS gather/scatter kernels _(source: research_finding)_
  - Fuse per-block gather/scatter launches into a single batched kernel across multiple HF3FS blocks _(source: research_finding)_
  - Persist token_indices in a device-resident ring buffer keyed by block layout to eliminate per-launch H2D copies _(source: agent_knowledge)_
  - Add contiguous-index fast paths that compute cache token offsets in-kernel _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0009](../modules/vllm_distributed_kv_transfer/kv_cache_scatter_kernel___kv_cache_gather_kernel_and_wrappers__cand-vllm_distributed_kv_transfer-0009.md)

### 85. `cand-vllm_v1_engine-0011` — DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task (score 57, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/core_client.py:1305` (`DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task`, region)
- **Description:** Async background stats receiver that maintains the frontend's local DP load-balancing snapshot from the coordinator XSUB socket.
- **Current approach:** Polls the stats and first-request sockets, drains all pending stats messages with nonblocking recv, keeps only the latest counts/wave/running tuple, slices counts to local managed ranks, and directly replaces lb_engines without smoothing or staleness metadata.
- **Why this impact / rank:** DP load-balancer stats freshness/smoothness affects DPLBAsyncMPClient routing; EWMA/hysteresis improvements reduce misroutes and tail TTFT in DP agentic bursts but the median effect is smaller. — Stale or noisy stats cause mis-routes that inflate per-engine queues; improving snapshot quality can reduce median TTFT tails in DP agentic bursts.
- **Proposals (4):**
  - Split lb_engines snapshot into prefill and decode load dimensions _(source: research_finding)_
  - Adopt short/long EWMA divergence to smooth DP load-balancing stats and detect queueing trends _(source: research_finding)_
  - Piggyback engine stats onto request-response path with sequence-numbered snapshots for tail freshness _(source: agent_knowledge)_
  - Bound stats drain work per poll to keep routing wakeups responsive _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0011](../modules/vllm_v1_engine/DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task__cand-vllm_v1_engine-0011.md)

### 86. `cand-vllm_v1_worker-0012` — preprocess_mamba (score 48, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/mamba_utils.py:1038` (`preprocess_mamba`, function)
- **Description:** Legacy per-step Mamba align-mode preprocessing that computes previous/current state block indices and stages copy metadata before forward.
- **Current approach:** Contains an explicit optimization TODO and loops over input_batch.req_ids, doing dict lookups, block arithmetic, per-request CPU buffer writes, and either fused-buffer staging or Python metadata collection.
- **Why this impact / rank:** preprocess_mamba is per-step and vectorization removes O(num_reqs x layer-groups) Python work, but the impact is gated by whether the served model is a hybrid Mamba/SSM architecture; strong within that regime, narrow across the fleet. — Hybrid Mamba/SSM models pay this on the per-step path; vectorizing scheduler-derived arrays reduces O(num_reqs x layer-groups) Python work and improves TPOT for long agent loops.
- **Proposals (4):**
  - Vectorize preprocess_mamba by making mamba_state_idx a persistent GPU-resident tensor indexed by MRV2-style stable request rows _(source: research_finding)_
  - Fold Mamba pre-copy staging into a fused selective_state_update with separate src/dst indices _(source: research_finding)_
  - Cache last-step curr_state_idx per stable row and short-circuit preprocess_mamba on unchanged steady-state decode steps _(source: agent_knowledge)_
  - Let fused precopy reset accepted-token GPU state in-kernel _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0012](../modules/vllm_v1_worker/preprocess_mamba__cand-vllm_v1_worker-0012.md)

### 87. `cand-vllm_v1_engine-0016` — InputProcessor.process_inputs (score 55, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/input_processor.py:244` (`InputProcessor.process_inputs`, method)
- **Description:** Per-request input-to-EngineCoreRequest converter that validates request parameters, preprocesses raw prompts, clones sampling or pooling params, normalizes multimodal features, and stamps admission metadata.
- **Current approach:** Runs validation on every call, preprocesses raw prompts synchronously through InputPreprocessor when needed, clones params, applies generation-config and tokenizer updates, sorts multimodal placeholders with argsort_mm_positions, builds MultiModalFeatureSpec objects, and returns a fresh EngineCoreRequest.
- **Why this impact / rank:** InputProcessor.process_inputs is per-request TTFT overhead for every new turn; fast-pathing pre-rendered EngineInput and deferring multimodal materialization is concrete but touches only new-turn admission wall time. — The work is per request rather than per token, so it mainly moves median TTFT; multi-turn agentic traffic creates many short requests where repeated validation, cloning, and preprocessing overhead is visible.
- **Proposals (2):**
  - Fast-path pre-rendered EngineInput to skip validation, cloning, and tokenizer updates for reused SamplingParams _(source: agent_knowledge)_
  - Defer multimodal feature materialization until EngineCore needs it _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0016](../modules/vllm_v1_engine/InputProcessor.process_inputs__cand-vllm_v1_engine-0016.md)

### 88. `cand-vllm_v1_engine-0017` — RequestOutputCollector.put/get_nowait/get (score 48, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/output_processor.py:62` (`RequestOutputCollector.put/get_nowait/get`, region)
- **Description:** Per-request async output handoff buffer that stores or merges RequestOutputs before generate() consumes them.
- **Current approach:** Maintains a single pending output plus an asyncio.Event; put either stores the output, raises readiness, or merges RequestOutput deltas with RequestOutput.add when the producer is ahead, while get/get_nowait clear the slot and event.
- **Why this impact / rank:** RequestOutputCollector is the AsyncLLM streaming handoff for every token batch; a lazy Future replacing asyncio.Event trims frontend event churn under many concurrent short generations. Improves frontend CPU but only indirectly shifts median TPOT. — Every streamed token batch crosses this handoff; reducing event and merge overhead can lower frontend CPU cost and median TPOT for many concurrent short generations.
- **Proposals (1):**
  - Replace asyncio.Event with lazy consumer Future in RequestOutputCollector _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0017](../modules/vllm_v1_engine/RequestOutputCollector.put_get_nowait_get__cand-vllm_v1_engine-0017.md)

### 89. `cand-vllm_v1_worker-0026` — sync_cudagraph_and_dp_padding (score 48, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/dp_utils.py:16` (`sync_cudagraph_and_dp_padding`, function)
- **Description:** New GPU runner DP coordination for cudagraph mode and token padding across data-parallel ranks.
- **Current approach:** Uses a CPU 3 x dp_size tensor, dist.all_reduce on the CPU group, then derives all-zero, minimum cudagraph mode, max token count, and uniform-token agreement through scalar tensor operations and .item() calls.
- **Why this impact / rank:** sync_cudagraph_and_dp_padding adds per-step CPU coordination in DP serving; packed collectives and idle-rank skipping trim scalar sync but the CPU-side floor is smaller than the sibling DP sync helpers in _run_ar. — The CPU collective avoids GPU sync but still adds per-step coordination latency in DP serving; reducing scalar post-processing or combining it with scheduler metadata improves median TPOT at scale.
- **Proposals (3):**
  - Amortize DP cudagraph/padding sync across multi-step decode windows _(source: research_finding)_
  - Replace SUM 3xdp_size all_reduce with packed MIN+MAX collectives that fold reductions into the wire _(source: agent_knowledge)_
  - Ignore idle DP ranks when agreeing on cudagraph mode and uniform-token padding _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0026](../modules/vllm_v1_worker/sync_cudagraph_and_dp_padding__cand-vllm_v1_worker-0026.md)

### 90. `cand-vllm_v1_core-0020` — BlockPool.cache_full_blocks (score 55, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/block_pool.py:225` (`BlockPool.cache_full_blocks`, method)
- **Description:** Caches newly full KV blocks by registering block hashes, promoting partial blocks when needed, applying sparse block masks, and optionally emitting KV cache events.
- **Current approach:** Slices new_full_blocks, resolves block hashes, loops per block to skip null/masked entries, removes old partial hashes, inserts new hash metadata into BlockHashToBlockMap, collects event hashes, then does a second loop to build per-block event extra keys when kv_cache_events are enabled.
- **Why this impact / rank:** cache_full_blocks per-block insertion runs when finalized tokens are cached and shapes future prefix hit availability; fusing hash-insert with events reduces scheduler overhead and improves reuse-driven TTFT. — Caching long prefills and generated blocks is on the scheduler path and determines future prefix hit availability. Reducing this loop lowers TTFT for long prompts and preserves TPOT under continuous batching.
- **Proposals (2):**
  - Precompute skip mask and fuse hash-insert with event-extra-keys into a single loop _(source: agent_knowledge)_
  - Split the fresh-block insert path to skip redundant containment lookups _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0020](../modules/vllm_v1_core/BlockPool.cache_full_blocks__cand-vllm_v1_core-0020.md)

### 91. `cand-vllm_v1_kv_offload-0015` — FileSystemTierManager.submit_store/submit_load (score 46, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/fs/manager.py:217` (`FileSystemTierManager.submit_store/submit_load`, region)
- **Description:** FS secondary-tier load/store submission builds paths and offsets, wraps the whole job as one task, and enqueues it into the dual-queue pool.
- **Current approach:** submit_store and submit_load each create one functools.partial over all block paths and offsets, then call enqueue_store or enqueue_load with n_tasks=1. Large jobs therefore occupy a single worker task, while small jobs are not coalesced across submissions.
- **Why this impact / rank:** FileSystemTierManager chunking and coalescing improve promotion latency on FS-backed reuse and are on the TTFT path when spill goes to disk; the benefit is real but only for deployments that actually use the FS tier, and rf_count=3 helps evidence. — For FS-backed multi-turn reuse, secondary-to-primary loads are on the TTFT path. Better task sizing can use configured read threads more effectively and reduce promotion latency, while store-side changes can reduce interference with read bursts.
- **Proposals (5):**
  - Progressive-batch chunking of submit_load with small early sub-tasks _(source: research_finding)_
  - Chunk FS tier jobs into multiple sub-tasks and add slack-aware load/store scheduling _(source: research_finding)_
  - Chunk FS submit_store/submit_load into per-worker batched I/O tasks _(source: research_finding)_
  - Cross-submission store coalescing with a short debounce and a job-fanout JobState _(source: agent_knowledge)_
  - Move path and offset materialization out of scheduler-thread submission _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0015](../modules/vllm_v1_kv_offload/FileSystemTierManager.submit_store_submit_load__cand-vllm_v1_kv_offload-0015.md)

### 92. `cand-vllm_v1_attention-0023` — FlexAttentionMetadata._build_block_mask_direct (score 52, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flex_attention.py:692` (`FlexAttentionMetadata._build_block_mask_direct`, method)
- **Description:** Builds FlexAttention BlockMask metadata directly from paged-KV block tables for the direct-build path.
- **Current approach:** Indexes block_table for every query token up to cdiv(max_seq_len, block_size), applies several masked_fill_ pruning passes for sequence length, causal, sliding-window, R-SWA, and custom sparsity hints, pads and reshapes by q_block_size, deduplicates with unique_static_unsorted, and copies kv_indices and kv_num_blocks into persistent buffers.
- **Why this impact / rank:** FlexAttention _build_block_mask_direct can dominate small-batch decode metadata cost; TPOT gain is meaningful but limited to the Flex backend rather than the default agentic serving stack. — For FlexAttention decode this metadata path runs before attention and can dominate small-batch per-step overhead. Reducing over-fetch, pruning passes, or dedup work can improve median TPOT for agentic decode batches using the Flex backend.
- **Proposals (3):**
  - Cache and reuse per-step BlockMask metadata across FlexAttention layers _(source: research_finding)_
  - Split fully-unmasked KV blocks into full_kv_indices to skip per-element mask_mod _(source: research_finding)_
  - Add a decode-specialized analytic fast path in _build_block_mask_direct that skips the batch-max gather and masked_fill_ pruning chain _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0023](../modules/vllm_v1_attention/FlexAttentionMetadata._build_block_mask_direct__cand-vllm_v1_attention-0023.md)

### 93. `cand-vllm_v1_executor-0008` — FutureWrapper.result (score 46, impact: medium)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/ray_utils.py:250` (`FutureWrapper.result`, method)
- **Description:** Ray executor future resolution path for non-blocking compiled-DAG execution; resolves Ray refs, detaches zero-copy buffers, and optionally aggregates all worker outputs.
- **Current approach:** result() calls ray.get(self.ref_or_refs, timeout=timeout). Without an aggregator it detaches the single output and returns it. With an aggregator it iterates outputs sequentially, detaches each output in Python, and calls self.aggregator.aggregate(outputs, output_rank=0).
- **Why this impact / rank:** FutureWrapper.result sits on the async Ray decode completion path with a sequential detach loop; ray.wait-based overlap helps TPOT under connectors/multi-output but non-connector single-output cases have only one detach. — This is directly on the async Ray decode completion path. It can move median TPOT under connector or multi-output cases, with medium scope because non-connector single-output runs perform only one detach.
- **Proposals (2):**
  - Use ray.wait to overlap detach with in-flight worker outputs in FutureWrapper.result _(source: research_finding)_
  - Skip zero-copy detach on non-output-rank worker outputs before aggregation _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0008](../modules/vllm_v1_executor/FutureWrapper.result__cand-vllm_v1_executor-0008.md)

### 94. `cand-vllm_v1_executor-0015` — RayDistributedExecutor._init_workers_ray (score 52, impact: high)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/ray_executor.py:143` (`RayDistributedExecutor._init_workers_ray`, method)
- **Description:** RayDistributedExecutor worker placement and rank-topology construction, including bundle selection, actor creation, driver-first sorting, rank remapping, per-node GPU mapping, and pp_tp_workers layout for compiled-DAG construction.
- **Current approach:** The method chooses bundle_indices from VLLM_RAY_BUNDLE_INDICES or the first GPU bundles in the placement group, creates one RayWorkerWrapper actor per rank, fetches worker IPs, sorts workers by driver-node preference, worker count per IP, and IP, adjusts worker ranks through collective_rpc('adjust_rank'), discovers physical GPU IDs, initializes workers, and finally builds pp_tp_workers as sequent…
- **Why this impact / rank:** Ray placement/rank layout is initialization code, but topology-aware TP intra-node / PP cross-node choices reduce cross-node transport per token step; steady-state TPOT gain is real but bounded to multi-node PP/TP setups. — Placement and rank layout determine the communication pattern used by every Ray compiled-DAG token step. Better topology choices can reduce cross-node synchronization and tensor transport time, moving median TPOT for multi-node PP/TP agentic workloads; the effect is smaller on single-node runs.
- **Proposals (4):**
  - Topology-aware rank remapping: pack TP groups intra-node, span PP across nodes _(source: research_finding)_
  - Hierarchical 2D rank assignment keeping TP intra-node and ordering PP stages for bandwidth-aware handoffs _(source: research_finding)_
  - NUMA/PCIe-affinity-aware bundle selection to cut host-to-device prefill staging latency for TTFT _(source: agent_knowledge)_
  - Order ranks within each TP group by local GPU interconnect topology _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0015](../modules/vllm_v1_executor/RayDistributedExecutor._init_workers_ray__cand-vllm_v1_executor-0015.md)

### 95. `cand-vllm_v1_engine-0012` — RequestState.make_request_output (score 46, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/output_processor.py:276` (`RequestState.make_request_output`, method)
- **Description:** Per-request RequestOutput factory that applies FINAL_ONLY and stream_interval gating, handles pooling vs completion output, and aggregates parallel-sampling children.
- **Current approach:** Branches on finish state, output kind, stream_interval, pooling output, and parent request state; may slice detokenizer.output_token_ids for DELTA mode and allocates CompletionOutput/RequestOutput objects on each emitted step.
- **Why this impact / rank:** make_request_output is allocated per request per output-processing step; hoisting FINAL_ONLY gating and deferring text decoding lowers frontend TPOT for many short concurrent generations, but constant-factor gains are bounded. — This is a per-request allocation and branching hotspot on the frontend path; reducing its constant factors can lower median TPOT, especially with many short concurrent generations.
- **Proposals (2):**
  - Hoist FINAL_ONLY / stream_interval gating out of make_request_output into a cached fast-reject predicate _(source: agent_knowledge)_
  - Defer text decoding for FINAL_ONLY requests with no stop strings _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0012](../modules/vllm_v1_engine/RequestState.make_request_output__cand-vllm_v1_engine-0012.md)

### 96. `cand-vllm_v1_executor-0006` — detach_zero_copy_from_model_runner_output (score 46, impact: medium)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/ray_utils.py:198` (`detach_zero_copy_from_model_runner_output`, function)
- **Description:** In-place detachment of read-only numpy arrays inside ModelRunnerOutput so Ray compiled-DAG shared-memory buffers are not retained across scheduler iterations.
- **Current approach:** The nested _copy_if_readonly checks isinstance(arr, np.ndarray) and arr.flags.writeable, then calls arr.copy() for read-only arrays. logprobs and routed_experts fields are unpacked, copied field-by-field, and their enclosing tuple/namedtuple is rebuilt only if at least one child array changed.
- **Why this impact / rank:** detach_zero_copy_from_model_runner_output is on the Ray output critical path but only bites when logprobs or routed_experts are requested; conditional agentic gains for Ray backends make this a mid-tier TPOT lever. — The copies are on the Ray output critical path and scale with emitted logprob/routing metadata. Optimizing them moves median TPOT for Ray backends when agentic workloads request logprobs or use expert routing, but has little effect when those fields are absent.
- **Proposals (4):**
  - Skip detach copies via opt-in, per-step routed-experts and logprobs payloads _(source: research_finding)_
  - Overlap zero-copy detach with outstanding worker outputs via ray.wait _(source: research_finding)_
  - Copy SHM arrays into a rank/field-keyed reusable pool instead of allocating fresh numpy buffers _(source: agent_knowledge)_
  - Detach only the aggregated rank’s surviving metadata _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0006](../modules/vllm_v1_executor/detach_zero_copy_from_model_runner_output__cand-vllm_v1_executor-0006.md)

### 97. `cand-vllm_v1_attention-0012` — make_local_attention_virtual_batches (score 50, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/utils.py:348` (`make_local_attention_virtual_batches`, function)
- **Description:** Splits local-attention requests into virtual batches and rebuilds local query lengths, KV lengths, computed-token counts, and block tables.
- **Current approach:** Performs multiple numpy repeat/cumsum/fancy-index steps, materializes batch_indices and block_indices, uploads both index arrays H2D, and then gathers block_table_local with PyTorch indexing.
- **Why this impact / rank:** make_local_attention_virtual_batches runs per step for local-attention Gemma models; coalescing four H2D uploads cuts CPU/H2D overhead but the win is scoped to that model family. — Local-attention Gemma-family models are relevant to agentic serving. This per-step metadata transformation feeds scheduling and attention launch, so reducing CPU/H2D work improves median TPOT.
- **Proposals (3):**
  - Coalesce four H2D uploads in make_local_attention_virtual_batches into one staged transfer _(source: research_finding)_
  - Compute batch_indices/block_indices on-device from small per-request tensors instead of materializing large numpy arrays and uploading them _(source: agent_knowledge)_
  - Add a single-local-block fast path for decode-style batches _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0012](../modules/vllm_v1_attention/make_local_attention_virtual_batches__cand-vllm_v1_attention-0012.md)

### 98. `cand-vllm_v1_core-0016` — BlockHashListWithBlockSize (score 45, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/kv_cache_utils.py:2229` (`BlockHashListWithBlockSize`, region)
- **Description:** Lazy view that maps hash-block-granularity BlockHash lists to larger target block sizes for heterogeneous KV cache groups.
- **Current approach:** Stores the original list and scale_factor. __getitem__, slicing, and __iter__ compute (idx + 1) * scale_factor - 1 in Python for every access; _get_value_at is dispatched per element.
- **Why this impact / rank:** BlockHashListWithBlockSize view overhead only bites hybrid models with mismatched block sizes; a strided-slice backing removes per-access attribute walks on prefix-lookup hot code, modestly reducing median TTFT for that specific configuration. — Hybrid models with mismatched block sizes can perform hundreds of hash accesses per admission. Reducing view overhead lowers prefix-lookup latency and median TTFT.
- **Proposals (1):**
  - Back BlockHashListWithBlockSize by a precomputed strided slice instead of computing (idx+1)*scale-1 per access _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0016](../modules/vllm_v1_core/BlockHashListWithBlockSize__cand-vllm_v1_core-0016.md)

### 99. `cand-vllm_v1_kv_offload-0017` — ObjectStoreSecondaryTierManager._submit_transfer/_poll_active_transfers (score 50, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/obj/manager.py:217` (`ObjectStoreSecondaryTierManager._submit_transfer/_poll_active_transfers`, region)
- **Description:** Object-store secondary tier builds NIXL OBJ descriptors, submits asynchronous transfers, polls in-flight handles, and releases transfer resources.
- **Current approach:** Each job converts block IDs, creates one OBJ descriptor per key with monotonically increasing dev IDs, registers object memory, prepares a transfer dlist, starts a NIXL transfer, and stores the handle in _transfers. Completion polling scans all active transfers, classifies states, releases handles/dlists/memory, and buffers one JobResult per job.
- **Why this impact / rank:** Object-store transfer submission/polling reduces secondary promotion delay before TTFT; helpful for cross-turn reuse but bounded by intrinsic network/object-store latency, indirect effect on median. — Object-store hits avoid recomputation but can add substantial secondary promotion delay before TTFT. Better batching or polling can reduce that delay for multi-turn agentic reuse, although network/object-store latency limits the median improvement.
- **Proposals (8):**
  - Offload NIXL OBJ transfer submission and polling to a dedicated background I/O thread _(source: research_finding)_
  - Split OBJ tier submissions into progressive sub-transfers to cut HOL blocking on long promotions _(source: research_finding)_
  - Batch NIXL OBJ transfers into chunked object I/O with slack-aware polling _(source: research_finding)_
  - Coalesce concurrent object-store jobs into a single batched NIXL submission _(source: research_finding)_
  - Reuse prepared NIXL OBJ dlist handles and batch descriptors across jobs _(source: research_finding)_
  - Compress KV blocks with adaptive bitrate before NIXL OBJ transfer to shrink promotion bytes _(source: research_finding)_
  - Issue parallel NIXL OBJ xfer stripes per job to saturate object-store concurrent-connection bandwidth _(source: agent_knowledge)_
  - Prioritize demand loads over cascade stores with OBJ transfer admission control _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0017](../modules/vllm_v1_kv_offload/ObjectStoreSecondaryTierManager._submit_transfer__poll_active_transfers__cand-vllm_v1_kv_offload-0017.md)

### 100. `cand-vllm_multimodal-0019` — MultiModalBudget._get_max_items (score 45, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/encoder_budget.py:147` (`MultiModalBudget._get_max_items`, method)
- **Description:** Computes per-modality max items per prompt and per batch from encoder budget, decoder budget, modality limits, max model length, and chunked-prefill settings.
- **Current approach:** Uses floor division by max_tokens_per_item and min/max clamps: encoder_budget // max_tokens_per_item, max_model_len // max_tokens_per_item, mm_limit, max_num_batched_tokens, and max_num_reqs. It assumes worst-case per-item token counts and does not account for cache hits or observed token distributions.
- **Why this impact / rank:** MultiModalBudget._get_max_items sets encoder-work admission ceilings; cache-aware and p95-driven sizing can reduce queueing-induced TTFT but only in multimodal-heavy mixes that the hint does not explicitly require. — This sets concurrency ceilings for multimodal encoder work. Conservative limits increase queueing and TTFT; overly loose limits risk OOM. Better admission improves throughput and TPOT under mixed multimodal bursts.
- **Proposals (4):**
  - Make _get_max_items cache-aware by discounting cached items from encoder budget _(source: research_finding)_
  - Adjust per-batch item ceilings by expected encoder-cache hit rate _(source: research_finding)_
  - Size encoder ceilings on observed p95 per-item token counts instead of worst-case max_tokens_per_item _(source: agent_knowledge)_
  - Add an aggregate mixed-modality encoder-token cap _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0019](../modules/vllm_multimodal/MultiModalBudget._get_max_items__cand-vllm_multimodal-0019.md)

### 101. `cand-vllm_v1_engine-0014` — DPCoordinator.run polling/publish loop (score 44, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/coordinator.py:257` (`DPCoordinator.run polling/publish loop`, loop)
- **Description:** DP coordinator publish loop that chooses when to broadcast engine load snapshots to frontends.
- **Current approach:** Computes wait_for as stats_update_interval_ms when stats changed or 5000 ms otherwise, optionally enforces a 50 ms minimum wait under wave coordination, then publishes either last_step_counts or current engine counts only on poll timeout.
- **Why this impact / rank:** DPCoordinator publish cadence controls DP routing snapshot freshness; adaptive/event-driven publication reduces stale-load misrouting TTFT in bursty DP serving, but only when data-parallel coordination is enabled. — Snapshot publication latency feeds directly into DP routing decisions; improving cadence can reduce stale-load misrouting and median TTFT for bursty multi-turn DP serving.
- **Proposals (3):**
  - Adaptive DP coordinator publish cadence via EWMA divergence of engine load _(source: research_finding)_
  - Event-driven publish on routing-decision threshold crossings in DPCoordinator.run _(source: agent_knowledge)_
  - Publish zero-wait snapshots immediately after FIRST_REQ wakeups _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0014](../modules/vllm_v1_engine/DPCoordinator.run_polling_publish_loop__cand-vllm_v1_engine-0014.md)

### 102. `cand-vllm_v1_core-0013` — SingleTypeKVCacheManager.get_num_blocks_to_allocate (score 48, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/single_type_kv_cache_manager.py:144` (`SingleTypeKVCacheManager.get_num_blocks_to_allocate`, method)
- **Description:** Computes how many KV blocks a single cache manager needs for a request, accounting for existing blocks, prefix hits, skipped-window blocks, partial-hit CoW, and evictable cached hits.
- **Current approach:** Performs Python arithmetic, computes skipped blocks, slices new_computed_blocks, scans remaining blocks with _get_num_evictable_blocks, and adds a partial-hit CoW reservation when needed. Running requests use a fast path when num_cached_block is set.
- **Why this impact / rank:** get_num_blocks_to_allocate can be called twice per waiting request in full_sequence_must_fit; memoizing the evictable scan trims admission overhead on TTFT, but gains are modest per turn. — Long shared-prefix requests can pass hundreds of cached blocks through this sizing path. Reducing repeated scans trims admission overhead and improves TTFT for deep-history agent turns.
- **Proposals (1):**
  - Memoize evictable-block scan across the twin full_sequence_must_fit sizing calls _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0013](../modules/vllm_v1_core/SingleTypeKVCacheManager.get_num_blocks_to_allocate__cand-vllm_v1_core-0013.md)

### 103. `cand-vllm_v1_worker-0025` — StagedWriteTensor and FusedStagedWriter apply path (score 46, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/buffer_utils.py:128` (`StagedWriteTensor and FusedStagedWriter apply path`, region)
- **Description:** Staged GPU-buffer write machinery used by the new runner to apply request and block-table mutations.
- **Current approach:** Accumulates staged indices, starts, contents, and cumulative lengths in Python lists; copies several metadata lists through UVA buffers; transfers contents separately; then launches _apply_write_kernel with fixed BLOCK_SIZE=1024.
- **Why this impact / rank:** StagedWriteTensor/FusedStagedWriter apply path scales with request churn and KV groups; persistent pinned/UVA ring buffers eliminate list rebuilds and a sync H2D. Real gains on admission/resume churn, though not on steady decode of static batches. — This cost scales with request churn and KV groups; better packed staging or persistent device-side append buffers improves TTFT for admitted/resumed requests and reduces TPOT during churn-heavy agent traffic.
- **Proposals (2):**
  - Stage writes directly into persistent pinned/UVA ring buffers to eliminate list rebuilding and the synchronous contents H2D _(source: agent_knowledge)_
  - Coalesce contiguous staged writes per row before applying _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0025](../modules/vllm_v1_worker/StagedWriteTensor_and_FusedStagedWriter_apply_path__cand-vllm_v1_worker-0025.md)

### 104. `cand-vllm_v1_attention-0021` — AttentionMetadataBuilder._init_reorder_batch_threshold (score 48, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backend.py:659` (`AttentionMetadataBuilder._init_reorder_batch_threshold`, config_block)
- **Description:** Initializes the per-backend threshold that classifies short requests as decode-like for batch reordering and metadata splitting.
- **Current approach:** Starts from a backend-provided reorder_batch_threshold, raises it to 1 + speculative-token count or parallel-drafting width when supports_spec_as_decode is true, and forces it back to 1 for DCP when varlen DCP support is unavailable.
- **Why this impact / rank:** Reorder-batch threshold routes short-extend/spec tokens between decode and prefill kernels; better routing keeps efficient decode kernels active, but effect is second-order relative to kernel-level tunables. — This threshold affects how agentic short extends and speculative tokens are routed through decode versus prefill paths. Better routing can reduce median TPOT by keeping efficient decode kernels active without misclassifying correctness-sensitive DCP cases.
- **Proposals (2):**
  - Include chunked-prefill tail-chunk size in reorder threshold for agentic short-extend turns _(source: agent_knowledge)_
  - Preserve disabled batch reordering under DCP override _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0021](../modules/vllm_v1_attention/AttentionMetadataBuilder._init_reorder_batch_threshold__cand-vllm_v1_attention-0021.md)

### 105. `cand-vllm_v1_worker-0020` — maybe_create_ubatch_slices (score 44, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/ubatch_utils.py:63` (`maybe_create_ubatch_slices`, function)
- **Description:** Chooses token split points and materializes legacy microbatch request/token slices for DBO.
- **Current approach:** Defaults to uniform token-count split_point = num_tokens_padded // num_ubatches and maps token split points back to request slices with np.searchsorted.
- **Why this impact / rank:** maybe_create_ubatch_slices governs DBO split points; compute-cost-weighted splitting improves overlap on mixed prefill/decode agent batches but the benefit is second-order and DBO must be enabled. — Cost-balanced slices can improve DBO overlap, moving TPOT for mixed prefill/decode agent batches where equal token counts do not imply equal runtime.
- **Proposals (2):**
  - Compute-cost-weighted DBO split using per-request query length and KV prefix _(source: agent_knowledge)_
  - Split DBO on actual tokens before padding _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0020](../modules/vllm_v1_worker/maybe_create_ubatch_slices__cand-vllm_v1_worker-0020.md)

### 106. `cand-vllm_v1_worker-0013` — postprocess_mamba_fused_kernel (score 42, impact: high)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/mamba_utils.py:154` (`postprocess_mamba_fused_kernel`, kernel)
- **Description:** Fused Triton kernel that updates accepted-token counts and copies Mamba/SSM/conv state blocks during spec-decode postprocess alignment.
- **Current approach:** Uses a grid over requests and flattened state types with COPY_BLOCK_SIZE fixed at 1024 at call sites, and the dim-first conv path loops over dim rows inside each program.
- **Why this impact / rank:** postprocess_mamba_fused_kernel tiling improvements shave hybrid Mamba spec-decode TPOT via reduced state-copy time; net effect is scoped to hybrid Mamba plus speculative decoding, so impact per-run is real but audience is small. — This kernel is on hybrid Mamba spec-decode TPOT; better tiling and vectorization can reduce memory-copy time for recurrent states in long-context agent workloads.
- **Proposals (3):**
  - Replace copy-based accepted-token alignment with ring-buffered input cache and matmul replay _(source: research_finding)_
  - Split into compacted decision prepass + 3D copy grid with per-state-type autotuned COPY_BLOCK_SIZE _(source: agent_knowledge)_
  - Write accepted-token outputs inside the fused kernel and drop the V1 pre-copy _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0013](../modules/vllm_v1_worker/postprocess_mamba_fused_kernel__cand-vllm_v1_worker-0013.md)

### 107. `cand-vllm_v1_attention-0010` — reorder_batch_to_split_decodes_and_prefills (score 45, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/utils.py:736` (`reorder_batch_to_split_decodes_and_prefills`, function)
- **Description:** Reorders InputBatch state into contiguous decode, short-extend, long-extend, and pure-prefill regions.
- **Current approach:** Builds numpy masks and a permutation, then follows cycles with input_batch.swap_states one pair at a time. decode_threshold is externally supplied and there is no local cost model for whether reordering is worthwhile on a given step.
- **Why this impact / rank:** reorder_batch_to_split_decodes_and_prefills runs every scheduler step; two-pointer minimum-swap and no-reorder fast paths reduce per-step overhead when agent traffic alternates decode and short extends. Small per-step savings. — Agentic workloads frequently alternate decode and short tool-response extends. Faster reorder work reduces scheduler-side per-step overhead and can improve median TPOT.
- **Proposals (2):**
  - Fast-path for already-ordered agentic batches + two-pointer minimum-swap reorder _(source: agent_knowledge)_
  - Add a conservative no-reorder cost gate for tiny mixed batches _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0010](../modules/vllm_v1_attention/reorder_batch_to_split_decodes_and_prefills__cand-vllm_v1_attention-0010.md)

### 108. `cand-vllm_v1_core-0023` — ChunkedLocalAttentionManager.find_longest_cache_hit (score 46, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/single_type_kv_cache_manager.py:1101` (`ChunkedLocalAttentionManager.find_longest_cache_hit`, method)
- **Description:** Chunked-local attention prefix-cache hit lookup that treats blocks before the local chunk as already computed and scans cache hits within the current chunk.
- **Current approach:** Computes the local attention chunk start, pre-fills computed block lists with null blocks up to the chunk start, then scans forward through in-window blocks with block_pool.get_cached_block until the first miss.
- **Why this impact / rank:** ChunkedLocalAttentionManager.find_longest_cache_hit avoids large null-list materialization on admission; TTFT-relevant but scoped to chunked-local models. — Only chunked-local models use it, but it runs during admission and can touch many chunk-prefix blocks for long contexts. Reducing scan and null-padding overhead lowers TTFT for those workloads.
- **Proposals (2):**
  - Reuse a shared null-prefix slice and localize the in-window probe in ChunkedLocalAttentionManager.find_longest_cache_hit _(source: agent_knowledge)_
  - Short-circuit chunk-boundary hits before resolving block hashes _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0023](../modules/vllm_v1_core/ChunkedLocalAttentionManager.find_longest_cache_hit__cand-vllm_v1_core-0023.md)

### 109. `cand-vllm_v1_kv_offload-0012` — CPUOffloadingManager.prepare_store (score 45, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/cpu/manager.py:166` (`CPUOffloadingManager.prepare_store`, method)
- **Description:** Prepares CPU-tier stores by applying admission filters, computing eviction needs, evicting victims, allocating blocks, and inserting pending writes.
- **Current approach:** Optionally filters by a fixed store_threshold, then filters already-stored keys, computes an all-or-nothing eviction budget, asks the cache policy for exactly that many victims, and returns None if the whole batch cannot fit.
- **Why this impact / rank:** CPU prepare_store admission improves future primary hit rate and reduces all-or-nothing store failures; complementary to eviction policy but effect on median TTFT/TPOT is indirect and multi-turn-cache-dependent. — Better store admission can avoid wasting primary capacity and reduce all-or-nothing store failures. It improves future primary hit rate and lowers promotion pressure, behind but complementary to the eviction policies.
- **Proposals (4):**
  - Workflow-value-aware admission and partial-batch store in prepare_store _(source: research_finding)_
  - TinyLFU-based admission with per-candidate victim comparison in prepare_store _(source: research_finding)_
  - Prefix-contiguity-aware admission and tail-trim partial acceptance in prepare_store _(source: agent_knowledge)_
  - Add group-completeness admission for multi-KV-group chunks _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0012](../modules/vllm_v1_kv_offload/CPUOffloadingManager.prepare_store__cand-vllm_v1_kv_offload-0012.md)

### 110. `cand-vllm_v1_attention-0020` — compute_mm_prefix_range_tensor (score 43, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/utils.py:49` (`compute_mm_prefix_range_tensor`, function)
- **Description:** Converts TritonAttention multimodal prefix ranges from a per-request dict into a padded device tensor.
- **Current approach:** Builds Python lists for every request, computes max_ranges on CPU, pads each request's range list, then uploads the nested list with async_tensor_h2d and reshapes it.
- **Why this impact / rank:** compute_mm_prefix_range_tensor is per-step multimodal prefill work with padded H2D uploads; a jagged CSR layout and pinned staging cut CPU list work on multimodal turns only. — Multimodal agent turns can hit this every prefill or extend step on the Triton backend. Reducing Python list work and H2D traffic improves multimodal TTFT and per-step overhead.
- **Proposals (3):**
  - Pinned staging + vectorized fill for mm_prefix_range tensor with single non-blocking H2D _(source: research_finding)_
  - Replace padded (num_seqs, max_ranges, 2) mm_prefix_range tensor with a jagged CSR layout (flat values + int32 offsets) _(source: agent_knowledge)_
  - Cache and update mm_prefix_range tensors only when request ranges change _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0020](../modules/vllm_v1_attention/compute_mm_prefix_range_tensor__cand-vllm_v1_attention-0020.md)

### 111. `cand-vllm_distributed_kv_transfer-0014` — HF3FSKVConnector._gather_or_scatter_kv_caches (score 44, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:984` (`HF3FSKVConnector._gather_or_scatter_kv_caches`, method)
- **Description:** Loops over each HF3FS block buffer, builds token indices for that block, and calls the gather or scatter helper once per block.
- **Current approach:** One Python iteration per block performs range/list allocation, wrapper call, CPU token_indices tensor creation, H2D copy, and Triton launch. Adjacent blocks are not batched, and device index tensors are not reused across calls.
- **Why this impact / rank:** HF3FS gather/scatter batching reduces per-block kernel launches on turn-2 fetch; helpful for HF3FS deployments but the win depends on many small blocks and doesn't touch the common non-HF3FS path. — HF3FS load/save latency contributes to turn-2 TTFT; many small blocks in agentic history reuse make per-block Python and H2D setup visible.
- **Proposals (4):**
  - Batch HF3FS gather/scatter into a single non-contiguous BatchTransfer-style call _(source: research_finding)_
  - Batch HF3FS block gather/scatter into a single fused kernel launch _(source: research_finding)_
  - Pipeline HF3FS storage I/O batches with per-batch scatter/gather to overlap disk and GPU compute _(source: agent_knowledge)_
  - Short-circuit HF3FS gather/scatter when no blocks are scheduled _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0014](../modules/vllm_distributed_kv_transfer/HF3FSKVConnector._gather_or_scatter_kv_caches__cand-vllm_distributed_kv_transfer-0014.md)

### 112. `cand-vllm_v1_attention-0015` — _make_mm_prefix_mask_mod (score 44, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flash_attn.py:1424` (`_make_mm_prefix_mask_mod`, function)
- **Description:** Builds the cached CuTE-DSL mask_mod for FA4 multimodal prefill, combining causal or sliding-window masking with mm_prefix ranges.
- **Current approach:** Caches by sliding-window parameters, defines nested cute.jit helpers, and loads two range entries for each query row before OR-ing the multimodal range with the causal/window mask.
- **Why this impact / rank:** _make_mm_prefix_mask_mod runs inside FA4 prefill blocks for multimodal prompts; precomputing full/partial KV block metadata skips per-element masking on fully-included blocks, helping image-heavy TTFT. Scope narrow (FA4 + mm_prefix). — For multimodal agent workloads this mask executes inside every FA4 prefill block. Reducing per-mask loads or simplifying the predicate can improve image-heavy TTFT.
- **Proposals (2):**
  - Precompute full/partial KV block metadata for mm_prefix so FA4 skips per-element masking on fully-included blocks _(source: research_finding)_
  - Pack mm_prefix range into a single int64 load and use range-length subtract for the predicate _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0015](../modules/vllm_v1_attention/_make_mm_prefix_mask_mod__cand-vllm_v1_attention-0015.md)

### 113. `cand-vllm_distributed_kv_transfer-0018` — NixlBaseConnectorWorker._compute_desc_ids (score 40, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:93` (`NixlBaseConnectorWorker._compute_desc_ids`, method)
- **Description:** Computes NIXL descriptor IDs for each READ/WRITE setup from grouped block IDs, allocating numpy arrays and concatenating descriptor vectors on the transfer-prep path.
- **Current approach:** The all-attention path concatenates block groups and broadcasts region_ids per call. Hybrid attention/SSM paths loop over groups, allocate np.asarray/np.arange intermediates, flatten per-group descriptors, and concatenate all_descs for every transfer side.
- **Why this impact / rank:** _compute_desc_ids pays per remote-prefill transfer setup; fused numpy or contiguous-range paths cut Python allocation, but the cost is a small fraction of NIXL transfer time and only shows up for HMA/multi-region setups with many blocks. — The cost is paid for every remote-prefill transfer setup; it affects turn-2 TTFT most when prompts span many blocks or heterogeneous/HMA groups multiply descriptor arrays.
- **Proposals (2):**
  - Fuse desc-id generation with np.add.outer into a preallocated int32 scratch buffer _(source: agent_knowledge)_
  - Skip descriptor synthesis for contiguous block runs by passing compact ranges to NIXL prep _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0018](../modules/vllm_distributed_kv_transfer/NixlBaseConnectorWorker._compute_desc_ids__cand-vllm_distributed_kv_transfer-0018.md)

### 114. `cand-vllm_v1_core-0005` — BlockPool.free_blocks (score 44, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/block_pool.py:719` (`BlockPool.free_blocks`, method)
- **Description:** Returns blocks whose ref_cnt reaches zero to the free queue and chooses their eviction ordering based on whether they still have prefix-cache hashes.
- **Current approach:** Loops through ordered blocks, decrements ref_cnt, classifies zero-ref non-null blocks into blocks_with_hash and blocks_without_hash lists, then prepends hashless blocks and appends hashed blocks to preserve the current LRU/hash-priority policy.
- **Why this impact / rank:** free_blocks retention ordering affects future prefix-cache hit rate; better retention helps follow-up-turn TTFT, but the loop itself is small and the mechanism is one policy signal among many governing hit rate. — The loop is small, but the ordering policy affects future prefix-cache hit rate. For multi-turn agents with overlapping histories, better retention ordering can improve TTFT on follow-up turns.
- **Proposals (9):**
  - Workflow-aware free ordering: prioritize retaining soon-needed agent-prefix blocks in free_blocks _(source: research_finding)_
  - Add priority-tier retention ordering to BlockPool.free_blocks via an auxiliary priority queue _(source: research_finding)_
  - Score-based eviction ordering in free_blocks using reuse-likelihood and compute-savings-per-block _(source: research_finding)_
  - Bias free_blocks eviction ordering with a learned continuation-probability score _(source: research_finding)_
  - Radix-tree-informed eviction ordering in BlockPool.free_blocks _(source: research_finding)_
  - Add W-TinyLFU frequency-aware admission to free_blocks placement _(source: research_finding)_
  - Apply SIEVE-style lazy promotion and quick demotion to free_blocks eviction ordering _(source: research_finding)_
  - Order free_blocks by observed per-block reuse-interval EMA to approximate Belady _(source: agent_knowledge)_
  - Batch free_blocks with allocation-free linked-list splicing _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0005](../modules/vllm_v1_core/BlockPool.free_blocks__cand-vllm_v1_core-0005.md)

### 115. `cand-vllm_v1_worker-0008` — GPUModelRunner._calc_mrope_positions (score 41, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu_model_runner.py:2795` (`GPUModelRunner._calc_mrope_positions`, method)
- **Description:** Populates legacy M-RoPE position buffers from per-request precomputed prompt positions and computes continuation positions.
- **Current approach:** Loops over input_batch.req_ids, performs per-request dict and scheduler lookups, computes prompt lengths, and writes request slices into a shared CPU/NumPy tensor.
- **Why this impact / rank:** _calc_mrope_positions is a per-request per-step loop on M-RoPE multimodal models; vectorizing helps multimodal TPOT/TTFT only when those models are being served. — The impact is concentrated in multimodal agent workloads, where per-turn image/video requests make M-RoPE active; reducing this loop lowers multimodal TPOT and TTFT.
- **Proposals (2):**
  - Vectorize decode-branch M-RoPE writes and drop per-request dict lookups _(source: agent_knowledge)_
  - Add an all-decode GPU fast path for M-RoPE positions _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0008](../modules/vllm_v1_worker/GPUModelRunner._calc_mrope_positions__cand-vllm_v1_worker-0008.md)

### 116. `cand-vllm_v1_core-0022` — MambaManager.reachable_block_mask (score 42, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/single_type_kv_cache_manager.py:1359` (`MambaManager.reachable_block_mask`, config_block)
- **Description:** Mamba sparse state-snapshot retention mask that selects segment-boundary and reachable-boundary recurrent-state blocks to register in the prefix cache.
- **Current approach:** Returns dense caching when retention is disabled, otherwise builds a boolean mask, marks one state per retention segment, and always marks replay/shared-prefix boundary states.
- **Why this impact / rank:** Mamba retention_interval affects long-context reuse for Mamba models; retention improvements matter but are scoped to Mamba/hybrid deployments rather than the default agentic serving stack. — Scoped to Mamba models, but recurrent-state retention strongly affects reuse for long multi-turn histories. Better snapshot selection improves prefix hit rate and reduces TTFT on later agent turns.
- **Proposals (5):**
  - Workflow-aware boundary prioritization for Mamba sparse retention _(source: research_finding)_
  - Score Mamba state retention by predicted-reuse × compute-savings-per-byte (Marconi-style) _(source: research_finding)_
  - Add connector-aware commit policy to Mamba retention mask selection _(source: research_finding)_
  - Co-residency-gated Mamba retention: only keep boundary states whose attention block is (or will remain) cached _(source: agent_knowledge)_
  - Use a sparse retained-index representation for Mamba retention masks _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0022](../modules/vllm_v1_core/MambaManager.reachable_block_mask__cand-vllm_v1_core-0022.md)

### 117. `cand-vllm_distributed_kv_transfer-0004` — _PUSH_WRITER_POLL_INTERVAL_MS (score 42, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:70` (`_PUSH_WRITER_POLL_INTERVAL_MS`, config_block)
- **Description:** Fixed 1.0 ms active-state poll cadence for the nixl-push-writer thread while unmatched push state exists.
- **Current approach:** A static sleep interval is used regardless of queue depth, recent notification rate, or number of pending PUSH_REG/finished-block matches. The writer is event-woken from idle but self-polls at this cadence while active.
- **Why this impact / rank:** _PUSH_WRITER_POLL_INTERVAL_MS is 1 ms and adds a bounded per-turn wait in push-mode NIXL; notification-driven wakeups remove that floor for turn-2 TTFT, but the absolute magnitude is small. — In push mode, D-side turn-2 TTFT can include this polling delay before registration and WRITE completion are matched; the bound is small but paid repeatedly under multi-turn reuse.
- **Proposals (4):**
  - Replace fixed 1 ms active-state poll with NIXL notification-driven wakeups for the push writer _(source: research_finding)_
  - Drain-to-empty per poll iteration to amortize the 1 ms cadence cost _(source: research_finding)_
  - Deadline-aware timed wait using per-request lease expirations instead of fixed 1 ms cadence _(source: agent_knowledge)_
  - Add a bounded fast-path poll window before backing off _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0004](../modules/vllm_distributed_kv_transfer/_PUSH_WRITER_POLL_INTERVAL_MS__cand-vllm_distributed_kv_transfer-0004.md)

### 118. `cand-vllm_v1_worker-0031` — MambaHybridModelState.preprocess_state (score 39, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/model_states/mamba_hybrid.py:160` (`MambaHybridModelState.preprocess_state`, method)
- **Description:** New GPU runner Mamba align-mode pre-forward state migration across block boundaries.
- **Current approach:** Runs a per-step preprocess_mamba_align_fused_kernel launch with fixed block=256 for all requests and then invokes ctx.run_fused_precopy, relying on GPU fast-exit when no state copy is needed.
- **Why this impact / rank:** MambaHybridModelState.preprocess_state amortization and fused-precopy fusion save decode-step launches, but the benefit is again limited to hybrid Mamba workloads and largely overlaps with the earlier mamba_utils candidates. — The impact is specific to hybrid Mamba/SSM models, but those workloads pay this on every decode step; avoiding no-op launches or improving tiling reduces median TPOT in long agent loops.
- **Proposals (4):**
  - Amortize preprocess_state launches across multi-step decode windows _(source: research_finding)_
  - Fold align-mode precopy into SSM update via separate src/dst state indices _(source: research_finding)_
  - Fuse preprocess and precopy Mamba-align kernels into a single launch _(source: agent_knowledge)_
  - Specialize the align preprocess launch for active request count buckets _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0031](../modules/vllm_v1_worker/MambaHybridModelState.preprocess_state__cand-vllm_v1_worker-0031.md)

### 119. `cand-vllm_v1_executor-0002` — FutureWrapper.result (score 42, impact: medium)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/multiproc_executor.py:88` (`FutureWrapper.result`, method)
- **Description:** FIFO synchronization point for non-blocking multiprocessing futures; resolving one future drains all earlier futures from the shared deque by calling _wait_for_response.
- **Current approach:** result(timeout=None) rejects timeouts, then repeatedly pops the oldest FutureWrapper from futures_queue and calls _wait_for_response until this future is done. _wait_for_response synchronously calls aggregate(get_response()) and stores either result or exception on the Future.
- **Why this impact / rank:** FutureWrapper.result drain loop only shows up for callers using non_block futures; polling all rank queues in one pass helps async decode sync but scope is narrower than collective_rpc. — The loop is on the async decode synchronization path. Improvements reduce median TPOT most when multiple steps are pipelined; scope is narrower than collective_rpc because it only appears when callers use non_block futures.
- **Proposals (3):**
  - Poll all per-rank response queues in one pass inside FutureWrapper drains _(source: research_finding)_
  - Overlap aggregate() with next future's get_response() in the drain loop via a single-slot prefetcher _(source: agent_knowledge)_
  - Move FIFO response draining to a dedicated completion thread _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0002](../modules/vllm_v1_executor/FutureWrapper.result__cand-vllm_v1_executor-0002.md)

### 120. `cand-vllm_v1_attention-0013` — fill_mm_prefix_query_ranges (score 40, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/utils.py:78` (`fill_mm_prefix_query_ranges`, function)
- **Description:** Builds per-query multimodal prefix range rows for FA4 multimodal prefill masking.
- **Current approach:** Loops in Python over request ranges, computes covered scheduled-token spans from CPU tensors, fills the staging array with -1, and writes each span with numpy slice assignments.
- **Why this impact / rank:** fill_mm_prefix_query_ranges runs on multimodal prefill/extend steps in the Triton backend; caching resolved spans avoids the O(num_actual_tokens) fill, but the workload trigger is narrow. — Image or multimodal agent turns can repeatedly hit this prefill/extend path. Vectorizing the range fill or moving it to device reduces per-step CPU preparation and can improve TTFT for multimodal turns.
- **Proposals (2):**
  - Cache resolved mm-prefix spans per request and use dirty-row tracking to skip the O(num_actual_tokens) -1 sentinel fill _(source: agent_knowledge)_
  - Filter mm-prefix ranges to active prefill chunks before FA4 staging _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0013](../modules/vllm_v1_attention/fill_mm_prefix_query_ranges__cand-vllm_v1_attention-0013.md)

### 121. `cand-vllm_multimodal-0017` — PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host (score 37, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/video.py:795` (`PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host`, method)
- **Description:** Decodes selected frames with PyNvVideoCodec, wraps DLPack outputs as tensors, converts to NHWC, copies to pinned CPU memory, synchronizes, and returns a numpy array.
- **Current approach:** Builds torch_frames with a Python list comprehension, stacks them into device_frames, converts layout, allocates a fresh pinned CPU tensor with torch.empty, copies non-blocking, then calls stream.synchronize before exposing host_frames.numpy().
- **Why this impact / rank:** PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host lowers NVDEC decode+D2H latency via pinned buffer pooling and stream pipelining, but only benefits video-ingesting multi-turn workloads and is off-path for text-only agent flows. — On NVDEC paths, decode plus D2H transfer is the dominant video ingress cost. Reducing pinned allocation and synchronization lowers bytes-to-frames latency and TTFT.
- **Proposals (4):**
  - Adopt PyNvVideoCodec ThreadedDecoder and memory demuxing in _decode_to_pinned_host _(source: research_finding)_
  - Pool pinned host buffers and stream-pipeline D2H in _decode_to_pinned_host _(source: research_finding)_
  - Eliminate on-device stack+contiguous by copying DLPack frames directly into a pre-allocated pinned NHWC host buffer _(source: agent_knowledge)_
  - Batch DLPack conversion with torch.utils.dlpack.from_dlpack to avoid transient Python tensor retention _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0017](../modules/vllm_multimodal/PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host__cand-vllm_multimodal-0017.md)

### 122. `cand-vllm_v1_core-0003` — SlidingWindowManager.find_longest_cache_hit (score 42, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/single_type_kv_cache_manager.py:897` (`SlidingWindowManager.find_longest_cache_hit`, method)
- **Description:** Sliding-window prefix-cache lookup that searches right-to-left for enough consecutive cached blocks to satisfy the attention window.
- **Current approach:** Allocates null-filled per-group lists of size max_num_blocks, scans candidate blocks in reverse via block_pool.get_cached_block, resets num_contiguous_blocks on each miss, trims trailing blocks after a valid run, and applies EAGLE/alignment trimming. An in-source TODO notes that misses could skip by sliding_window_contiguous_blocks.
- **Why this impact / rank:** SlidingWindowManager.find_longest_cache_hit reverse-scan is O(max_num_blocks) with a known asymptotic win to O(max_num_blocks/K + K); helps SWA admission TTFT but confined to SWA/SWA-hybrid models. — Applies only to sliding-window models, but those pay this scan at admission. Cold or low-hit first turns are common, so reducing reverse-scan work moves TTFT for SWA and SWA-hybrid workloads.
- **Proposals (2):**
  - Workload-aware start-index hint short-circuits reverse SWA prefix-cache scan _(source: agent_knowledge)_
  - Scan only alignment-eligible SWA tail blocks before validating the run _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0003](../modules/vllm_v1_core/SlidingWindowManager.find_longest_cache_hit__cand-vllm_v1_core-0003.md)

### 123. `cand-vllm_v1_kv_offload-0010` — AsyncLookupManager._worker (score 40, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/async_lookup.py:214` (`AsyncLookupManager._worker`, method)
- **Description:** Background lookup worker groups a flushed lookup batch by request and calls the tier-specific batch_lookup.
- **Current approach:** A single worker thread consumes one full-step queue item at a time, groups with a fresh dict keyed by req_id, calls batch_lookup once per request, and converts exceptions into all-False results for that request.
- **Why this impact / rank:** AsyncLookupManager batching/dedup shortens the retry window before promotion, tightening TTFT tail; median gain is bounded by tier backend latency and depends on secondary-hit frequency. — Faster async secondary lookups shorten the retry window before promotion can start. This improves TTFT tail and can move median TTFT when secondary-tier hits are common, but the tier backend's own lookup latency bounds the gain.
- **Proposals (2):**
  - Deduplicate keys across the flushed batch before invoking batch_lookup _(source: agent_knowledge)_
  - Process short request lookup groups first and publish each group immediately _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0010](../modules/vllm_v1_kv_offload/AsyncLookupManager._worker__cand-vllm_v1_kv_offload-0010.md)

### 124. `cand-vllm_v1_executor-0016` — RayExecutorV2._init_executor (score 40, impact: medium)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/ray_executor_v2.py:304` (`RayExecutorV2._init_executor`, method)
- **Description:** RayExecutorV2 actor placement, MessageQueue topology, GPU mapping, worker initialization, and response-queue setup for the MQ-based Ray backend.
- **Current approach:** The method chooses bundle assignments from VLLM_RAY_BUNDLE_INDICES or get_bundles_sorted_by_node, counts driver-node-local workers to size the broadcast MessageQueue local readers, creates one RayWorkerProc actor per bundle, gathers physical GPU IDs with ray.get, initializes workers with local ranks and assigned_physical_gpu_ids, collects response MessageQueue handles, starts each actor's run loo…
- **Why this impact / rank:** RayExecutorV2._init_executor is mostly startup code affecting TTFT only during bring-up; the MessageQueue locality does bleed into TPOT for multi-node MQ deployments, but single-node runs see little movement. — The method is initialization code, but it fixes the MessageQueue locality and worker placement used for every RayExecutorV2 decode step. Optimizing it can reduce median TPOT in multi-node MQ-backed Ray deployments and reduce TTFT during startup; impact is medium because single-node or already-local…
- **Proposals (5):**
  - Precompile Ray actor graph for steady-state token loop in RayExecutorV2 _(source: research_finding)_
  - Topology-aware bundle-to-rank assignment: keep TP intra-node, PP across nodes _(source: research_finding)_
  - Hybrid intra-node shm / inter-node Ray RPC broadcast for RayExecutorV2 _(source: research_finding)_
  - Pipeline actor bring-up: overlap GPU discovery, worker init, and run() startup to eliminate straggler barriers _(source: agent_knowledge)_
  - Cache and reuse per-node physical GPU discovery across RayExecutorV2 restarts _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0016](../modules/vllm_v1_executor/RayExecutorV2._init_executor__cand-vllm_v1_executor-0016.md)

### 125. `cand-vllm_multimodal-0002` — MultiModalHasher.iter_item_to_bytes/hash_kwargs (score 38, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/hasher.py:145` (`MultiModalHasher.iter_item_to_bytes/hash_kwargs`, region)
- **Description:** Walks nested list/tuple/dict kwargs, encodes path keys, serializes leaves, and feeds each yielded byte chunk into the configured hash object.
- **Current approach:** Uses recursive Python generators with f-string path construction such as f'{key}.{i}' and f'{key}.{k}'. hash_kwargs loops over every yielded chunk and calls hasher.update one chunk at a time, including many tiny key bytes.
- **Why this impact / rank:** MultiModalHasher runs on cache-key construction; batching hasher updates and iterative traversal shave overhead but savings are far smaller than avoiding tensor copies and only visible on cache hits. — This is on every multimodal cache-key construction. The savings are smaller than avoiding image/tensor copies, but they reduce TTFT on cache hits and on agentic turns with many small metadata fields.
- **Proposals (2):**
  - Batch small metadata updates before entering the BLAKE3 C hasher _(source: research_finding)_
  - Fold traversal into hasher: replace recursive generators with a direct-hash walker plus interned index-key table _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0002](../modules/vllm_multimodal/MultiModalHasher.iter_item_to_bytes_hash_kwargs__cand-vllm_multimodal-0002.md)

### 126. `cand-vllm_multimodal-0018` — @VIDEO_LOADER_REGISTRY.register("opencv") (score 36, impact: high)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/video.py:986` (`@VIDEO_LOADER_REGISTRY.register("opencv")`, plugin_seam)
- **Description:** Registers the default VideoBackend implementation in VIDEO_LOADER_REGISTRY; sibling decorator registrations add model-specific video sampling backends under names such as pynvvideocodec, qwen2_vl, qwen3_vl, glm46v, glmga, molmo2, nemotron_vl, and openpangu.
- **Current approach:** The interface is VideoLoader.compute_frames_index_to_sample and VideoLoader.load_bytes in vllm/multimodal/video.py. Reference implementations include VideoBackend in vllm/multimodal/video.py and Qwen2VLVideoBackend/Qwen3VLVideoBackend in the same file. Runtime selection comes from --media-io-kwargs, VLLM_VIDEO_LOADER_BACKEND, or processor-name mappings registered through VIDEO_LOADER_REGISTRY.reg…
- **Why this impact / rank:** Alternative VIDEO_LOADER_REGISTRY backends can cut per-clip encoder FLOPs by sampling fewer frames, potentially large TTFT wins for long-clip video prompts, but the objective's workload hint doesn't call out video ingestion. — Sampled frame count directly drives video encoder FLOPs and prefill work. Better registered samplers can cut TTFT substantially for long clips while preserving answer quality.
- **Proposals (3):**
  - Add a memory-demux + ThreadedDecoder PyNvVideoCodec sibling backend to VIDEO_LOADER_REGISTRY _(source: research_finding)_
  - Register a content-hash memoizing sibling backend for repeated video attachments in multi-turn sessions _(source: agent_knowledge)_
  - Add an index-only predecode sampler wrapper to avoid full-video decode for sparse frame requests _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0018](../modules/vllm_multimodal/_VIDEO_LOADER_REGISTRY.register__opencv____cand-vllm_multimodal-0018.md)

### 127. `cand-vllm_distributed_kv_transfer-0008` — OffloadingScheduler._build_store_jobs (score 38, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1248` (`OffloadingScheduler._build_store_jobs`, method)
- **Description:** Builds store transfer jobs each scheduler step by scanning scheduled and finished requests, computing offloadable tokens, filtering reachable chunks, calling manager.prepare_store, and packing GPU source block metadata.
- **Current approach:** Nested Python loops over requests, KV groups, chunks, and blocks perform list slicing, reachability checks, set membership tests, event recording, and GPULoadStoreSpec packing. prepare_store is called per request rather than batched across the scheduler step.
- **Why this impact / rank:** OffloadingScheduler._build_store_jobs adds per-step CPU cost that scales with active requests; batching prepare_store trims scheduler overhead but the win is small vs runner-side hot paths. — Adds CPU time to every scheduler step and therefore TPOT; with many concurrent agentic sessions, store bookkeeping can also delay scheduling of TTFT-critical loads.
- **Proposals (2):**
  - Cross-request batched prepare_store with single-pass reachability + block-list build _(source: agent_knowledge)_
  - Defer active-request stores on load-heavy scheduler steps _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0008](../modules/vllm_distributed_kv_transfer/OffloadingScheduler._build_store_jobs__cand-vllm_distributed_kv_transfer-0008.md)

### 128. `cand-vllm_multimodal-0020` — BaseMultiModalField.reduce_data (score 40, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/inputs.py:462` (`BaseMultiModalField.reduce_data`, method)
- **Description:** Common wrapper for all multimodal field reductions: validates field types, applies keep_on_cpu/pin_memory policy, collects per-element data, calls the field-specific reducer, and optionally moves the nested result to the target device.
- **Current approach:** Builds a list of field types and a set to reject mixed field classes, builds another list of elem.data, may traverse the nested batch with _nested_tensors_are_cpu, calls _reduce_data, then maps the output through _nested_tensors_h2d for device transfer.
- **Why this impact / rank:** BaseMultiModalField.reduce_data does repeated list/set/traversal work per field per grouped prefill; a fused single-pass path is a clean CPU win but small in absolute TTFT terms and multimodal-only. — This wrapper runs once per multimodal field in every grouped prefill. Cutting repeated list/set/traversal work reduces CPU prefill preparation and therefore TTFT, especially with many small fields per agentic turn.
- **Proposals (2):**
  - Collapse reduce_data wrapper into a single-pass fast path with skip-on-noop device/pin checks _(source: agent_knowledge)_
  - Add a reducer no-op path for singleton multimodal fields _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0020](../modules/vllm_multimodal/BaseMultiModalField.reduce_data__cand-vllm_multimodal-0020.md)

### 129. `cand-vllm_multimodal-0022` — MultiModalKwargsItems.from_hf_inputs (score 37, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/inputs.py:964` (`MultiModalKwargsItems.from_hf_inputs`, method)
- **Description:** Converts a Hugging Face BatchFeature plus field configs into per-modality sequences of MultiModalKwargsItem objects used by caching and batching.
- **Current approach:** Builds elems_by_key and keys_by_modality, then for each modality builds elems_in_modality, a batch_sizes dict, a set of sizes, and a nested list/dict comprehension that allocates one MultiModalKwargsItem per item.
- **Why this impact / rank:** from_hf_inputs churns Python objects per modality item; transposed assembly reduces cache-miss TTFT for multimodal prompts but is narrow and constant-factor. — The cost scales with number of modalities, fields, and media items. Reducing object churn lowers cache-miss TTFT for multi-image, audio, and video prompts in agentic workloads.
- **Proposals (2):**
  - Transpose per-item assembly with zip(*values) and drop the batch_sizes dict on the fast path _(source: agent_knowledge)_
  - Bypass UserDict's copying constructor for freshly built item dicts _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0022](../modules/vllm_multimodal/MultiModalKwargsItems.from_hf_inputs__cand-vllm_multimodal-0022.md)

### 130. `cand-vllm_v1_engine-0013` — LogprobsProcessor._update_sample_logprobs (score 38, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/logprobs.py:69` (`LogprobsProcessor._update_sample_logprobs`, method)
- **Description:** Sample-logprobs converter that appends generated-token logprob alternatives and cumulative logprob state for each produced token position.
- **Current approach:** For each generated logprob position, converts rank/logprobs/token_ids tensors or arrays to Python lists, optionally tokenizes ids, recomputes sampled context ids, verifies UTF-8 correction, updates cumulative_logprob, and appends one Logprob container.
- **Why this impact / rank:** LogprobsProcessor per-token conversion cost is real but only paid when logprobs are enabled; TPOT gain is conditional on request-level flags rather than baseline agentic traffic. — When logprobs are enabled for agentic telemetry or ranking, this work is paid on every generated token; reducing Python conversion and tokenizer overhead can lower median TPOT for those requests.
- **Proposals (2):**
  - Batch `.tolist()` across all sample-logprob positions and cache sampled-context IDs incrementally _(source: agent_knowledge)_
  - Cache per-request decoded token strings for sample logprobs _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0013](../modules/vllm_v1_engine/LogprobsProcessor._update_sample_logprobs__cand-vllm_v1_engine-0013.md)

### 131. `cand-vllm_distributed_kv_transfer-0016` — MoRIIOConnectorWorker._read_blocks (score 34, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:2541` (`MoRIIOConnectorWorker._read_blocks`, method)
- **Description:** Posts MoRIIO READ operations for a request by iterating every registered layer, computing offsets, synchronously calling read_remote_data, and retrying SQ-full responses with exponential sleep backoff.
- **Current approach:** Layer transfers are submitted sequentially. Each iteration recomputes list(self.layer_name_to_local_kv_cache_metadata.keys()).index(layer_name), calls _compute_block_transfer_offsets, then blocks in a while True retry loop with time.sleep backoff up to 50 ms before recording status under moriio_wrapper.lock.
- **Why this impact / rank:** MoRIIOConnectorWorker._read_blocks batching helps remote-prefill TTFT on MoRIIO READ mode; impact is real but the connector is niche, and NIXL-based paths dominate most disaggregated deployments. — MoRIIO READ mode is connector-specific, but when enabled this sequential per-layer posting is on remote-prefill TTFT and can delay layer barriers during decode.
- **Proposals (4):**
  - Batch MoRIIO READ descriptors per request with topology-aware slicing _(source: research_finding)_
  - Pipeline MoRIIO layer READs with async submission and completion polling _(source: research_finding)_
  - Layer-order-prioritized progressive READ completion for remote-prefill TTFT _(source: agent_knowledge)_
  - Prune MoRIIO READ ranges for chunked-local attention layers _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0016](../modules/vllm_distributed_kv_transfer/MoRIIOConnectorWorker._read_blocks__cand-vllm_distributed_kv_transfer-0016.md)

### 132. `cand-vllm_v1_worker-0016` — KVBlockZeroer (score 40, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/utils.py:47` (`KVBlockZeroer`, config_block)
- **Description:** Builds segment metadata and launch parameters for the KV block zeroing Triton kernel, then launches zeroing for newly allocated blocks.
- **Current approach:** Derives a single blk_size and MAX_CHUNKS from all segment page sizes, launches n_blocks x n_segs x max_chunks programs, and relies on an early return when a segment has fewer chunks.
- **Why this impact / rank:** KVBlockZeroer segment-aware tiling shaves wasted programs during new-turn allocation; helps TTFT/tail when page sizes vary across cache segments, but allocation-time cost isn't a dominant term for steady serving. — Fresh KV blocks are allocated during new turns and prefills; segment-aware tiling reduces wasted programs and improves TTFT/tail latency when page sizes vary across KV cache segments.
- **Proposals (2):**
  - Replace rectangular zero-kernel grid with a precomputed flat work-item table sized to exact per-segment chunk counts _(source: agent_knowledge)_
  - Skip KV zeroing for CUDA allocations that are already guaranteed clean _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0016](../modules/vllm_v1_worker/KVBlockZeroer__cand-vllm_v1_worker-0016.md)

### 133. `cand-vllm_v1_worker-0022` — copy_kv_cache_blocks_inplace (score 36, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/utils.py:565` (`copy_kv_cache_blocks_inplace`, function)
- **Description:** Copies logical KV cache blocks in-place across backing storage tensors for prefix-cache promotion/copy-on-write.
- **Current approach:** Builds a unique storage list, transfers copy indices to GPU, then performs one advanced-index assignment per storage tensor on a uint8 view.
- **Why this impact / rank:** copy_kv_cache_blocks_inplace fusion saves launches during promotion/copy, but the copy path isn't every step and the raw cost is small relative to input prep and attention. — Multi-turn agents reuse and promote prefixes frequently; fusing copies across storages reduces repeated launches and improves TPOT when many KV storage tensors are present.
- **Proposals (2):**
  - Fuse per-storage KV block copies into a single CUDA-graph-captured copy kernel _(source: agent_knowledge)_
  - Add overlap-safe ordering for in-place KV block copies _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0022](../modules/vllm_v1_worker/copy_kv_cache_blocks_inplace__cand-vllm_v1_worker-0022.md)

### 134. `cand-vllm_v1_kv_offload-0011` — DualQueueThreadPool._worker (score 36, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/fs/thread_pool.py:153` (`DualQueueThreadPool._worker`, method)
- **Description:** File-system tier worker loop services load/store queues with static per-thread priority and shared condition signaling.
- **Current approach:** Each worker has fixed load-priority or store-priority behavior, waits on one condition variable, pops from its primary queue if available, otherwise falls back to the secondary queue. There is no adaptive rebalancing, batching, or store backpressure under load contention.
- **Why this impact / rank:** DualQueueThreadPool._worker governs FS-tier read/write balance; slack-aware scheduling helps load-burst TTFT but the FS bandwidth ceiling bounds gains and the path is optional. — For FS-backed secondary tiers, load latency is on the promotion path while stores compete for the same workers. Adaptive scheduling can reduce load-burst latency and TTFT variance, with gains bounded by filesystem bandwidth.
- **Proposals (4):**
  - Slack-aware worker scheduling with batched task pickup for FS tier _(source: research_finding)_
  - Batch adjacent I/O tasks per worker pop to amortize per-submission overhead _(source: research_finding)_
  - Split shared condition per queue with hysteresis-guarded cross-queue stealing _(source: agent_knowledge)_
  - Apply OS I/O priority hints per worker class _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0011](../modules/vllm_v1_kv_offload/DualQueueThreadPool._worker__cand-vllm_v1_kv_offload-0011.md)

### 135. `cand-vllm_v1_kv_offload-0007` — compute_sub_block_ptrs (score 35, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/cpu/gpu_worker.py:73` (`compute_sub_block_ptrs`, function)
- **Description:** Expands block IDs into raw byte pointers for transfer descriptors, including sub-block fanout and partial-block skip handling.
- **Current approach:** Fast path handles blocks_per_chunk == 1. General path allocates sub_offsets, materializes a full (num_blocks, blocks_per_chunk) pointer matrix, flattens it, then slices by skip_count and output length.
- **Why this impact / rank:** compute_sub_block_ptrs eliminates small per-call materialization inside transfer submission; ceiling bounded by surrounding scheduling and memcpy cost, so effect on TTFT is minor. — This fixed overhead is paid twice per group/data-ref pair inside transfer submission. It can noticeably reduce small-promotion CPU overhead, though the ceiling is bounded by the surrounding transfer scheduling and memory copy cost.
- **Proposals (2):**
  - Eliminate matrix materialization in compute_sub_block_ptrs via direct strided writes with cached invariants _(source: agent_knowledge)_
  - Reuse per-group relative pointer offsets across data refs _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0007](../modules/vllm_v1_kv_offload/compute_sub_block_ptrs__cand-vllm_v1_kv_offload-0007.md)

### 136. `cand-vllm_v1_executor-0003` — WorkerProc.worker_busy_loop (score 34, impact: medium)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/multiproc_executor.py:1008` (`WorkerProc.worker_busy_loop`, method)
- **Description:** Worker-side dispatcher loop for multiprocessing and RayExecutorV2 workers: receives broadcast RPC tuples, resolves the target method, invokes it, and forwards output or exceptions to the response path.
- **Current approach:** Every dequeued message performs an isinstance dispatch, does getattr(self.worker, method) for string methods or cloudpickle.loads plus partial for bytes methods, calls func(*args, **kwargs), then conditionally emits output for the requested rank. Exceptions are logged, annotated, stringified for transport, and routed through handle_output when this rank is expected to reply.
- **Why this impact / rank:** worker_busy_loop dispatch overhead is real per step but small relative to model execution and queue transport; specializing the dispatch gives a thin TPOT gain concentrated in short-generation regimes. — The cost is paid on every worker and scales with TP/PP world size, so it can move median TPOT in large or short-generation deployments. The rating is medium because dispatch overhead is still smaller than model execution and message-queue transport.
- **Proposals (2):**
  - Precompile worker dispatch path to eliminate per-step RPC overhead _(source: research_finding)_
  - Hoist hot-loop attribute lookups to local variables in worker_busy_loop _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0003](../modules/vllm_v1_executor/WorkerProc.worker_busy_loop__cand-vllm_v1_executor-0003.md)

### 137. `cand-vllm_v1_kv_offload-0002` — ARCCachePolicy.touch (score 33, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/cpu/policies/arc.py:75` (`ARCCachePolicy.touch`, method)
- **Description:** ARC touch handling promotes T1 hits, refreshes T2 hits, and adapts target_t1_size on B1/B2 ghost hits.
- **Current approach:** Materializes list(keys) and iterates in reverse. Ready T1 hits move to T2, T2 hits move to MRU, and ghost hits adjust target_t1_size using max(1, len(B_other) / len(B_self)) without smoothing or momentum.
- **Why this impact / rank:** ARCCachePolicy.touch tuning influences future eviction targets and therefore primary-cache hit rate, but the connection to median TTFT/TPOT is indirect and capped by downstream eviction decisions and steady-state hit rate. — Touch-time adaptation influences the future eviction target and therefore primary-cache hit rate, especially during bursty agent turns. It is meaningful, but downstream eviction decisions cap the immediate TTFT/TPOT effect.
- **Proposals (3):**
  - Bias ARC touch promotion and ghost-hit adaptation with agent-supplied workflow priority _(source: research_finding)_
  - Dampen ARC ghost-hit adaptation per touch batch with EMA-smoothed target_t1_size _(source: agent_knowledge)_
  - Avoid copying reversible touch key batches _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0002](../modules/vllm_v1_kv_offload/ARCCachePolicy.touch__cand-vllm_v1_kv_offload-0002.md)

### 138. `cand-vllm_distributed_kv_transfer-0017` — MooncakeStoreWorker.lookup (score 40, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:1760` (`MooncakeStoreWorker.lookup`, method)
- **Description:** Mooncake store prefix lookup expands each candidate block hash across KV groups and rank namespaces, sends one batch_is_exist request, reduces results to an exists_set, and asks the coordinator for the longest usable hit.
- **Current approach:** Candidate keys are materialized as Python strings with PoolKey.build_key_string inside nested loops over groups, chunks, and key prefixes. Result reduction builds a Python set of present (group, hash) pairs, then may call coord.find_longest_cache_hit twice when the hit reaches the request end.
- **Why this impact / rank:** MooncakeStoreWorker.lookup adds Python key expansion cost on every prefix-hit lookup; skipping redundant find_longest_cache_hit passes trims TTFT for Mooncake-backed multi-turn prompts, but backend-specific and bounded by RPC time. — Lookup latency contributes directly to TTFT for Mooncake store prefix hits; long multi-turn prompts multiply groups, chunks, and rank namespaces, making Python key expansion a visible CPU cost.
- **Proposals (2):**
  - Skip the second find_longest_cache_hit when the initial hit already spans the request; precompute per-group key-prefix byte templates and d… _(source: agent_knowledge)_
  - Add a bounded positive-existence cache to skip repeated Mooncake lookups for already-confirmed prefix blocks _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0017](../modules/vllm_distributed_kv_transfer/MooncakeStoreWorker.lookup__cand-vllm_distributed_kv_transfer-0017.md)

### 139. `cand-vllm_multimodal-0003` — nested_tensors_equal (score 36, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/inputs.py:232` (`nested_tensors_equal`, function)
- **Description:** Recursively compares NestedTensors structures used by batching and shared-field compatibility checks.
- **Current approach:** Builds a check_dtype lambda on every call, has symmetric duplicated branches for tensor/list/tuple cases, and reaches torch.equal for tensor leaves before explicit dtype/device/shape or object-identity short-circuits.
- **Why this impact / rank:** nested_tensors_equal is called during multimodal batching; storage-identity short-circuits avoid CUDA-syncing torch.equal calls, but effect is confined to prefill batching with repeated shared fields. — Shared multimodal fields are compared during batching on every prefill. Avoiding elementwise tensor comparisons, especially CUDA comparisons that can synchronize, reduces TTFT for batched or broadcast multimodal inputs.
- **Proposals (2):**
  - Short-circuit nested_tensors_equal via storage-view identity and per-call memoization _(source: agent_knowledge)_
  - Replace recursive container traversal with an iterative stack walk _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0003](../modules/vllm_multimodal/nested_tensors_equal__cand-vllm_multimodal-0003.md)

### 140. `cand-vllm_v1_worker-0021` — WorkspaceManager._ensure_workspace_size (score 33, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/workspace.py:119` (`WorkspaceManager._ensure_workspace_size`, region)
- **Description:** Lazy per-ubatch GPU scratch workspace allocation and resize policy.
- **Current approach:** When a larger request arrives, drops the old workspace, calls torch.accelerator.empty_cache(), and allocates exactly required_bytes for the requesting ubatch.
- **Why this impact / rank:** WorkspaceManager growth policy is a tail-latency lever (avoiding empty_cache reallocation stalls on long tool-response prefills), not a median TPOT driver in steady state. — This is a tail-latency and occasional TTFT lever rather than steady-state TPOT; exponential high-water growth can avoid repeated allocator stalls for long tool-response prefills.
- **Proposals (2):**
  - Geometric high-water growth with lazy empty_cache to eliminate re-allocation stalls on the hot path _(source: agent_knowledge)_
  - Consolidate ubatch scratch buffers into one growable workspace slab _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0021](../modules/vllm_v1_worker/WorkspaceManager._ensure_workspace_size__cand-vllm_v1_worker-0021.md)

### 141. `cand-vllm_distributed_kv_transfer-0010` — KVConnectorFactory.register_connector registrations (score 40, impact: medium)

- **Module:** vllm/distributed/kv_transfer
- **Location:** `vllm/distributed/kv_transfer/kv_connector/factory.py:152` (`KVConnectorFactory.register_connector registrations`, plugin_seam)
- **Description:** Registry maps external connector-name strings to module/class pairs at import time, and KVConnectorFactory.create_connector lazily instantiates the implementation selected by KVTransferConfig.
- **Current approach:** The interface is KVConnectorBase_V1 in vllm/distributed/kv_transfer/kv_connector/v1/base.py, reached through KVConnectorBase compatibility in vllm/distributed/kv_transfer/kv_connector/base.py. Existing implementations include NixlConnector in vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py, OffloadingConnector in vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py,…
- **Why this impact / rank:** The KVConnectorFactory registry itself is cold; rf_count=9 reflects proposal breadth but the seam is a plumbing point, not a hot loop. Direct TTFT/TPOT movement depends entirely on which connector variant ships behind it, so treat this as an enabler rather than a mover. — The registry is not hot, but it enables low-risk replacement of connector routing, batching, or transfer scheduling logic that can move turn-2 TTFT without invasive scheduler changes.
- **Proposals (11):**
  - Add a scored KV-routing MultiConnector selected via the factory registry _(source: research_finding)_
  - Power-of-two connector selection for cache-affinity load balancing _(source: research_finding)_
  - Add NIXL-optimized connector variant with batched progress and cached remote metadata _(source: research_finding)_
  - Register async DEALER/ROUTER NIXL connector variant to pipeline handshake and metadata round-trips _(source: research_finding)_
  - Register a batched, topology-aware scatter-gather KV connector variant _(source: research_finding)_
  - Add a CacheBlend-style selective-recompute connector that overlaps KV load with partial recomputation _(source: research_finding)_
  - Add a CacheGen-style compressed KV connector as a new plugin for slow-tier reuse _(source: research_finding)_
  - Register a RadixAttention-inspired prefix-tree connector for multi-turn agentic KV reuse _(source: research_finding)_
  - Add latency-predicting connector router to KVConnectorFactory for multi-connector selection _(source: research_finding)_
  - Register a hedged fetch-vs-recompute racing connector for tail-TTFT bounding _(source: agent_knowledge)_
  - Register a TPOT-aware async writeback connector _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_distributed_kv_transfer-0010](../modules/vllm_distributed_kv_transfer/KVConnectorFactory.register_connector_registrations__cand-vllm_distributed_kv_transfer-0010.md)

### 142. `cand-vllm_multimodal-0026` — PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec (score 32, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/video.py:842` (`PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec`, method)
- **Description:** Stages raw video bytes for PyNvVideoCodec, reads metadata, computes sampled frame indices, gates raw decoded frame memory, decodes to pinned host memory, and returns frames plus valid indices.
- **Current approach:** Creates a temporary .mp4 file with tempfile.mkstemp, writes the whole input bytes to it, uses that path for metadata and decode, computes raw_frame_bytes from sampled frames, optionally acquires the global GPU IPC pool, and deletes the temp file in finally.
- **Why this impact / rank:** Removing per-request temp-file staging in PyNvVideoCodec avoidance is a clear TTFT win for repeated video ingestion, but video is not the described workload, so the score reflects narrow applicability. — For NVDEC video, byte staging happens before any frame reaches the processor. Removing whole-file temp I/O reduces video TTFT, especially for long clips and multi-turn workloads that repeatedly submit video bytes.
- **Proposals (3):**
  - Replace per-request temp-file staging with PyNvVideoCodec memory demuxing _(source: research_finding)_
  - Fuse metadata + decode into a single decoder-slot borrow, with content-hash metadata cache for multi-turn reuse _(source: agent_knowledge)_
  - Stream PyNvVideoCodec frames into pinned host memory without GPU batch stacking _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0026](../modules/vllm_multimodal/PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec__cand-vllm_multimodal-0026.md)

### 143. `cand-vllm_v1_worker-0014` — stage_postprocess_inputs_to_gpu (score 32, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/mamba_utils.py:1246` (`stage_postprocess_inputs_to_gpu`, function)
- **Description:** Stages per-request scalar inputs for the fused legacy Mamba postprocess kernel after forward.
- **Current approach:** Loops over num_reqs, performs several dict lookups per request, writes four pinned NumPy buffers, and then issues four copy_to_gpu calls.
- **Why this impact / rank:** Mamba postprocess H2D coalescing helps hybrid spec-decode workloads; TPOT gain is real but confined to that model/decoding path. — The cost is specific to hybrid spec-decode runs; collapsing four per-step copies into one packed buffer reduces launch overhead and improves TPOT for that workload slice.
- **Proposals (2):**
  - Reuse runner's already-staged num_scheduled/num_computed GPU tensors and pack the remaining two per-request values into a single H2D copy _(source: agent_knowledge)_
  - Overlap postprocess input staging copies with the model forward pass _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0014](../modules/vllm_v1_worker/stage_postprocess_inputs_to_gpu__cand-vllm_v1_worker-0014.md)

### 144. `cand-vllm_v1_executor-0011` — Executor.get_class (score 34, impact: medium)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/abstract.py:50` (`Executor.get_class`, plugin_seam)
- **Description:** Runtime executor implementation selection surface keyed by parallel_config.distributed_executor_backend, with an additional env-controlled Ray v2 branch and qualified-name extension path.
- **Current approach:** The interface symbol is Executor in vllm/v1/executor/abstract.py. Reference implementations include MultiprocExecutor in vllm/v1/executor/multiproc_executor.py, RayDistributedExecutor in vllm/v1/executor/ray_executor.py, RayExecutorV2 in vllm/v1/executor/ray_executor_v2.py, and UniProcExecutor in vllm/v1/executor/uniproc_executor.py. The selector is parallel_config.distributed_executor_backend wi…
- **Why this impact / rank:** Executor.get_class is a selector, not a hot path; it enables alternative executor topologies rather than performing the optimization itself, so impact is indirect and gated on downstream implementations. — The selector itself is not hot, but it is the lowest-blast-radius path to evaluate alternative control-plane and data-plane executor designs. Such variants can move median TTFT/TPOT substantially; this site is medium because it enables the optimization rather than performing it directly.
- **Proposals (6):**
  - Add Ray Compiled Graph executor with overlapped GPU communication for pipeline-parallel handoffs _(source: research_finding)_
  - Register a poll-based multiproc executor backend that drains all rank response queues in one pass _(source: research_finding)_
  - Add a topology-aware device-mesh executor backend to Executor.get_class _(source: research_finding)_
  - Register a hybrid intra-/inter-node executor variant via Executor.get_class _(source: research_finding)_
  - Register an io_uring/eventfd-driven multiproc executor that batches worker wakeups via a single kernel syscall _(source: agent_knowledge)_
  - Register a fused multiproc execute-and-sample executor backend _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0011](../modules/vllm_v1_executor/Executor.get_class__cand-vllm_v1_executor-0011.md)

### 145. `cand-vllm_multimodal-0005` — MultiModalFlatField._reduce_data (score 30, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/inputs.py:569` (`MultiModalFlatField._reduce_data`, method)
- **Description:** Concatenates or pads variable-length NestedTensors into one flat tensor for fields whose batch dimension is represented by slices.
- **Current approach:** For same non-concat shapes it allocates torch.empty and calls torch.concat(out=...). For variable non-concat shapes it computes max_sizes in Python, creates torch.zeros, builds a list of slice objects per tensor, and assigns each tensor into the output one by one.
- **Why this impact / rank:** MultiModalFlatField._reduce_data reduces per-item slice-assign overhead for variable-length audio/video features; TTFT gain only for multimodal requests, no effect on text-only agent turns. — This affects audio and video workloads with variable feature lengths. Fewer per-item copies or launches reduces multimodal prefill preparation latency, moving TTFT for those requests.
- **Proposals (3):**
  - Use torch.nested packed values+offsets and to_padded_tensor for variable-length _reduce_data _(source: research_finding)_
  - Fuse per-item slice-assign into a single torch._foreach_copy_ over narrow views, and elide zero-fill where padding is empty _(source: agent_knowledge)_
  - Coalesce contiguous same-padding runs before copying into the padded output _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0005](../modules/vllm_multimodal/MultiModalFlatField._reduce_data__cand-vllm_multimodal-0005.md)

### 146. `cand-vllm_multimodal-0024` — PyNvVideoCodecVideoBackendMixin._configure_decoder_slots/_borrow_decoder_slot (score 38, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/video.py:689` (`PyNvVideoCodecVideoBackendMixin._configure_decoder_slots/_borrow_decoder_slot`, region)
- **Description:** Configures and borrows retained PyNvVideoCodec decoder slots guarded by a process-wide condition variable.
- **Current approach:** The first _configure_decoder_slots call fixes _max_decoder_slots. _borrow_decoder_slot pops an idle slot from a LIFO list, creates a new slot while _active_decoder_slots is below the max, or waits on _decoder_slot_cond until a release appends a slot and notify wakes one waiter.
- **Why this impact / rank:** PyNvVideoCodec decoder-slot admission (LIFO wait loop, cond.notify) affects concurrent video prefill overlap; FIFO/priority queues and per-device sharding help TTFT tails but only when NVDEC contention is real, narrow slice of agentic workloads. — Under concurrent NVDEC video requests, decoder-slot admission determines whether media preprocessing queues or overlaps. Better scheduling reduces TTFT tail and can improve median TPOT by avoiding frontend GPU contention spikes.
- **Proposals (3):**
  - Adopt PyNvVideoCodec ThreadedDecoder and decoder-reuse primitives for slot admission _(source: research_finding)_
  - Priority- and deadline-aware decoder-slot admission with cancellation propagation _(source: agent_knowledge)_
  - Shard decoder-slot pools by CUDA device _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0024](../modules/vllm_multimodal/PyNvVideoCodecVideoBackendMixin._configure_decoder_slots__borrow_decoder_slot__cand-vllm_multimodal-0024.md)

### 147. `cand-vllm_v1_attention-0024` — get_kernel_options (score 30, impact: medium)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flex_attention.py:1422` (`get_kernel_options`, function)
- **Description:** Selects PyTorch FlexAttention kernel options, especially BLOCK_M and BLOCK_N, from logical block sizes, dtype, direct-build mode, and shared-memory capacity.
- **Current approach:** For direct build it forwards block_m/block_n directly. Otherwise it uses preferred_block=32 for fp32 and 64 for other dtypes, enforces divisibility with gcd, lowers both candidates when shared memory is below 144 KiB, clamps to a lower bound of 16, and forces FlexAttention through FORCE_USE_FLEX_ATTENTION.
- **Why this impact / rank:** FlexAttention BLOCK_M/BLOCK_N tuning can shift compiled kernel occupancy for TTFT prefills and repeated decode, but the effect is scoped to Flex backend and depends on device/shape. — BLOCK_M and BLOCK_N control FlexAttention kernel occupancy, memory use, and compilation choices. Better per-device and per-shape values can improve TTFT for compiled prefills and median TPOT for repeated decode on the Flex backend.
- **Proposals (1):**
  - Split BLOCK_M/BLOCK_N selection by decode vs prefill regime with shared-memory-aware sizing _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0024](../modules/vllm_v1_attention/get_kernel_options__cand-vllm_v1_attention-0024.md)

### 148. `cand-vllm_v1_worker-0018` — EncoderCudaGraphManager._execute_local (score 30, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/encoder_cudagraph.py:360` (`EncoderCudaGraphManager._execute_local`, region)
- **Description:** Greedy multimodal item packer that batches encoder items under max_batch_size and per-path token budgets before graph replay or eager fallback.
- **Current approach:** Sorts items by output tokens and first-fits them until max batch size or max path budgets are reached, without budget-cliff awareness or lookahead.
- **Why this impact / rank:** EncoderCudaGraphManager packing moves near-budget encoder batches from eager to cudagraph, cutting multimodal TTFT; scoped to mixed-modality agent turns, so limited relevance to the described text-heavy multi-turn workload. — Better packing can turn near-budget eager batches into graph hits, reducing multimodal TTFT for mixed image/video agent turns.
- **Proposals (2):**
  - Make encoder cudagraph packing bucket-aware to avoid budget-cliff eager fallbacks _(source: agent_knowledge)_
  - Add per-modality graph-path lanes before greedy packing _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0018](../modules/vllm_v1_worker/EncoderCudaGraphManager._execute_local__cand-vllm_v1_worker-0018.md)

### 149. `cand-vllm_v1_core-0011` — Scheduler._mamba_block_aligned_split (score 28, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/sched/scheduler.py:349` (`Scheduler._mamba_block_aligned_split`, method)
- **Description:** Mamba align-mode chunk-boundary heuristic that clips prefill chunks to cacheable recurrent-state boundaries.
- **Current approach:** Computes start/end, last cacheable block boundary with EAGLE back-off, prompt-tail hash boundary, next block boundary, and shared-prefix boundary, then chooses the earliest mandatory stop strictly inside the proposed chunk.
- **Why this impact / rank:** Mamba block-aligned split is scoped strictly to hybrid-Mamba align-mode models; potentially meaningful TTFT/TPOT for those workloads but a small slice of agentic serving overall. — Scoped to Mamba align-mode models, but those workloads can gain or lose substantial reuse from chunk boundaries. Better choices reduce repeated prefill work and improve TTFT/TPOT for hybrid-Mamba agent turns.
- **Proposals (2):**
  - Cache last-turn Mamba boundary decisions per request to skip re-scanning shared prefixes _(source: agent_knowledge)_
  - Gate shared-prefix stops on actual reuse demand _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0011](../modules/vllm_v1_core/Scheduler._mamba_block_aligned_split__cand-vllm_v1_core-0011.md)

### 150. `cand-vllm_v1_core-0019` — MambaManager.find_longest_cache_hit (score 32, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/single_type_kv_cache_manager.py:1280` (`MambaManager.find_longest_cache_hit`, method)
- **Description:** Mamba prefix-cache hit lookup using fine-grained partial-state lookup or a right-to-left block scan for the deepest reusable recurrent state.
- **Current approach:** Resolves block hashes to Mamba block size, allocates per-group computed block lists, scans partial hash units or full blocks from right to left, pads skipped positions with null blocks, and returns the first deepest hit.
- **Why this impact / rank:** MambaManager.find_longest_cache_hit accelerates admission cache-hit lookup but only for Mamba/hybrid models; agentic workloads on the mainstream attention family will not exercise this path. — Impact is model-family scoped, but Mamba/hybrid agent workloads depend on this path to reuse long recurrent-state prefixes. Faster deepest-hit lookup reduces TTFT on follow-up turns.
- **Proposals (2):**
  - Retention-aware reverse candidate scan for MambaManager.find_longest_cache_hit _(source: agent_knowledge)_
  - Memoize repeated Mamba prefix-hit searches per cache epoch _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0019](../modules/vllm_v1_core/MambaManager.find_longest_cache_hit__cand-vllm_v1_core-0019.md)

### 151. `cand-vllm_multimodal-0010` — find_split_point (score 26, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/audio.py:400` (`find_split_point`, function)
- **Description:** Finds the quietest audio window in a search segment to choose a chunk split point.
- **Current approach:** Loops in Python over non-overlapping windows and computes RMS with (window ** 2).mean() ** 0.5 for each window, tracking the minimum non-NaN energy.
- **Why this impact / rank:** Vectorizing find_split_point cuts audio preprocessing wall time on TTFT, but only for voice-agent/ASR workloads and not the described general multi-turn agentic path. — Long audio inputs are chunked before processing. Vectorizing split search reduces audio preprocessing wall time and lowers TTFT for voice-agent and ASR-style requests.
- **Proposals (3):**
  - Vectorize find_split_point using frame-wise RMS reduction _(source: research_finding)_
  - Coarse-to-fine strided window search with zero-energy early exit in find_split_point _(source: agent_knowledge)_
  - Use reduceat sum-of-squares for exact split search _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0010](../modules/vllm_multimodal/find_split_point__cand-vllm_multimodal-0010.md)

### 152. `cand-vllm_v1_engine-0008` — LogprobsProcessor._update_prompt_logprobs (score 36, impact: medium)

- **Module:** vllm/v1/engine
- **Location:** `vllm/v1/engine/logprobs.py:121` (`LogprobsProcessor._update_prompt_logprobs`, method)
- **Description:** Prompt-logprobs converter that turns prompt logprob tensors into per-position Logprob containers.
- **Current approach:** Flattens token ids, converts all ids to tokens, Pythonizes ranks/logprobs/token_ids with .tolist(), then loops over prompt positions to slice decoded tokens, recompute sampled context ids, verify UTF-8, and append per-position logprobs.
- **Why this impact / rank:** LogprobsProcessor._update_prompt_logprobs is a per-position Python loop paid only when prompt_logprobs is enabled; gated feature limits reach, and per-token savings are small even when active. — Prompt-logprobs users pay this on every new turn; reducing per-position Python work can lower median TTFT for prefill-heavy multi-turn traffic.
- **Proposals (2):**
  - Fast-path prompt logprobs when no UTF-8 replacement chars are present _(source: agent_knowledge)_
  - Trim prompt logprob rows before Python conversion _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_engine-0008](../modules/vllm_v1_engine/LogprobsProcessor._update_prompt_logprobs__cand-vllm_v1_engine-0008.md)

### 153. `cand-vllm_v1_worker-0032` — RopeState.prepare_positions and _prepare_rope_positions_kernel (score 29, impact: medium)

- **Module:** vllm/v1/worker
- **Location:** `vllm/v1/worker/gpu/mm/rope.py:110` (`RopeState.prepare_positions and _prepare_rope_positions_kernel`, region)
- **Description:** New GPU runner M-RoPE/XD-RoPE position preparation from staged prefill positions, deltas, and per-request query ranges.
- **Current approach:** Launches one Triton program per request with BLOCK_SIZE=1024 and an inner static loop over RoPE dimensions, even when decode queries are one token, and reads staged UVA-backed prefill positions for prefill rows.
- **Why this impact / rank:** RopeState.prepare_positions removes wasted lanes on one-token decode for M-RoPE/XD-RoPE; only helps multimodal RoPE workloads and gains are small relative to the full decode step. — The cost is concentrated in multimodal agent workloads using M-RoPE/XD-RoPE; adaptive tiling or batching removes wasted lanes in one-token decode and improves multimodal median TPOT.
- **Proposals (2):**
  - Split M-RoPE/XD-RoPE position preparation into a fused decode kernel plus per-request prefill kernel _(source: agent_knowledge)_
  - Add a no-op fast path when staged RoPE positions are already current _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_worker-0032](../modules/vllm_v1_worker/RopeState.prepare_positions_and__prepare_rope_positions_kernel__cand-vllm_v1_worker-0032.md)

### 154. `cand-vllm_multimodal-0023` — BaseMultiModalReceiverCache.get_and_update_features (score 26, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/cache.py:589` (`BaseMultiModalReceiverCache.get_and_update_features`, method)
- **Description:** Updates engine-side multimodal features with cached encoder outputs while preserving receiver cache eviction order.
- **Current approach:** Runs two full loops over mm_features: the first recomputes each cache key and touches the receiver cache, and the second recomputes the same key and calls get_and_update_item to mutate feature.data.
- **Why this impact / rank:** Encoder-cache get_and_update_features deduplication reduces Python overhead on multimodal ingress; effect on TTFT only for multimodal-heavy agent batches, small per-call work. — This path runs for multimodal encoder cache interaction in the engine. Reducing duplicate cache work shortens TTFT on cache hits and misses, especially when agentic batches carry many multimodal features.
- **Proposals (1):**
  - Deduplicate mm_features by cache_key to collapse redundant touch/get_and_update_item calls _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0023](../modules/vllm_multimodal/BaseMultiModalReceiverCache.get_and_update_features__cand-vllm_multimodal-0023.md)

### 155. `cand-vllm_v1_kv_offload-0005` — NUM_SMS/THRESHOLD_BYTES/MIN_N (score 30, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/cpu/swap_blocks_triton.py:12` (`NUM_SMS/THRESHOLD_BYTES/MIN_N`, config_block)
- **Description:** Triton swap-block tuning constants define SM count, descriptor-size crossover, and minimum descriptor batch size.
- **Current approach:** Fixed H100 PCIe Gen5 constants: NUM_SMS = 12, THRESHOLD_BYTES = 28 * 1024, and MIN_N = 16, applied uniformly across devices, page sizes, and workloads.
- **Why this impact / rank:** Tuning NUM_SMS/THRESHOLD_BYTES/MIN_N adjusts CPU-to-GPU Triton swap crossover on non-H100 links; strictly a tuning knob for the promotion path chosen by _select_swap_blocks_fn. — Non-H100 devices and different host links may have different DMA/Triton crossovers. Tuning these values can improve CPU-to-GPU promotion latency, though the effect is limited to the Triton branch selected by _select_swap_blocks_fn.
- **Proposals (2):**
  - Online one-shot autotune of NUM_SMS/THRESHOLD_BYTES/MIN_N at handler init, with soft per-call MIN_N _(source: agent_knowledge)_
  - Add runtime overrides for swap-block Triton tuning constants _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0005](../modules/vllm_v1_kv_offload/NUM_SMS_THRESHOLD_BYTES_MIN_N__cand-vllm_v1_kv_offload-0005.md)

### 156. `cand-vllm_multimodal-0008` — MultiModalCache.get_leaf_size/get_item_size (score 24, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/cache.py:98` (`MultiModalCache.get_leaf_size/get_item_size`, region)
- **Description:** Computes byte-size accounting for nested multimodal cache values used as LRUCache getsizeof callbacks.
- **Current approach:** get_item_size builds a parallel tree with json_map_leaves(cls.get_leaf_size, value), then walks that tree again with json_reduce_leaves(operator.add, ...). Tensor leaves use nbytes; other leaves fall back to sys.getsizeof.
- **Why this impact / rank:** MultiModalCache size accounting runs on cache inserts/evictions; touches TTFT for new-media cache misses but only in multimodal workloads and with small per-operation savings. — Size accounting runs on multimodal cache inserts and evictions. Reducing this cost shortens the cache-miss ingress path, which affects TTFT for new media in multi-turn workloads.
- **Proposals (2):**
  - Replace json_map_leaves with a type-dispatched iterative size accumulator specialized for MultiModalCacheValue _(source: agent_knowledge)_
  - Bypass generic sizing for cache wrappers with stored item_size _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0008](../modules/vllm_multimodal/MultiModalCache.get_leaf_size_get_item_size__cand-vllm_multimodal-0008.md)

### 157. `cand-vllm_multimodal-0021` — MultiModalGPUMemoryPool.acquire/_release (score 27, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/gpu_ipc_memory.py:54` (`MultiModalGPUMemoryPool.acquire/_release`, region)
- **Description:** Implements the frontend GPU multimodal memory budget as a blocking byte-counting semaphore used by GPU video decode paths.
- **Current approach:** acquire validates nbytes, then waits on a threading.Condition while available bytes are insufficient. _release returns bytes and calls notify_all, waking every waiter. There is no fairness, request-size ordering, timeout/backoff policy, or batching of wakeups.
- **Why this impact / rank:** MultiModalGPUMemoryPool admission scheduling cuts thundering-herd wakeups for concurrent video ingress, improving tail more than median and only when video traffic is bursty; small median TTFT effect for the described workload. — This affects concurrent GPU video ingress. Better admission scheduling reduces TTFT queueing for video requests and avoids CPU wakeup storms that can interfere with token generation, improving median and tail latency under agentic bursts.
- **Proposals (2):**
  - Replace notify_all with FIFO wait queue and targeted per-waiter notify _(source: agent_knowledge)_
  - Add an adaptive small-request bypass to the memory admission policy _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0021](../modules/vllm_multimodal/MultiModalGPUMemoryPool.acquire__release__cand-vllm_multimodal-0021.md)

### 158. `cand-vllm_v1_kv_offload-0016` — batch_store_block/batch_load_block (score 36, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/fs/io.py:168` (`batch_store_block/batch_load_block`, region)
- **Description:** Filesystem batch I/O implementation validates offsets, dispatches to the C extension when available, and falls back to serial Python per-block store/load loops.
- **Current approach:** Each batch validates every offset, then the C path materializes memoryview slices and temp paths before one extension call. The Python fallback loops over paths serially, stops on the first load error, and annotates load failures with num_succeeded for partial promotion recovery.
- **Why this impact / rank:** batch_store_block/batch_load_block for FS-tier offload sees Python fallback loops parallelized with a ThreadPoolExecutor. Only activates when reused prefixes spill past CPU primary memory, and even then FS bandwidth is often the true ceiling. — FS-backed secondary-to-primary loads sit on the TTFT path when reused blocks spill beyond CPU primary memory. Reducing Python loop overhead or improving batch I/O can lower promotion latency, with gains bounded by filesystem bandwidth and the optional C path.
- **Proposals (2):**
  - Parallelize the Python fallback in batch_store_block / batch_load_block with a bounded ThreadPoolExecutor _(source: agent_knowledge)_
  - Add duplicate-offset coalescing to batch_load_block before dispatch _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0016](../modules/vllm_v1_kv_offload/batch_store_block_batch_load_block__cand-vllm_v1_kv_offload-0016.md)

### 159. `cand-vllm_multimodal-0027` — DeepStreamVideoBackendMixin.decode_indices (score 28, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/video.py:929` (`DeepStreamVideoBackendMixin.decode_indices`, method)
- **Description:** Submits raw video bytes to the DeepStream decode pool, validates the result, copies CUDA frames into pinned CPU memory, synchronizes, and returns a numpy frame batch.
- **Current approach:** Calls cls._get_pool(pool_size).decode with target_indices and max_frames, derives valid indices from result.n_kept, allocates a fresh pinned CPU tensor for CUDA results, copies non-blocking, calls torch.cuda.current_stream().synchronize(), and exposes host.numpy().
- **Why this impact / rank:** DeepStream decode_indices affects video TTFT on deployments using that backend; pinned buffers and stream-aware handles reduce blocking D2H but the trigger workload is very narrow. — DeepStream deployments pay this D2H copy and synchronization on every decoded video. Reducing the blocking copy path lowers video TTFT and avoids CPU/GPU stalls that can perturb TPOT under concurrent agentic video traffic.
- **Proposals (3):**
  - Pool pinned host buffers and issue D2H on a dedicated stream in DeepStream decode_indices _(source: research_finding)_
  - Return a stream-aware lazy frame handle to overlap D2H with downstream preprocessing _(source: agent_knowledge)_
  - Copy only the kept frame slice back to CPU _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0027](../modules/vllm_multimodal/DeepStreamVideoBackendMixin.decode_indices__cand-vllm_multimodal-0027.md)

### 160. `cand-vllm_multimodal-0007` — PlaceholderRange.embeds_cumsum/extract_embeds_range (score 24, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/inputs.py:150` (`PlaceholderRange.embeds_cumsum/extract_embeds_range`, region)
- **Description:** Computes cumulative embedding counts and extracts contiguous embedded-token ranges from a boolean mask.
- **Current approach:** embeds_cumsum calls torch cumsum then tolist. extract_embeds_range converts the mask to int, runs torch.diff and torch.nonzero twice, stacks starts and ends, then converts the result to a Python list.
- **Why this impact / rank:** PlaceholderRange cumsum/extract avoids small tensor ops on placeholder masks; only exercised by multi-image prompts, and each call is already tiny. — Placeholder masks are processed during request ingress and scheduling. The per-call work is small, but it repeats for each multimodal item, so avoiding tiny tensor ops and syncs can reduce TTFT for multi-image prompts.
- **Proposals (2):**
  - Materialize is_embed on CPU once and compute cumsum + ranges in pure Python _(source: agent_knowledge)_
  - Derive embed ranges from the cached cumulative counts _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0007](../modules/vllm_multimodal/PlaceholderRange.embeds_cumsum_extract_embeds_range__cand-vllm_multimodal-0007.md)

### 161. `cand-vllm_multimodal-0009` — ShmObjectStoreSenderCache.get_and_update_item/remove_dangling_items (score 34, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/cache.py:488` (`ShmObjectStoreSenderCache.get_and_update_item/remove_dangling_items`, method)
- **Description:** Handles sender-side shared-memory processor-cache hits and misses, including opportunistic pruning of P0 prompt-update metadata.
- **Current approach:** On misses, it prunes when len(_p0_cache) is at least 2 * len(_shm_cache.key_index). remove_dangling_items then builds a set from all P0 keys, subtracts shared-memory keys, and deletes dangling entries one by one.
- **Why this impact / rank:** ShmObjectStoreSenderCache.remove_dangling_items O(N) prune bursts create TTFT outliers under long-running churn; incremental cursor-based pruning smooths tails rather than shifts the median much. — This path runs on every IPC processor-cache miss and may trigger O(N) work. Smoothing prune cost reduces TTFT outliers and can improve median latency under long-running agentic cache churn.
- **Proposals (2):**
  - Amortize dangling-key pruning via a bounded incremental scan cursor _(source: agent_knowledge)_
  - Prune P0 metadata from explicit shared-memory eviction results _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0009](../modules/vllm_multimodal/ShmObjectStoreSenderCache.get_and_update_item_remove_dangling_items__cand-vllm_multimodal-0009.md)

### 162. `cand-vllm_multimodal-0011` — resample_audio_pyav (score 25, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/audio.py:174` (`resample_audio_pyav`, function)
- **Description:** Resamples audio to a target sample rate using PyAV/libswresample, including short-input padding, frame creation, resampler invocation, flush, and output-frame concatenation.
- **Current approach:** Rounds rates, recursively resamples each channel for 2D audio, pads short mono arrays to 1024 samples, constructs a fresh av.AudioResampler and AudioFrame per call, materializes each output frame with to_ndarray, and concatenates the frame arrays.
- **Why this impact / rank:** resample_audio_pyav single-pass multi-channel resample cuts audio ingress CPU, but audio isn't the target workload and per-call setup is a small fraction of overall TTFT. — Audio resampling runs at ingress for mismatched sample rates. Cutting per-call setup and per-channel recursion reduces TTFT for audio-heavy agentic workloads and lowers CPU contention.
- **Proposals (2):**
  - Resample multi-channel audio in a single PyAV pass instead of per-channel recursion _(source: agent_knowledge)_
  - Skip PyAV setup when the rounded sample rates already match _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0011](../modules/vllm_multimodal/resample_audio_pyav__cand-vllm_multimodal-0011.md)

### 163. `cand-vllm_multimodal-0025` — reserve_mm_ipc_gpu_memory (score 22, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/gpu_ipc_memory.py:155` (`reserve_mm_ipc_gpu_memory`, function)
- **Description:** Subtracts frontend multimodal GPU decode reservations from available KV-cache memory based on raw-frame budget, API process count, video backend, and PyNvVideoCodec decoder constants.
- **Current approach:** Computes raw_frame_reserved_bytes from mm_ipc_gpu_memory_gb, detects PyNvVideoCodec use from media_io_kwargs or VLLM_VIDEO_LOADER_BACKEND, validates hw_decoders, adds num_api_servers times a fixed per-server decoder reservation when use_gpu_video_backend is true, and raises if the remaining KV cache is non-positive.
- **Why this impact / rank:** mm_ipc_gpu_memory reservation trades KV headroom vs decode concurrency for video deployments; effect on TTFT/TPOT is real but tied to video-heavy configurations, not the stated workload. — This directly controls KV-cache capacity in GPU video deployments. Less pessimistic but safe reservation can improve TPOT and batching capacity, while adequate reservation prevents TTFT-damaging OOM retries.
- **Proposals (2):**
  - Adapt mm_ipc_gpu_memory reservation to observed peak decode residency in agentic multi-turn workloads _(source: agent_knowledge)_
  - Make PyNvVideoCodec reservation conditional on actual local API processes _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0025](../modules/vllm_multimodal/reserve_mm_ipc_gpu_memory__cand-vllm_multimodal-0025.md)

### 164. `cand-vllm_multimodal-0012` — OpenCVVideoBackendMixin._read_frames_with_recovery (score 26, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/video.py:328` (`OpenCVVideoBackendMixin._read_frames_with_recovery`, method)
- **Description:** Reads selected OpenCV frames with forward-scan recovery for failed target frames.
- **Current approach:** Walks frame by frame from zero to the max target index with cap.grab, checks target membership in a set, retrieves selected or recovery frames, converts each frame with cv2.cvtColor, appends to a Python list, and stacks at the end.
- **Why this impact / rank:** OpenCV frame-read recovery loop is preprocessing overhead confined to sampled video prompts; keyframe seek reduces CPU cost but does not touch the mainline agentic decode/prefill critical path. — OpenCV video ingress can dominate CPU preprocessing for sampled video prompts. Reducing per-frame conversion and allocation overhead lowers TTFT in video workloads.
- **Proposals (2):**
  - Replace linear grab-scan with keyframe seek + bounded recovery window _(source: agent_knowledge)_
  - Avoid recovery-induced extra frames for target indices _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0012](../modules/vllm_multimodal/OpenCVVideoBackendMixin._read_frames_with_recovery__cand-vllm_multimodal-0012.md)

### 165. `cand-vllm_multimodal-0014` — PyAVVideoBackendMixin.decode_frames (score 22, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/video.py:547` (`PyAVVideoBackendMixin.decode_frames`, method)
- **Description:** Decodes requested frame indices from a PyAV container using seek and forward decode to target PTS.
- **Current approach:** Sets stream.thread_type to SLICE, loops over frame_indices, converts each index to timestamp and PTS, seeks when the requested PTS does not advance, decodes forward until a frame reaches the target PTS, converts each chosen frame to an RGB ndarray, appends to a list, and stacks at the end.
- **Why this impact / rank:** PyAV video decode seek/GOP coalescing reduces preprocessing wall time for video ingress; TTFT gain only for video-heavy requests, off the multi-turn text agent hot path. — PyAV is a common video ingress path. Reducing seek/decode churn and per-frame allocation lowers media preprocessing wall time and therefore TTFT for video-heavy requests.
- **Proposals (2):**
  - Batch-sort frame_indices and coalesce seeks by GOP with preallocated output buffer _(source: agent_knowledge)_
  - Memoize duplicate target frames before seeking _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0014](../modules/vllm_multimodal/PyAVVideoBackendMixin.decode_frames__cand-vllm_multimodal-0014.md)

### 166. `cand-vllm_multimodal-0015` — PYNVVIDEOCODEC_* module-level constants (score 20, impact: medium)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/video.py:217` (`PYNVVIDEOCODEC_* module-level constants`, config_block)
- **Description:** Defines PyNvVideoCodec resource-sizing defaults: backend name, per-decoder GPU memory reservation, decoder cache size, default HW decoder slots, and CUDA-context reservation.
- **Current approach:** Uses hard-coded constants: 128 MiB per decoder, decoder cache size 2, default HW decoders 2, and a 1.8 GiB CUDA-context reservation. These feed decoder-slot construction and GPU memory reservation logic.
- **Why this impact / rank:** PyNvVideoCodec reservation constants affect NVDEC slot counts and KV headroom for video bursts; scope is narrow (NVDEC deployments) and it is a policy-constant tune with weak evidence. — For NVDEC deployments, decoder slot count and reserved memory affect whether concurrent video requests queue or overlap. Better tuning can reduce TTFT tail and improve TPOT under video bursts.
- **Proposals (2):**
  - Replace static PyNvVideoCodec reservation with floor/ceiling + idle-teardown decoder pool _(source: agent_knowledge)_
  - Make PyNvVideoCodec memory reservation tunable per deployment _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0015](../modules/vllm_multimodal/PYNVVIDEOCODEC___module-level_constants__cand-vllm_multimodal-0015.md)

### 167. `cand-vllm_v1_core-0014` — MambaManager.allocate_new_blocks (score 32, impact: medium)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/single_type_kv_cache_manager.py:1532` (`MambaManager.allocate_new_blocks`, method)
- **Description:** Mamba align-mode block allocation path covering running-state blocks, speculative-block reuse, partial-hit CoW, and producer partial-tail handoff bookkeeping.
- **Current approach:** Branches on align mode, mutates req_blocks with null padding and recycled speculative blocks, calls block_pool.get_new_blocks for the delta, handles partial-hit CoW by moving cache hashes or applying a local copy, records pending copies/offloads, and updates per-request align-mode state.
- **Why this impact / rank:** MambaManager.allocate_new_blocks reuse policy affects TPOT/follow-up TTFT for Mamba align-mode models only; recycling speculative blocks and lazy allocation help, but model-scope narrow. — Only Mamba align-mode models use this path, but for them it runs during allocation and shapes recurrent-state reuse. Cleaner reuse and lower Python overhead reduce TPOT and follow-up-turn TTFT.
- **Proposals (2):**
  - Lazy speculative-block allocation with unused-tail reclamation in Mamba align mode _(source: agent_knowledge)_
  - Add a memory-pressure fallback that drops partial-hit CoW preservation _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0014](../modules/vllm_v1_core/MambaManager.allocate_new_blocks__cand-vllm_v1_core-0014.md)

### 168. `cand-vllm_v1_kv_offload-0019` — ServerRole.serve_external_requests/_process_inbound_lookup (score 24, impact: medium)

- **Module:** vllm/v1/kv_offload
- **Location:** `vllm/v1/kv_offload/tiering/p2p/session/server.py:449` (`ServerRole.serve_external_requests/_process_inbound_lookup`, region)
- **Description:** P2P server-side LookupMsg handling batches inbound key probes, re-polls pending keys, pins hits, and emits one aggregated LookupRespMsg.
- **Current approach:** Uses a per-request _serve_pending work list, de-duplicates keys within a LookupMsg using dict.fromkeys, calls parent.lookup per unique key, pins newly resolved hits through parent.create_store_job, re-polls HIT_PENDING/RETRY keys once per serve window, and forces unresolved keys to MISS after _LOOKUP_PENDING_TIMEOUT_S.
- **Why this impact / rank:** P2P ServerRole lookup batching helps peer-cache-hit TTFT via earlier LookupResp emission, but the effect is bounded by scheduler-step cadence and only matters for symmetric P2P deployments, which are uncommon. — Symmetric P2P consumers cannot start promotion until lookup responses arrive. Faster or better-batched inbound lookup resolution can reduce TTFT for peer-cache hits, while the effect is bounded by scheduler-step cadence and remote data-transfer time.
- **Proposals (2):**
  - Emit an early head-prefix LookupRespMsg for the contiguous HIT run before deadline _(source: agent_knowledge)_
  - Throttle pending-key re-polls with a per-serve lookup budget _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_kv_offload-0019](../modules/vllm_v1_kv_offload/ServerRole.serve_external_requests__process_inbound_lookup__cand-vllm_v1_kv_offload-0019.md)

### 169. `cand-vllm_v1_attention-0014` — get_dcp_local_seq_lens (score 22, impact: low)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/utils.py:958` (`get_dcp_local_seq_lens`, function)
- **Description:** Computes per-rank local sequence lengths for Decode Context Parallelism.
- **Current approach:** Converts seq_lens to int32, allocates rank_offsets with torch.arange or torch.tensor on each call, applies integer arithmetic and torch.clip, and returns a fresh tensor.
- **Why this impact / rank:** get_dcp_local_seq_lens is per-DCP-metadata-build scalar arithmetic and small allocation; caching rank_offsets is real but per-call cost is tiny and DCP must be enabled, so impact on median TTFT/TPOT is minimal. — Per-call cost is small, so impact is low. It still runs on the DCP per-step path, so caching offsets or fusing arithmetic can modestly improve TPOT in DCP-enabled agentic serving.
- **Proposals (2):**
  - Cache DCP rank_offsets tensor across metadata builds in get_dcp_local_seq_lens _(source: research_finding)_
  - Add fast paths in get_dcp_local_seq_lens for dcp_size==1, interleave==1, and int32 inputs _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0014](../modules/vllm_v1_attention/get_dcp_local_seq_lens__cand-vllm_v1_attention-0014.md)

### 170. `cand-vllm_multimodal-0016` — DeepStreamVideoBackendMixin._get_pool (score 18, impact: low)

- **Module:** vllm/multimodal
- **Location:** `vllm/multimodal/video.py:897` (`DeepStreamVideoBackendMixin._get_pool`, region)
- **Description:** Lazily initializes a process-wide DeepStream DecodePool and chooses its worker count.
- **Current approach:** Returns an existing singleton pool if present. Otherwise it reads pool_size from the caller or VLLM_MEDIA_LOADING_THREAD_COUNT with default 8, clamps to [1, 16], logs, and constructs DecodePool(num_workers=pool_size). The first caller fixes the process-wide pool size.
- **Why this impact / rank:** DeepStream pool sizing only affects DeepStream video deployments; narrow scope and unrelated to the multi-turn text agentic hot path implied by the workload hint. — This only affects DeepStream deployments, but worker sizing can materially change decode queueing under concurrent video requests, moving TTFT and sometimes TPOT for that backend.
- **Proposals (2):**
  - Size DeepStream decode pool from measured NVDEC engine count with async warm-start _(source: agent_knowledge)_
  - Make DeepStream pool singleton initialization race-free _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_multimodal-0016](../modules/vllm_multimodal/DeepStreamVideoBackendMixin._get_pool__cand-vllm_multimodal-0016.md)

### 171. `cand-vllm_v1_attention-0016` — FlashInferMetadataBuilder._get_workspace_buffer (score 18, impact: low)

- **Module:** vllm/v1/attention
- **Location:** `vllm/v1/attention/backends/flashinfer.py:940` (`FlashInferMetadataBuilder._get_workspace_buffer`, method)
- **Description:** Lazily allocates the FlashInfer workspace buffer using an environment default and a head-footprint estimate.
- **Current approach:** Allocates once at max(env buffer size, max_num_batched_tokens * num_qo_heads * head_dim * 16), never shrinks, and does not adapt to observed batch shapes after allocation.
- **Why this impact / rank:** FlashInfer workspace sizing mainly touches TTFT outliers on long prefills rather than median latency; self-labeled low impact and no supporting research-finding proposals. — Workspace sizing mainly affects memory footprint and edge-case long prefills rather than steady-state median latency. It can still reduce TTFT outliers by avoiding FlashInfer workspace failures or fallback behavior.
- **Proposals (2):**
  - Share workspace buffer across FlashInfer wrappers and reuse a per-device singleton to cut allocation and TTFT jitter _(source: agent_knowledge)_
  - Grow the FlashInfer workspace on demand after runtime size failures _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_attention-0016](../modules/vllm_v1_attention/FlashInferMetadataBuilder._get_workspace_buffer__cand-vllm_v1_attention-0016.md)

### 172. `cand-vllm_v1_core-0015` — KVCacheCoordinator.get_num_blocks_to_allocate (score 24, impact: low)

- **Module:** vllm/v1/core
- **Location:** `vllm/v1/core/kv_cache_coordinator.py:130` (`KVCacheCoordinator.get_num_blocks_to_allocate`, method)
- **Description:** Coordinator fanout that sums per-manager block-allocation requirements, with a cross-attention special case using encoder-token sizing.
- **Current approach:** Loops over single_type_managers, checks isinstance(manager, CrossAttentionManager) on every group, dispatches to either encoder-token sizing or standard request-token sizing, and accumulates the sum.
- **Why this impact / rank:** KVCacheCoordinator.get_num_blocks_to_allocate is called twice per waiting request in admission but absolute cost is tiny; useful as part of broader admission-loop tuning, not a standalone TTFT mover. — Absolute cost is small because hybrid models usually have few groups, so it is mainly useful as part of broader admission-loop optimization. It can still shave TTFT overhead in request-heavy scheduling steps.
- **Proposals (2):**
  - Precompute per-group dispatch schedule to remove per-call isinstance checks in get_num_blocks_to_allocate _(source: agent_knowledge)_
  - Cache allocation-size results across admission retries within a scheduler step _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_core-0015](../modules/vllm_v1_core/KVCacheCoordinator.get_num_blocks_to_allocate__cand-vllm_v1_core-0015.md)

### 173. `cand-vllm_v1_executor-0009` — RayDistributedExecutor.collective_rpc (score 20, impact: low)

- **Module:** vllm/v1/executor
- **Location:** `vllm/v1/executor/ray_executor.py:470` (`RayDistributedExecutor.collective_rpc`, method)
- **Description:** Broadcast an RPC to all Ray worker actors: cloudpickle-serializes callable methods per call, fans out execute_method.remote across every worker, then either wraps refs in FutureWrapper or ray.gets synchronously on all ranks.
- **Current approach:** For non-string methods, cloudpickle.dumps is invoked on every call. A list comprehension issues worker.execute_method.remote for every worker in self.workers, then ray.get blocks on the full ref list when non_block is False because the collective_rpc contract returns one result per rank.
- **Why this impact / rank:** RayDistributedExecutor.collective_rpc is off the steady-state decode path (Ray uses compiled DAG); improvements only touch startup, warmup, and occasional control RPCs, so median TTFT/TPOT movement is minimal by design. — The path can improve Ray startup, warmup, and occasional control-call latency, which may move TTFT around engine setup or reconfiguration. It is low for median TPOT because steady-state Ray execute_model/sample_tokens use the compiled DAG rather than this RPC path.
- **Proposals (2):**
  - Cache cloudpickle payloads for repeated callable methods in collective_rpc _(source: agent_knowledge)_
  - Put large shared RPC arguments once before worker fanout _(source: agent_knowledge)_
- **Full detail:** [cand-vllm_v1_executor-0009](../modules/vllm_v1_executor/RayDistributedExecutor.collective_rpc__cand-vllm_v1_executor-0009.md)

## Method note

- Sharding: 5 shards, ≤ 40 cards each (round-robin over pre-score order so each shard spans the pre-score range).
- Judge: one general-purpose sub-agent per shard produced a 0–100 impact score per card; a single merge sub-agent then produced the global top-40 ordering with fresh 0–100 scores.
- Tail (rank 41–173): ordered by (shard-rank percentile, then pre_score), deterministic tie-break on `(module_qualified_name, id)`.
- Repair: 0 unknown IDs dropped from the merge judge; 0 duplicates removed; 0 candidates appended by repair (both judges covered every card assigned to them).
