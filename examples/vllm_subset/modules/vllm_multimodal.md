# vllm/multimodal

[← All modules](../index.md)

## Module
- **Path:** `vllm/multimodal`
- **Description:** Multimodal input handling: per-modality loaders (image/audio/video), prompt parsing, HF processor plumbing, feature cache, and encoder-budget accounting.
- **Depends on:** torch, transformers, pillow, librosa, vllm/config, vllm/inputs, vllm/utils
- **Main files:**
  - `vllm/multimodal/inputs.py` — MultiModalKwargs and data types
  - `vllm/multimodal/registry.py` — Model to processor registry
  - `vllm/multimodal/parse.py` — Extract multimodal items from prompts
  - `vllm/multimodal/cache.py` — Processed-feature cache
  - `vllm/multimodal/encoder_budget.py` — Encoder work budgeting
- **Run status:** SUCCEEDED
- **Findings:** 10
- **Issues:** 0

## Candidates

| Candidate | Impact | Deep research proposals |
|---|---|---:|
| [`MultiModalHasher.serialize_item`](vllm_multimodal/MultiModalHasher.serialize_item__cand-vllm_multimodal-0001.md) | high | 2 |
| [`PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host`](vllm_multimodal/PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host__cand-vllm_multimodal-0017.md) | medium | 2 |
| [`MultiModalBudget._get_max_items`](vllm_multimodal/MultiModalBudget._get_max_items__cand-vllm_multimodal-0019.md) | medium | 2 |
| [`MultiModalHasher.iter_item_to_bytes/hash_kwargs`](vllm_multimodal/MultiModalHasher.iter_item_to_bytes_hash_kwargs__cand-vllm_multimodal-0002.md) | medium | 1 |
| [`MultiModalBatchedField._reduce_data`](vllm_multimodal/MultiModalBatchedField._reduce_data__cand-vllm_multimodal-0004.md) | high | 1 |
| [`MultiModalFlatField._reduce_data`](vllm_multimodal/MultiModalFlatField._reduce_data__cand-vllm_multimodal-0005.md) | medium | 1 |
| [`find_split_point`](vllm_multimodal/find_split_point__cand-vllm_multimodal-0010.md) | medium | 1 |
| [`@VIDEO_LOADER_REGISTRY.register("opencv")`](vllm_multimodal/_VIDEO_LOADER_REGISTRY.register__opencv____cand-vllm_multimodal-0018.md) | high | 1 |
| [`PyNvVideoCodecVideoBackendMixin._configure_decoder_slots/_borrow_decoder_slot`](vllm_multimodal/PyNvVideoCodecVideoBackendMixin._configure_decoder_slots__borrow_decoder_slot__cand-vllm_multimodal-0024.md) | medium | 1 |
| [`PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec`](vllm_multimodal/PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec__cand-vllm_multimodal-0026.md) | medium | 1 |
| [`DeepStreamVideoBackendMixin.decode_indices`](vllm_multimodal/DeepStreamVideoBackendMixin.decode_indices__cand-vllm_multimodal-0027.md) | medium | 1 |
| [`nested_tensors_equal`](vllm_multimodal/nested_tensors_equal__cand-vllm_multimodal-0003.md) | medium | 0 |
| [`_can_batch_mm_items/_batch_mm_items/group_and_batch_mm_items/group_and_batch_mm_kwargs`](vllm_multimodal/_can_batch_mm_items__batch_mm_items_group_and_batch_mm_items_group_and_batch_mm_kwargs__cand-vllm_multimodal-0006.md) | high | 0 |
| [`PlaceholderRange.embeds_cumsum/extract_embeds_range`](vllm_multimodal/PlaceholderRange.embeds_cumsum_extract_embeds_range__cand-vllm_multimodal-0007.md) | medium | 0 |
| [`MultiModalCache.get_leaf_size/get_item_size`](vllm_multimodal/MultiModalCache.get_leaf_size_get_item_size__cand-vllm_multimodal-0008.md) | medium | 0 |
| [`ShmObjectStoreSenderCache.get_and_update_item/remove_dangling_items`](vllm_multimodal/ShmObjectStoreSenderCache.get_and_update_item_remove_dangling_items__cand-vllm_multimodal-0009.md) | medium | 0 |
| [`resample_audio_pyav`](vllm_multimodal/resample_audio_pyav__cand-vllm_multimodal-0011.md) | medium | 0 |
| [`OpenCVVideoBackendMixin._read_frames_with_recovery`](vllm_multimodal/OpenCVVideoBackendMixin._read_frames_with_recovery__cand-vllm_multimodal-0012.md) | medium | 0 |
| [`PyAVVideoBackendMixin.decode_frames`](vllm_multimodal/PyAVVideoBackendMixin.decode_frames__cand-vllm_multimodal-0014.md) | medium | 0 |
| [`PYNVVIDEOCODEC_* module-level constants`](vllm_multimodal/PYNVVIDEOCODEC___module-level_constants__cand-vllm_multimodal-0015.md) | medium | 0 |
| [`DeepStreamVideoBackendMixin._get_pool`](vllm_multimodal/DeepStreamVideoBackendMixin._get_pool__cand-vllm_multimodal-0016.md) | low | 0 |
| [`BaseMultiModalField.reduce_data`](vllm_multimodal/BaseMultiModalField.reduce_data__cand-vllm_multimodal-0020.md) | medium | 0 |
| [`MultiModalGPUMemoryPool.acquire/_release`](vllm_multimodal/MultiModalGPUMemoryPool.acquire__release__cand-vllm_multimodal-0021.md) | medium | 0 |
| [`MultiModalKwargsItems.from_hf_inputs`](vllm_multimodal/MultiModalKwargsItems.from_hf_inputs__cand-vllm_multimodal-0022.md) | medium | 0 |
| [`BaseMultiModalReceiverCache.get_and_update_features`](vllm_multimodal/BaseMultiModalReceiverCache.get_and_update_features__cand-vllm_multimodal-0023.md) | medium | 0 |
| [`reserve_mm_ipc_gpu_memory`](vllm_multimodal/reserve_mm_ipc_gpu_memory__cand-vllm_multimodal-0025.md) | medium | 0 |

## Findings (full list)

1. **Efficiently Serving Large Multimodal Models Using EPD Disaggregation**
   - Source type: paper
   - URL: <https://www.emergentmind.com/papers/2501.05460>
   - Technique: Adopt encode-prefill-decode separation as a design target for multimodal ingress: treat media encoding as its own schedulable resource with its own cache and admission decisions rather than coupling it to text prefill. For multi-turn agentic workloads, this suggests encoder-cache-aware batching and budget policies that prioritize cached or cheap multimodal work to reduce TTFT and avoid TPOT interference.
   - Evidence: Quote: "separates the encoding, prefill, and decode stages onto dedicated resources." Pointer: abstract, lines 33-36.
2. **GitHub - vbdi/epdserve: [ICML 2025] Efficiently Serving Large Multimodal Models Using EPD Disaggregation · GitHub**
   - Source type: codebase
   - URL: <https://github.com/vbdi/epdserve>
   - Technique: EPDServe turns the paper idea into an implementation with independent stage schedulers, cache managers, intra-request encoding parallelism, CUDA IPC transfer, and dynamic role switching. The transferable idea is to make multimodal encoder work independently batchable and movable, so image/audio/video preprocessing can scale or drain without stalling decode-critical work.
   - Evidence: Quote: "Each stage operates independently with its own compute resources, scheduler, cache manager, and GPU workers." Pointer: README, lines 173-178.
3. **VideoDecoder — TorchCodec 0.16.0.dev20260807+cu126 Documentation**
   - Source type: docs
   - URL: <https://meta-pytorch.org/torchcodec/main/generated/torchcodec.decoders.VideoDecoder.html>
   - Technique: Use TorchCodec's batched frame APIs, in-memory byte/tensor sources, NHWC output, CUDA/NVDEC device mode, and exact-vs-approximate seek mode as a lower-overhead video decode path. This can reduce per-frame Python seek/stack/cvtColor work and avoid tempfile staging while preserving a backend-specific correctness contract.
   - Evidence: Quote: "If you need to decode multiple frames, we recommend using the batch methods instead, since they are faster." Pointer: __getitem__/get_frame_at notes and constructor parameters, lines 70-82.
4. **PyNvVideoCodec API Programming Guide**
   - Source type: docs
   - URL: <https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html>
   - Technique: Use PyNvVideoCodec's memory demuxing, decoder caching/reconfiguration, ThreadedDecoder, and batch frame retrieval as first-class policies rather than repeatedly staging to files and rebuilding decode state. This directly targets video ingress latency, decoder-slot contention, and fixed setup cost in GPU decode paths.
   - Evidence: Quote: "High-Throughput Pipelines Using ThreadedDecoder" and "Decoder Reuse". Pointer: table of contents, lines 117-153.
5. **A guide on good usage of non_blocking and pin_memory() in PyTorch**
   - Source type: docs
   - URL: <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>
   - Technique: Adopt measured pinned-memory policy: allocate or reuse pinned buffers in the right lifecycle, use non-default streams for overlap, and explicitly synchronize GPU-to-CPU copies before exposing CPU arrays. This supports replacing per-request pinned allocations and blocking D2H transfers with pooled, stream-aware transfers for multimodal tensors and decoded video frames.
   - Evidence: Quote: "Optimizing the transfer of tensors from the CPU to the GPU can be achieved through asynchronous transfers and memory pinning." Pointer: introduction and practical benchmark, lines 300-305 and 679-684.
6. **torch.nested — PyTorch 2.9 documentation**
   - Source type: docs
   - URL: <https://docs.pytorch.org/docs/2.9/nested.html>
   - Technique: Use jagged nested tensors or values+offsets packing for variable-length audio/video feature batches instead of eagerly zero-padding every item. The transferable technique is to represent ragged multimodal features as a packed values buffer plus offsets, deferring padding only where a downstream processor actually requires dense layout.
   - Evidence: Quote: "such data is stored underneath in an efficient packed representation". Pointer: Introduction and Data Layout sections.
7. **GitHub - BLAKE3-team/BLAKE3: the official Rust and C implementations of the BLAKE3 cryptographic hash function · GitHub**
   - Source type: codebase
   - URL: <https://github.com/BLAKE3-team/BLAKE3>
   - Technique: Exploit BLAKE3's chunked Merkle-tree structure for multimodal cache-key hashing: hash large tensor/media byte spans in larger contiguous chunks with SIMD or worker parallelism, and batch small metadata updates before entering the C hasher. This targets hashing overhead before cache-hit detection while retaining deterministic digests.
   - Evidence: Quote: "Highly parallelizable across any number of threads and SIMD lanes, because it's a Merkle tree on the inside." Pointer: README feature list.
8. **librosa.feature.rms — librosa 1.0.0dev documentation**
   - Source type: docs
   - URL: <https://librosa.org/doc/main/generated/librosa.feature.rms.html>
   - Technique: Replace per-window Python RMS loops in audio split selection with vectorized frame-wise RMS over a search segment and an argmin/tie policy. This preserves silence-aware splitting while reducing Python overhead for long audio prompts.
   - Evidence: Quote: "Compute root-mean-square (RMS) value for each frame." Pointer: function description and parameters.
9. **Cache-aware prefill–decode disaggregation (CPD) for up to 40% faster long-context LLM serving**
   - Source type: blog
   - URL: <https://www.together.ai/blog/cache-aware-disaggregated-inference>
   - Technique: Adopt cache-aware routing/admission as a policy pattern for multimodal encoder budgeting: classify requests by reusable context or multimodal-cache hit likelihood, then give warm work a fast path while isolating expensive cold media processing. For agentic multi-turn workloads, this can reduce TTFT by avoiding cached-turns waiting behind uncached media-heavy turns.
   - Evidence: Quote: "don't let expensive cold prefills block the fast path for reusable context." Pointer: How CPD works, lines 63-72.
10. **M* (M-star): A Modular, Extensible, Serving System for Multimodal Models**
   - Source type: other
   - URL: <https://mstar.stanford.edu/>
   - Technique: Model multimodal processing as component graph walks with explicit loops, parallel blocks, streaming edges, and placement policies. The target module could adapt the idea at the input/processor layer by carrying richer modality-stage metadata and scheduling hints, enabling only-needed modality processing, stream chunk policies, and component-aware batching.
   - Evidence: Quote: "A request is a series of Walks, chosen by a small state machine." Pointer: The Walk Graph section, lines 49-60.

## Issues

_No issues._
