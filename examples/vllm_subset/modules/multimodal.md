# multimodal

[← All modules](../index.md)

## Module
- **Path:** `vllm/multimodal`
- **Description:** Multi-modal input/processing: image/audio/video parsing, hashing, tokenization, encoder budgets, and registry of modality processors.
- **Depends on:** torch, transformers, pillow, opencv-python-headless, mistral_common
- **Main files:**
  - `vllm/multimodal/registry.py` — Registers per-model multimodal processors.
  - `vllm/multimodal/processing/processor.py` — Generic multimodal processing pipeline.
  - `vllm/multimodal/inputs.py` — Multimodal input dataclasses.
  - `vllm/multimodal/cache.py` — Multimodal feature cache.
  - `vllm/multimodal/hasher.py` — Stable hashing of multimodal inputs.
- **Run status:** DEGRADED
- **Findings:** 15
- **Issues:** 2

## Candidates

| Candidate | Impact | Deep research proposals |
|---|---|---:|
| [`VIDEO_LOADER_REGISTRY.register("opencv")`](multimodal/VIDEO_LOADER_REGISTRY.register__opencv____cand-0012.md) | high | 5 |
| [`OpenCVVideoBackendMixin._read_frames_no_recovery`](multimodal/OpenCVVideoBackendMixin._read_frames_no_recovery__cand-0003.md) | high | 3 |
| [`PyAVVideoBackendMixin.decode_frames`](multimodal/PyAVVideoBackendMixin.decode_frames__cand-0004.md) | high | 2 |
| [`VideoMediaIO.load_base64`](multimodal/VideoMediaIO.load_base64__cand-0016.md) | medium | 2 |
| [`_iter_placeholders`](multimodal/_iter_placeholders__cand-0006.md) | high | 1 |
| [`MediaConnector.load_from_url_async`](multimodal/MediaConnector.load_from_url_async__cand-0014.md) | medium | 1 |
| [`MultiModalHasher serialization/hash pipeline`](multimodal/MultiModalHasher_serialization_hash_pipeline__cand-0001.md) | high | 0 |
| [`iter_token_matches`](multimodal/iter_token_matches__cand-0005.md) | medium | 0 |
| [`compute_retention_mask`](multimodal/compute_retention_mask__cand-0007.md) | high | 0 |
| [`recompute_mrope_positions`](multimodal/recompute_mrope_positions__cand-0008.md) | high | 0 |
| [`MultiModalCache.get_leaf_size/get_item_size`](multimodal/MultiModalCache.get_leaf_size_get_item_size__cand-0009.md) | medium | 0 |
| [`find_split_point`](multimodal/find_split_point__cand-0010.md) | medium | 0 |
| [`_get_group_hash/group_and_batch_mm_items`](multimodal/_get_group_hash_group_and_batch_mm_items__cand-0011.md) | medium | 0 |
| [`MediaConnector._maybe_evict`](multimodal/MediaConnector._maybe_evict__cand-0013.md) | medium | 0 |
| [`MultiModalFlatField._reduce_data`](multimodal/MultiModalFlatField._reduce_data__cand-0015.md) | medium | 0 |
| [`resample_audio_pyav`](multimodal/resample_audio_pyav__cand-0017.md) | medium | 0 |

## Findings (full list)

1. **Image processors - Hugging Face**
   - Source type: docs
   - URL: <https://huggingface.co/docs/transformers/en/image_processors>
   - Technique: Use fast image processors where available, keeping inputs as batched torch tensors and selecting device-backed processing so resize, crop, normalize, and tensor conversion run through torchvision instead of PIL/numpy. This targets media TTFT by shortening image preprocessing before the multimodal encoder runs.
   - Evidence: Quote: "BaseImageProcessorFast is based on torchvision and is significantly faster, especially when processing on a GPU. For a batch of torch.Tensor inputs, this can be up to 33x faster." Pointer: Fast image processors section.
2. **qwen-vl-utils · PyPI**
   - Source type: codebase
   - URL: <https://pypi.org/project/qwen-vl-utils/>
   - Technique: Adopt budget-aware visual sizing and video frame controls as first-class multimodal processor inputs, deriving max pixels, resized dimensions, FPS, and frame caps from encoder budgets. This can reduce median TPOT by limiting visual token counts before placeholders and encoder work are scheduled.
   - Evidence: Quote: "You can set the maximum tokens for a video through the environment variable VIDEO_MAX_PIXELS # based on the maximum tokens that the model can accept." Pointer: Qwen2.5VL usage section.
3. **VideoDecoder — TorchCodec 0.12.0+cu126 Documentation**
   - Source type: docs
   - URL: <https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html>
   - Technique: Add a TorchCodec video decode path that retrieves sampled frame batches by frame index or presentation time and returns PyTorch tensors directly. This would reduce video TTFT by avoiding repeated per-frame decode loops and numpy/PIL round trips for sparse frame sampling.
   - Evidence: Quote: "get_frames_at(indices: Tensor | list[int]) → FrameBatch [source] # Return frames at the given indices." Pointer: VideoDecoder API, get_frames_at method.
4. **NVIDIA DALI Documentation**
   - Source type: docs
   - URL: <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html>
   - Technique: Use a GPU-accelerated decode and preprocessing pipeline for image, video, and audio bytes, with mixed CPU/GPU stages and prefetching. This can move expensive decode, resize, and normalization work out of the API CPU critical path and overlap media preprocessing with serving work.
   - Evidence: Quote: "The NVIDIA Data Loading Library (DALI) is a GPU-accelerated library for data loading and pre-processing to accelerate deep learning applications. It provides a collection of highly optimized building blocks for loading and processing image, video and audio data." Pointer: DALI documentation overview.
5. **ETag header - HTTP | MDN**
   - Source type: docs
   - URL: <https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/ETag>
   - Technique: Extend media URL caching with HTTP validators such as ETag and conditional GET requests, not just local TTL by URL. In multi-turn agentic workloads that repeatedly reference the same external media, this can avoid full body transfers and lower media fetch TTFT.
   - Evidence: Quote: "The HTTP ETag (entity tag) response header is an identifier for a specific version of a resource. It lets caches be more efficient and save bandwidth, as a web server does not need to resend a full response if the content has not changed." Pointer: ETag header overview.
6. **singleflight package - golang.org/x/sync/singleflight - Go Packages**
   - Source type: docs
   - URL: <https://pkg.go.dev/golang.org/x/sync/singleflight>
   - Technique: Apply the singleflight duplicate-suppression pattern to media fetches, hashing, and processor-cache misses keyed by URL or multimodal hash. This prevents concurrent turns or requests from redundantly downloading or preprocessing the same media during cache misses.
   - Evidence: Quote: "Package singleflight provides a duplicate function call suppression mechanism." Quote: "If a duplicate comes in, the duplicate caller waits for the original to complete and receives the same results." Pointer: Package overview and Group.Do documentation.
7. **Vips – 8.0: Using > Checklist for programmers using libvips**
   - Source type: docs
   - URL: <https://www.libvips.org/API/current/developer-checklist.html>
   - Technique: Use libvips thumbnail or shrink-on-load paths for large image bytes before handing them to model-specific processors. Decode-time downscaling can cut CPU time and memory pressure for large screenshots, scans, and camera images in agentic media loops.
   - Evidence: Quote: "The thumbnail operation combines load and resize into one step. This lets it take advantage of format library features, such as shrink on load, and can lead to a large improvement in speed and a large drop in memory use." Pointer: Developer checklist, thumbnail guidance.
8. **Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference**
   - Source type: paper
   - URL: <https://arxiv.org/html/2510.14624v1>
   - Technique: Generalize EVS-style temporal redundancy pruning as a video-token budget policy: compare consecutive frame patches, prune static patches, and preserve positional identity. This can reduce visual token counts for long or static videos, improving both TTFT and TPOT.
   - Evidence: Quote: "We introduce Efficient Video Sampling (EVS), a simple, plug-and-play method for reducing token redundancy in videos by identifying and pruning temporally static patches - spatial regions that remain unchanged across consecutive frames. EVS preserves positional identity, requires no architectural changes or retraining." Pointer: Abstract.
9. **PruneVid: Visual Token Pruning for Efficient Video Large Language Models**
   - Source type: paper
   - URL: <https://aclanthology.org/2025.findings-acl.1024/>
   - Technique: Adopt a two-stage video-token reduction design: merge temporally static and spatially similar tokens, then prune query-irrelevant visual tokens when safe. The multimodal pipeline could expose pruned token counts and placeholder metadata so scheduler and encoder budgets reflect reduced video input size.
   - Evidence: Quote: "PruneVid (1) reduces intrinsic video redundancy by merging temporally static and spatially similar tokens, and (2) leverages LLMs' inherent ability to selectively prune visual tokens irrelevant to specific queries, thereby improving model efficiency." Pointer: Abstract.
10. **GitHub - snakers4/silero-vad: Silero VAD: pre-trained enterprise-grade Voice Activity Detector**
   - Source type: codebase
   - URL: <https://github.com/snakers4/silero-vad>
   - Technique: Run voice activity detection before audio feature extraction to trim silence or split long audio into speech-bearing spans. This can reduce audio preprocessing time and downstream audio token counts for multi-turn voice or meeting-agent workloads.
   - Evidence: Quote: "One audio chunk (30+ ms) takes less than 1ms to be processed on a single CPU thread. Using batching or GPU can also improve performance considerably." Pointer: README, Key Features.
11. **pyahocorasick — ahocorasick documentation**
   - Source type: docs
   - URL: <https://pyahocorasick.readthedocs.io/>
   - Technique: Compile multimodal placeholder targets into an Aho-Corasick automaton per processor and scan prompt text or token strings in one pass. This can reduce prompt-update overhead when prompts contain many image, video, or audio placeholders.
   - Evidence: Quote: "pyahocorasick is a fast and memory efficient library for exact or approximate multi-pattern string search meaning that you can find multiple key strings occurrences at once in some input text." Pointer: Documentation overview.
12. **GitHub - huggingface/tokenizers: Fast State-of-the-Art Tokenizers optimized for Research and Production**
   - Source type: codebase
   - URL: <https://github.com/huggingface/tokenizers>
   - Technique: Use tokenizer alignment and offset mappings to map multimodal placeholder substrings to token spans during tokenization, avoiding decode-to-text and re-encode fallback paths. This directly targets TTFT overhead in prompt update and placeholder range discovery.
   - Evidence: Quote: "Normalization comes with alignments tracking. It's always possible to get the part of the original sentence that corresponds to a given token." Pointer: README, Main features.
13. **10. Introducing Decord: an efficient video reader - Gluon**
   - Source type: docs
   - URL: <https://cv.gluon.ai/build/examples_action_recognition/decord_loader.html>
   - Technique: Use a Decord-backed optional video loader for sparse random access and batched frame retrieval, especially when model processors request non-contiguous sampled frames. This can reduce TTFT for video prompts on platforms where Decord's GPU or batch path is available.
   - Evidence: Quote: "It supports get_batch, GPU loading, fast random access, etc, which is perfectly designed for training video deep neural networks." Pointer: GluonCV Decord loader example introduction.
14. **GitHub - uploadcare/pillow-simd: The friendly PIL fork**
   - Source type: codebase
   - URL: <https://github.com/uploadcare/pillow-simd>
   - Technique: Offer an optional SIMD-accelerated PIL-compatible backend for image resize and conversion paths that must stay in Pillow. This is a low-intrusion way to reduce CPU preprocessing latency for image-heavy serving deployments on x86 CPUs.
   - Evidence: Quote: "Pillow-SIMD is highly optimized version of Pillow library for x86 architecture (mainly Intel and AMD CPUs)." Quote: "Pillow-SIMD, in turn, is even faster than the original Pillow by the factor of 4-6." Pointer: README.
15. **FastCDC: A Fast and Efficient Content-Defined Chunking Approach for Data Deduplication**
   - Source type: paper
   - URL: <https://www.usenix.org/conference/atc16/technical-sessions/presentation/xia>
   - Technique: Use content-defined chunking for large media cache keys, enabling partial reuse for long videos/audio or repeated media with small changes instead of relying only on whole-object hashes. This could improve cache hit behavior in multi-turn agentic workloads that repeatedly reference related media segments.
   - Evidence: Quote: "Content-Defined Chunking (CDC) has been playing a key role in data deduplication systems in the past 15 years or so due to its high redundancy detection ability." Quote: "we propose FastCDC, a Fast and efficient CDC approach." Pointer: USENIX ATC 2016 abstract.

## Issues

### proposal_from_finding_creator
- **[error, recoverable]** agent failure (candidate_id=cand-0009, finding_id=find-0014): claude timed out after 600.0s
- **[error, recoverable]** agent failure (candidate_id=cand-0009, finding_id=find-0015): claude timed out after 600.0s
