# @VIDEO_LOADER_REGISTRY.register("opencv")

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/video.py`](vllm/multimodal/video.py) (lines 986–986)
- **Symbol:** `@VIDEO_LOADER_REGISTRY.register("opencv")`
- **Kind:** plugin_seam
- **Estimated impact:** high
- **Id:** `cand-vllm_multimodal-0018`

## Description
Registers the default VideoBackend implementation in VIDEO_LOADER_REGISTRY; sibling decorator registrations add model-specific video sampling backends under names such as pynvvideocodec, qwen2_vl, qwen3_vl, glm46v, glmga, molmo2, nemotron_vl, and openpangu.

## Current approach
The interface is VideoLoader.compute_frames_index_to_sample and VideoLoader.load_bytes in vllm/multimodal/video.py. Reference implementations include VideoBackend in vllm/multimodal/video.py and Qwen2VLVideoBackend/Qwen3VLVideoBackend in the same file. Runtime selection comes from --media-io-kwargs, VLLM_VIDEO_LOADER_BACKEND, or processor-name mappings registered through VIDEO_LOADER_REGISTRY.register(..., video_processor=...).

## Estimated impact explanation
Sampled frame count directly drives video encoder FLOPs and prefill work. Better registered samplers can cut TTFT substantially for long clips while preserving answer quality.

## Evolve rationale
The concrete seam is the VIDEO_LOADER_REGISTRY.register decorator used for backend registration at line 986 and mirrored by sibling backend registrations. New sibling implementations can change sampling policy while reusing existing decode backends, making frame-count reduction an isolated evolutionary target. Correctness oracle: tests/multimodal/test_video.py plus model-specific HF-parity tests; sampled frames/metadata must match the selected backend contract, and quality-sensitive proposals need downstream video QA evaluation.

## Deep research proposals

### 1. Add a memory-demux + ThreadedDecoder PyNvVideoCodec sibling backend to VIDEO_LOADER_REGISTRY
- **Finding:** `find-vllm_multimodal-0004` — *PyNvVideoCodec API Programming Guide*
- **Source URL:** <https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

At the plugin seam anchored by @VIDEO_LOADER_REGISTRY.register("opencv") in vllm/multimodal/video.py:986, register a new sibling PyNvVideoCodec loader (e.g. VIDEO_LOADER_REGISTRY.register("pynvvideocodec_threaded")) that replaces two remaining fixed costs in the current pynvvideocodec path. First, drop the temp-file stage in PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec (vllm/multimodal/video.py:842-875), which currently writes bytes to tempfile.mkstemp(suffix=".mp4") before every decode. Instead, use PyNvVideoCodec's in-memory demuxer to feed the raw container bytes directly into the decoder (no filesystem round-trip), which the PyNvVideoCodec API Programming Guide documents as first-class input. Second, layer the guide's ThreadedDecoder over the existing PyNvVideoCodecDecoderSlot pool (vllm/multimodal/video.py:237-276, 661-745) so decode of one clip's frames overlaps with demux of the next request in the multi-turn agentic queue; keep the existing decoder-reuse/reconfigure_decoder path so per-request CUVID parser + surface-pool construction stays amortized. Preserve the existing load_bytes signature and NHWC uint8 output contract, and continue to route the pinned-host copy through get_mm_gpu_ipc_pool. Expose the new backend via --media-io-kwargs and VLLM_VIDEO_LOADER_BACKEND identically to the current pynvvideocodec entry so operators can opt in without touching processor mappings.

**Proposal rationale.**

The candidate seam is the VIDEO_LOADER_REGISTRY decorator, whose sibling registrations already include a PyNvVideoCodec backend. The finding pinpoints two concrete optimizations from NVIDIA's PyNvVideoCodec guide — in-memory demuxing and ThreadedDecoder — that map directly onto observable gaps in the current sibling implementation: decode_frames_pynvvideocodec stages every request to a tempfile before decode (vllm/multimodal/video.py:852-872) and there is no threaded/pipelined decoder around the existing per-slot serial path. Removing the tempfile stage cuts a fixed per-request I/O cost (open/write/fsync/unlink plus VFS caching pressure) that scales with clip size; adding ThreadedDecoder overlaps NVDEC decode with demux and PCIe copy so decoder slots stay saturated under the multi-turn agentic workload. Because video decode sits on the prefill critical path, shrinking this fixed setup and increasing decode overlap plausibly reduces median TTFT for video-carrying turns, matching the caller objective. The change is isolated behind a new sibling registration at the same decorator seam, so quality-sensitive sampling behavior (compute_frames_index_to_sample) can be reused unchanged and validated by the existing tests/multimodal/test_video.py oracle.

---

## Agent proposals

### 1. Register a content-hash memoizing sibling backend for repeated video attachments in multi-turn sessions
- **Agent:** claude

**Detailed description.**

At the plugin seam anchored by @VIDEO_LOADER_REGISTRY.register("opencv") in vllm/multimodal/video.py:986, add a new sibling registration (e.g. VIDEO_LOADER_REGISTRY.register("memoized")) whose load_bytes wraps an underlying decode backend (opencv/pynvvideocodec/deepstream) with a keyed cache keyed on (blake3(data), num_frames, fps, max_duration, sampling-suffix, backend, and any kwargs that affect compute_frames_index_to_sample such as min_frames/max_frames/temporal_patch_size). On a hit, return the previously produced (frames_array, hf_metadata) tuple directly, skipping demux + decode + PCIe copy entirely; on a miss, delegate to the wrapped backend's load_bytes and populate the cache before returning. Use a process-local LRU (bounded by aggregate host-pinned bytes rather than entry count) that stores frames in the same pinned-host NHWC uint8 layout produced today by VideoBackend/PyNvVideoCodecVideoBackendMixin/DeepStreamVideoBackendMixin so downstream callers observe identical arrays and identical HF metadata (create_hf_metadata output). Expose the wrapped codec via --media-io-kwargs (e.g. {"video": {"backend": "memoized", "inner": "pynvvideocodec", "cache_bytes": ...}}) and via VLLM_VIDEO_LOADER_BACKEND, and provide a compute_frames_index_to_sample that just delegates to the inner backend's implementation so Qwen2/Qwen3/Glm/etc. sampling policies remain intact when composed. Preserve the load_bytes signature, the NHWC uint8 output contract, and the create_hf_metadata metadata dict; validate against tests/multimodal/test_video.py to confirm byte-for-byte parity with the wrapped backend and add a hit/miss test asserting decode is called exactly once for repeated identical inputs.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_multimodal-0004) targets decode-path infrastructure inside the pynvvideocodec backend: eliminating the tempfile stage in decode_frames_pynvvideocodec and layering PyNvVideoCodec's ThreadedDecoder over the existing decoder-slot pool. It optimizes the per-decode fixed cost and pipeline depth. This proposal is orthogonal: it introduces a cache-first sibling registration that entirely skips demux + decode + PCIe copy on cache hits for repeated video attachments across turns of a multi-turn agentic session, which the existing proposal does not do and does not preclude — the two compose (memoized wrapping ThreadedDecoder-pynvvideocodec). It directly targets the caller's stated multi-turn agentic workload and TTFT/TPOT objective by exploiting the observation that the same video bytes are frequently referenced across successive turns, whereas the existing proposal only accelerates the first (and every) decode.

---

### 2. Add an index-only predecode sampler wrapper to avoid full-video decode for sparse frame requests
- **Agent:** codex

**Detailed description.**

At the VIDEO_LOADER_REGISTRY.register seam in vllm/multimodal/video.py, register a sibling backend such as "sparse_seek" that composes an existing backend's compute_frames_index_to_sample policy with a decode path that seeks only to the selected frame indices instead of decoding the full clip and slicing afterward. Keep the public load_bytes contract unchanged: first probe stream metadata to compute the same frame_count/fps/duration metadata used today, call the selected sampler to obtain the exact frame indices, then use OpenCV/PyAV/PyNvVideoCodec random seek plus short forward decode windows around keyframes to materialize only those frames into the existing NHWC uint8 output and create_hf_metadata result. Expose the inner sampler/codec through media-io kwargs so model-specific policies such as qwen2_vl or qwen3_vl can be reused unchanged, and add parity tests in tests/multimodal/test_video.py that compare selected indices, frame tensor shape, and metadata against the wrapped backend on short and long clips.

**Novelty rationale.**

The deep_research proposal optimizes the PyNvVideoCodec decode implementation by removing tempfile I/O and adding threaded decode, but it still accelerates a decode operation rather than changing how much of the video is decoded. Agent A's proposal memoizes completed decode results for repeated identical inputs, which helps only on cache hits. This proposal is distinct because it targets first-use long-video requests with sparse sampling: it preserves the registered sampling policies while avoiding full sequential decode when only a small subset of frames will be consumed.

---
