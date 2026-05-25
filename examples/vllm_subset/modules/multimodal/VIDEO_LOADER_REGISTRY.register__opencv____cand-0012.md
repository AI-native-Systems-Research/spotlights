# VIDEO_LOADER_REGISTRY.register("opencv")

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/video.py`](vllm/multimodal/video.py) (lines 419–419)
- **Symbol:** `VIDEO_LOADER_REGISTRY.register("opencv")`
- **Kind:** plugin_seam
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0012`

## Description
Registration site for the default VideoLoader backend in the video loader registry, demonstrating the extension point used to add alternate frame sampling/decoding backends.

## Current approach
The interface is VideoLoader in vllm/multimodal/video.py:78, and the registered reference implementation is VideoBackend in vllm/multimodal/video.py:420. VideoMediaIO selects a backend through vllm/multimodal/media/video.py:63-67 using media_io_kwargs.video.video_backend or envs.VLLM_VIDEO_LOADER_BACKEND, then calls VIDEO_LOADER_REGISTRY.load(...). Existing registrations include opencv, opencv_dynamic, molmo2, nemotron_vl, and openpangu.

## Estimated impact explanation
Video workloads have widely different optimal decode paths. A backend implementing GPU decode or keyframe-aware sparse extraction can reduce video TTFT substantially while keeping the same VideoMediaIO caller contract.

## Evolve rationale
A new backend can be added with a VIDEO_LOADER_REGISTRY.register("name") decorator and a class implementing compute_frames_index_to_sample, load_bytes, and create_hf_metadata-compatible output. This is a concrete plugin seam for packet-aware PyAV, adaptive sparse seeking, or hardware decode policies without changing callers. Oracle: tests/multimodal/test_video.py and tests/multimodal/media/test_video.py must validate frames_array shape, frames_indices, do_sample_frames, total_num_frames/fps/duration metadata, and backend selection through video_backend.

## Deep research proposals

### 1. Register a budget-aware VideoLoader backend that caps frames and pixels from encoder token budget
- **Finding:** `find-0002` — *qwen-vl-utils · PyPI*
- **Source URL:** <https://pypi.org/project/qwen-vl-utils/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new backend class registered via @VIDEO_LOADER_REGISTRY.register("budget_aware") in vllm/multimodal/video.py alongside the existing opencv registration at line 419. Subclassing VideoLoader (vllm/multimodal/video.py:78), the backend exercises the same plugin seam that opencv/opencv_dynamic/molmo2/nemotron_vl/openpangu use today and is selectable through media_io_kwargs.video.video_backend or VLLM_VIDEO_LOADER_BACKEND in vllm/multimodal/media/video.py:63-67 — no caller changes required. The backend overrides compute_frames_index_to_sample to derive the sample count from a per-request visual token budget (max_pixels, max_frames, target_fps), inspired by qwen-vl-utils' VIDEO_MAX_PIXELS / FPS / FRAME caps. Concretely: read budget knobs from kwargs (with env fallback such as VLLM_VIDEO_MAX_PIXELS, VLLM_VIDEO_MAX_FRAMES, VLLM_VIDEO_TARGET_FPS); compute frames_to_sample = min(max_frames, ceil(duration * target_fps), floor(max_pixels / frame_pixels)); then return uniformly spaced indices subject to that cap. load_bytes can reuse PyAVVideoBackendMixin/OpenCVVideoBackendMixin to decode only the chosen indices, and create_hf_metadata is unchanged so total_num_frames/fps/duration/frames_indices/do_sample_frames remain populated as the existing oracle in tests/multimodal/test_video.py and tests/multimodal/media/test_video.py expects. Backend selection through video_backend continues to be the only integration surface, keeping the change isolated to this plugin seam.

**Proposal rationale.**

The candidate is the registration site for new video loader backends, and its evolve_rationale explicitly invites alternate sampling/decoding strategies behind the same VideoMediaIO contract. Finding find-0002 contributes a concrete, transferable idea from qwen-vl-utils — budget-aware visual sizing where max pixels, FPS, and frame caps are first-class processor inputs derived from encoder budgets — that is not currently expressed by any registered backend (opencv/opencv_dynamic uniformly sample without consulting a token/pixel budget). For the stated multi-turn agentic workload, capping decoded frames and pixels before placeholders and encoder work are scheduled directly attacks both media TTFT (less decode and encoder compute) and median TPOT (fewer visual tokens occupying KV/attention), which matches the candidate's high-impact estimate. Implementing this as a separate registered backend is faithful to the plugin seam (no caller changes) and preserves all metadata fields the existing test oracle validates.

---

### 2. Add a TorchCodec-backed VideoLoader for direct-tensor sparse frame decode
- **Finding:** `find-0003` — *VideoDecoder — TorchCodec 0.12.0+cu126 Documentation*
- **Source URL:** <https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Register a new backend at the existing plugin seam in vllm/multimodal/video.py:419 (e.g., VIDEO_LOADER_REGISTRY.register("torchcodec")) implementing the VideoLoader interface from vllm/multimodal/video.py:78. The backend wraps torchcodec.decoders.VideoDecoder and implements: (1) compute_frames_index_to_sample using existing fps/num_frames policy as in the OpenCV backend; (2) load_bytes by constructing a VideoDecoder over the input bytes/path and calling get_frames_at(indices) to retrieve only the sampled frames as a single FrameBatch, then exposing the .data tensor (already on CPU/GPU, NCHW or NHWC per device option) instead of decoding every packet through a per-frame loop and converting via numpy/PIL; (3) populate create_hf_metadata-compatible fields (total_num_frames, fps, duration, frames_indices, do_sample_frames) using VideoDecoder.metadata. Selection flows through the unchanged VideoMediaIO path at vllm/multimodal/media/video.py:63-67 via media_io_kwargs.video.video_backend="torchcodec" or VLLM_VIDEO_LOADER_BACKEND=torchcodec. No caller-side changes. Validation reuses tests/multimodal/test_video.py and tests/multimodal/media/test_video.py to assert frames_array shape, frames_indices, do_sample_frames, total_num_frames/fps/duration, and backend selection.

**Proposal rationale.**

The candidate is explicitly a plugin seam whose evolve_rationale calls out hardware/keyframe-aware decode backends as the high-impact axis, and TorchCodec's get_frames_at(indices) is a direct match for the sparse-sampling contract the existing OpenCV backend implements with a per-frame read loop plus numpy/PIL conversions. Replacing that loop with a single indexed decode that returns PyTorch tensors removes two TTFT contributors that the candidate's current_approach explicitly tolerates: repeated per-frame container seeks/decodes and host-side array conversions. This addresses the caller-context objective of reducing media TTFT for multi-turn agentic workloads where video prefill latency dominates, while preserving the VideoLoader contract so no caller changes are required.

---

### 3. Add a DALI-backed VideoLoader backend for GPU-accelerated decode and frame sampling
- **Finding:** `find-0004` — *NVIDIA DALI Documentation*
- **Source URL:** <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new VideoLoader implementation registered via VIDEO_LOADER_REGISTRY.register("dali") in vllm/multimodal/video.py alongside the existing opencv backend at line 419. The new class subclasses VideoLoader (vllm/multimodal/video.py:78) and uses NVIDIA DALI's video decoding operators (e.g., fn.experimental.decoders.video / fn.readers.video) to perform demux, decode, and frame sampling on the GPU. Implement compute_frames_index_to_sample to translate the requested num_frames/fps/duration into DALI sequence_length and stride parameters; implement load_bytes to push the encoded bytes into a DALI Pipeline (built once and reused/prefetched) that returns a tensor of shape (num_frames, H, W, C) plus the same frames_indices, total_num_frames, fps, and duration metadata that VideoMediaIO (vllm/multimodal/media/video.py:63-67) expects from create_hf_metadata. Selection remains driven by media_io_kwargs.video.video_backend or VLLM_VIDEO_LOADER_BACKEND=dali, so callers are unchanged. Resize/normalize stages can be fused into the same DALI pipeline so decode + preprocessing overlap serving work and run off the API CPU critical path. Validate via tests/multimodal/test_video.py and tests/multimodal/media/test_video.py for frames_array shape, frames_indices, do_sample_frames, metadata fields, and backend selection.

**Proposal rationale.**

This candidate is exactly a plugin seam for new video decode backends, and the finding describes a concrete GPU-accelerated decode + preprocessing library (DALI) that targets the same media-decode CPU bottleneck the candidate's evolve_rationale calls out ("GPU decode or keyframe-aware sparse extraction"). For the stated objective of reducing media TTFT in a multi-turn agentic workload, moving JPEG/H.264 demux+decode and resize/normalize off the CPU and overlapping them with prefetch directly addresses the dominant pre-prefill latency for video inputs. DALI's video reader natively supports stride/sequence_length sampling, which maps cleanly onto VideoLoader.compute_frames_index_to_sample without changing VideoMediaIO callers, so the finding is not merely topically adjacent but contributes a transferable, drop-in backend implementation idea.

---

### 4. Add a temporal-redundancy-aware VideoLoader backend that drops near-static frames at sample time
- **Finding:** `find-0008` — *Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference*
- **Source URL:** <https://arxiv.org/html/2510.14624v1>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Register a new backend at the VIDEO_LOADER_REGISTRY plugin seam in vllm/multimodal/video.py (alongside the opencv registration at line 419) that adapts the EVS idea of temporal-redundancy pruning to the frame-sampling stage controlled by VideoLoader. Concretely, implement a class (e.g., `EvsVideoBackend`) decorated with `@VIDEO_LOADER_REGISTRY.register("evs")` that subclasses the VideoLoader interface (vllm/multimodal/video.py:78) and: (1) in compute_frames_index_to_sample, perform an initial uniform/dense candidate index proposal, then on a fast decoded-thumbnail pass (e.g., low-resolution OpenCV grab) compute per-frame patch-grid difference signatures between consecutive candidates and drop candidates whose aggregate patch-change ratio is below a configurable threshold, while preserving the positional identity of retained frames so downstream rotary/positional encodings remain correct; (2) in load_bytes, decode only the surviving indices and emit frames_array, frames_indices, do_sample_frames, and total_num_frames/fps/duration in the same shape that VideoMediaIO and the existing opencv/opencv_dynamic backends produce, so VideoMediaIO selection in vllm/multimodal/media/video.py:63-67 (via media_io_kwargs.video.video_backend or VLLM_VIDEO_LOADER_BACKEND) needs no caller-side changes. Expose tuning knobs (similarity threshold, min frames floor, thumbnail resolution) through media_io_kwargs.video so the policy degenerates to standard uniform sampling when the threshold is 0. Validate via tests/multimodal/test_video.py and tests/multimodal/media/test_video.py for frames_array shape, frames_indices monotonicity, metadata correctness, and backend-selection plumbing.

**Proposal rationale.**

The candidate explicitly calls out adaptive sparse seeking as a target use of this plugin seam, and the finding's core insight - that consecutive-frame regions are often temporally redundant and can be pruned without retraining while preserving positional identity - is directly applicable to the frame-selection responsibility of VideoLoader.compute_frames_index_to_sample. While EVS in the paper prunes patches downstream of the vision encoder, the same temporal-redundancy signal is already detectable at decode time and can be exploited one stage earlier (frame-level dropping), which is the only knob exposed by this seam. For the stated multi-turn agentic workload with media TTFT and TPOT objectives, dropping decode and downstream encoder/prefill work for near-duplicate frames in long or static videos plausibly reduces both metrics, and the change is contained to a new backend registration with no caller impact, matching the candidate's plugin-seam contract.

---

### 5. Add a Decord-backed VideoLoader for sparse random-access frame sampling
- **Finding:** `find-0013` — *10. Introducing Decord: an efficient video reader - Gluon*
- **Source URL:** <https://cv.gluon.ai/build/examples_action_recognition/decord_loader.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Register a new video backend (e.g., "decord") at the plugin seam in vllm/multimodal/video.py:419 alongside the existing VIDEO_LOADER_REGISTRY.register("opencv") entry. Implement a DecordVideoBackend class that conforms to the VideoLoader interface defined at vllm/multimodal/video.py:78 (compute_frames_index_to_sample, load_bytes, and HF-metadata-compatible output). Internally, wrap a decord.VideoReader: use len(vr) for total_num_frames, vr.get_avg_fps() for fps, derive duration, and use vr.get_batch(frames_indices) to retrieve only the sampled frame indices in a single batched call instead of decoding sequentially. Where Decord is built with GPU support, expose a ctx=decord.gpu(device_id) option via media_io_kwargs.video so requests can opt into hardware-decoded reads. The selection path remains unchanged: VideoMediaIO at vllm/multimodal/media/video.py:63-67 already routes via media_io_kwargs.video.video_backend or envs.VLLM_VIDEO_LOADER_BACKEND, so users can flip to "decord" without caller changes. Make the import lazy and mark the backend optional (raise an actionable error at registration use, not at import time, if decord is not installed). Validate via tests/multimodal/test_video.py and tests/multimodal/media/test_video.py: frames_array shape, frames_indices, do_sample_frames, total_num_frames/fps/duration metadata, and backend selection through video_backend.

**Proposal rationale.**

The candidate is explicitly a plugin seam for new video decode strategies, and the finding contributes a concrete, transferable backend choice that targets the exact gap the candidate calls out: "adaptive sparse seeking" and "hardware decode policies." Decord's get_batch and fast random access map directly onto compute_frames_index_to_sample + load_bytes, avoiding OpenCV's sequential per-frame grab/retrieve loop when processors request non-contiguous sampled frames - which is the dominant pattern for video prompts. Reducing wall-clock decode for the sampled frame subset directly attacks media TTFT (the stated workload objective) without altering VideoMediaIO callers, and the optional-import pattern keeps the seam safe for environments that lack Decord.

---

## Agent proposals

### 1. Register a content-addressed caching VideoLoader that memoizes decoded frame tensors across multi-turn requests
- **Agent:** claude

**Detailed description.**

At the plugin seam in vllm/multimodal/video.py:419, register a new wrapper backend (e.g., @VIDEO_LOADER_REGISTRY.register("cached")) that subclasses the VideoLoader interface defined at vllm/multimodal/video.py:78 and delegates the actual decode to an inner backend (configurable via media_io_kwargs.video.cached_inner_backend, defaulting to opencv) while interposing a process-local LRU cache keyed by a stable content-addressed signature. The cache key is (blake2b(input_bytes), num_frames, requested_fps, target_duration, inner_backend_name) so that identical (video, sampling spec) tuples encountered across different requests/turns short-circuit decode and metadata construction entirely. Concretely: (1) compute_frames_index_to_sample first probes the cache with a partial key derived from the input bytes hash and the sampling args, returning the cached frames_indices when present so even the index computation is skipped; (2) load_bytes returns the cached frames_array when present, otherwise calls the inner backend and stores (frames_array, frames_indices, total_num_frames, fps, duration, do_sample_frames) in the cache; (3) create_hf_metadata mirrors the inner backend so VideoMediaIO at vllm/multimodal/media/video.py:63-67 sees the same field set the existing oracle in tests/multimodal/test_video.py and tests/multimodal/media/test_video.py validates. Cache budget is governed by env (VLLM_VIDEO_DECODE_CACHE_BYTES, default e.g. 2 GiB) with size accounting on the numpy array nbytes; eviction is LRU. The hash is computed once per input and reused. Selection remains driven entirely through media_io_kwargs.video.video_backend or VLLM_VIDEO_LOADER_BACKEND=cached, so no caller in VideoMediaIO or upstream changes. The backend can also be composed with any existing or future backend (opencv, opencv_dynamic, torchcodec, decord, dali, evs, budget_aware) by changing cached_inner_backend.

**Novelty rationale.**

All five existing deep_research_proposals target a single decode of one video: find-0002 caps frame/pixel count by budget, find-0003 swaps in TorchCodec for sparse indexed decode, find-0004 moves decode to GPU via DALI, find-0008 prunes temporally redundant frames at sample time, and find-0013 uses Decord's batched random access. None of them exploit reuse of the same video across requests or turns, which is the dominant temporal-locality pattern of the stated multi-turn agentic workload (an agent re-reads the same uploaded video each turn while reasoning, calling tools, etc.). A content-addressed decode cache attacks media TTFT on every turn after the first by reducing it to a hash + dictionary lookup, is orthogonal to all listed backends (it composes with any of them as the inner decoder), and lives at the same VIDEO_LOADER_REGISTRY plugin seam so it preserves the candidate's contract without caller changes.

---

### 2. Register a GOP-aware PyAV VideoLoader backend for exact sparse frame extraction
- **Agent:** codex

**Detailed description.**

Add a new backend at vllm/multimodal/video.py:419, for example @VIDEO_LOADER_REGISTRY.register("pyav_gop"), implemented as a VideoBackend subclass that reuses VideoBackend.compute_frames_index_to_sample but replaces the current PyAV per-target seek path with packet-aware GOP grouping. In load_bytes, open the bytes with av.open(BytesIO(data)), build lightweight source metadata with PyAVVideoBackendMixin.get_metadata, compute frame_idx, then map requested frame indices to timestamps and group adjacent targets by their preceding keyframe packet. For each group, seek once to the keyframe, decode forward only until the last requested frame in that GOP, and emit frames exactly when decoded frame timestamps match the requested indices. Return the same create_hf_metadata fields: total_num_frames, fps, duration, video_backend="pyav_gop", frames_indices, and do_sample_frames. Selection stays on the existing VideoMediaIO path through media_io_kwargs.video.video_backend="pyav_gop" or VLLM_VIDEO_LOADER_BACKEND=pyav_gop. Validate in tests/multimodal/test_video.py and tests/multimodal/media/test_video.py with a synthetic video containing multiple sampled frames per GOP to assert shape, monotonic frames_indices, backend selection, and exact target-frame extraction rather than repeatedly returning the first frame after each seek.

**Novelty rationale.**

This is not the budget-aware proposal, because it does not change frame count or pixel policy. It is not TorchCodec, Decord, or DALI, because it uses the existing PyAV/FFmpeg dependency path already present in vllm/multimodal/video.py and improves the seek/decode algorithm rather than introducing a new decoder library. It is not EVS, because it does not drop frames based on visual redundancy. It is not Claude's cache, because it optimizes the first decode of a new video instead of memoizing repeated decodes. The specific new idea is GOP-level batching for sparse PyAV extraction, which reduces repeated keyframe seeks and redundant decode while preserving the current VideoLoader metadata contract.

---
