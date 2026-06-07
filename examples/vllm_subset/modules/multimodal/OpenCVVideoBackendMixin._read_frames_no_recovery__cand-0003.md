# OpenCVVideoBackendMixin._read_frames_no_recovery

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/video.py`](vllm/multimodal/video.py) (lines 274–323)
- **Symbol:** `OpenCVVideoBackendMixin._read_frames_no_recovery`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0003`

## Description
Reads requested frame indices from an OpenCV VideoCapture without recovery, returning the loaded RGB frames and their valid source indices.

## Current approach
The method linearly scans from frame 0 through max_frame_idx, calling cap.grab() for every intermediate frame and cap.retrieve() only for target indices. Sparse sampling therefore still pays a grab/demux path for every skipped frame.

## Estimated impact explanation
Video TTFT is often dominated by frame extraction. For long clips with sparse uniform sampling, this loop performs work proportional to the last requested frame rather than the number of requested frames, so a seek-aware policy can substantially reduce first-token latency.

## Evolve rationale
The loop over range(max_frame_idx + 1) at lines 288-311 is a concrete hand-rolled frame access policy. A hybrid seek/scan strategy could use cap.set(CAP_PROP_POS_FRAMES, idx) or keyframe-aware seeking when the gap to the next requested frame is large, while retaining the current grab loop for dense targets. Oracle: tests/multimodal/test_video.py and tests/multimodal/media/test_video.py should compare frames_array shape, valid_frame_indices, and sampled pixel content against the current linear scan for supported codecs, with any seek tolerance explicitly bounded.

## Deep research proposals

### 1. Add a TorchCodec-backed sparse frame path to OpenCV video reader
- **Finding:** `find-0003` — *VideoDecoder — TorchCodec 0.12.0+cu126 Documentation*
- **Source URL:** <https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce an alternative decode path that, when TorchCodec is available, replaces the linear grab/retrieve loop in OpenCVVideoBackendMixin._read_frames_no_recovery (vllm/multimodal/video.py:274-323) with a single call to torchcodec.decoders.VideoDecoder(...).get_frames_at(indices). Concretely: build a VideoDecoder over the same source the OpenCV backend currently opens, pass the requested frame_indices list directly to get_frames_at, and return the resulting FrameBatch's .data as the RGB frames_array along with the indices that the decoder reports as valid. Keep the existing OpenCV grab/retrieve loop as the fallback for codecs/containers TorchCodec cannot handle, and gate the new path behind a capability check plus the same valid_frame_indices contract used today (so callers and the no_recovery/ recovery split in this mixin remain unchanged). The oracle described in evolve_rationale (tests/multimodal/test_video.py and tests/multimodal/media/test_video.py) should validate frames_array shape, valid_frame_indices, and per-pixel agreement against the current linear scan within an explicitly bounded tolerance.

**Proposal rationale.**

The candidate's bottleneck is that sparse sampling still pays cap.grab() for every intermediate frame from 0 to max_frame_idx, so cost scales with the last requested index rather than the number of sampled frames. The finding documents a directly applicable primitive — VideoDecoder.get_frames_at(indices) — that is purpose-built for retrieving an arbitrary set of frame indices and returns torch tensors, eliminating both the per-frame grab/demux loop and the numpy/PIL round trips that follow it. This addresses exactly the gap flagged in evolve_rationale (a hybrid seek/scan strategy for sparse targets) and targets the media TTFT objective for the multi-turn agentic workload, while leaving the existing OpenCV path intact as a compatibility fallback.

---

### 2. Replace OpenCV linear-scan frame reader with DALI GPU-accelerated frame decoder for sparse video sampling
- **Finding:** `find-0004` — *NVIDIA DALI Documentation*
- **Source URL:** <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the linear grab/retrieve loop in OpenCVVideoBackendMixin._read_frames_no_recovery (vllm/multimodal/video.py:274-323) with an NVIDIA DALI-based decoding path for sparse frame sampling. Concretely: introduce an alternative backend (selected when DALI and a CUDA device are available) that builds a small DALI pipeline using a video reader operator (e.g. fn.experimental.readers.video / fn.decoders.image for the per-frame fallback) configured to read only the requested frame indices from the video bytes. The pipeline runs decode on GPU via NVDEC and returns the decoded RGB frames already on device (or copied back to host to match the existing np.ndarray contract at line 320). The current OpenCV path remains the fallback when DALI is unavailable, when codecs are unsupported, or when the requested target set is dense enough that the existing grab loop is competitive. Validate against the same oracle suggested in evolve_rationale: tests/multimodal/test_video.py and tests/multimodal/media/test_video.py must show matching frames_array shape, valid_frame_indices, and pixel content (within an explicit decoder tolerance) for supported codecs.

**Proposal rationale.**

The candidate's bottleneck is that sparse frame sampling pays demux/grab cost proportional to max_frame_idx rather than to len(frame_indices), directly inflating media TTFT. DALI's video readers are explicitly designed to decode specified frame indices on GPU (NVDEC), removing both the per-frame Python-side grab loop and the CPU decode cost from the critical path, and they support overlapping preprocessing (resize/normalize already done downstream) with serving work via prefetching. This addresses the same constraint identified in evolve_rationale (avoid touching every intermediate frame) but with a stronger mechanism than CAP_PROP_POS_FRAMES seeking, which is codec-dependent and often silently approximate. The finding is concrete and transferable: DALI provides the building blocks needed (GPU video reader with frame-index selection, mixed CPU/GPU stages) to plausibly reduce TTFT on long, sparsely sampled clips in the multi-turn agentic workload.

---

### 3. Add an optional Decord-backed sparse frame reader alongside the OpenCV linear scan
- **Finding:** `find-0013` — *10. Introducing Decord: an efficient video reader - Gluon*
- **Source URL:** <https://cv.gluon.ai/build/examples_action_recognition/decord_loader.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce an alternative video backend (or a sibling classmethod next to OpenCVVideoBackendMixin._read_frames_no_recovery in vllm/multimodal/video.py:274-323) that uses Decord's VideoReader.get_batch(frame_indices) when Decord is importable. The new path replaces the range(max_frame_idx + 1) cap.grab() loop with a single random-access batched call: vr = decord.VideoReader(path); arr = vr.get_batch(sorted_indices).asnumpy(); then map the rows back to valid_frame_indices, falling back to the existing OpenCV scan when Decord is unavailable, the codec is unsupported, or get_batch raises. Preserve the current contract: return frames as RGB uint8 with shape (num_valid, H, W, 3) plus the list of valid_frame_indices, and keep the existing per-frame failure logging by detecting indices missing from Decord's batch result. Gate the backend behind an env var or config knob (mirroring existing backend selection in this file) so the default behavior is unchanged, and reuse the recovery path that wraps _read_frames_no_recovery so corrupted clips still degrade gracefully. Tests in tests/multimodal/test_video.py and tests/multimodal/media/test_video.py should be extended to assert frames_array shape, valid_frame_indices, and pixel parity against the OpenCV baseline within an explicitly bounded tolerance for supported codecs.

**Proposal rationale.**

The candidate's core inefficiency is that sparse sampling still pays a grab/demux cost for every frame from 0 to max_frame_idx, making cost proportional to clip length rather than to the number of requested frames. The finding documents Decord as a video reader explicitly designed for fast random access and batched frame retrieval via get_batch, which is exactly the primitive missing here. Substituting Decord's get_batch for the linear cap.grab() loop on the sparse-sampling path directly attacks the high-impact TTFT gap called out in evolve_rationale, while leaving the OpenCV implementation as a fallback preserves current behavior on platforms or codecs where Decord is unavailable.

---

## Agent proposals

### 1. Hybrid OpenCV seek+scan using CAP_PROP_POS_FRAMES with keyframe-aware fallback
- **Agent:** claude

**Detailed description.**

Modify OpenCVVideoBackendMixin._read_frames_no_recovery in vllm/multimodal/video.py:274-323 to replace the unconditional range(max_frame_idx + 1) cap.grab() loop with a hybrid policy that stays inside the existing OpenCV backend (no new library dependency). Concretely: (1) sort the requested frame_indices; (2) iterate over them in order, and for each consecutive pair compute the gap; (3) when the gap exceeds a tunable threshold (e.g. >= GOP_HEURISTIC, default 32 or read from a constant), call cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx) to jump near the target — OpenCV internally seeks to the nearest preceding keyframe and decodes forward — then verify with cap.get(cv2.CAP_PROP_POS_FRAMES); (4) when the actual position lands before target_idx (typical, since seek snaps to a keyframe), advance with cap.grab() until the position matches, then cap.retrieve(); (5) for small gaps (dense regions of the requested set), retain the existing grab/retrieve loop unchanged because it is faster than re-seeking inside a GOP. Preserve the existing valid_frame_indices contract by treating any seek/grab failure exactly like the current code's failure path (skip the index, log, continue). Gate the new behavior behind an env var (e.g. VLLM_VIDEO_OPENCV_SEEK=1) defaulting to off initially so adopters can opt in per codec, and use the same oracle from evolve_rationale (tests/multimodal/test_video.py and tests/multimodal/media/test_video.py) to assert frames_array shape, valid_frame_indices, and pixel-level agreement with the linear scan within a bounded tolerance — codecs where seek is unsafe (e.g. open-GOP H.264 with B-frames in some containers) can be excluded via a small denylist or by verifying the post-seek position matches the target before accepting frames.

**Novelty rationale.**

All three existing deep_research_proposals (find-0003 TorchCodec, find-0004 DALI, find-0013 Decord) replace OpenCV with a different library that exposes a batched random-access primitive. None of them address the alternative explicitly suggested in evolve_rationale — using OpenCV's own cap.set(CAP_PROP_POS_FRAMES) with keyframe-aware seeking, retaining the grab loop only for dense targets. This proposal requires zero new dependencies (works on every platform vLLM already supports), composes with the existing recovery wrapper unchanged, and ships independently of TorchCodec/DALI/Decord availability, so it is materially different in mechanism, deployment surface, and dependency footprint from all listed proposals.

---

### 2. Cache sampled OpenCV frame batches for repeated video inputs
- **Agent:** codex

**Detailed description.**

Add a bounded process-local LRU cache around the no-recovery OpenCV path so repeated requests for the same video bytes and same resolved sampling parameters bypass OpenCVVideoBackendMixin._read_frames_no_recovery entirely. Concretely, in the load_bytes path after computing source metadata and frame_idx, and before calling read_frames with frame_recovery=False/backend="opencv", look up a key such as (sha256(data), loader class/sampling suffix, tuple(frame_idx), backend, relevant media kwargs). On miss, run the existing _read_frames_no_recovery path unchanged and store (frames, valid_frame_indices, metadata) with size accounting based on frames.nbytes; on hit, return a copy of the cached frames plus a copied metadata dict so downstream processors cannot mutate the cache entry. Gate the cache behind a small configurable byte budget or reuse the existing multimodal cache sizing conventions, and add tests that monkeypatch _read_frames_no_recovery to assert identical second calls avoid decode, changed sampling kwargs miss the cache, and mutating the first returned array does not corrupt later cache hits.

**Novelty rationale.**

The listed deep_research_proposals and Agent A all optimize a single sparse decode by replacing or augmenting the frame access policy: TorchCodec, DALI, Decord, or OpenCV seek+scan. This proposal does not introduce another decoder and does not rely on seeking; it avoids invoking the candidate method at all when multi-turn agentic workloads resend the same video with the same sampling request. That repeated-input media-cache behavior is not covered by any of the existing proposals and targets a different TTFT failure mode than per-video sparse extraction cost.

---
