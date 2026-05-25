# VideoMediaIO.load_base64

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/media/video.py`](vllm/multimodal/media/video.py) (lines 74–140)
- **Symbol:** `VideoMediaIO.load_base64`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0016`

## Description
Loads base64 video inputs, including comma-separated JPEG frame sequences, into a frame array and metadata dict.

## Current approach
For media_type video/jpeg, the method splits the full data string into frame strings, decodes each frame serially through ImageMediaIO.load_base64 in a list comprehension, converts each PIL image with np.asarray, and stacks the resulting list with np.stack before validating metadata.

## Estimated impact explanation
This path matters for clients that send videos as JPEG frame sequences rather than container bytes. Serial frame decode and stack allocation scale with frame count, so improving it reduces video media TTFT for those API workloads.

## Evolve rationale
The data.split plus serial decode/np.stack block at lines 83-92 is a concrete decode batching target. Headroom includes streaming delimiter parsing to avoid materializing all unused frame strings, parallel frame decode for large JPEG sequences, preallocating the output frame array after the first decode, and stopping exactly at num_frames without extra list churn. Oracle: tests/multimodal/media/test_video.py JPEG-sequence cases must preserve frame pixels, frames_indices validation, total_num_frames/duration/fps metadata, do_sample_frames, and num_frames error behavior.

## Deep research proposals

### 1. Offload base64 JPEG-sequence video decode to a GPU DALI pipeline
- **Finding:** `find-0004` — *NVIDIA DALI Documentation*
- **Source URL:** <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the serial PIL-based decode path in VideoMediaIO.load_base64 (vllm/multimodal/media/video.py:74-140, specifically the data.split + ImageMediaIO.load_base64 list comprehension + np.asarray + np.stack block at lines 83-92) with a NVIDIA DALI pipeline for the video/jpeg branch. The change: (1) base64-decode each frame string into raw JPEG bytes (or stream the comma-delimited input lazily, stopping after num_frames items rather than materializing the full split list), (2) feed the byte buffers into a DALI ExternalSource and use fn.decoders.image (mixed/GPU backend, nvJPEG) to batch-decode the JPEG frames in parallel, (3) optionally fuse downstream resize/normalization stages that today run after load_base64, and (4) return a single contiguous frame tensor/array that replaces the np.stack output, preserving the existing metadata dict (total_num_frames, duration, fps, frames_indices) and do_sample_frames / num_frames error semantics that tests/multimodal/media/test_video.py exercises. Gate the DALI path behind a feature flag and an availability check, falling back to the current PIL path when DALI or a CUDA device is not present, so CPU-only deployments and existing oracles remain green.

**Proposal rationale.**

The candidate's hot path is exactly what DALI is designed to accelerate: a batch of independent JPEG decodes feeding a stacked frame tensor. The current implementation is serial, CPU-bound, and forces a per-frame PIL->numpy->np.stack round trip on the API request thread, which directly inflates media TTFT for JPEG-sequence video inputs. DALI's nvJPEG-backed mixed pipeline parallelizes decode across frames, can prefetch and overlap with model execution (helping median TPOT in the multi-turn agentic workload), and produces a contiguous device buffer that avoids the np.stack allocation. The finding contributes a concrete, transferable mechanism (GPU decode + prefetching pipeline) that addresses the specific bottlenecks named in the candidate's evolve_rationale (serial decode and stack allocation scaling with frame count), rather than merely restating the current approach.

---

### 2. Use Pillow-SIMD as drop-in PIL backend to accelerate JPEG frame decode in VideoMediaIO.load_base64
- **Finding:** `find-0014` — *GitHub - uploadcare/pillow-simd: The friendly PIL fork*
- **Source URL:** <https://github.com/uploadcare/pillow-simd>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

For VideoMediaIO.load_base64 in vllm/multimodal/media/video.py:74-140, the per-frame JPEG decode at lines 83-92 calls ImageMediaIO.load_base64 in a serial list comprehension; each call internally invokes PIL.Image.open + decode on the base64-decoded bytes. Make Pillow-SIMD an optional, supported backend on x86: detect at import time whether the active Pillow distribution is Pillow-SIMD (e.g. by checking PIL.__version__ for the '.postN' suffix or PIL.features), and document/recommend installing 'pillow-simd' in CPU-image-heavy deployment environments where this video/jpeg frame-sequence path is hot. No call-site code change is required because Pillow-SIMD is API-compatible with Pillow; the speedup comes from the SIMD-accelerated JPEG decode and the np.asarray conversion path that ImageMediaIO already uses. Combine with the candidate's existing parallel-decode / preallocation refactor so each worker thread benefits from the faster per-frame decode. Preserve all oracle behaviors validated by tests/multimodal/media/test_video.py (frame pixels, frames_indices, total_num_frames/duration/fps metadata, do_sample_frames, num_frames error behavior) since pixel output is bit-equivalent for JPEG decode under Pillow-SIMD.

**Proposal rationale.**

The candidate's bottleneck on the video/jpeg branch is repeated PIL-based JPEG decoding per frame, and Pillow-SIMD is documented to be 4-6x faster than stock Pillow on x86 for exactly these decode/convert operations. Because it is a drop-in replacement, it directly addresses the per-frame CPU cost component of media TTFT for the JPEG-frame-sequence workload that this candidate targets, without altering the API contract or the metadata validation logic. It is complementary to (not a substitute for) the parallel-decode and preallocation evolutions, multiplying their effective throughput on x86 CPU hosts that dominate CPU-side preprocessing in serving deployments.

---

## Agent proposals

### 1. Content-addressed LRU cache of decoded frame arrays in VideoMediaIO.load_base64
- **Agent:** claude

**Detailed description.**

Wrap the JPEG-sequence decode block at vllm/multimodal/media/video.py:83-92 (the data.split + serial ImageMediaIO.load_base64 list comprehension + np.asarray + np.stack) with a bounded, content-addressed cache keyed on a fast hash of the raw base64 payload combined with the decode-shaping parameters that affect the output (num_frames, do_sample_frames, and any frames_indices / sampling knobs read inside load_base64). Concretely: (1) before splitting/decoding, compute a cheap fingerprint of the base64 string (e.g. blake3 or xxhash over the bytes, or a (len, head, tail, mid) sketch when full hashing is too expensive) plus a tuple of the sampling parameters; (2) consult a process-level LRU (e.g. functools.lru_cache-style or a small custom OrderedDict guarded by a lock, sized via an env var like VLLM_VIDEO_FRAME_CACHE_BYTES with a sensible default and per-entry byte accounting so we evict by total ndarray nbytes, not entry count) that maps fingerprint -> (frames_ndarray, metadata_dict); (3) on hit, return a shallow copy of the metadata dict and the cached ndarray (read-only / np.ascontiguousarray view) so callers cannot mutate the cached buffer; (4) on miss, run the existing decode/stack path, then insert the result. Preserve the existing metadata validation (total_num_frames, duration, fps, frames_indices) and num_frames error behavior by performing validation before insertion so that error-raising inputs are not cached. Make the cache opt-in via env var and disabled in tests by default so tests/multimodal/media/test_video.py oracles continue to pass deterministically.

**Novelty rationale.**

Neither existing proposal addresses cross-call reuse: find-0004 (DALI GPU decode) and find-0014 (Pillow-SIMD) both still re-decode every frame on every call, just with faster per-frame primitives. This proposal eliminates decode work entirely on cache hits, which is exactly the regime the caller context (multi-turn agentic workload, with the same video media frequently re-sent across turns) selects for. It is also complementary to either accelerator: the cache sits above the decode primitive and multiplies its benefit on repeated payloads, and it specifically targets median TPOT for later turns (where TTFT-style decode optimizations have already paid off once) rather than only first-turn TTFT.

---

### 2. Deduplicate repeated JPEG frames within each base64 sequence before decoding
- **Agent:** codex

**Detailed description.**

In `vllm/multimodal/media/video.py` inside `VideoMediaIO.load_base64`'s `video/jpeg` branch, add a small within-request deduplication step after computing the truncated `frame_parts` and before the current `ImageMediaIO.load_base64` decode loop. Build an ordered map from each frame's base64 string to the positions where it appears. If every frame is unique, keep the current decode path. If duplicates exist, decode each unique frame string exactly once, convert it with `np.asarray`, allocate the final `(len(frame_parts), H, W, C)` array from the first decoded frame's shape/dtype, and copy the decoded array into all duplicate positions. Keep the same metadata validation and error behavior, including `frames_indices`, `total_num_frames`, `duration`, `fps`, `do_sample_frames`, and `num_frames=0`; for inconsistent unique-frame shapes, raise an error equivalent to the current `np.stack` failure. Add a focused test that sends a sequence with repeated identical JPEG base64 frames and asserts pixel equivalence and unchanged metadata, ideally with a spy/mocked `ImageMediaIO.load_base64` call count to verify duplicates are decoded once.

**Novelty rationale.**

The DALI proposal accelerates every frame decode with a GPU pipeline, and the Pillow-SIMD proposal accelerates the per-frame PIL backend, but both still treat every position in the JPEG sequence as a separate decode. Agent A's cache is cross-call and keyed on the whole video payload, so the first request for a repeated-frame sequence still decodes duplicates. This proposal is a stateless, within-call optimization that specifically exploits repeated frames common in agentic screen/video workloads, reducing decode work even on the first occurrence of a payload and remaining complementary to GPU decode, Pillow-SIMD, or whole-payload caching.

---
