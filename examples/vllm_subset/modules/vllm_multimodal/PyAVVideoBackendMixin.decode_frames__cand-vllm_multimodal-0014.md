# PyAVVideoBackendMixin.decode_frames

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/video.py`](vllm/multimodal/video.py) (lines 547–590)
- **Symbol:** `PyAVVideoBackendMixin.decode_frames`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0014`

## Description
Decodes requested frame indices from a PyAV container using seek and forward decode to target PTS.

## Current approach
Sets stream.thread_type to SLICE, loops over frame_indices, converts each index to timestamp and PTS, seeks when the requested PTS does not advance, decodes forward until a frame reaches the target PTS, converts each chosen frame to an RGB ndarray, appends to a list, and stacks at the end.

## Estimated impact explanation
PyAV is a common video ingress path. Reducing seek/decode churn and per-frame allocation lowers media preprocessing wall time and therefore TTFT for video-heavy requests.

## Evolve rationale
The concrete constructs are container.seek, the nested decoder loop, per-frame to_ndarray(format='rgb24'), and final np.stack. Tracking keyframe/GOP intervals to choose seek-vs-continue decisions and preallocating the output array can reduce repeated decode and allocation while preserving selected-frame semantics. Correctness oracle: tests/multimodal/test_video.py; sampled indices, valid indices, NHWC uint8 shape, and empty-output behavior must remain correct.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Batch-sort frame_indices and coalesce seeks by GOP with preallocated output buffer
- **Agent:** claude

**Detailed description.**

In PyAVVideoBackendMixin.decode_frames (vllm/multimodal/video.py:547-590), replace the per-index seek-or-continue loop with a two-part optimization tailored to the actual TTFT bottleneck (repeated GOP re-decode + list-append-then-stack allocations):

1) Sort-and-restore ordering. Compute a stable argsort of frame_indices so the physical decode walks PTS monotonically (single forward pass through the container), then scatter decoded frames back into their original request order at the end. Sampled frame_indices in vLLM's video pipeline are already usually monotonic, but the current code has no guarantee — a single out-of-order index forces a fresh seek + GOP-prefix redecode. Monotonic traversal removes that class of wasted work entirely and preserves selected-frame semantics (identical output for identical inputs) which tests/multimodal/test_video.py asserts against.

2) Coarse GOP-gap heuristic to avoid useless seeks. When last_pts is set and the next target pts is only a few frames ahead (delta * time_base < a small threshold such as ~0.5s, or fewer than ~stream.gop_size frames if exposed), skip the container.seek() entirely and keep draining the existing decoder — PyAV's seek() snaps backward to the nearest keyframe, so seeking forward by a small delta is strictly slower than just decoding through. When the gap is large, keep the current seek path. This is a pure local change to the seek-vs-continue predicate on line 573; correctness is preserved because we still walk forward to frame.pts >= pts.

3) Preallocate the output ndarray. Resolve stream.codec_context.width/height once (or peek the first decoded frame's shape), then allocate frames_out = np.empty((len(frame_indices), H, W, 3), dtype=np.uint8) up front and write each decoded RGB frame into frames_out[original_position] via to_ndarray(format='rgb24') — eliminating the intermediate frames_list of len(N) ndarrays and the final np.stack copy. On dropped/invalid frames, trim to frames_out[:len(valid_indices)] before returning; empty-output behavior (np.empty((0,), dtype=np.uint8)) at line 588-589 is unchanged.

All three sub-changes are localized to this method and keep the NHWC uint8 contract that tests/multimodal/test_video.py validates (sampled-indices semantics, valid-indices ordering, empty-output). Expected wins on the multi-turn agentic video workload: fewer redundant GOP prefix decodes per request (largest saving), one fewer full-tensor allocation-and-copy at stack time, and lower peak memory during preprocessing — all of which shave preprocessing wall-time and therefore median TTFT for video-heavy turns.

**Novelty rationale.**

The candidate lists zero existing deep_research_proposals, so any concrete change is novel by construction. Beyond that: the evolve_rationale gestures at 'tracking keyframe/GOP intervals' and 'preallocating the output array' as directions, but does not prescribe (a) sorting the request order to guarantee monotonic PTS traversal (which is the actual mechanism that eliminates seek-caused GOP redecode), (b) a concrete seek-skip predicate based on PTS delta vs. a small time threshold, or (c) scatter-write into a preallocated NHWC buffer keyed by original request position to eliminate np.stack. This proposal names those three specific, localized code edits at lines 567-590 with a correctness argument tied to the existing test oracle.

---

### 2. Memoize duplicate target frames before seeking
- **Agent:** codex

**Detailed description.**

In `PyAVVideoBackendMixin.decode_frames` (`vllm/multimodal/video.py:547-590`), add a small per-call cache keyed by the computed target `pts` (or by the clamped timestamp/PTS plus frame index semantics) so repeated requested frames are copied from an already decoded RGB ndarray instead of forcing another seek/decode. This matters when sampling asks for more frames than the source can uniquely provide, or when timestamp clamping maps several high indices to the same final `max_ts`: the current `pts <= last_pts` branch seeks again for equal PTS and then re-decodes the same keyframe prefix. After converting `chosen.to_ndarray(format="rgb24")`, store it in `decoded_by_pts[pts]`; at the start of each loop, if `pts` is already present, append a copy or reference consistent with downstream immutability expectations and add the current `idx` to `valid_indices` without touching the container. Preserve empty-output behavior and the existing `(N, H, W, 3)` uint8 return contract.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A covers sorting frame indices, GOP-aware seek coalescing, and output preallocation, but it does not explicitly handle equal target PTS values as a deduplication case. Sorting alone still needs special handling for duplicate PTS because the decoder cannot move forward to produce the same already-consumed frame again; this proposal is a separate cache-based fast path for duplicate or clamped targets.

---
