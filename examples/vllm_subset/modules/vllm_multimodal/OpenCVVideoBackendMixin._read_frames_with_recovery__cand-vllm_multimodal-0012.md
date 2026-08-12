# OpenCVVideoBackendMixin._read_frames_with_recovery

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/video.py`](vllm/multimodal/video.py) (lines 328–432)
- **Symbol:** `OpenCVVideoBackendMixin._read_frames_with_recovery`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0012`

## Description
Reads selected OpenCV frames with forward-scan recovery for failed target frames.

## Current approach
Walks frame by frame from zero to the max target index with cap.grab, checks target membership in a set, retrieves selected or recovery frames, converts each frame with cv2.cvtColor, appends to a Python list, and stacks at the end.

## Estimated impact explanation
OpenCV video ingress can dominate CPU preprocessing for sampled video prompts. Reducing per-frame conversion and allocation overhead lowers TTFT in video workloads.

## Evolve rationale
The concrete hot constructs are per-frame cv2.cvtColor, frames_list append plus final np.stack, and next_target_map lookups during the scan. Preallocating the output when dimensions are known, batching BGR-to-RGB after stacking, and simplifying recovery state can reduce copies and Python overhead. Correctness oracle: tests/multimodal/test_video.py; recovered/no-recovery frame counts, valid indices, dtype, and NHWC shape invariants must hold.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace linear grab-scan with keyframe seek + bounded recovery window
- **Agent:** claude

**Detailed description.**

In OpenCVVideoBackendMixin._read_frames_with_recovery (vllm/multimodal/video.py:328-432), replace the current O(max_frame_idx) linear cap.grab() walk with a per-target seek-and-decode path, keeping the existing scan only as a bounded recovery/fallback:

1. Preallocate output as (len(frame_indices), H, W, 3) uint8 (like _read_frames_no_recovery at lines 442-444) instead of appending to a Python list and np.stack'ing at the end.
2. For each target idx in frame_indices, call cap.set(cv2.CAP_PROP_POS_FRAMES, idx) then cap.retrieve() (or cap.read()). OpenCV/FFmpeg resolve this to the nearest keyframe and decode forward internally, so sampling N frames from an M-frame video costs O(N * keyframe_gap) instead of O(M) grabs.
3. On retrieve failure, perform a bounded forward grab scan up to min(next_target_idx, idx + K) frames (K small, e.g. 16) to preserve the current recovery semantics — the first successful decoded frame in that window populates recovered_map[idx] = source_idx, matching the invariants asserted in tests/multimodal/test_video.py.
4. Probe seek reliability once at entry: after cap.set(CAP_PROP_POS_FRAMES, 0), if CAP_PROP_POS_FRAMES does not reflect the set value or CAP_PROP_FRAME_COUNT is 0/unknown (VFR/live/malformed containers), fall back to the existing linear grab-scan implementation unchanged.
5. Defer BGR→RGB: write raw BGR frames into the preallocated buffer, then do a single contiguous frames[..., ::-1] view or one bulk cv2.cvtColor on the stacked buffer after the loop, eliminating N per-frame Python cvtColor call overhead.

Keep valid_frame_indices, recovered_map, dtype (uint8), and NHWC shape identical to today's contract so tests/multimodal/test_video.py and read_frames' downstream logging at lines 494-520 are unaffected.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, and the candidate's own evolve_rationale calls out only per-frame cv2.cvtColor, list-append + np.stack, and next_target_map lookups — i.e. constant-factor cleanups inside the grab loop. The seek-based sampling change replaces the O(max_frame_idx) grab loop itself with an O(N_targets) seek path, attacking a different and larger cost (dominant for sparsely-sampled long videos, which is exactly the TTFT-relevant multi-turn agentic video case). Preallocation and batched BGR→RGB are retained here but are secondary; the seek strategy plus bounded recovery-window fallback is the novel lever not covered by the candidate's rationale.

---

### 2. Avoid recovery-induced extra frames for target indices
- **Agent:** codex

**Detailed description.**

Tighten `OpenCVVideoBackendMixin._read_frames_with_recovery` so a successfully retrieved target frame is never also consumed as the recovery source for an earlier failed target. Today `can_recover` is computed independently from `is_target_frame`, so if frame N failed and the next target frame M retrieves successfully, the method appends M once, records `valid_frame_indices.append(M)`, and then maps `recovered_map[N] = M` while popping the failed frame. That collapses two logical requested frames onto one physical output row, making `valid_frame_indices` and `recovered_map` ambiguous for downstream prompt/image-token accounting. Change the recovery predicate to only allow non-target source frames, e.g. `can_recover = (not is_target_frame) and cls._can_use_for_recovery(...)`, and keep failed targets pending until a non-target frame before the next target is decoded; if none exists, leave them unrecovered. Add or adjust a focused case in `tests/multimodal/test_video.py` where a target fails immediately before another target succeeds, asserting that the later target remains a normal valid frame and is not reported as a recovery source.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal focuses on replacing the linear scan with keyframe seeking, preallocation, bounded recovery, and batched BGR-to-RGB conversion. This proposal is a narrower correctness fix in the existing recovery state machine: preventing a target frame from simultaneously satisfying its own request and recovering a previous failed request. It does not depend on seek strategy, allocation strategy, or color-conversion batching, and it addresses an ambiguity not mentioned in the candidate rationale or Agent A's proposal.

---
