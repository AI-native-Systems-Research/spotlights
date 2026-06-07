# find_split_point

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/audio.py`](vllm/multimodal/audio.py) (lines 349–396)
- **Symbol:** `find_split_point`
- **Kind:** function
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0010`

## Description
Finds a low-RMS-energy split point in an audio search window for long-audio chunking.

## Current approach
A Python loop walks non-overlapping windows, slices each window, computes (window**2).mean() ** 0.5, and tracks the first minimum energy index.

## Estimated impact explanation
Long-audio preprocessing calls this once per chunk boundary. Vectorizing the window reductions reduces audio media TTFT for long clips, but it does not affect steady-state TPOT after preprocessing.

## Evolve rationale
The loop over range(..., min_energy_window) at lines 389-394 is a vectorization target. Headroom includes reshaping the scanned segment into windows and computing all RMS values in one NumPy reduction, or using cumulative sum of squares for large search regions, while preserving the current non-overlapping-window and first-tie semantics. Oracle: tests/multimodal/test_audio.py split_audio/find_split_point cases must return the same quietest index for uniform, silence, and quiet-region inputs.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Vectorize via argmin on sum-of-squares (skip sqrt/mean) and fix last-axis slicing for multi-dim audio
- **Agent:** claude

**Detailed description.**

Replace the Python loop at lines 389-394 with a single vectorized reduction that argmins on sum-of-squares instead of RMS, and correct the segment slicing to use the documented last-axis (time) convention.

Concretely, in `find_split_point` (vllm/multimodal/audio.py:349-396):

1. Change `segment = wav[start_idx:end_idx]` (line 383) to `segment = wav[..., start_idx:end_idx]`. The function's docstring and `split_audio`'s contract (lines 300-301: 'Splits along the last dimension (time axis)') say wav may be multi-dimensional, but the current axis-0 slice silently truncates the channel/batch dimension when wav is 2D and corrupts the search region. `split_audio` itself uses `audio_data[..., i:split_point]` (line 343), so the convention is mismatched.

2. Reshape the trimmed last axis into non-overlapping windows and reduce in one step:
   ```
   n = (segment.shape[-1] - min_energy_window) // min_energy_window
   if n <= 0:
       return start_idx  # match loop semantics: no iterations -> quietest_idx stays 0+start_idx
   trimmed = segment[..., : n * min_energy_window]
   windows = trimmed.reshape(*segment.shape[:-1], n, min_energy_window)
   # sum-of-squares over the window axis (and any leading channel axes)
   sumsq = np.einsum('...ij,...ij->...i', windows, windows)
   if sumsq.ndim > 1:
       sumsq = sumsq.sum(axis=tuple(range(sumsq.ndim - 1)))
   quietest_window = int(np.argmin(sumsq))  # argmin returns first index on ties
   return quietest_window * min_energy_window + start_idx
   ```

Why argmin on sum-of-squares is semantically equivalent and faster than the candidate's stated approach: for fixed-size non-overlapping windows, `argmin(rms) == argmin(mean(x**2)) == argmin(sum(x**2))` because `sqrt` and the constant `1/min_energy_window` are monotone. Skipping the per-window `** 0.5` and `.mean()` removes a sqrt and a divide per window relative to the candidate's 'compute all RMS values in one NumPy reduction' phrasing, and `np.einsum('...ij,...ij->...i', w, w)` materializes no intermediate `w**2` array, halving memory traffic on the squaring step. `np.argmin` is documented to return the first occurrence on ties, preserving the loop's first-minimum-wins semantics required by the oracle (uniform-input test must return 0).

The loop bound `range(0, len(segment) - min_energy_window, min_energy_window)` is `n = (L - W) // W` iterations, exactly matching the reshape arithmetic above.

This should be a drop-in replacement and is verifiable against `tests/multimodal/test_audio.py`.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate, so the bar is overlap with the candidate's own evolve_rationale. The candidate proposes 'reshaping into windows and computing all RMS values in one NumPy reduction' or 'cumulative sum of squares'. Both still implicitly compute RMS. My proposal contributes two distinct items the rationale does not state: (a) the observation that for equal-size windows argmin on sum-of-squares is identical to argmin on RMS, so sqrt and mean can be dropped entirely (and `np.einsum` avoids materializing `windows**2`); and (b) a correctness fix that the current `wav[start_idx:end_idx]` slice violates the last-axis convention documented for multi-dim audio in `split_audio`, which any vectorized rewrite must also correct or it will regress on 2D inputs. Neither point is present in the evolve_rationale.

---

### 2. Preserve the exact loop-bound semantics in the vectorized window count
- **Agent:** codex

**Detailed description.**

When vectorizing `find_split_point` in `vllm/multimodal/audio.py:389-394`, derive the number of windows from the existing loop shape rather than using `(segment_len - min_energy_window) // min_energy_window`. The current loop is `range(0, len(segment) - min_energy_window, min_energy_window)`, so for a segment length `L` and window size `W` it scans `len(range(0, L - W, W))`, which is `max(0, (L - 1) // W)`, not `(L - W) // W`. For example, `L = 2*W + 1` currently scans starts `0` and `W`; the floor formula scans only `0` and can return a different split point if the second window is quieter. Use `num_windows = len(range(0, segment_len - min_energy_window, min_energy_window))` or the equivalent guarded formula, then trim to `num_windows * min_energy_window` before reshaping. Add a focused regression case where the quietest window begins at `min_energy_window` and the segment has a short tail, so the vectorized path cannot silently drop that candidate window.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A covers vectorizing with sum-of-squares, avoiding RMS/sqrt, and fixing last-axis slicing, but it also states an incorrect loop-count formula. This proposal is specifically about preserving the existing `range(...)` boundary semantics during any vectorized rewrite and adding a regression for that off-by-one case, which is not covered by Agent A or the candidate rationale.

---
