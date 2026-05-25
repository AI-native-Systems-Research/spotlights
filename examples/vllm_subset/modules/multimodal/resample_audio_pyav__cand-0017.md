# resample_audio_pyav

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/audio.py`](vllm/multimodal/audio.py) (lines 169–224)
- **Symbol:** `resample_audio_pyav`
- **Kind:** function
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0017`

## Description
Resamples NumPy audio arrays to a target sample rate through PyAV/libswresample while preserving expected output length.

## Current approach
For 2D audio, the function recursively resamples each channel independently and stacks the results. For 1D audio it converts to float32, pads short inputs to 1024 samples, creates a new av.AudioResampler and AudioFrame, collects resampled output frames into a list, concatenates them with np.concatenate, and trims to expected_len.

## Estimated impact explanation
Audio prompts that arrive at a non-model sample rate pass through this during parsing or media loading. The cost scales with clip length and channel count, so reducing per-channel PyAV setup and concatenate overhead lowers audio media TTFT for long or multi-channel inputs without affecting steady-state TPOT after preprocessing.

## Evolve rationale
The per-channel recursion at lines 193-201 and output-frame list plus np.concatenate at lines 220-224 are the optimization unit. Headroom includes feeding multi-channel planar data through a single resampler, avoiding one AudioResampler construction per channel, and pre-sizing or streaming the output buffer instead of accumulating frames then concatenating. Oracle: tests/multimodal/test_audio.py resample_audio_pyav/AudioResampler cases and MultiModalDataParser audio resampling must preserve output shape, channel order, expected_len trimming, same-rate passthrough, short-input padding behavior, and numeric equivalence within existing tolerances.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Add short-input scipy polyphase fast path that bypasses PyAV setup for typical agentic voice clips
- **Agent:** claude

**Detailed description.**

In `resample_audio_pyav` at vllm/multimodal/audio.py:169-224, add an early dispatch that short-circuits to `scipy.signal.resample_poly` (already imported and used by `resample_audio_scipy` at line 227) when the input is short relative to the cost of constructing an `av.AudioResampler`. Concretely, before the 2D-recursion branch at line 193, compute `g = math.gcd(orig_sr_int, target_sr_int)`, then `up, down = target_sr_int // g, orig_sr_int // g`. If `audio.shape[-1] < THRESHOLD` (e.g. ~32k samples ≈ 2s at 16 kHz) and `max(up, down) <= POLY_MAX` (e.g. 512, to bound polyphase filter length for irrational-feeling ratios like 44100→16000 = 160:441), apply `scipy.signal.resample_poly(audio, up, down, axis=-1).astype(audio.dtype, copy=False)[..., :expected_len]` and return — handling 1D and 2D in one shot via `axis=-1`, which also eliminates the per-channel recursion at lines 193-201 and the AudioResampler construction at line 215 for the common short-clip case. Pad to expected_len with zeros if `resample_poly` returns fewer samples (it can return ceil(N*up/down) which already matches `expected_len`). Keep the existing PyAV path unchanged for long clips and exotic ratios. Multi-turn agentic voice prompts (1–10s utterances at 16/24/48 kHz) will skip the swresample filter-graph init entirely, where init cost dominates per-call latency for clips under a few seconds; long-form audio still benefits from FFmpeg's optimized swresample. The oracle in tests/multimodal/test_audio.py is preserved because (a) same-rate passthrough still hits line 190-191 first, (b) `resample_poly` produces output of identical length and within numeric tolerance for the rates the existing scipy path already validates, (c) trimming to `expected_len` uses the identical formula at line 203, and (d) short-input padding is no longer needed because polyphase has no minimum-input requirement.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, and the candidate's own evolve_rationale enumerates only intra-PyAV optimizations (single multi-channel resampler, avoiding per-channel construction, pre-sizing/streaming the output buffer). It does not consider dispatching outside PyAV based on input length, nor leveraging the already-imported `scipy_signal.resample_poly` to avoid swresample filter-graph initialization for short clips — which is exactly the regime that matters for media TTFT in multi-turn agentic voice workloads. It also addresses a specific defect not raised in the rationale: the `_MIN_SAMPLES = 1024` zero-padding workaround at line 209-212 becomes unnecessary on the fast path, removing wasted work for sub-second audio.

---

### 2. Pre-reduce mono-target audio before PyAV resampling
- **Agent:** codex

**Detailed description.**

For the common path where `MultiModalDataParser` is configured with `target_channels=1`, move channel normalization ahead of `resample_audio_pyav` so stereo or multi-channel inputs are reduced to mono before reaching `vllm/multimodal/audio.py:193-201`. With the current default `AudioSpec(target_channels=1)` reduction this means averaging channels first, then calling `resample_audio_pyav` once on the resulting 1D array, instead of resampling every channel and averaging afterward. Keep the existing ordering as a fallback for non-mono targets or non-linear reductions. Add a test that a 2D `(channels, samples)` input with `target_channels=1` produces the same shape and numerically close output as the old resample-then-normalize order.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposed bypassing PyAV with a SciPy short-clip fast path; this proposal keeps the PyAV backend and reduces how often the candidate's 2D recursive path is entered by exploiting the fact that linear mono reduction commutes with linear resampling. It is also distinct from the candidate's PyAV-internal optimization ideas: it avoids unnecessary channel resampling at the parser level for mono-target models rather than batching channels through one resampler or changing output-buffer allocation.

---
