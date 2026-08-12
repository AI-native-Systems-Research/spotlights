# resample_audio_pyav

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/audio.py`](vllm/multimodal/audio.py) (lines 174–229)
- **Symbol:** `resample_audio_pyav`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0011`

## Description
Resamples audio to a target sample rate using PyAV/libswresample, including short-input padding, frame creation, resampler invocation, flush, and output-frame concatenation.

## Current approach
Rounds rates, recursively resamples each channel for 2D audio, pads short mono arrays to 1024 samples, constructs a fresh av.AudioResampler and AudioFrame per call, materializes each output frame with to_ndarray, and concatenates the frame arrays.

## Estimated impact explanation
Audio resampling runs at ingress for mismatched sample rates. Cutting per-call setup and per-channel recursion reduces TTFT for audio-heavy agentic workloads and lowers CPU contention.

## Evolve rationale
The concrete constructs are the per-channel np.stack recursion, av.AudioResampler construction, AudioFrame.from_ndarray, and np.concatenate over output frames. Batching channels through a multi-channel frame or selecting a lower-overhead in-repo method path can reduce fixed setup and Python materialization while preserving output length and numeric closeness. Correctness oracle: tests/multimodal/test_audio.py; output shape and resampled length must satisfy the existing reference tolerances.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Resample multi-channel audio in a single PyAV pass instead of per-channel recursion
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/audio.py:174-229 (resample_audio_pyav), replace the 2D branch that recursively calls resample_audio_pyav on each row and re-stacks the results with a single-pass multi-channel resample: for a (C, N) input, keep the array as planar (C, N) float32, build one av.AudioResampler(format='fltp', layout=<mono|stereo|N-channel layout matching C>, rate=target_sr_int) and one av.AudioFrame.from_ndarray(audio_f32, format='fltp', layout=<same>) with frame.sample_rate = orig_sr_int, then run resampler.resample(frame) + flush and np.concatenate the output planar frames along the sample axis, trimming to expected_len. This eliminates C separate AudioResampler and AudioFrame constructions (each of which spins up libswresample state), C separate flush passes, and the outer np.stack copy for stereo/multi-channel inputs, which is the common case for real audio. For C==1 the fast path is unchanged. Correctness is preserved because libswresample's polyphase resampler is applied per channel internally, and the existing short-input _MIN_SAMPLES=1024 zero-pad and expected_len trimming can be applied to the planar array in one shot. The output shape and length contract (2D in → 2D out, trimmed to ceil(N * target/orig)) is unchanged, so tests/multimodal/test_audio.py tolerances continue to hold.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate (the list is empty), so any concrete change is novel by construction. Specifically, this proposal targets the per-channel recursion + repeated AudioResampler/AudioFrame construction — the dominant fixed-overhead cost for stereo/multi-channel inputs at ingress — rather than switching backends or caching, and preserves the current numeric behavior via libswresample's own per-channel handling.

---

### 2. Skip PyAV setup when the rounded sample rates already match
- **Agent:** codex

**Detailed description.**

In vllm/multimodal/audio.py:174-229 (resample_audio_pyav), add an early return immediately after computing the integer-rounded original and target sample rates: if orig_sr_int == target_sr_int, return the input audio unchanged (or a cheap dtype-preserving view/copy only if the surrounding contract requires it) instead of padding, constructing an AudioFrame, creating an AudioResampler, flushing, concatenating frames, and trimming. This preserves shape and values for the no-op resample case while avoiding all libswresample setup and Python frame materialization for inputs whose declared rates are already equal after the function's existing rounding behavior. This matters at multimodal ingress because callers may route audio through the same normalization path even when the source model/input rate already matches the target rate.

**Novelty rationale.**

There are no deep_research_proposals for this candidate, and Claude's proposal specifically optimizes multi-channel inputs by replacing per-channel recursion with one multi-channel PyAV pass. This proposal targets a different branch of work: eliminating the entire PyAV resampling path for no-op same-rate calls, including mono inputs, without changing multi-channel batching behavior.

---
