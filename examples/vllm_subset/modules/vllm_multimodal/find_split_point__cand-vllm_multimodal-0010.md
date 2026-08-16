# find_split_point

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/audio.py`](vllm/multimodal/audio.py) (lines 400–447)
- **Symbol:** `find_split_point`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0010`

## Description
Finds the quietest audio window in a search segment to choose a chunk split point.

## Current approach
Loops in Python over non-overlapping windows and computes RMS with (window ** 2).mean() ** 0.5 for each window, tracking the minimum non-NaN energy.

## Estimated impact explanation
Long audio inputs are chunked before processing. Vectorizing split search reduces audio preprocessing wall time and lowers TTFT for voice-agent and ASR-style requests.

## Evolve rationale
The hot construct is the for loop over range(0, len(segment) - min_energy_window, min_energy_window). Reshaping to windows and computing RMS with one numpy reduction, or using a running sum for overlapping variants, preserves the split-point contract with much lower Python overhead. Correctness oracle: tests/multimodal/test_audio.py and the docstring example; selected split points must remain in the same quiet window under defined tie-breaking.

## Deep research proposals

### 1. Vectorize find_split_point using frame-wise RMS reduction
- **Finding:** `find-vllm_multimodal-0008` — *librosa.feature.rms — librosa 1.0.0dev documentation*
- **Source URL:** <https://librosa.org/doc/main/generated/librosa.feature.rms.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/multimodal/audio.py (find_split_point, lines 400-447), replace the Python-level for-loop over non-overlapping windows with a single vectorized numpy computation, mirroring librosa.feature.rms's frame-wise RMS pattern. Concretely: take segment = wav[start_idx:end_idx]; compute the number of full non-overlapping frames n = (len(segment) - min_energy_window) // min_energy_window (matching the current range()); reshape the leading segment[: n * min_energy_window] into a (n, min_energy_window) view; compute per-frame RMS as np.sqrt((frames ** 2).mean(axis=1)) in one reduction. Preserve the current NaN handling and tie-breaking by masking NaNs to +inf and using np.argmin (which returns the first occurrence on ties, matching the '<' comparison in the loop). Map the winning frame index k back to quietest_idx = k * min_energy_window + start_idx. If n <= 0, return start_idx as today. Keep the function signature, return semantics, and docstring/example unchanged; tests in tests/multimodal/test_audio.py and the docstring example serve as the correctness oracle.

**Proposal rationale.**

The librosa.feature.rms documentation confirms the standard pattern of computing RMS per frame as a single vectorized reduction over a framed 2D view, which is exactly the operation find_split_point performs serially in Python. Vectorizing eliminates per-window Python overhead and math.isnan checks in the loop, reducing audio preprocessing wall time for long inputs that trigger chunking. This directly addresses the candidate's evolve_rationale (the hot Python loop) and supports the caller's TTFT objective for voice/ASR-style multi-turn workloads, while the non-overlapping window structure and integer arithmetic guarantee the same split index is chosen under the current tie-breaking rule.

---

## Agent proposals

### 1. Coarse-to-fine strided window search with zero-energy early exit in find_split_point
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/audio.py::find_split_point (lines 400-447), replace the per-sample-window Python loop with a two-stage strided search plus a cheap absolute-zero (digital-silence) short-circuit, both operating on numpy without changing the return contract.

Stage 0 (silence fast path): compute m = np.max(np.abs(segment[: n * min_energy_window])) once. If m == 0 (true digital silence, common in padded/agent turn-taking audio), immediately return start_idx (equivalent to picking the first window under the current '<' tie-break, since all RMS values are 0 and argmin returns index 0). This avoids any squaring/reduction for zero-padded chunks that agent pipelines routinely produce between turns.

Stage 1 (coarse pass): compute RMS on a strided subset of windows using np.lib.stride_tricks.sliding_window_view over segment with window shape (min_energy_window,) but stepped by S * min_energy_window (e.g. S=4). Take np.sqrt(np.mean(view**2, axis=-1)), mask NaN to +inf, and pick k_coarse = np.argmin.

Stage 2 (local refine): only evaluate the S non-overlapping windows in the neighborhood [k_coarse - S + 1, k_coarse + S] intersected with [0, n) as a single small reshape+mean reduction, then argmin among those to obtain k_final. Return k_final * min_energy_window + start_idx; if n <= 0 return start_idx.

Correctness under tie-breaking: because the coarse pass is a subset of the exact non-overlapping windows the current loop visits, and the refine pass covers every original window whose RMS could beat the coarse winner (given non-overlap and identical per-window RMS values), the chosen quiet window matches the current implementation whenever a strictly quieter window exists; for exact ties, using np.argmin (first occurrence) matches the '<' comparison in the loop. Preserve function signature, docstring/example, and the tests/multimodal/test_audio.py oracle.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_multimodal-0008) only reshapes all non-overlapping windows into a single (n, W) tensor and does one np.sqrt/mean/argmin reduction — it still touches every window and performs the full O(n*W) squaring/reduction. This proposal adds two orthogonal optimizations it does not cover: (1) an O(n*W) but branch-cheap np.max(|segment|)==0 short-circuit that returns without any squaring for digital-silence regions (very common in multi-turn agent audio with padded gaps, directly relevant to the TTFT objective), and (2) a coarse-to-fine strided search that reduces the reduction work by roughly a factor of S while provably preserving the current tie-breaking behavior. Neither the silence fast path nor the strided coarse/refine strategy is mentioned in the existing proposal, which is a straight full-reduction vectorization.

---

### 2. Use reduceat sum-of-squares for exact split search
- **Agent:** codex

**Detailed description.**

In vllm/multimodal/audio.py::find_split_point, replace the per-window RMS loop with an exact vectorized reduction that avoids both the 2D framed temporary and the unnecessary sqrt. After computing segment = wav[start_idx:end_idx] and n matching the current range semantics, take usable = segment[:n * min_energy_window], compute per-window sum of squares with np.add.reduceat(usable * usable, np.arange(0, usable.size, min_energy_window)), and separately compute per-window NaN presence with np.add.reduceat(np.isnan(usable), same_starts) > 0. Mask NaN-containing windows to +inf, then use np.argmin to preserve the current first-minimum tie behavior and return start_idx + k * min_energy_window. Because every compared window has the same length, minimizing sum of squares is equivalent to minimizing RMS, so the sqrt and division by window length can be skipped without changing the selected split point.

**Novelty rationale.**

The deep_research_proposal covers reshaping all windows into a 2D frame matrix and computing np.sqrt((frames ** 2).mean(axis=1)); agent A covers a silence fast path plus coarse-to-fine strided search. This proposal is a distinct exact implementation strategy: use one-dimensional segmented reductions via np.add.reduceat, compare sum-of-squares directly, and avoid the framed view, mean, and sqrt while preserving the current exhaustive search and tie-breaking semantics.

---
