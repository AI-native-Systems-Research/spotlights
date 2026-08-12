# split_decodes_and_prefills

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/utils.py`](vllm/v1/attention/backends/utils.py) (lines 635–706)
- **Symbol:** `split_decodes_and_prefills`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0011`

## Description
Finds the decode/prefill boundary in an already reordered batch and returns request and token counts for each region.

## Current approach
Uses CPU tensor diffs, scalar .item() reads, torch.any, and argmax().item() to find the first prefill. Data is already CPU-side, so the main cost is repeated tensor/Python scalar work rather than GPU synchronization.

## Estimated impact explanation
Called by multiple metadata builders every step. Reducing repeated CPU tensor overhead improves median TPOT when decode shapes repeat across many agentic turns.

## Evolve rationale
Concrete constructs are query_lens[0].item(), torch.any(is_prefill), argmax().item(), and query_start_loc[first_prefill].item(). A numpy/searchsorted or cached CPU-array implementation can be validated by exact equality of the returned integers on randomized reordered batches and existing backend tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Use numpy searchsorted on cumulative query_start_loc to find decode/prefill boundary without materializing query_lens
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/utils.py:635-706, replace the per-call materialization of `query_lens = query_start_loc[1:] - query_start_loc[:-1]` followed by boolean-tensor construction, `torch.any`, and `argmax().item()` with a numpy view over the already-CPU `query_start_loc_cpu` and use `numpy.searchsorted`.

Concretely: obtain `qsl_np = common_attn_metadata.query_start_loc_np` if the CommonAttentionMetadata already carries a numpy alias (or a one-time `query_start_loc_cpu.numpy()` view — no copy, since the tensor is CPU/contiguous). For the common `require_uniform=False, treat_short_extends_as_decodes=True` path, the boundary is exactly the smallest index `i` such that `qsl_np[i+1] - qsl_np[i] > decode_threshold`. This can be computed in a single vectorized numpy pass: `diffs = qsl_np[1:] - qsl_np[:-1]; first_prefill = np.argmax(diffs > decode_threshold)` where `np.argmax` on a boolean array returns 0 if no True exists — combined with a cheap `diffs[first_prefill] > decode_threshold` scalar check this collapses the current three tensor ops (diff, `torch.any`, `.int().argmax().item()`) into one small numpy kernel with no Python-level per-scalar `.item()` synchronization overhead.

Better still, avoid the diff materialization entirely for the common case by observing that `query_start_loc` is monotonically increasing and decode tokens all have length ≤ `decode_threshold`. Use `np.searchsorted(qsl_np, qsl_np[0] + something, ...)` — or, more directly: precompute once per builder invocation the small boolean numpy array `is_prefill_np` and reuse it inside `split_decodes_and_prefills`, `reorder_batch_to_split_decodes_and_prefills`, and the extend variant (lines 570-632) which currently redoes very similar work. All three call sites already sit on the CPU-side hot path invoked by every attention metadata builder every step.

Also replace the trailing `query_start_loc[first_prefill].item()` with `int(qsl_np[first_prefill])` — this avoids the PyTorch 0-dim tensor → Python-int conversion overhead which shows up in profiles for small batches. Preserve identical return values: cast final ints via `int(...)` so mypy and callers relying on Python int types continue to work. Add a fast-path branch when `qsl_np.shape[0] <= 2` (single request) that skips numpy entirely.

Validation: property-test that for randomized `query_start_loc_cpu` tensors covering all-decode, all-prefill, mixed, `require_uniform`, and short-extend edge cases, the new implementation returns integers exactly equal to the current implementation. Reuse existing backend tests in tests/v1/attention/ that construct CommonAttentionMetadata to guard behavior. Benchmark with `benchmarks/kernels/benchmark_*.py` style microbench on the attention metadata builder for a synthetic 128-request multi-turn agentic batch to demonstrate median-TPOT-relevant reduction of per-step CPU overhead.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, so any concrete proposal is novel by construction. This specific approach — using a numpy view over the existing CPU tensor plus `np.argmax`/`np.searchsorted` on a boolean mask, sharing the resulting `is_prefill` numpy array across the sibling split/reorder functions, and eliminating the 0-dim-tensor `.item()` conversions — is a distinct implementation direction from the candidate's own `evolve_rationale` hint (which mentions numpy/searchsorted or cached CPU-array only in the abstract, without the multi-callsite sharing, the boolean-argmax fast path, or the `int(qsl_np[...])` conversion optimization).

---

### 2. Cache split counts upstream when the batch is reordered
- **Agent:** codex

**Detailed description.**

Add optional precomputed decode/prefill split metadata to `CommonAttentionMetadata` and populate it in the CPU-side batch preparation/reorder path that already has `num_scheduled_tokens`, `is_prefilling_np`, and the decode/short-extend/long-extend/prefill region counts. Then make `split_decodes_and_prefills` in `vllm/v1/attention/backends/utils.py:635-706` return the cached `(num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)` for the common argument combination (`require_uniform=False`, `treat_short_extends_as_decodes=True`, matching `decode_threshold`) instead of rescanning `query_start_loc_cpu` for every backend metadata builder. The cache should store the threshold it was computed for, because some callers pass non-default `decode_threshold`; fall back to the existing computation when `require_uniform=True`, `treat_short_extends_as_decodes=False`, or the threshold does not match. This moves the boundary decision to the point where the regions are already classified and prevents repeated per-backend work in multi-layer or multi-backend attention metadata construction.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes optimizing `split_decodes_and_prefills` itself with numpy views/search/argmax and possibly sharing a boolean mask across sibling helpers. This proposal is different: compute and cache the final split integers upstream during the existing reorder/batch-preparation classification, then bypass `split_decodes_and_prefills` entirely for the common cached case rather than replacing its tensor operations with numpy operations.

---
