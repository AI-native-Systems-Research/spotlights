# PlaceholderRange.embeds_cumsum/extract_embeds_range

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/inputs.py`](vllm/multimodal/inputs.py) (lines 150–205)
- **Symbol:** `PlaceholderRange.embeds_cumsum/extract_embeds_range`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0007`

## Description
Computes cumulative embedding counts and extracts contiguous embedded-token ranges from a boolean mask.

## Current approach
embeds_cumsum calls torch cumsum then tolist. extract_embeds_range converts the mask to int, runs torch.diff and torch.nonzero twice, stacks starts and ends, then converts the result to a Python list.

## Estimated impact explanation
Placeholder masks are processed during request ingress and scheduling. The per-call work is small, but it repeats for each multimodal item, so avoiding tiny tensor ops and syncs can reduce TTFT for multi-image prompts.

## Evolve rationale
The specific constructs are self.is_embed.cumsum(...).tolist(), torch.diff, torch.nonzero, and ranges.tolist(). For small placeholder masks, a single Python or numpy scan can avoid multiple tensor dispatches and device synchronization if the mask is on CUDA. Correctness oracle: tests/multimodal/test_inputs.py and the documented examples; extracted ranges and cumulative indices must be identical.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Materialize is_embed on CPU once and compute cumsum + ranges in pure Python
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/inputs.py (lines 150-205), replace the two independent tensor pipelines in `embeds_cumsum` and `extract_embeds_range` with a single cached CPU view of `is_embed` (as a Python `bytes`/`memoryview` or a small `np.ndarray[uint8]`), computed lazily on first access. Concretely: add a `_is_embed_cpu` cached_property that does `self.is_embed.detach().to('cpu', non_blocking=False).contiguous().view(torch.uint8).numpy()` exactly once. Then rewrite `embeds_cumsum` to `np.cumsum(mask, dtype=np.int32).tolist()` on that CPU array, and rewrite `extract_embeds_range` to a pure-Python run-length scan: walk the uint8 buffer once, emit `(start+offset, end+offset)` tuples on 0→1 and 1→0 transitions (or use `np.flatnonzero(np.diff(padded_mask))` on the small CPU array and pair adjacent transitions). This collapses the current cost — one CUDA cumsum + one `.tolist()` sync in `embeds_cumsum`, plus two `torch.diff`, two `torch.nonzero`, a `torch.stack`, and another `.tolist()` sync in `extract_embeds_range` — into at most one device→host transfer per PlaceholderRange, reused by both methods and by `get_num_embeds` / `get_embeds_indices_in_range`. For typical placeholder masks (tens to low thousands of positions, often already effectively CPU-bound work) the Python/numpy scan is faster than dispatching several tiny CUDA kernels and blocks the caller only once instead of two-plus times. Correctness is preserved because both outputs are functions of `is_embed` alone and are validated by `tests/multimodal/test_inputs.py` and the docstring examples on lines 168-172 and 185-187. Guard: if `is_embed.device.type == 'cpu'` already, skip the copy and operate directly on the underlying tensor's numpy view.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete proposal is novel by construction. Beyond that, the specific idea — unifying the two methods behind a single cached CPU materialization of `is_embed` so that both `embeds_cumsum` and `extract_embeds_range` share one device→host sync and then use numpy/pure-Python for the tiny arithmetic — is not stated in the candidate's `evolve_rationale`, which only vaguely mentions 'a single Python or numpy scan' without addressing the shared-cache angle, the CPU-fast-path guard, or the combined elimination of the second sync in `extract_embeds_range`.

---

### 2. Derive embed ranges from the cached cumulative counts
- **Agent:** codex

**Detailed description.**

In `vllm/multimodal/inputs.py`, add a private cached property such as `_embeds_ranges` and rewrite `extract_embeds_range()` to build ranges by scanning `self.embeds_cumsum` instead of launching a separate `torch.diff`/`torch.nonzero`/`torch.stack` pipeline. For each position, compare the current cumulative count with the previous count to recover whether that placeholder position is embedded; open a range on a false-to-true transition and close it on a true-to-false transition, adding `self.offset` to the emitted inclusive `(start, end)` pairs. Keep the `is_embed is None` fast path as the full placeholder range, store the cached result as an immutable tuple of tuples, and have `extract_embeds_range()` return `list(self._embeds_ranges)` so callers keep the current list-returning API without being able to mutate the cache. This makes repeated calls from prefix-LM and model-specific paths cheap, and when `get_embeds_indices_in_range()` has already populated `embeds_cumsum`, range extraction performs no tensor work or device synchronization at all.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes materializing `is_embed` on CPU once and using numpy or a Python scan over that CPU mask for both cumsum and range extraction. This proposal instead reuses the existing `embeds_cumsum` cached list as the single source of truth and adds a cached range result while preserving the current public method shape; it does not require a new CPU mask cache or numpy view.

---
