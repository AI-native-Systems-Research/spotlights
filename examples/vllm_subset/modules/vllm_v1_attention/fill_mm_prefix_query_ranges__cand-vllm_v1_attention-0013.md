# fill_mm_prefix_query_ranges

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/utils.py`](vllm/v1/attention/backends/utils.py) (lines 78–146)
- **Symbol:** `fill_mm_prefix_query_ranges`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0013`

## Description
Builds per-query multimodal prefix range rows for FA4 multimodal prefill masking.

## Current approach
Loops in Python over request ranges, computes covered scheduled-token spans from CPU tensors, fills the staging array with -1, and writes each span with numpy slice assignments.

## Estimated impact explanation
Image or multimodal agent turns can repeatedly hit this prefill/extend path. Vectorizing the range fill or moving it to device reduces per-step CPU preparation and can improve TTFT for multimodal turns.

## Evolve rationale
Concrete hot constructs are the nested Python loop over mm_prefix_range and the per-span numpy fills. Correctness oracle is exact equality of the output staging buffer and returned row count on randomized multimodal prefix ranges, plus existing multimodal prefix attention tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Cache resolved mm-prefix spans per request and use dirty-row tracking to skip the O(num_actual_tokens) -1 sentinel fill
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/utils.py:78-146 (fill_mm_prefix_query_ranges), two things run every scheduling step even when nothing about a request's multimodal prefix has changed: (1) the nested Python loop over mm_prefix_range.items() re-resolves every (start,end) span against context_len/query_len from scratch, and (2) `out[:num_actual_tokens] = -1` broadcasts a full sentinel fill across the entire staging buffer prefix, which is O(max_num_batched_tokens) and typically dominates because the spans that actually get written cover only a small fraction of tokens (a handful of image regions inside a much larger text prompt).

Proposal: introduce a small module-level LRU-like cache on the attention metadata builder that owns `out`, keyed by request id, storing (mm_prefix_range_signature, context_len, last_row_ranges: list[tuple[row_start,row_end,start,end]]). On each call:
  1. For each req_idx in mm_prefix_range, compute the current context_len from query_start_loc/seq_lens. If the cached signature matches and context_len equals the cached context_len, reuse the cached resolved spans directly (skip the per-range clamping loop). Otherwise recompute and update the cache. This eliminates the Python-level per-span math for requests whose chunk boundary hasn't advanced (common under chunked prefill retries and for decode-phase requests still carrying mm ranges).
  2. Track a persistent `dirty_rows: list[tuple[int,int]]` set on the builder that records exactly which row spans in `out` currently hold non-(-1) values. Replace the blanket `out[:num_actual_tokens] = -1` with a targeted reset: for each cached dirty span, write -1 into that slice only. Then write the newly-resolved spans and record them as the next step's dirty set. Because scheduled-token row indices shift each step, the reset uses last step's row indices before writing this step's spans; correctness is preserved because any row not in the new spans and not in the old dirty set is already -1 from initialization (allocate `out` filled with -1 once at builder construction).

Correctness oracle is unchanged: the returned row count and the contents of out[:num_actual_tokens] must match a from-scratch fill on randomized mm_prefix_range inputs, chunk boundaries, and interleaved decode/prefill batches; a fuzz test comparing against the current implementation guards this. Expected wins: for typical multimodal turns where images occupy <10% of tokens, the sentinel-fill cost drops from O(num_actual_tokens) to O(sum of span lengths), and repeated chunked-prefill steps for the same prompt skip the resolve loop entirely, cutting CPU prep on the TTFT critical path.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, so any concrete idea qualifies. Beyond that, this proposal is distinct from the candidate's own evolve_rationale, which only mentions generic 'vectorizing the range fill or moving it to device.' The two novel mechanisms here — (a) per-request signature-keyed caching of already-resolved spans across scheduling steps, exploiting the observation that mm_prefix_range is a property of the prompt and rarely changes between chunked-prefill steps, and (b) persistent -1-initialized staging with dirty-row tracking so the blanket sentinel broadcast is replaced by a targeted reset over previously-written rows — attack a different bottleneck (the unconditional O(num_actual_tokens) fill and repeated per-span math) than vectorization or GPU offload would. They are also complementary to vectorization: a future vectorized/device implementation still benefits from skipping work on unchanged requests and from not clearing the full prefix each step.

---

### 2. Filter mm-prefix ranges to active prefill chunks before FA4 staging
- **Agent:** codex

**Detailed description.**

Add an active-range pruning step before `fill_mm_prefix_query_ranges` is called for FA4. For each request with `mm_prefix_range`, track or cheaply compute that request's `[min_start, max_end)` multimodal prefix bounds, then skip the request entirely when its scheduled query window `[context_len, context_len + query_len)` cannot intersect those bounds. If no request survives, return `0` without entering the per-range loop or touching the staging buffer. This is especially useful after a multimodal prompt has moved into decode: `mm_req_doc_ranges` can still be present, but every decode token is past the prompt-local image ranges, so FA4 currently rechecks ranges on every token step only to produce no spans. The change can live as a small helper around `fill_mm_prefix_query_ranges` or as an optional precomputed bounds argument to the function, with a randomized equivalence test comparing filtered-vs-unfiltered outputs across prefill, chunked prefill, and decode-only batches.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. This is distinct from Agent A's cache and dirty-row reset proposal: it does not persist resolved spans, cache signatures, or optimize sentinel clearing. Instead, it removes whole requests from consideration when the current scheduled query window cannot overlap any multimodal prefix range, targeting the repeated decode/late-chunk no-op calls that matter for multi-turn TPOT.

---
