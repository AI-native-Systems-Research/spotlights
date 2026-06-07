# split_decodes_prefills_and_extends/split_decodes_and_prefills

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/backends/utils.py`](vllm/v1/attention/backends/utils.py) (lines 425–565)
- **Symbol:** `split_decodes_prefills_and_extends/split_decodes_and_prefills`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0010`

## Description
Host-side boundary classifiers that split an already-reordered batch into decode, extend, and prefill counts for backend metadata builders.

## Current approach
Uses max_query_len fast paths, then builds query_lens from query_start_loc_cpu and uses torch CPU tensor predicates, torch.any, argmax, and .item calls to find first extend/prefill boundaries. split_decodes_and_prefills also has require_uniform and treat_short_extends_as_decodes policy branches.

## Estimated impact explanation
For small decode-heavy agentic batches, host metadata overhead can be visible in median TPOT, and incorrect boundary policy can route short extends to slower kernels. Impact is medium because the work is CPU-side and small relative to large GPU attention kernels.

## Evolve rationale
These functions run in many backend build paths on every scheduler step and define dispatch boundaries that choose decode vs prefill kernels. They have concrete headroom in avoiding repeated CPU tensor scans and scalar extraction by carrying scheduler-side counts or using cheaper array operations at the same contract. Correctness oracle: tests/v1/attention/test_attention_splitting.py covers boundary cases, and tests/v1/attention/test_attention_backends.py catches backend-equivalence regressions.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Fuse reorder-side region counts into CommonAttentionMetadata to make split_decodes_* O(1)
- **Agent:** claude

**Detailed description.**

Today, `reorder_batch_to_split_decodes_and_prefills` (utils.py:595-672) already computes the four exact region counts on the input batch using cheap numpy boolean sums: `num_decodes = int(is_decode.sum())`, `num_short_extends`, `num_long_extends`, `num_prefills` (lines 641-644). These are computed from per-request scalars the worker already has on hand (`num_scheduled_tokens`, `num_computed_tokens_cpu`, `num_prompt_tokens`) and are exact, not approximations. The function then discards them and returns only a bool. Per scheduler step, every attention backend's metadata builder later re-derives the same boundaries inside `split_decodes_prefills_and_extends` / `split_decodes_and_prefills` (utils.py:425-565) via a torch CPU `query_start_loc[1:] - query_start_loc[:-1]` allocation, two predicate tensors (`is_prefill_or_extend`, `is_prefill`), two `torch.any` reductions, two `argmax(dim=-1).item()` calls, and several `query_start_loc[i].item()` lookups. Hybrid models (Mamba+attention, GDN+attention, MLA+full attn) hit this path K times per step, once per attention group. Concrete change: (1) extend `reorder_batch_to_split_decodes_and_prefills` to also compute and return `(num_short_extends_tokens, num_long_extends_tokens, num_prefill_tokens)` from `num_scheduled_tokens_np[is_*].sum()` — these are already-on-CPU O(N) numpy ops that reuse the same masks already built at lines 630-633. (2) Cache the resulting six-tuple on the `InputBatch` (or the per-step bookkeeping object that gpu_model_runner already maintains in `_prepare_inputs`), keyed by the `decode_threshold` actually used. (3) Have each backend's `CommonAttentionMetadata` builder consume those cached counts and assign them onto the metadata as `num_decodes_hint`, etc. (4) Add a fast path at the top of both `split_decodes_*` functions: if these hints are populated and the `(decode_threshold, require_uniform, treat_short_extends_as_decodes)` triple matches the configuration the reorder used, return them directly — skipping all torch tensor work. Fall back to the current scan path when hints are absent (e.g., backends that don't go through the standard reorder, or for the 3-way variant when the reorder was called with a different threshold). The `treat_short_extends_as_decodes=False` and `require_uniform=True` policy branches are still expressible as thin policies on top of the four region counts (e.g., short_extends-as-prefills just adds `num_short_extends` to prefill side). Correctness oracle is exactly what the candidate cites: `tests/v1/attention/test_attention_splitting.py` and `tests/v1/attention/test_batch_reordering.py`. Net effect for agentic decode-heavy steps that mix in occasional tool-turn extends: eliminates per-backend torch CPU scan + 3-5 `.item()` syncs at metadata-build time.

**Novelty rationale.**

The candidate's `evolve_rationale` mentions "carrying scheduler-side counts" as a possible direction. This proposal is materially different in three ways the rationale does not cover: (a) the source of the counts is `reorder_batch_to_split_decodes_and_prefills` in the worker (utils.py), not the scheduler — the function already computes them but throws them away; no scheduler/SchedulerOutput plumbing is required, which is a much smaller blast radius than scheduler-side counts; (b) it identifies the specific intermediate quantities to also compute alongside the existing `is_*.sum()` calls (the per-region token sums via `num_scheduled_tokens_np[is_*].sum()`), so the entire six-tuple in `split_decodes_prefills_and_extends` becomes O(1) — not just the request counts; (c) it specifies a cache-with-key fallback strategy (keyed on `(decode_threshold, require_uniform, treat_short_extends_as_decodes)`) so backends that diverge from the reorder's threshold transparently fall back to the existing scan, avoiding correctness regressions in non-uniform configurations. The other rationale hint, "using cheaper array operations at the same contract," is about replacing torch scans with numpy in place — orthogonal to and weaker than eliminating the scan entirely by reusing already-computed numpy results. There are no listed deep_research_proposals on this candidate.

---

### 2. Add a small-batch scalar split fast path in utils.py
- **Agent:** codex

**Detailed description.**

Add a local fast path inside `vllm/v1/attention/backends/utils.py` for `split_decodes_prefills_and_extends` and `split_decodes_and_prefills` when `num_reqs` is small, e.g. <=128, and the needed CPU metadata is already resident. Convert `query_start_loc_cpu`, `seq_lens_cpu_upper_bound`, and, when needed, `is_prefilling` to NumPy views and perform a single Python/NumPy scalar scan that short-circuits at the first boundary. For the 3-way splitter, scan once for the first `query_len > decode_threshold`, then continue only until the first true prefill row (`seq_len == query_len`) and compute token counts from the cumulative start offsets. For the 2-way splitter, mirror the existing `require_uniform` and `treat_short_extends_as_decodes` semantics exactly, including the padded-uniform zero-query case. Keep the current torch-vectorized implementation as the large-batch fallback. This targets the multi-turn agentic case directly: tiny or modest decode-heavy batches pay PyTorch dispatcher, temporary tensor, `torch.any`, `argmax`, and `.item()` overhead that is often larger than a short scalar scan.

**Novelty rationale.**

There are no deep_research_proposals to duplicate. This is not Agent A's reorder-count caching proposal: it does not plumb or cache scheduler/reorder-side region counts, does not add hint fields, and does not make the split O(1). Instead, it optimizes the existing scan contract itself for the common small-batch fallback path, including cases where reorder hints are unavailable, keyed differently, disabled, or where `require_uniform` / `treat_short_extends_as_decodes` must be evaluated from the actual metadata.

---
