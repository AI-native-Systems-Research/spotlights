# MultiModalBudget._get_max_items

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/encoder_budget.py`](vllm/multimodal/encoder_budget.py) (lines 147–187)
- **Symbol:** `MultiModalBudget._get_max_items`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0019`

## Description
Computes per-modality max items per prompt and per batch from encoder budget, decoder budget, modality limits, max model length, and chunked-prefill settings.

## Current approach
Uses floor division by max_tokens_per_item and min/max clamps: encoder_budget // max_tokens_per_item, max_model_len // max_tokens_per_item, mm_limit, max_num_batched_tokens, and max_num_reqs. It assumes worst-case per-item token counts and does not account for cache hits or observed token distributions.

## Estimated impact explanation
This sets concurrency ceilings for multimodal encoder work. Conservative limits increase queueing and TTFT; overly loose limits risk OOM. Better admission improves throughput and TPOT under mixed multimodal bursts.

## Evolve rationale
The concrete heuristic is the floor-division and clamp formula in _get_max_items. Percentile-based sizing, cache-aware admission, or queue/backpressure-aware limits can tune encoder utilization while preserving safety. Correctness oracle: tests/v1/core/test_encoder_cache_manager.py and multimodal budget tests; max_items_per_batch must stay >= 1 when budget exists and encoder cache must not overflow.

## Deep research proposals

### 1. Make _get_max_items cache-aware by discounting cached items from encoder budget
- **Finding:** `find-vllm_multimodal-0001` — *Efficiently Serving Large Multimodal Models Using EPD Disaggregation*
- **Source URL:** <https://www.emergentmind.com/papers/2501.05460>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend MultiModalBudget._get_max_items in vllm/multimodal/encoder_budget.py (lines 147-187) to treat cached multimodal items as effectively free against the encoder compute/cache budget, consistent with an EPD-style view where media encoding is its own schedulable resource with its own cache. Concretely: rather than deriving max_encoder_items_per_batch strictly as encoder_budget // max_tokens_per_item under worst-case per-item tokens, expose an effective_encoder_budget that accounts for the fraction of items expected to hit the encoder cache (self.cache) in this batch, and/or split the return into (uncached_capacity, cached_capacity) so the scheduler can admit more items when a large share are already encoded. Preserve current invariants: max_items_per_batch stays >= 1 when budget exists, mm_limit and max_model_len // max_tokens_per_item still cap per-prompt items, and the decoder-side max_num_reqs clamp under non-chunked-prefill is unchanged. Fall back to today's worst-case formula when no cache is present or the hit-rate signal is unavailable, so the change is safe by default and only relaxes limits when cache-aware evidence supports it. Correctness is checked against tests/v1/core/test_encoder_cache_manager.py and multimodal budget tests; the encoder cache must not overflow and max_items_per_batch must remain >= 1 whenever encoder_budget > 0.

**Proposal rationale.**

The finding argues for treating multimodal encoding as its own resource with cache- and admission-aware policy, and highlights multi-turn agentic workloads where prioritizing cached/cheap multimodal work reduces TTFT and avoids TPOT interference. This directly addresses the candidate's stated gap: _get_max_items currently assumes worst-case per-item token counts and ignores cache hits, so ceilings are set for the uncached-cold-batch scenario. In multi-turn agentic traffic (the caller's workload), the same media are re-referenced across turns and dominate the encoder cache, so a cache-aware discount can safely raise the per-batch item ceiling without OOM risk — reducing queueing at the encoder stage (median TTFT) while leaving decoder-side clamps intact (protecting median TPOT). It is a concrete, transferable idea (cache-aware admission on encoder budget) rather than a topical restatement, and matches the evolve_rationale's explicit correctness oracle.

---

### 2. Adjust per-batch item ceilings by expected encoder-cache hit rate
- **Finding:** `find-vllm_multimodal-0009` — *Cache-aware prefill–decode disaggregation (CPD) for up to 40% faster long-context LLM serving*
- **Source URL:** <https://www.together.ai/blog/cache-aware-disaggregated-inference>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/multimodal/encoder_budget.py at MultiModalBudget._get_max_items (lines 147-187), replace the raw `encoder_budget // max_tokens_per_item` computation of `max_encoder_items_per_batch` with a cache-hit-aware effective budget. Concretely: (1) plumb an expected encoder-cache hit rate `h` per modality into MultiModalBudget (default 0.0, preserving current behavior; overridable via config or an EMA maintained by the encoder cache manager that reports observed hits/misses). (2) Compute `effective_encoder_items = floor(encoder_budget / ((1 - h) * max_tokens_per_item))` so when a fraction of admitted items are expected to be served from the encoder cache and skip fresh encoder compute, the ceiling scales up proportionally. (3) Keep the existing `max(1, ...)` safety clamp and continue to cap by `max_num_reqs * max_items_per_prompt` so decoder-side limits are unchanged. (4) Preserve current worst-case behavior when `h == 0` or the cache is disabled; gate the new path behind a config knob so tests in tests/v1/core/test_encoder_cache_manager.py and existing multimodal budget tests continue to see today's numbers by default. Optionally expose the estimator on MultiModalBudget so upstream admission (which decides whether to enqueue a media-heavy request) can distinguish warm-context requests from cold ones, giving warm multi-turn agentic requests a fast path while cold media-heavy requests are subject to the stricter raw-budget ceiling.

**Proposal rationale.**

The candidate's current heuristic assumes worst-case per-item encoder tokens and ignores that many multimodal items in multi-turn agentic workloads recur across turns and hit the encoder cache — exactly the gap the candidate's evolve_rationale calls out (cache hits and observed distributions are not accounted for). The finding's core idea — classify by cache-hit likelihood, then let warm work bypass the cold-work bottleneck — maps directly onto encoder budgeting: cached items consume cache space but no fresh encoder compute, so the compute-bound ceiling is systematically too tight when hit rates are non-trivial. Scaling the effective encoder budget by `1/(1 - h)` is the minimal, safe expression of the finding's policy in this specific method and improves median TTFT/TPOT for the stated multi-turn agentic workload without changing decoder-side clamps or the mm_limit invariant. The correctness oracle (max_items_per_batch >= 1, encoder cache non-overflow) is preserved because the encoder cache size clamp already inside get_encoder_budget bounds the outstanding working set, and the max(1, ...) clamp is retained.

---

## Agent proposals

### 1. Size encoder ceilings on observed p95 per-item token counts instead of worst-case max_tokens_per_item
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/encoder_budget.py at MultiModalBudget._get_max_items (lines 147-187), replace the worst-case `max_tokens_per_item` denominator in the `encoder_budget // max_tokens_per_item` and `max_model_len // max_tokens_per_item` divisions with a percentile-based per-modality estimate (e.g., p95 of recently observed per-item encoder token counts) while retaining worst-case as a hard safety fallback. Concretely: (1) Maintain a lightweight per-modality streaming quantile sketch (P^2 algorithm or a small reservoir + sort) on MultiModalBudget, updated by the encoder cache manager / scheduler each time an item's actual token count is materialized during preprocessing. (2) Expose `effective_tokens_per_item(modality) = clamp(p95_observed, floor=min_tokens_per_item, ceil=max_tokens_per_item)` with a warmup guard that returns `max_tokens_per_item` until N samples are collected (N configurable, default ~64). (3) Use `effective_tokens_per_item` for computing `max_encoder_items_per_batch` and the per-prompt `max_model_len // tokens_per_item` cap, but keep an admission-time recheck: when a specific request's realized token count exceeds the p95 estimate, fall back to per-request worst-case accounting so an unusually large item cannot overflow the encoder cache or exceed `max_model_len`. (4) Preserve `max(1, ...)` and `mm_limit` clamps and leave the non-chunked-prefill `max_num_reqs` clamp unchanged. (5) Gate behind a config knob (default off) so existing tests in tests/v1/core/test_encoder_cache_manager.py and multimodal budget tests see identical numbers, and add tests that exercise the sketch's warmup, the p95 ceiling under bimodal item-size distributions, and the per-request fallback path. Rationale for the workload: in multi-turn agentic traffic, per-item token counts for images/audio/video within a session are typically far below the modality's `max_tokens_per_item` (which reflects the largest resolution / longest clip the model supports); pricing the encoder ceiling on p95 rather than max lets more items co-schedule per batch, cutting encoder queue depth and thus median TTFT, while the per-request fallback prevents pathological overshoot.

**Novelty rationale.**

The two existing deep_research_proposals both attack the same lever: discounting the encoder budget by an expected cache-hit rate `h`, so cached items are treated as free (proposal 1 splits into cached/uncached capacity; proposal 9 scales the budget by `1/(1-h)`). Both leave `max_tokens_per_item` — the worst-case per-item denominator — untouched, so their ceilings still assume every uncached item is a max-sized item. My proposal targets an orthogonal source of conservatism explicitly called out in the evolve_rationale ("does not account for ... observed token distributions"): the per-item size assumption itself. It uses a percentile of the observed per-item token distribution as the denominator, with a per-request worst-case fallback for safety. This composes with (rather than duplicates) either cache-aware proposal — one could apply both a hit-rate discount and a p95 denominator — and it also improves the cold/no-cache path that the cache-aware proposals explicitly leave at today's worst-case numbers.

---

### 2. Add an aggregate mixed-modality encoder-token cap
- **Agent:** codex

**Detailed description.**

Extend `MultiModalBudget._get_max_items` in `vllm/multimodal/encoder_budget.py` so the per-modality `max_items_per_batch` values cannot each independently spend the full encoder budget when a model supports multiple tower modalities. After computing each modality's current per-item cap, derive an aggregate mixed-modality admission invariant such as `sum(admitted_items[m] * max_tokens_per_item[m]) <= get_encoder_budget()` and expose it alongside the existing per-modality item ceilings, or precompute conservative per-modality shares for the scheduler to enforce. Keep the current single-modality behavior unchanged, but add tests for a model/config with two modalities where each individual `max_items_per_batch` fits yet admitting both at their individual maxima would exceed the encoder cache/compute budget. This avoids over-admitting mixed image/audio/video bursts, reducing encoder backpressure and protecting median TPOT under multi-turn agentic traffic that can reference several modalities in the same scheduling window.

**Novelty rationale.**

The existing deep_research proposals are cache-hit-aware: they relax the encoder ceiling when items are likely cached. Agent A's proposal changes the denominator from worst-case item size to an observed p95 size. This proposal targets a different accounting gap: `_get_max_items` computes independent per-modality ceilings, each against the full encoder budget, but does not express an aggregate cross-modality budget invariant for mixed-modality batches. It composes with both cache-aware budgeting and percentile sizing rather than duplicating either.

---
