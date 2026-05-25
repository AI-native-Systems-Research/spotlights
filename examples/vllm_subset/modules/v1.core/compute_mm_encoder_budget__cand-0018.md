# compute_mm_encoder_budget

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/encoder_cache_manager.py`](vllm/v1/core/encoder_cache_manager.py) (lines 269–316)
- **Symbol:** `compute_mm_encoder_budget`
- **Kind:** config_block
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0018`

## Description
Budget policy for multimodal encoder scheduling: validates disable_chunked_mm_input against the largest modality item, then sets encoder_compute_budget and encoder_cache_size to max(configured value, max_tokens_per_mm_item).

## Current approach
Static per-model calculation based only on the largest single multimodal item and two SchedulerConfig fields. It does not adapt to expected concurrent media items, cache reuse rate, modality mix, or max_num_batched_tokens beyond the disabled-chunking validation.

## Estimated impact explanation
This moves media TTFT by controlling whether multimodal encoder inputs can be scheduled or hit in cache under pressure. Impact is medium because it is critical for media-heavy agentic workloads but irrelevant to text-only decode.

## Evolve rationale
The concrete policy-defining construct is the max_tokens_per_mm_item calculation and the max(...) assignments for encoder_compute_budget and encoder_cache_size. These capacities directly constrain Scheduler._try_schedule_encoder_inputs and EncoderCacheManager.can_allocate for media turns. Headroom includes concurrency-aware cache sizing, separate compute/cache budgets per modality, tying compute budget to batched-token headroom, and adaptive defaults for repeated media in agentic sessions. Correctness oracles include tests/v1/core/test_scheduler.py and tests/v1/core/test_encoder_cache_manager.py, which exercise scheduler multimodal budgets, chunking behavior, and encoder-cache capacity invariants.

## Deep research proposals

### 1. Adapt encoder budget policy to EPD-style cache admission and decoupled compute sizing
- **Finding:** `find-0001` — *Efficiently Serving Large Multimodal Models Using EPD Disaggregation*
- **Source URL:** <https://arxiv.org/html/2501.05460v2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend `compute_mm_encoder_budget` in vllm/v1/core/encoder_cache_manager.py:269-316 so the two returned capacities reflect EPD-style separation rather than a flat max() over a single largest item:

1) Concurrency- and reuse-aware `encoder_cache_size`: instead of `max(scheduler_config.encoder_cache_size, max_tokens_per_mm_item)`, scale by an expected-concurrent-and-reused-media factor `N_mm` derived from workload hints (e.g. multi-turn agentic sessions revisiting the same images/audio). Compute it as roughly `max(scheduler_config.encoder_cache_size, N_mm * sum_or_weighted_mix(mm_max_toks_per_item.values()))` so repeated media in an agentic session stay resident across turns and convert media TTFT into a cache hit. The factor should be configurable (new SchedulerConfig field, default 1 to preserve current behavior) and clipped against memory.

2) Decoupled `encoder_compute_budget`: treat encoder compute as an EPD-disaggregated stage rather than a sub-budget of `max_num_batched_tokens`. Allow the budget to be sized independently of prefill (e.g. `max(scheduler_config.max_num_encoder_input_tokens, k * max_tokens_per_mm_item)` with k>=1 reflecting per-request intra-encoder parallelism), so `Scheduler._try_schedule_encoder_inputs` can admit encoder work without serializing behind prefill tokens. Keep the existing `disable_chunked_mm_input` validation, but evaluate it against the (possibly larger) decoupled compute budget rather than `max_num_batched_tokens` alone.

3) Preserve modality awareness: when `mm_max_toks_per_item` has more than one modality, optionally return per-modality budgets (or use the weighted mix above) instead of collapsing to a single `max(...)`, so a heavy modality does not displace a frequently-cached lighter one in agentic mixes.

Validation should reuse tests/v1/core/test_encoder_cache_manager.py and tests/v1/core/test_scheduler.py to confirm budget invariants, chunking behavior, and that defaults reproduce the current numbers when the new knobs are at their identity values.

**Proposal rationale.**

The candidate's gap is exactly that its capacities depend only on the single largest item and two static SchedulerConfig fields, ignoring concurrent media, reuse, and the encoder-vs-prefill compute split. EPD disaggregation contributes two transferable ideas that map directly to the two returned numbers: (a) multimedia-token caching, which argues for sizing `encoder_cache_size` around expected reused/concurrent media (high-leverage in the stated multi-turn agentic workload, where the same media reappear across turns and dominate media TTFT), and (b) separating encode from prefill, which argues for sizing `encoder_compute_budget` independently of the prefill token budget so encoder admission in `Scheduler._try_schedule_encoder_inputs` is not bottlenecked by prefill contention. Both ideas are concrete edits to the existing `max(...)` assignments rather than topical adjacency, and both plausibly reduce media TTFT (and median TPOT, by avoiding stalls when encoder work would otherwise block decode steps) on the targeted workload.

---

### 2. Tie encoder_compute_budget to stall-free batched-token headroom (Sarathi-style)
- **Finding:** `find-0007` — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- **Source URL:** <https://arxiv.org/abs/2403.02310>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/core/encoder_cache_manager.py compute_mm_encoder_budget (lines 269-316), replace the static max(configured, max_tokens_per_mm_item) policy for encoder_compute_budget with a stall-free headroom calculation inspired by Sarathi-Serve's chunked-prefill scheduling. Concretely: compute a per-step encoder compute cap as a fraction of SchedulerConfig.max_num_batched_tokens (the same budget that bounds a scheduling step), then take encoder_compute_budget = max(max_tokens_per_mm_item, min(configured, headroom_fraction * max_num_batched_tokens)). The floor at max_tokens_per_mm_item preserves the existing correctness invariant that a single largest item must still be schedulable when chunked MM input is disabled, while the headroom cap prevents a media-heavy step from consuming the entire token budget and pausing ongoing decodes. Keep encoder_cache_size decoupled (it is a capacity, not a per-step compute knob) and only adapt encoder_compute_budget. The disable_chunked_mm_input validation against max_tokens_per_mm_item is unchanged.

**Proposal rationale.**

The candidate's evolve_rationale explicitly lists "tying compute budget to batched-token headroom" as headroom, and Sarathi-Serve's core contribution is exactly that: sizing prefill-side work per step so it does not stall ongoing decodes. The current policy sets encoder_compute_budget purely from the largest single multimodal item and a configured value, with no reference to max_num_batched_tokens beyond the disabled-chunking validation, so a large image/video encode can monopolize a step in a multi-turn agentic workload and inflate median TPOT. Bounding encoder compute by a fraction of the step's batched-token budget transfers Sarathi-Serve's stall-free reasoning to the multimodal encoder path and directly targets the caller's stated TPOT objective without harming media TTFT, since the floor at max_tokens_per_mm_item keeps the largest item schedulable.

---

### 3. Size encoder cache for cross-turn media reuse, inspired by AttentionStore's multi-turn hierarchical cache
- **Finding:** `find-0009` — *AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving*
- **Source URL:** <https://arxiv.org/html/2403.19708v2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In compute_mm_encoder_budget (vllm/v1/core/encoder_cache_manager.py:269-316), replace the current encoder_cache_size = max(configured, max_tokens_per_mm_item) with a reuse-aware sizing that targets expected unique media items across an agentic session window, not just the single largest item. Concretely: (1) decouple encoder_compute_budget from encoder_cache_size — keep encoder_compute_budget tied to per-step compute headroom (still bounded by max_tokens_per_mm_item and chunking constraints), but raise encoder_cache_size to N * max_tokens_per_mm_item where N is a small concurrency/reuse factor derived from SchedulerConfig (e.g. expected concurrent media-bearing requests and a multi-turn reuse hint); (2) leave a hook in the returned budget for a tiered fallback — a host-side staging area for evicted encoder outputs that can be re-promoted on a turn that re-references the same media, mirroring AttentionStore's layer-wise asynchronous preload/save pattern but applied to per-modality encoder outputs in EncoderCacheManager. The candidate file remains the policy site; downstream changes in EncoderCacheManager.can_allocate / Scheduler._try_schedule_encoder_inputs are out of scope for this proposal but are the consumers the new budget must remain compatible with.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'concurrency-aware cache sizing' and 'adaptive defaults for repeated media in agentic sessions' as headroom; the current code sizes the encoder cache to a single largest item, which under-provisions reuse in multi-turn agentic workloads where the same images/media recur across turns. AttentionStore's central insight — that multi-turn serving benefits from a cache sized and tiered around cross-turn reuse with scheduler-aware fetch/eviction, rather than per-step capacity — transfers directly to the encoder-output cache: media items referenced again in later turns can hit instead of re-encoding, improving media TTFT, which is exactly the caller objective. The finding addresses the gap that the current static per-model max(...) calculation has no notion of expected reuse or concurrent media items.

---

### 4. SLO-aware, concurrency-adaptive encoder cache sizing for multi-turn agentic media
- **Finding:** `find-0011` — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*
- **Source URL:** <https://arxiv.org/abs/2407.00079>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the static max(configured, max_tokens_per_mm_item) policy in compute_mm_encoder_budget (vllm/v1/core/encoder_cache_manager.py:269-316) with a Mooncake-inspired, SLO-aware sizing policy that decouples encoder_compute_budget from encoder_cache_size. Concretely: (1) keep encoder_compute_budget tied to per-step compute headroom (a function of max_num_batched_tokens and the largest single mm item), but (2) compute encoder_cache_size as a function of expected concurrent in-flight multimodal items and an estimated reuse factor for multi-turn agentic sessions (e.g., expected_concurrent_items * max_tokens_per_mm_item * reuse_multiplier), capped by an explicit TTFT/TPOT-driven ceiling. Expose two new SchedulerConfig knobs (e.g., mm_encoder_cache_concurrency and mm_encoder_cache_reuse_factor) with adaptive defaults derived from max_num_seqs and modality mix, and validate that the resulting cache size keeps EncoderCacheManager.can_allocate from rejecting the steady-state concurrent media set. The disabled-chunking validation and the per-modality max_tokens_per_mm_item floor are preserved. Tests in tests/v1/core/test_encoder_cache_manager.py and tests/v1/core/test_scheduler.py should be extended to cover the concurrency- and reuse-driven cache size and the separate compute/cache budgets.

**Proposal rationale.**

The candidate currently sizes encoder cache only off the largest single mm item and ignores concurrency, reuse, and SLO context, which is precisely the gap the evolve_rationale flags (concurrency-aware cache sizing, separate compute/cache budgets, adaptive defaults for repeated media in agentic sessions). Mooncake's central thesis - treat the cache as a first-class scheduling resource and size/admit it against TTFT/TPOT SLOs and reuse patterns rather than local free-block heuristics - transfers directly to encoder cache for multimodal: in multi-turn agentic media workloads, the same image/audio embeddings are reused across turns, so a cache sized only to the single-largest item systematically forces re-encode and inflates media TTFT. Decoupling compute budget (per-step admission) from cache size (cross-step residency), and letting cache size scale with expected concurrent items and reuse, is the encoder-side analogue of Mooncake's KVCache-centric capacity policy and plausibly reduces media TTFT and median TPOT for the stated caller workload.

---

### 5. Size encoder cache for cross-turn media reuse in agentic workloads
- **Finding:** `find-0013` — *VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference*
- **Source URL:** <https://arxiv.org/pdf/2512.12977>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In compute_mm_encoder_budget (vllm/v1/core/encoder_cache_manager.py:269-316), decouple encoder_cache_size from encoder_compute_budget and grow the cache target beyond max_tokens_per_mm_item so multiple recent media items can be retained across turns. Concretely: keep encoder_compute_budget = max(configured, max_tokens_per_mm_item) (a single-item peak driven by chunking/disable_chunked_mm_input validation), but compute encoder_cache_size as max(configured, k * max_tokens_per_mm_item) where k is an agentic-reuse factor derived from a new SchedulerConfig knob (default >=2, surfaced for tuning) and optionally clamped by a fraction of max_num_batched_tokens. This mirrors the candidate's evolve_rationale items 'concurrency-aware cache sizing', 'separate compute/cache budgets per modality', and 'adaptive defaults for repeated media in agentic sessions', and keeps invariants exercised by tests/v1/core/test_encoder_cache_manager.py and tests/v1/core/test_scheduler.py (the new size only relaxes capacity, never lowers it below today's value). No reuse algorithm is implemented here; this finding only motivates the sizing target so a downstream reuse path (exact-hit today, possibly partial-recompute later) has room to retain prior-turn encoder outputs.

**Proposal rationale.**

The candidate sizes encoder_cache_size from a single-item peak, which is correct for one-shot media but leaves no room to retain prior-turn encoder outputs - exactly the regime VLCache shows is most valuable. The finding's headline result (1.2x-16x TTFT speedup by reusing 95-98% of recurring vision tokens) is contingent on the encoder cache actually holding past media; if encoder_cache_size == max_tokens_per_mm_item, at most one item fits and any reuse-aware policy is starved before it can act. The caller context (multi-turn agentic, optimize media TTFT) matches VLCache's target workload directly, and the proposed change is a minimal, conservative sizing adjustment that unlocks the reuse benefit without committing to a specific reuse algorithm or altering compute-budget semantics that the scheduler relies on for chunking decisions.

---

### 6. Size encoder cache for cross-turn multimodal reuse in agentic sessions
- **Finding:** `find-0017` — *Stateful Large Language Model Serving with Pensieve*
- **Source URL:** <https://arxiv.org/abs/2312.05516>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify compute_mm_encoder_budget in vllm/v1/core/encoder_cache_manager.py (lines 269-316) so that encoder_cache_size is no longer just max(configured, max_tokens_per_mm_item) but instead also reserves capacity for retained encoder outputs across turns. Concretely: keep encoder_compute_budget tied to per-step throughput (largest item / chunk headroom), but compute encoder_cache_size as max(configured, max_tokens_per_mm_item * expected_retained_items), where expected_retained_items is derived from a new SchedulerConfig hint (e.g., max_concurrent_mm_sessions and mm_history_retention_turns, defaulting to 1 to preserve current behavior). The intent is that finished turns' encoder outputs for media reused later in the same session are not evicted under pressure, so the cache acts as a stateful owner of per-session media embeddings. Downstream, EncoderCacheManager.can_allocate / has_cache lookups (which already key on (request_id, input_id)) benefit from the larger cache without changing their interface; only the budget sizing is touched. Wire the new fields through SchedulerConfig and document them in the function's docstring; tests/v1/core/test_encoder_cache_manager.py and tests/v1/core/test_scheduler.py should be extended to assert that with retention>1 the budget grows linearly and that single-turn behavior is unchanged when retention=1.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'concurrency-aware cache sizing' and 'adaptive defaults for repeated media in agentic sessions' as headroom, and the caller context is a multi-turn agentic workload optimizing media TTFT. Pensieve's core idea — treat conversation state as a cross-request cache owner so repeated history is not reprocessed — maps directly onto the encoder cache: in agentic loops, the same images/audio recur turn after turn, but today encoder_cache_size is provisioned only for a single largest item, forcing re-encoding under pressure. The finding contributes the concrete, transferable shift from 'size for one item' to 'size for retained per-session items', addressing the gap that the current static max(...) policy does not adapt to expected retained media across turns. This is the budget-side complement to runtime retention and is the smallest change to compute_mm_encoder_budget that lets the existing EncoderCacheManager hit-path actually serve repeated media without thrashing.

---

## Agent proposals

### 1. Denominate encoder_compute_budget in modality-weighted compute-equivalent tokens, not raw tokens
- **Agent:** claude

**Detailed description.**

In compute_mm_encoder_budget (vllm/v1/core/encoder_cache_manager.py:269-316), change the unit in which encoder_compute_budget is expressed from raw multimodal-token counts to modality-cost-weighted equivalent tokens, while leaving encoder_cache_size in raw token units (since cache footprint is genuinely per-token).

Concretely:

1) Introduce a per-modality compute-cost coefficient `c_mod` representing the encoder forward FLOPs per output token of modality `mod` divided by a reference modality's FLOPs per token (image-patch encoder = 1.0 by default). Source the coefficients in this order: (a) an explicit SchedulerConfig.mm_encoder_cost_coeffs dict, (b) a per-model registry keyed on the multimodal config (vision tower depth, video temporal stride, audio mel-spectrogram window) computed once at startup, (c) fallback 1.0 to preserve current behavior.

2) Replace the line that sets `max_tokens_per_mm_item = max(mm_max_toks_per_item.values())` for the *compute* path with a weighted form: `max_compute_equiv_per_item = max(c_mod * mm_max_toks_per_item[mod] for mod in mm_max_toks_per_item)`. Use this quantity for the `encoder_compute_budget = max(configured_compute, max_compute_equiv_per_item)` assignment and for the `disable_chunked_mm_input` validation, so a video item that is cheap-per-token but token-numerous does not reserve as much per-step encoder compute as a smaller-but-FLOP-heavier image item, and vice versa.

3) Keep `encoder_cache_size = max(configured_cache, max(mm_max_toks_per_item.values()))` in raw token units. The cache stores encoder outputs whose memory footprint is linear in tokens regardless of producing modality, so weighting cache sizing would be incorrect.

4) Plumb the coefficients to Scheduler._try_schedule_encoder_inputs so its admission test charges encoder work in the same compute-equivalent unit; an explicit `encoder_tokens_to_compute_equiv(num_tokens, modality)` helper on EncoderCacheManager keeps the conversion in one place. Defaults of 1.0 reproduce today's numbers exactly, so existing tests in tests/v1/core/test_encoder_cache_manager.py and tests/v1/core/test_scheduler.py pass unchanged; new tests should verify that with non-unit coefficients the per-step admitted token count tracks the FLOP budget (e.g., a 0.3x-cost audio modality admits ~3.3x more audio tokens per step than the image baseline).

This is a unit-system change to the existing static max(...) policy, not a new dynamic mechanism, so it composes cleanly with any of the listed concurrency/reuse-based cache-sizing proposals: those would still operate on raw token counts for cache_size, while encoder_compute_budget would now be expressed in compute-equivalent units that reflect actual stall risk.

**Novelty rationale.**

Every listed deep_research_proposal arithmetic-operates on raw multimodal token counts: find-0001 separates compute vs cache and offers per-modality budgets but still in token units; find-0007 caps compute at a fraction of max_num_batched_tokens (token-denominated); find-0009/0011/0013/0017 all multiply max_tokens_per_mm_item by concurrency or reuse factors. None addresses that a token of one modality costs a different number of encoder FLOPs than a token of another modality - so a token-denominated compute budget systematically over-admits cheap-per-token modalities (e.g., audio mel patches) and under-admits expensive ones (e.g., dense video temporal attention), which directly affects TPOT in mixed-modality agentic workloads. The proposal switches the *unit* of the compute budget from raw tokens to modality-weighted compute-equivalent tokens; this is orthogonal to the existing proposals' headroom/concurrency/reuse multipliers and complements them rather than restating them. It also explicitly preserves raw-token sizing for cache_size, separating a unit concern that all prior findings conflate.

---

### 2. Validate disabled MM chunking against placeholder span, not encoder-embed count
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/encoder_cache_manager.py:269-316`, separate the unit used for the `disable_chunked_mm_input` validation from the unit used for encoder compute/cache capacity. Today `mm_max_toks_per_item` is derived from `PlaceholderRange.get_num_embeds()`, but `_try_schedule_encoder_inputs` decides whether an MM item is chunked using `mm_position.length`. For masked or embedding-sparse placeholders, `length` can be much larger than `get_num_embeds()`, so startup can accept `disable_chunked_mm_input=True` even when the scheduler cannot fit the full placeholder span in one `max_num_batched_tokens` step. Add a companion per-modality max placeholder-span value from the multimodal dummy placeholders, pass it into `compute_mm_encoder_budget`, and use `max_placeholder_tokens_per_mm_item > scheduler_config.max_num_batched_tokens` for the disabled-chunking `ValueError`. Keep `encoder_compute_budget` and `encoder_cache_size` based on encoder-embed counts, since those are the resources consumed by encoder execution and cache storage. Add tests covering a placeholder with `length=128`, `get_num_embeds()=8`, `max_num_batched_tokens=64`, and `disable_chunked_mm_input=True`, which should now fail fast; the same case with chunking enabled should preserve current budget values.

**Novelty rationale.**

The deep research proposals focus on concurrency, reuse, SLO-aware cache sizing, compute/cache decoupling, and batched-token headroom. Agent A focuses on modality-weighted compute-equivalent tokens. None of them addresses the distinct unit mismatch between decoder placeholder span (`mm_position.length`, used by the scheduler to define chunk boundaries) and encoder embedding count (`get_num_embeds()`, used for compute/cache capacity). This proposal changes only the disabled-chunking validation unit while preserving raw embed-token budget sizing, so it is orthogonal to both cache-reuse policies and modality compute weighting.

---
