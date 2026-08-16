# SlidingWindowManager.reachable_block_mask

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 996–1055)
- **Symbol:** `SlidingWindowManager.reachable_block_mask`
- **Kind:** config_block
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0021`

## Description
Sliding-window sparse-retention mask policy that decides which newly full blocks are worth registering in the prefix cache under retention_interval, alignment, and EAGLE settings.

## Current approach
Computes the contiguous run length needed for a hit, applies an EAGLE shift, allocates a boolean mask, marks segment-boundary tails based on retention_interval, and explicitly preserves replay/shared-prefix reachable-boundary tails.

## Estimated impact explanation
The policy trades cache memory for future hit rate. For multi-turn agents with long overlapping prefixes, better SWA retention improves follow-up-turn cache hits and reduces median TTFT.

## Evolve rationale
Concrete policy behind sparse sliding-window checkpoint retention. Headroom in adaptive interval selection, workload-aware boundary retention, and cheaper mask construction for large block ranges. Correctness oracle: for any start/end/alignment/window/use_eagle combination, every True block must be in a tail that can satisfy find_longest_cache_hit, every reachable boundary tail must be retained, and None must remain equivalent to all blocks cacheable.

## Deep research proposals

### 1. Score sliding-window tail retention by predicted reuse × compute-saved / memory
- **Finding:** `find-vllm_v1_core-0004` — *Marconi: Prefix Caching for the Era of Hybrid LLMs*
- **Source URL:** <https://huggingface.co/papers/2411.19379>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `SlidingWindowManager.reachable_block_mask` (vllm/v1/core/single_type_kv_cache_manager.py:996-1055), replace the fixed-interval segment-tail policy driven solely by `retention_interval` with a Marconi-style admission score that ranks each candidate `need`-block tail by (predicted reuse likelihood) × (compute saved on a hit) / (memory footprint of the tail), and marks tails True in descending score order until a per-request or per-cache memory budget is met. Concretely: keep the existing geometry (contiguous `need`-length tails ending at aligned boundaries, with EAGLE `shift`, plus the always-retained `reachable_boundaries` tails from step (2)) but derive a per-boundary score from lightweight signals already available at cache-manager scope — recent hit history on that alignment offset, distance from the request's live sliding window, and the number of tokens that would be saved on a hit (≈ `need * block_size` minus overhead). Expose `retention_interval` as the coarse fallback when no scorer is configured, so `None` (dense) and `0` (no dense tails) behavior is preserved. The correctness oracle in the candidate description is unchanged: every True block still lies in a tail that can satisfy `find_longest_cache_hit`, `reachable_boundaries` tails remain retained explicitly, and `alignment_tokens is None` still short-circuits to `None`.

**Proposal rationale.**

The candidate's current approach uses a fixed `retention_interval` that treats every segment boundary as equally worth caching, which is a recency/geometry heuristic — exactly the class of policy Marconi argues against. The finding's admission/eviction scoring (`reuse likelihood` × `compute savings` / memory) is directly transferable: sliding-window tails are the unit of admission, they have a known compute-saved-on-hit (`need` blocks) and a known memory cost, and the multi-turn agentic workload hint implies non-uniform reuse across boundaries (shared prefixes and replay boundaries dominate). Scoring-based retention plausibly improves follow-up-turn hit rate at fixed memory, which addresses the estimated impact (median TTFT via better SWA hits on overlapping prefixes) without changing the reachability invariants the candidate must preserve.

---

### 2. Gate sliding-window mask tails with a TinyLFU frequency sketch to skip one-off boundaries
- **Finding:** `find-vllm_v1_core-0009` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://doczz.net/doc/8550019/tinylfu--a-highly-efficient-cache-admission-policy>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend SlidingWindowManager.reachable_block_mask (vllm/v1/core/single_type_kv_cache_manager.py:996-1055) with an optional TinyLFU-style admission filter that decides whether each candidate True bit — both the segment-boundary tails computed in section (1) and the reachable-boundary tails from `reachable_boundaries` in section (2) — should actually be registered. Concretely: (a) maintain a compact Count-Min sketch keyed by a stable prefix identifier (e.g., the hash of the block or its aligned boundary token position) plus a small doorkeeper Bloom filter, updated when find_longest_cache_hit observes a hit and aged with a periodic halving pass; (b) plumb an optional admission-oracle callable (or the sketch itself) into reachable_block_mask via a new keyword argument, defaulting to None to preserve current behavior and the `None means all cacheable` contract; (c) when the oracle is provided, for each block index i that the structural policy would set to True, query estimated frequency for the boundary that tail ends on and only keep the mask bit set if the estimate exceeds a small threshold (first-time-seen boundaries pass through the doorkeeper so cold prefixes are not permanently starved). The replay boundary (num_prompt - 1) should be exempt from the filter — that tail is needed for correctness of the current request — so the filter only trims segment-boundary tails and shared-prefix-junction tails whose value is speculative. Keep the fast-path `alignment_tokens is None -> None` branch and the `need >= per_segment -> None` shortcut unchanged.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out `workload-aware boundary retention` as a gap, and the current mask policy is purely structural: every segment boundary and every shared-prefix junction gets a tail regardless of whether that prefix is ever revisited. Under multi-turn agentic workloads with long overlapping prefixes and memory pressure, that uniformly-dense retention lets low-reuse tails from one-off segments compete with hot shared prefixes for prefix-cache slots, dragging follow-up-turn hit rate and median TTFT. TinyLFU's contribution — approximate frequency-based admission with a compact sketch and optional doorkeeper — directly addresses this: it lets the mask preserve tails that carry demonstrated reuse and drop tails whose boundaries have never been hit, without requiring an exact per-block reuse counter. The supporting quote (`TinyLFU decides if replacing the cache victim with the new item is expected to increase the hit-ratio`) maps cleanly onto the register-vs-skip decision this mask encodes. The correctness oracle in the candidate is preserved because the replay boundary tail (required for the current request) is exempted and the fast paths that return None remain intact; the filter only tightens speculative tails, so every True bit still ends on a boundary that can satisfy find_longest_cache_hit.

---

## Agent proposals

### 1. Build the sliding-window mask in closed form over tail intervals instead of a per-block Python loop
- **Agent:** claude

**Detailed description.**

In `SlidingWindowManager.reachable_block_mask` (vllm/v1/core/single_type_kv_cache_manager.py:996-1055), replace the per-index Python loop over `range(start_block, end_block)` that evaluates `(i - shift) % per_segment >= per_segment - need` with a closed-form enumeration of the True-valued tail intervals, and materialize the result via a NumPy `bool` array with vectorized slice assignments (converted to a `list[bool]` at the boundary only if callers still require one). Concretely: (a) compute the first aligned boundary index `first_end = shift + per_segment * ceil((start_block - shift + per_segment - need) / per_segment)` and iterate boundaries `end_i = first_end + k * per_segment` while `end_i - need < end_block`, setting `mask[max(start_block, end_i - need) - start_block : min(end_block, end_i) - start_block] = True`; this replaces `end_block - start_block` Python-level modulo checks with `ceil((end_block - start_block) / per_segment)` slice writes. (b) Apply the same interval-slice pattern to the `reachable_boundaries` loop (section (2)) — it is already interval-shaped, so drop the inner `range(...)` and use a single slice assignment per boundary. (c) Preserve the existing fast paths byte-for-byte: `alignment_tokens is None -> None`, `segment_tokens is None -> skip section (1)`, and `need >= per_segment -> return None`. Because the mask a boolean array whose True positions are a deterministic arithmetic function of `(start_block, end_block, alignment_tokens, block_size, sliding_window, use_eagle, retention_interval, reachable_boundaries)`, every produced bit is provably identical to the current implementation's output; the correctness oracle in the candidate description is unchanged.

**Novelty rationale.**

Both listed deep_research_proposals are ADMISSION-policy changes (Marconi scoring in find-0004; TinyLFU frequency-sketch gating in find-0009) that add work per candidate True bit to improve hit-rate under memory pressure. Neither addresses the third gap the candidate's evolve_rationale explicitly names — 'cheaper mask construction for large block ranges' — and both, if anything, add per-bit overhead. This proposal is orthogonal: it is a pure construction-cost optimization that leaves the retention policy (which bits are True) bit-identical to today's implementation, and it composes cleanly with either scoring or TinyLFU gating layered on top. It targets the exact same file/symbol/lines but a different axis (mask assembly speed on the hot path of block-registration), so it does not overlap with the reuse-vs-memory scoring axis of find-0004 or the reuse-frequency admission axis of find-0009.

---

### 2. Validate sparse retention intervals against scheduler alignment
- **Agent:** codex

**Detailed description.**

In `SlidingWindowManager.reachable_block_mask` (`vllm/v1/core/single_type_kv_cache_manager.py:996-1055`), add an explicit guard for positive `retention_interval` values before deriving `per_segment`: require `retention_interval % alignment_tokens == 0` (and therefore also block-aligned, given the existing `alignment_tokens % block_size == 0` assertion), or fail fast with a clear `ValueError` at the caller/config boundary. The current code floors `retention_interval // block_size` and then treats those block-period boundaries as reachable, but `find_longest_cache_hit` only accepts hits whose post-EAGLE boundary is aligned to `alignment_tokens`. A non-multiple interval can therefore register segment-tail blocks that the hit path will later skip, adding prefix-cache hash-map pressure without improving TTFT. Add focused tests for `retention_interval` values that are valid multiples, `0`, `None`, and invalid non-multiples, including `use_eagle=True`, to lock the invariant that every sparse segment tail ends on a boundary the hit path can actually use.

**Novelty rationale.**

The deep research proposals change admission policy using reuse scoring or TinyLFU frequency filtering, and Agent A optimizes mask construction cost with closed-form interval writes. This proposal is neither a scoring policy nor a construction-speed optimization: it is a configuration/correctness guard that prevents structurally unreachable tails from being admitted in the first place when `retention_interval` is misaligned with the scheduler hit boundary. It targets wasted cache occupancy and invariant enforcement, not adaptive reuse ranking or vectorized materialization.

---
