# KVCacheManager.allocate_slots

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/kv_cache_manager.py`](vllm/v1/core/kv_cache_manager.py) (lines 345–566)
- **Symbol:** `KVCacheManager.allocate_slots`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_core-0012`

## Description
Central KV slot allocation entry point for admitted or extending requests: fit checks, skipped-block removal, local/external prefix blocks, new block allocation, and cache insertion.

## Current approach
Computes local and total computed tokens, applies watermark and reserved-block gates, optionally performs a full-sequence fit pre-check that duplicates get_num_blocks_to_allocate work, removes skipped blocks, recomputes required blocks, allocates computed and new blocks, and caches finalized tokens.

## Estimated impact explanation
Every scheduled request that needs KV capacity goes through this method. Removing redundant admission work and reducing allocation bookkeeping lowers TTFT for prefills and TPOT for decode steps that allocate blocks.

## Evolve rationale
The full_sequence_must_fit path duplicates get_num_blocks_to_allocate work, and the ordering of remove_skipped_blocks, allocation sizing, and caching exposes batching and memoization opportunities across coordinator sizing calls. Correctness oracle: tests/v1/core/test_kv_cache_manager.py can assert allocation size, returned blocks, ref_cnt updates, cache-key registration, and None-on-insufficient-capacity behavior across admission paths.

## Deep research proposals

### 1. Add a commit-policy parameter to allocate_slots to gate exact-prefix caching of external/approximate KV
- **Finding:** `find-vllm_v1_core-0008` — *[RFC]: Semantic KV Cache Reuse Interface*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/44223>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend `KVCacheManager.allocate_slots` (vllm/v1/core/kv_cache_manager.py:345-566) with an explicit cache-commit policy argument (e.g. `commit_policy: CommitPolicy = CommitPolicy.EXACT_COMMIT`, with a second variant such as `REQUEST_ONLY`). Today the method's final caching stage (lines 552-564) has only two modes: either skip caching entirely via `delay_cache_blocks`/`not self.enable_caching`, or cache `min(total_computed_tokens + num_new_tokens, request.num_tokens)` through `self.coordinator.cache_blocks(request, num_tokens_to_cache)`. That binary flow always registers external-connector KV (accounted through `num_external_computed_tokens` at lines 459-462, 516-517, 536-541) into the exact prefix-cache map, which is unsafe for semantic/approximate reuse.

The change threads the commit policy through allocation as follows: (1) when the policy is `REQUEST_ONLY`, cap `num_tokens_to_cache` to the strictly-local finalized-token count (i.e. `min(num_local_computed_tokens + num_new_tokens, request.num_tokens)`) so external tokens are made usable for this request via `allocate_new_computed_blocks` (lines 536-541) but are not committed to the exact-match prefix cache via `cache_blocks`; (2) plumb the policy from the connector-facing scheduler path (P/D and semantic connectors that supply `num_external_computed_tokens`) so donors can request per-request-only reuse; (3) keep the default `EXACT_COMMIT` behavior byte-identical, so the existing `test_kv_cache_manager.py` oracles for allocation size, returned blocks, `ref_cnt`, cache-key registration, and None-on-insufficient-capacity remain unchanged. The fit checks, watermark/reserved-block gates, `remove_skipped_blocks`, and `get_num_blocks_to_allocate` paths are untouched — the policy only alters the terminal commit step.

**Proposal rationale.**

The finding identifies a concrete upstream gap — a connector-controlled cache-commit policy — that maps directly onto the one branch of `allocate_slots` where external KV becomes visible to the global prefix cache. Adding an EXACT_COMMIT/REQUEST_ONLY switch at the `cache_blocks` call site is a minimal, transferable change that unblocks safe use of approximate/semantic donor KV in multi-turn agentic workloads (the stated caller objective), where speculative or fuzzy reuse can materially reduce TTFT without polluting the exact-hit cache that other requests rely on. It respects the candidate's existing structure (all fit-checks, skipped-block removal, and coordinator sizing calls are unchanged) and is testable against the same correctness oracle the candidate already cites.

---

## Agent proposals

### 1. Reuse the full_sequence_must_fit sizing result to eliminate a duplicate coordinator.get_num_blocks_to_allocate call
- **Agent:** claude

**Detailed description.**

In `KVCacheManager.allocate_slots` (vllm/v1/core/kv_cache_manager.py:345-566), remove the duplicate coordinator sizing traversal that happens whenever `full_sequence_must_fit=True`. Today lines 477-486 call `self.coordinator.get_num_blocks_to_allocate(request_id, num_tokens=full_num_tokens, new_computed_blocks=new_computed_block_list, num_encoder_tokens=num_encoder_tokens, total_computed_tokens=total_computed_tokens, num_local_computed_tokens=num_local_computed_tokens, num_tokens_main_model=full_num_tokens, apply_admission_cap=True)` as an admission gate, and lines 511-520 immediately call `get_num_blocks_to_allocate` again with `num_tokens=num_tokens_need_slot` and the same request_id / new_computed_block_list / num_encoder_tokens / num_local_computed_tokens plus the same `num_local_computed_tokens + num_external_computed_tokens` for total_computed_tokens. For the common admission case where `num_tokens_need_slot == full_num_tokens` (no lookahead tokens, no sliding-window truncation, chunked prefill has stabilized to the full sequence), the second call recomputes an identical result across all KV cache groups.

The change: (1) In the `full_sequence_must_fit` branch, keep the admission call but also record the resulting `num_blocks_to_allocate` (call it `admission_num_blocks`) and the tuple of inputs used (`full_num_tokens`, `new_computed_block_list` identity, `num_encoder_tokens`, `total_computed_tokens`, `num_local_computed_tokens`). (2) After computing `num_tokens_need_slot` at lines 491-494, compare it against `full_num_tokens`; if equal, and the `apply_admission_cap` semantics of the admission call are a superset of the second call's requirements (they are — `apply_admission_cap=True` only tightens sizing), reuse `admission_num_blocks` instead of calling `get_num_blocks_to_allocate` a second time. If different (spec-decode lookahead, sliding-window shrink, encoder-token divergence), fall back to the current second call so semantics are preserved. (3) Keep `remove_skipped_blocks` (lines 505-509) exactly where it is — it must run before the second sizing in the fallback path — and keep all downstream `allocate_new_computed_blocks` / `allocate_new_blocks` / `cache_blocks` calls untouched.

Because the free-block check at lines 488-489 already uses the admission call's sizing to reject infeasible requests, the reused number is guaranteed to be a valid upper bound on what the coordinator will allocate for the shorter or equal `num_tokens_need_slot`; the subsequent `available_blocks` check at lines 524-528 remains correct. The correctness oracle cited by the candidate (tests/v1/core/test_kv_cache_manager.py — allocation size, returned blocks, ref_cnt updates, cache-key registration, and None-on-insufficient-capacity) continues to apply unchanged. Expected TTFT win comes from cutting a coordinator-wide traversal across all KV cache groups on every admission where `full_sequence_must_fit=True`, which in the caller's multi-turn agentic workload is the hot admission path for chunked-prefill re-entries.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_core-0008) adds a `commit_policy` parameter that only alters the terminal `cache_blocks` step (lines 552-564) and explicitly states that the fit checks, watermark/reserved-block gates, `remove_skipped_blocks`, and `get_num_blocks_to_allocate` paths are untouched. This proposal targets exactly those untouched sizing paths (lines 473-520) — specifically the duplicated `get_num_blocks_to_allocate` traversal that the candidate's own evolve_rationale flags. It is non-overlapping in code location (sizing vs. commit), mechanism (memoize/short-circuit vs. new policy argument), and rationale (redundant coordinator work vs. cache-safety for approximate KV).

---

### 2. Move skipped-block removal before the full-sequence fit gate
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/kv_cache_manager.py:473-509`, call `self.coordinator.remove_skipped_blocks(...)` before the `full_sequence_must_fit` free-block admission check instead of only after that check passes. Today the method returns `None` at lines 488-489 without releasing blocks that are already outside the request's active attention window, even though the later comment says skipped blocks can be freed even when the request cannot be scheduled. For long multi-turn requests with sliding window or hybrid attention, this can cause false admission failures and extra scheduler turns because the fit gate compares `required_blocks` against a free-block count that still includes reclaimable blocks held by the same request. The change should compute `total_computed_tokens` as it does today, run `remove_skipped_blocks(request.request_id, max(0, total_computed_tokens - request.num_in_flight_tokens), num_prompt_tokens=request.num_prompt_tokens)` before either full-sequence or normal capacity checks, then leave the existing sizing and allocation calls intact. Add a regression test that constructs a request with reclaimable skipped blocks and `full_sequence_must_fit=True`, with free capacity only becoming sufficient after skipped-block removal, and asserts `allocate_slots` succeeds rather than returning `None`.

**Novelty rationale.**

This is not covered by the deep_research_proposal, which changes only the terminal cache commit policy for external or approximate KV and explicitly leaves fit checks and skipped-block removal untouched. It also does not duplicate Agent A's proposal, which reuses the duplicated `get_num_blocks_to_allocate` result while explicitly keeping `remove_skipped_blocks` in its current position. This proposal targets the ordering of skipped-block reclamation relative to the full-sequence admission gate to avoid avoidable capacity rejects.

---
