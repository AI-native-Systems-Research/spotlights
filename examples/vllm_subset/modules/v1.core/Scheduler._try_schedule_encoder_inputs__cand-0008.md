# Scheduler._try_schedule_encoder_inputs

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 1103–1264)
- **Symbol:** `Scheduler._try_schedule_encoder_inputs`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0008`

## Description
Determines which multimodal encoder inputs overlap the current token range, checks local and external encoder caches, enforces encoder compute/cache budgets, rolls back token scheduling when chunked multimodal input is disabled, and returns scheduled/local-load indexes.

## Current approach
Linear scan over request.mm_features on every call. Each candidate performs range checks, duplicate detection with a per-call set, EncoderCacheManager cache and capacity probes, ECConnector lookup, and chunk rollback logic.

## Estimated impact explanation
This moves TTFT for multimodal agentic turns by reducing repeated scheduler-side scans and cache probes. Impact is workload-dependent because text-only requests bypass it.

## Evolve rationale
The concrete hot construct is the for i, mm_feature in enumerate(mm_features) scan. Multimodal multi-turn requests can call it on every admission and decode step, while mm_features may be long. Headroom includes indexing mm_features by start_pos, carrying the next relevant feature index on the request, separating cache-hit dedupe from budget checks, and short-circuiting external-cache probes. Correctness oracles include tests/v1/core/test_scheduler.py and tests/v1/core/test_encoder_cache_manager.py, which cover encoder cache accounting, eviction, and scheduler multimodal invariants.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Event-driven per-request encoder state replaces per-step cache probes
- **Agent:** claude

**Detailed description.**

In `vllm/v1/core/sched/scheduler.py:1103-1264` (`Scheduler._try_schedule_encoder_inputs`), replace the per-call probes into `encoder_cache_manager.check_and_update_cache`, `encoder_cache_manager.can_allocate`, and `ec_connector.has_cache_item` with reads of state precomputed on the `Request` object and pushed to it by event callbacks. Concretely: (1) at request admission, build on the `Request` an `_encoder_plan` consisting of mm_features sorted by `mm_position.offset` with cached `num_embeds`, `item_identifier`, and a per-feature `_encoder_state` enum (`UNCONSUMED` / `LOCAL_HIT` / `EXTERNAL_HIT` / `SCHEDULED_THIS_STEP` / `RETIRED_TO_KV`); (2) extend `EncoderCacheManager` to maintain a per-mm_hash subscriber set and notify subscribed requests on `add`/`free`/`evict`, flipping `_encoder_state` to/from `LOCAL_HIT`; (3) extend `ECConnector` to push `EXTERNAL_HIT`/miss deltas into requests that registered for an mm_hash on admission, instead of being polled each step; (4) reduce the hot loop to: advance a `next_active_idx` cursor (lower bound by `start_pos + num_encoder_tokens > num_computed_tokens`), iterate forward only while `start_pos < num_computed_tokens + num_new_tokens + shift_computed_tokens`, and dispatch on `_encoder_state` — the only per-step work that remains is the budget/capacity arithmetic and the `disable_chunked_mm_input` rollback. The per-step `mm_hashes_to_schedule` set becomes redundant because `_encoder_state == SCHEDULED_THIS_STEP` (cleared at the end of `schedule()`) is the dedupe signal. Validate against `tests/v1/core/test_scheduler.py` and `tests/v1/core/test_encoder_cache_manager.py`; for performance, exercise a multi-turn agentic workload with many recurring images per request and measure media TTFT plus per-decode-step scheduler latency before/after.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's `evolve_rationale` lists pull-side optimizations only — sorted-index lookup of mm_features, a per-request cursor, separating dedupe from budget checks, and short-circuiting external-cache probes when locally satisfied. This proposal is materially different because it replaces probing with a push/subscription model: `EncoderCacheManager` and `ECConnector` notify requests of cache state changes, so the scheduler hot path performs zero cache lookups per feature — it reads a cached enum. It also introduces a per-feature state machine on the `Request`, which is finer-grained than a single cursor and removes the need for the per-call `mm_hashes_to_schedule` set.

---

### 2. Make encoder cache allocation transactional
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/sched/scheduler.py:1103-1264`, stop treating `encoder_cache_manager.can_allocate()` as a mutating probe. Today that call can evict `freeable` encoder cache entries and append `freed` before the scheduler knows the request will actually get KV slots; if `allocate_slots()` later fails, token scheduling rolls back, or the request is skipped, those evictions are still emitted through `free_encoder_mm_hashes` and can force later turns to reload or recompute reusable media. Add a non-mutating reservation path, for example `EncoderCacheManager.plan_allocate(...) -> EncoderCacheReservation`, that checks compute/cache capacity and records the eviction plan without changing `cached`, `freeable`, `num_free_slots`, or `freed`. Have `_try_schedule_encoder_inputs` return the reservation alongside `encoder_inputs_to_schedule` and `external_load_encoder_input`; commit it only after KV allocation succeeds and immediately before the existing `allocate()` / `update_state_after_alloc()` calls. Drop the reservation on all zero-token, rollback, skip, and failed-KV-allocation paths. Validate with a regression in `tests/v1/core/test_scheduler.py` that forces KV allocation failure after an encoder item appears and asserts no encoder cache entries are freed, plus `tests/v1/core/test_encoder_cache_manager.py` coverage for plan-vs-commit accounting.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A proposes a request-local event/subscription state machine to replace repeated local and external cache probes, plus a cursor over active features. This proposal is different: it keeps the existing pull-based scheduling shape but makes the cache-capacity side effect transactional, specifically addressing premature `can_allocate()` evictions when scheduling is still tentative. Agent A does not cover deferred eviction commit or rollback-safe encoder cache reservations.

---
