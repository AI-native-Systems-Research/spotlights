# AsyncLookupManager._worker

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/tiering/async_lookup.py`](vllm/v1/kv_offload/tiering/async_lookup.py) (lines 214–249)
- **Symbol:** `AsyncLookupManager._worker`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0010`

## Description
Background lookup worker groups a flushed lookup batch by request and calls the tier-specific batch_lookup.

## Current approach
A single worker thread consumes one full-step queue item at a time, groups with a fresh dict keyed by req_id, calls batch_lookup once per request, and converts exceptions into all-False results for that request.

## Estimated impact explanation
Faster async secondary lookups shorten the retry window before promotion can start. This improves TTFT tail and can move median TTFT when secondary-tier hits are common, but the tier backend's own lookup latency bounds the gain.

## Evolve rationale
Grouping, batch-size caps, bounded coalescing, and worker parallelism are explicit async scheduling knobs. Correctness oracle: AsyncLookupManager tests and the invariant that every enqueued key receives exactly one cached boolean result unless cleaned up by request finish.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Deduplicate keys across the flushed batch before invoking batch_lookup
- **Agent:** claude

**Detailed description.**

In `AsyncLookupManager._worker` (vllm/v1/kv_offload/tiering/async_lookup.py:214-249), when grouping the flushed batch by req_id, the same OffloadKey can appear under multiple requests within a single step (a common pattern in multi-turn agentic workloads where several concurrent requests share prefix blocks). Today the worker issues a separate `batch_lookup` per request, and the same key is probed once per requesting request across the step's batch, wasting secondary-tier IOPS/roundtrips and inflating tail latency for the whole step's drain window.

Change the worker to first deduplicate keys across the entire flushed `pending` list, choosing a single canonical `ReqContext` per unique key (e.g., the first-seen context, which mirrors the scheduler-side invariant that `_lookup_batch` only appends on first sighting per step). Then group deduped keys by their canonical `ReqContext.req_id` and issue `batch_lookup` once per group, exactly as today. Because `_pending_results` is applied by `drain_results()` via `_lookup_state[key]` — a single dict entry regardless of how many requests share it — one verdict per unique key is sufficient and preserves the enqueue-once invariant asserted at lines 177-181. Additionally, if `req_context` is only used by concrete tiers as a hint (verify per-tier — otherwise keep per-req grouping only for keys with divergent contexts), collapse further into a single `batch_lookup` call per step to maximize backend batching.

On the fast/error path the semantics are unchanged: the exception → all-False fallback (lines 235-242) still maps one-to-one over the deduped `keys` list, and every originally-enqueued key still receives exactly one cached boolean when `drain_results()` runs, because deduped results still write through `_lookup_state.get(key)`. Add a lightweight counter/log for `duplicate_keys_dropped_per_step` to make the win observable.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, and the evolve_rationale explicitly enumerates the standard knobs it expects follow-ups to explore — grouping, batch-size caps, bounded coalescing, and worker parallelism. This proposal is orthogonal to all four: it does not change the number of workers, cap batch size, coalesce across steps, or alter the grouping key. It targets a distinct waste — intra-step duplicate probing of the same OffloadKey across multiple requests in multi-turn agentic workloads where prefix sharing is high — and directly reduces backend work per step, which shortens the retry window before promotion for shared prefixes and improves TTFT tail without adding threads.

---

### 2. Process short request lookup groups first and publish each group immediately
- **Agent:** codex

**Detailed description.**

In `AsyncLookupManager._worker` (`vllm/v1/kv_offload/tiering/async_lookup.py:214-249`), after building the per-`req_id` `batches`, process those request groups in ascending `len(keys)` order and enqueue each group's `(key, hit)` results to `_pending_results` as soon as that `batch_lookup` completes, instead of accumulating all request groups into one `results` list and publishing only after the whole flushed step is done. This reduces head-of-line blocking when one request contributes a large lookup group or hits a slow backend path: smaller groups can become visible to `drain_results()` on the next scheduler step without waiting for the largest group in the same flushed batch. Keep the existing exception-to-all-False behavior per group, and add/extend AsyncLookupManager tests to cover that multiple result queue items from one flushed batch are drained correctly and that a slow/large earlier group no longer blocks a later small group when ordering is applied.

**Novelty rationale.**

There are no deep_research_proposals listed for this candidate. Agent A's proposal removes duplicate `OffloadKey` probes across requests before lookup; this proposal does not deduplicate or change key identity semantics. It targets a separate scheduling issue: request-level head-of-line blocking caused by holding completed group results until every group in the flushed step has finished, and by processing large groups before small ones.

---
