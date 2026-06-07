# Scheduler.schedule (common-prefix computation)

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 861–869)
- **Symbol:** `Scheduler.schedule (common-prefix computation)`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0017`

## Description
Per-step region that computes num_common_prefix_blocks for all KV cache groups by taking the first running request and delegating to KVCacheManager.get_num_common_prefix_blocks before building SchedulerOutput.

## Current approach
The scheduler computes common-prefix metadata whenever self.running is non-empty, independent of whether downstream cascade attention is enabled or whether the common-prefix boundary changed since the previous step. The delegated full-attention path scans blocks until a ref_cnt mismatch.

## Estimated impact explanation
This affects median TPOT by reducing scheduler-side work that is repeated every decode step for long shared-prefix batches. Impact is medium because it is most visible when many running requests share long prefixes or cascade attention is disabled downstream.

## Evolve rationale
The optimization unit is the schedule: get_num_common_prefix_blocks region and its unconditional per-step call. High-concurrency agentic workloads often share long system prompts, so recomputing the same common-prefix boundary can add scheduler CPU on every decode step. Headroom includes gating the call when cascade attention is disabled, maintaining an incremental/common-prefix watermark as requests enter or leave running, and invalidating only on block allocation/free events. Correctness oracles include tests/v1/e2e/general/test_cascade_attention.py, tests/v1/worker/test_gpu_model_runner.py, and tests/v1/core/test_scheduler.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Cache num_common_prefix_blocks across decode steps with event-driven invalidation
- **Agent:** claude

**Detailed description.**

At vllm/v1/core/sched/scheduler.py:861-869, replace the unconditional per-step call to KVCacheManager.get_num_common_prefix_blocks with a cached result invalidated only by events that can actually change the common-prefix boundary.

Concretely:

1. Add scheduler state: `self._common_prefix_blocks: list[int] | None = None` and `self._common_prefix_dirty: bool = True`. Initialize per kv_cache_groups length.

2. Mark dirty in exactly the small set of places where the result can change:
   - When a request transitions into `self.running` (new request scheduled, preempted-then-resumed). The existing `scheduled_new_reqs`/`scheduled_resumed_reqs` paths already enumerate these.
   - When a request is removed from `self.running` (finish_requests, abort, preempt). All these paths already exist in scheduler.py; flip the flag there.
   - When the anchor request's block list grows during this step. New blocks at the tail of `self.running[0]` can extend the shared prefix only if all other running requests already hold those blocks (rare and handled by point 2 above), so allocating new tail blocks for the anchor alone is sufficient to invalidate.

3. In lines 861-869, only call `self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)` when `self._common_prefix_dirty` is True (or cache is None). Otherwise reuse `self._common_prefix_blocks`. After (re)computation, clear the dirty flag.

4. Add an early-out: if the model has no cascade-attention-capable layer (FullAttentionManager is the only group type that returns non-zero from `get_num_common_prefix_blocks`; SlidingWindow/ChunkedLocal/Mamba/CrossAttention all return 0 — see vllm/v1/core/single_type_kv_cache_manager.py:496/634/787/887/1088), precompute on Scheduler.__init__ a boolean `self._any_group_supports_cascade` by checking the manager classes in `kv_cache_coordinator.single_type_managers`. When False, skip the region entirely and keep the all-zeros default.

In steady-state decode (no requests entering/leaving, no new blocks for the anchor), the cache hits every step and the FullAttentionManager prefix scan — which walks blocks until a ref_cnt mismatch — is avoided. For long shared system prompts (the multi-turn agentic workload), this scan length grows with prompt length and runs every decode step, so the savings scale with prefix length × decode steps. Correctness oracles: tests/v1/e2e/general/test_cascade_attention.py, tests/v1/core/test_scheduler.py, tests/v1/worker/test_gpu_model_runner.py.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete proposal is novel by construction. Beyond that, this proposal goes further than the bullet sketch in the candidate's evolve_rationale by (a) precisely enumerating the minimal invalidation event set (running-set membership changes plus anchor-only block growth — not all alloc/free events, which would over-invalidate), (b) deriving a static early-out at __init__ from the manager class composition rather than relying on a runtime cascade-attention flag the scheduler does not currently see, and (c) keying the cache off the existing anchor (`self.running[0]`) so no new identity tracking is needed.

---

### 2. Compute common prefixes over the scheduled batch only
- **Agent:** codex

**Detailed description.**

Change the common-prefix region in `vllm/v1/core/sched/scheduler.py:861-869` so it asks the KV cache for the common prefix among the request IDs actually scheduled in this step, i.e. `num_scheduled_tokens.keys()`, rather than using `self.running[0]` plus `ref_cnt == len(req_to_blocks)` over every allocated running request. Add a `KVCacheManager.get_num_common_prefix_blocks_for_requests(request_ids)` path through the coordinator and implement it for `FullAttentionManager` by walking the first scheduled request's block list index-by-index, stopping when any other scheduled request has no block at that index, has a different block id/object, or the anchor block is null. Non-cascade-capable managers can keep returning 0. This aligns the metadata with the model runner, which computes cascade eligibility from the scheduled batch arrays, not the entire `self.running` set. It also fixes the documented edge case where an unscheduled running request with a divergent prefix forces the current `ref_cnt == len(req_to_blocks)` check to return 0 even though every request in the current forward pass shares a long prefix. Add a scheduler or KV manager test with three allocated requests where two scheduled requests share prefix blocks and one unscheduled request diverges; assert the scheduled-batch API reports the shared prefix while the legacy all-allocated check would not.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal focuses on caching the existing all-running/all-allocated computation and invalidating it on membership or block-growth events, plus a static no-cascade early-out. This proposal changes the set over which the prefix is computed to match the actual scheduled forward batch, enabling cascade attention in cases the existing computation incorrectly suppresses; it is not a cache, invalidation scheme, or manager-class early-out.

---
