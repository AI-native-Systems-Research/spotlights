# KVCacheCoordinator.get_num_blocks_to_allocate

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/kv_cache_coordinator.py`](vllm/v1/core/kv_cache_coordinator.py) (lines 130–190)
- **Symbol:** `KVCacheCoordinator.get_num_blocks_to_allocate`
- **Kind:** method
- **Estimated impact:** low
- **Id:** `cand-vllm_v1_core-0015`

## Description
Coordinator fanout that sums per-manager block-allocation requirements, with a cross-attention special case using encoder-token sizing.

## Current approach
Loops over single_type_managers, checks isinstance(manager, CrossAttentionManager) on every group, dispatches to either encoder-token sizing or standard request-token sizing, and accumulates the sum.

## Estimated impact explanation
Absolute cost is small because hybrid models usually have few groups, so it is mainly useful as part of broader admission-loop optimization. It can still shave TTFT overhead in request-heavy scheduling steps.

## Evolve rationale
In the allocate_slots admission loop and may be invoked twice per waiting request when full-sequence admission is enabled. Headroom in precomputed dispatch metadata, grouped manager calls, and cache reuse between the two admission passes. Correctness oracle: result must equal the sum of current per-manager calls for all KVCacheConfig group layouts.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Precompute per-group dispatch schedule to remove per-call isinstance checks in get_num_blocks_to_allocate
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/kv_cache_coordinator.py, replace the runtime isinstance(manager, CrossAttentionManager) check inside KVCacheCoordinator.get_num_blocks_to_allocate (lines 167-189) with a precomputed dispatch schedule built once in __init__. Concretely: after self.single_type_managers is constructed (around line 120), add self._alloc_dispatch = tuple((i, manager, isinstance(manager, CrossAttentionManager)) for i, manager in enumerate(self.single_type_managers)) and, if useful, split it further into two tuples self._alloc_standard_dispatch and self._alloc_cross_attn_managers so the hot loop can iterate a single tuple with no branch when no cross-attention group is present (the common case). Then rewrite get_num_blocks_to_allocate to iterate the precomputed schedule(s): sum manager.get_num_blocks_to_allocate(request_id, num_tokens, new_computed_blocks[i], total_computed_tokens, num_local_computed_tokens, num_tokens_main_model, apply_admission_cap=apply_admission_cap) over the standard schedule and add a separate short loop over cross-attn managers passing the fixed (num_encoder_tokens, [], 0, 0, num_encoder_tokens) arg tuple. Because kv_cache_config and single_type_managers are immutable after construction, the schedule is safe to freeze. The correctness oracle from the candidate is preserved: for every group layout the returned sum is bitwise identical to the current loop. For models with no cross-attention group (the vast majority), the fast path becomes a plain for-loop over a tuple with zero isinstance calls, which is the dominant micro-cost when the admission loop invokes this twice per waiting request under full-sequence admission on a multi-turn agentic workload where waiting queues can be nontrivial each scheduler step. Optionally memoize the standard-branch argument tuple per call at the coordinator level so both invocations of the two-pass admission gate reuse a single constructed args tuple, but the precomputed dispatch is the primary win.

**Novelty rationale.**

The candidate has no existing deep_research_proposals, so there is no prior proposal to overlap with. The idea is specific to the observation that manager identities and cross-attention membership are fixed at __init__ time, making the per-call isinstance dispatch pure overhead; freezing that dispatch as precomputed metadata is a concrete, correctness-preserving micro-optimization distinct from any general 'batch admission' or 'cache admission results' idea one might imagine.

---

### 2. Cache allocation-size results across admission retries within a scheduler step
- **Agent:** codex

**Detailed description.**

In vllm/v1/core/kv_cache_coordinator.py, add a tiny coordinator-local memoization path for KVCacheCoordinator.get_num_blocks_to_allocate so the second admission pass can reuse the exact sum computed for the same request and token-sizing inputs. Key the cache by the immutable inputs that affect the result: request_id, num_tokens, tuple(len(blocks) or block ids for each new_computed_blocks group if needed by manager semantics), total_computed_tokens, num_local_computed_tokens, num_tokens_main_model, num_encoder_tokens, and apply_admission_cap. Clear or scope the cache at the start/end of each scheduler allocation attempt, or keep only the last key/value on the coordinator to avoid stale growth. This targets the documented case where full-sequence admission may invoke this method twice per waiting request: the first call computes the per-manager sum normally, and a repeated call with identical sizing arguments returns immediately without fanning out to every KV cache manager. Tests should assert parity with the uncached path and that different apply_admission_cap or computed-block inputs miss the cache.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal removes per-call type dispatch overhead by precomputing standard vs cross-attention manager schedules; this proposal instead avoids the entire repeated per-manager calculation when the admission loop calls get_num_blocks_to_allocate twice with the same inputs. It is a separate optimization axis: reuse of a completed aggregate result, not precomputed dispatch metadata or isinstance elimination.

---
