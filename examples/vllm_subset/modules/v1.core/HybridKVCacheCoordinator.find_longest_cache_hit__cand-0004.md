# HybridKVCacheCoordinator.find_longest_cache_hit

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/kv_cache_coordinator.py`](vllm/v1/core/kv_cache_coordinator.py) (lines 487–591)
- **Symbol:** `HybridKVCacheCoordinator.find_longest_cache_hit`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0004`

## Description
Fixed-point prefix-cache lookup for hybrid KV cache groups. It repeatedly asks each attention-group manager for the longest hit at the current candidate length, shrinks the candidate when a group disagrees, handles EAGLE one-block verification, and truncates full-attention blocks to the final hit length.

## Current approach
Outer while True fixed-point loop with an inner pass over self.attention_groups. Any shrink restarts the pass. Full attention gets a cached downward-closed fast path, simple hybrid exits after one shrink, and block-size conversions use BlockHashListWithBlockSize wrappers on demand.

## Estimated impact explanation
Every cached admission on hybrid models pays this lookup. Reducing repeated group scans cuts TTFT for long shared agentic prefixes, especially when prefix hits are large.

## Evolve rationale
The specific optimization unit is the fixed-point loop over attention_groups and manager_cls.find_longest_cache_hit calls. Hybrid models with full attention plus SWA/Mamba make this the admission-time prefix lookup. Headroom exists in ordering groups by likely bottleneck, skipping already-converged non-full groups, memoizing block-size converted hash lists, and batching lookups at LCM-aligned lengths. Correctness oracles include tests/v1/core/test_prefix_caching.py, tests/v1/core/test_single_type_kv_cache_manager.py, and tests/v1/core/test_kv_cache_utils.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace fixed-point loop with a single LCM-capped pass plus bounded EAGLE post-step
- **Agent:** claude

**Detailed description.**

In `HybridKVCacheCoordinator.find_longest_cache_hit` (vllm/v1/core/kv_cache_coordinator.py:487-591), eliminate the outer `while True` fixed-point loop by exploiting the fact that each manager's `find_longest_cache_hit` already returns the group's true longest hit ≤ max_length, and that per-group prefix hits are downward-closed.

Concrete change:

1. Pre-floor the cap once: `cap = (max_cache_hit_length // self.lcm_block_size) * self.lcm_block_size`. Because `cap` is a multiple of every group's block size, each manager's reply at this cap is its true maximum hit subject to `cap` — there is no alignment-induced disagreement that requires re-querying.
2. Single pass over `self.attention_groups` for non-EAGLE groups: call `manager_cls.find_longest_cache_hit(..., max_length=cap, alignment_tokens=self.lcm_block_size)` exactly once per group, store `hit_blocks_by_group`, and track each group's `hit_X = len(hit_blocks[0]) * spec.block_size`.
3. Compute `joint = (min(hit_X) // self.lcm_block_size) * self.lcm_block_size`.
4. EAGLE post-step (bounded, no loop): for each `idx in self.eagle_attn_group_indices`, query its manager with `max_length = min(joint + spec.block_size, max_cache_hit_length)` and `use_eagle=True`. If the eagle drop reduces that group's hit below `joint`, set `joint = floor_to_lcm(new_hit)`. Because each EAGLE group drops at most one block, at most one re-floor is needed; iterate over EAGLE groups in a fixed order and short-circuit when none changes.
5. Final truncate: walk `self.attention_groups` once and trim each group's `hit_blocks_by_group` entry to `joint // spec.block_size` blocks. This subsumes the existing tail-only trim of full-attention blocks at lines 581-587.

Keep the existing `BlockHashListWithBlockSize` wrapper for non-matching block sizes; only the control flow changes. The `is_simple_hybrid` fast path becomes redundant and can be removed. `eagle_verified` and the inner `if isinstance(spec, FullAttentionSpec) and cached_blocks is not None` shortcut also become unnecessary since each group is touched exactly once in step 2.

Worst-case manager calls drop from O(G²) (G shrinks × G groups per inner pass) to O(G + |eagle_groups|), which on hybrids with full + SWA + Mamba (G=3) is a 3× reduction in per-admission lookup cost. Correctness is anchored on the existing oracles: tests/v1/core/test_prefix_caching.py, tests/v1/core/test_single_type_kv_cache_manager.py, and tests/v1/core/test_kv_cache_utils.py — particularly EAGLE coverage around issue #32802.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, and the change is not implied by any item in `evolve_rationale`. That list enumerates micro-optimizations *within* the fixed-point structure (group ordering, skipping converged non-full groups, memoizing block-size wrappers, batching at LCM-aligned lengths). This proposal instead removes the fixed-point loop entirely by pre-aligning the cap to LCM and isolating EAGLE's one-block adjustment as a bounded post-step — an algorithmic simplification, not a tuning of the existing loop.

---

### 2. Split hybrid lookup into length discovery and final block materialization
- **Agent:** codex

**Detailed description.**

In `HybridKVCacheCoordinator.find_longest_cache_hit`, stop storing full `hit_blocks_by_group` during the fixed-point search. Add a length-only lookup path for the single-type managers, for example a `find_longest_cache_hit_length(...) -> int` classmethod or a `materialize_blocks=False` mode that runs the same cache-hit logic, including alignment and EAGLE pop semantics, but returns only the token length. The coordinator would run the existing fixed-point algorithm over integer lengths, then after convergence call each manager once at the final length to materialize the actual `KVCacheBlock` lists returned to `KVCacheManager.get_computed_blocks`. For EAGLE groups, the final materialization should use the same one-block verification cap as the search path: `min(final_hit_length + spec.block_size, max_cache_hit_length)`, with `use_eagle=True`, and assert or clamp if the materialized length is shorter than the fixed-point length. This targets a hot allocation cost that remains even when manager-call count is optimized: `SlidingWindowManager.find_longest_cache_hit` currently allocates `[null_block] * max_num_blocks` for every provisional candidate, and `MambaManager` builds dummy null prefixes before the coordinator knows the final joint hit length. On long multi-turn agentic prompts, this can move repeated large list construction and trimming out of the fixed-point loop while preserving the same correctness oracle from the existing prefix-caching, hybrid, different-block-size, and EAGLE tests.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. This is not the same as Agent A's proposal, which changes the fixed-point control flow into a single LCM-capped pass plus bounded EAGLE post-step to reduce the number of manager calls. This proposal keeps the fixed-point semantics compatible with the current algorithm and instead removes repeated block-list materialization and null-padding work from provisional lookups. It also is not covered by the candidate rationale items: it is not group ordering, skipping converged groups, memoizing converted hash-list wrappers, or batching LCM-aligned lookups.

---
