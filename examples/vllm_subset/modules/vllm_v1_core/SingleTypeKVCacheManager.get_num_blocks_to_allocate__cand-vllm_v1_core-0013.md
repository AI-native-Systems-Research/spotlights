# SingleTypeKVCacheManager.get_num_blocks_to_allocate

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 144–230)
- **Symbol:** `SingleTypeKVCacheManager.get_num_blocks_to_allocate`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0013`

## Description
Computes how many KV blocks a single cache manager needs for a request, accounting for existing blocks, prefix hits, skipped-window blocks, partial-hit CoW, and evictable cached hits.

## Current approach
Performs Python arithmetic, computes skipped blocks, slices new_computed_blocks, scans remaining blocks with _get_num_evictable_blocks, and adds a partial-hit CoW reservation when needed. Running requests use a fast path when num_cached_block is set.

## Estimated impact explanation
Long shared-prefix requests can pass hundreds of cached blocks through this sizing path. Reducing repeated scans trims admission overhead and improves TTFT for deep-history agent turns.

## Evolve rationale
Called from allocate_slots, sometimes twice per waiting request when full_sequence_must_fit is on. Headroom in reusing the evictable-count result across calls, avoiding scans for impossible evictions, and specializing hot no-partial-hit/no-skip cases. Correctness oracle: returned counts must exactly match blocks later consumed by allocate_new_blocks plus evictable touched blocks across existing cache-manager tests and randomized inputs.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Memoize evictable-block scan across the twin full_sequence_must_fit sizing calls
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/single_type_kv_cache_manager.py lines 144-230, `get_num_blocks_to_allocate` recomputes `_get_num_evictable_blocks(new_computed_blocks[num_skipped_new_computed_blocks:])` from scratch on every invocation. When `full_sequence_must_fit=True` (see kv_cache_manager.py:473-489 then again at :511), this scan runs twice per waiting request on the very same `new_computed_blocks` list — no ref-count mutations happen between the two calls (`remove_skipped_blocks` at line 505 only touches this manager's already-allocated `req_to_blocks`, not the new prefix-hit list). For long shared-prefix agentic turns (hundreds of cached blocks), that doubles a hot Python-level O(N) scan on every admission attempt.

Concrete change (all local to the candidate symbol and its caller path):

1. Introduce a tiny per-request memo on the manager: `self._evictable_scan_cache: dict[str, tuple[int, int, int]]` keyed by `request_id`, storing `(new_computed_blocks_id, num_skipped_new_computed_blocks, num_evictable_blocks)` where `new_computed_blocks_id = id(new_computed_blocks)`. Populate it on first computation inside `get_num_blocks_to_allocate` and consult it before scanning. Because the twin call in `KVCacheManager.allocate_slots` passes the exact same `new_computed_block_list` object (see kv_cache_manager.py:480 and :514), `id()` equality is a sufficient and correct cache key — no accidental reuse across different lists is possible. Invalidate/pop the entry at the end of `allocate_slots` (or lazily on the next successful allocation for this request), and in `free`/`remove_request`. Because `num_cached_block[request_id]` is set only for running requests (which take the fast path at line 194 and never touch the scan), there is no interaction with the running-request fast path.

2. Add a cheap early-exit inside `_get_num_evictable_blocks`: it is only summed with `num_new_blocks` to decide free-pool capacity in the caller. Threading a `stop_at: int | None = None` argument (the remaining free-pool headroom, computable in the caller as `block_pool.get_num_free_blocks() - num_new_blocks - watermark_blocks + 1`) lets the scan bail out once it has counted enough evictable blocks to guarantee admission failure — turning a full pass into an early return on the frequent "prefix cache hit dwarfs free pool" path. Keep the current unconstrained call site (`num_skipped_new_computed_blocks:` slice) intact for correctness paths that require the exact count, or expose both a `count_evictable(blocks)` and `count_evictable_at_least(blocks, threshold)` variant.

3. Specialize the common no-skip / no-partial-hit path. `num_skipped_blocks` and `_has_partial_local_hit` are almost always zero for full-attention groups (the dominant KV manager for agent workloads); detect `num_skipped_tokens == 0 and num_local_computed_tokens % self.block_size == 0` and take a short-circuit branch that returns `max(num_required_blocks - num_local_computed_blocks, 0) + num_evictable`, skipping the max/max arithmetic ladder and the CoW +1 reservation branch. This is pure code-motion and adds no new behavior.

Correctness oracle: the returned count must remain identical to the current implementation across the existing tests in tests/v1/core/test_kv_cache_manager.py, tests/v1/core/test_kv_cache_coordinator.py, and tests/v1/core/test_prefix_caching.py; additionally add a randomized property test that (a) calls `get_num_blocks_to_allocate` twice in the `full_sequence_must_fit` shape and asserts equality with a from-scratch recomputation, and (b) asserts the `stop_at` variant equals `min(threshold, true_count)`.

Expected win: for a 400-block shared-prefix agent turn on full attention, the double-scan cost drops from ~2×400 attribute accesses to ~400 (first call) plus O(1) (cached lookup). On admission-failing calls with the early exit, the first-call scan itself can terminate at the first block that pushes the count over the free-pool headroom. This is admission-path overhead, so it lands directly on TTFT for waiting requests in multi-turn agentic workloads.

**Novelty rationale.**

The candidate's `deep_research_proposals` list is empty, so any concrete change is novel by construction. The `evolve_rationale` already flags "reusing the evictable-count result across calls" and "avoiding scans for impossible evictions" as headroom, but does not commit to a design; this proposal supplies the specific mechanism (id-keyed per-request memo tied to the identity of `new_computed_block_list` threaded from `KVCacheManager.allocate_slots`, plus a `stop_at`-guarded early-exit variant and a specialized no-skip/no-partial-hit fast path) with concrete invalidation points and a correctness oracle, none of which are described in the candidate.

---
