# BlockPool.cache_full_blocks

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/block_pool.py`](vllm/v1/core/block_pool.py) (lines 225–342)
- **Symbol:** `BlockPool.cache_full_blocks`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0020`

## Description
Caches newly full KV blocks by registering block hashes, promoting partial blocks when needed, applying sparse block masks, and optionally emitting KV cache events.

## Current approach
Slices new_full_blocks, resolves block hashes, loops per block to skip null/masked entries, removes old partial hashes, inserts new hash metadata into BlockHashToBlockMap, collects event hashes, then does a second loop to build per-block event extra keys when kv_cache_events are enabled.

## Estimated impact explanation
Caching long prefills and generated blocks is on the scheduler path and determines future prefix hit availability. Reducing this loop lowers TTFT for long prompts and preserves TPOT under continuous batching.

## Evolve rationale
Per-block insertion loop runs whenever finalized tokens are cached and drives future prefix-cache availability. Headroom in fusing event and insertion passes, batching hash-key construction and inserts, avoiding block-hash view slicing, and specializing the common no-events/no-mask path. Correctness oracle: cached_block_hash_to_block and cached_block_hashes_by_block after caching must match a reference implementation, including partial-to-full promotion and masked/null skips; emitted events must preserve the same hash/token ranges.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Precompute skip mask and fuse hash-insert with event-extra-keys into a single loop
- **Agent:** claude

**Detailed description.**

In BlockPool.cache_full_blocks (vllm/v1/core/block_pool.py:225-342), collapse the two passes over new_full_blocks into a single pass and hoist invariant work out of the hot loop.

Specific changes:
1. Hoist `self.enable_kv_cache_events` into a local `emit_events` bool, and preallocate `new_hashes: list = [None] * n` and `extra_keys_list: list = [None] * n` sized to the number of non-skipped blocks only if `emit_events`. Avoid `if new_hashes is not None` checks inside the loop by binding a local `append_hash = new_hashes.append if emit_events else None`.
2. Build a boolean `keep` array once via a list comprehension: `keep = [not b.is_null and (block_mask is None or block_mask[i]) for i, b in enumerate(new_full_blocks)]`. This lets the inner loop become a single branch on `keep[i]` and removes the redundant re-evaluation of the same skip predicate in the second (event-extras) loop, which currently reads `blocks[i].is_null` and `block_mask[i - num_cached_blocks]` a second time.
3. Replace the `block_hashes[num_cached_blocks:]` slice with direct indexing (`block_hashes[num_cached_blocks + i]`) to avoid materializing the sliced list (`resolve_block_hashes` may return a list; slicing copies it).
4. When `emit_events` is True, compute `extra_keys` inside the same loop iteration that inserts the hash — passing `curr_mm_idx` through — so multi-modal `curr_mm_idx` progression happens exactly once per kept block, and the second `for i in range(num_cached_blocks, num_full_blocks)` loop plus its `blocks[i].is_null` / `block_mask[...]` re-checks are removed entirely.
5. Specialize the fast path when `emit_events is False and block_mask is None`: run a tight loop that only checks `blk.is_null`, skips the `maybe_convert_block_hash` call, and never touches `extra_keys_list`. This is the dominant scheduler path in multi-turn agentic serving where KV events are typically disabled.
6. Cache bound-method locals (`insert = self._insert_block_hash`, `remove = self._remove_cached_block_hashes`, `emit_removed = self._emit_block_removed_events`, `make_key = make_block_hash_with_group_id`) before the loop to skip attribute lookups per block.

Correctness oracle: `cached_block_hash_to_block` and `cached_block_hashes_by_block` after caching, and the `BlockStored` event's `block_hashes`/`extra_keys`/`parent_block_hash`/token range, must be byte-identical to the current implementation across (a) all-null-blocks, (b) fully masked-out ranges, (c) partial-to-full promotion (pre-existing `blk.block_hash`), (d) multi-modal requests with `curr_mm_idx` advancing across blocks including cache_salt-on-first-block, and (e) `num_cached_blocks == 0` vs. `> 0` parent-hash selection. A parametrized regression over these five shapes should compare `dict(cached_block_hash_to_block._map)` and the queued event tuples against the pre-refactor implementation.

Expected impact on the caller's objective: for long-prefill and long-generation multi-turn agentic workloads with KV events disabled, this halves the per-block Python overhead of caching newly full blocks (one loop instead of two, no re-evaluation of the skip predicate, no slice copy, no per-iteration attribute lookups). Because caching happens on the scheduler thread between engine steps, reducing this fixed per-block cost directly lowers scheduler wall time per step, improving TPOT under continuous batching and shaving TTFT on long prompts where hundreds of blocks are cached in one shot.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, so any concrete change qualifies as novel. The specific fusion of the insertion loop with the event-extra-keys loop, the precomputed `keep` mask that eliminates re-evaluating `is_null`/`block_mask` in a second pass, and the specialized no-events/no-mask fast path with cached bound-method locals collectively target the exact headroom the candidate's evolve_rationale calls out (fusing event and insertion passes, avoiding block-hash view slicing, specializing the common no-events/no-mask path) with a concrete, verifiable implementation sketch and correctness oracle.

---

### 2. Split the fresh-block insert path to skip redundant containment lookups
- **Agent:** codex

**Detailed description.**

In `BlockPool.cache_full_blocks` the dominant path is caching a newly allocated full block whose `blk.block_hash is None`. Today that still calls `_insert_block_hash`, which first probes `cached_block_hash_to_block.contain(key, block_id)` before setting the block hash, and then `BlockHashToBlockMap.insert` probes the map again to attach the block. Refactor `_insert_block_hash` so the `block.block_hash is None` case sets the primary hash and inserts immediately, returning before the `contain` check. Keep the existing `block.block_hash == key` and `contain` checks for blocks that already have a primary hash, where partial entries or duplicate registration are possible. This preserves collision handling because `BlockHashToBlockMap.insert` still merges same-hash different-block entries, while removing one per-block cache-map lookup from the hot fresh-cache path driven by `cache_full_blocks`. Add a focused regression around `cache_full_blocks` with two fresh blocks sharing a hash and a partial-to-full promoted block to verify `cached_block_hash_to_block` and `cached_block_hashes_by_block` remain unchanged.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal focuses on fusing the two loops inside `cache_full_blocks`, avoiding block-hash slicing, precomputing skip masks, event-extra-key generation, and bound-method locals. This proposal targets a different cost: the helper-level duplicate containment probe performed during insertion of fresh blocks. It does not require changing the loop structure or event pass and is therefore not covered by Agent A.

---
