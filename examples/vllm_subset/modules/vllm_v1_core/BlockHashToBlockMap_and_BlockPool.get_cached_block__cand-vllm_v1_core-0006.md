# BlockHashToBlockMap and BlockPool.get_cached_block

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/block_pool.py`](vllm/v1/core/block_pool.py) (lines 33–223)
- **Symbol:** `BlockHashToBlockMap and BlockPool.get_cached_block`
- **Kind:** region
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_core-0006`

## Description
Prefix-cache hash lookup stack: maps hash/group keys to cached KV blocks and resolves a requested hash across one or more KV cache group ids.

## Current approach
BlockPool.get_cached_block allocates a result list, builds a BlockHashWithGroupId per group, and calls BlockHashToBlockMap.get_one_block. The map wraps a Python dict whose values are either a single KVCacheBlock or a dict of duplicate blocks, with isinstance dispatch in get_one_block, contain, insert, and pop.

## Estimated impact explanation
Every cache-hit probe performs at least one lookup here, often hundreds per admission for long agent histories. Micro-optimizations reduce prefix-lookup wall time and median TTFT.

## Evolve rationale
Innermost data structure and facade for every prefix-cache probe and eviction. Headroom in single-group fast paths, precomputed group-key construction, split singleton/duplicate maps to remove isinstance dispatch, cached hot keys, or native implementations of hot operations. Correctness oracle: insert/get/pop/contain behavior for singleton and duplicate hashes, multi-group all-hit/all-miss behavior, and existing block-pool duplicate insertion and eviction tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Split-map + single-group fast path for prefix-cache probes
- **Agent:** claude

**Detailed description.**

Refactor `BlockHashToBlockMap` (block_pool.py:33-140) and `BlockPool.get_cached_block` (block_pool.py:198-223) to eliminate three per-probe overheads that dominate prefix-cache lookups in long multi-turn agentic histories:

1) Split-storage map. In `BlockHashToBlockMap.__init__`, replace the single union-valued `_cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]` with two dicts: `_singleton: dict[BlockHashWithGroupId, KVCacheBlock]` for the common single-block-per-hash case, and `_duplicates: dict[BlockHashWithGroupId, dict[int, KVCacheBlock]]` for the rare duplicate case. Rewrite `get_one_block` as `blk = self._singleton.get(key); return blk if blk is not None else (next(iter(d.values())) if (d := self._duplicates.get(key)) else None)` — eliminating the `isinstance(..., KVCacheBlock)` / `isinstance(..., dict)` dispatch and the `_unexpected_blocks_type` branch on every probe. Adjust `contain`, `insert`, `pop` symmetrically (insert promotes singleton→duplicate by popping from `_singleton` into `_duplicates`, pop demotes back). `__len__` returns `len(self._singleton) + len(self._duplicates)`. This matches the existing NOTE #2 rationale for the union type (GC pressure), while removing branch cost.

2) Precomputed group-id suffixes. Add `self._group_id_suffix: dict[int, bytes] = {}` on `BlockPool` (populated lazily on first use or eagerly from a known `kv_cache_group_ids` list). Each entry is `gid.to_bytes(4, "big", signed=False)`. Inside the `get_cached_block` loop, build the composite key as `block_hash + self._group_id_suffix[gid]` rather than calling `make_block_hash_with_group_id` (which repeats `to_bytes(4, "big")` on every call — `int.to_bytes` is a C call but still ~100 ns per invocation in CPython, invoked O(prefix-blocks × groups) times per admission).

3) Single-group fast path. Add `BlockPool.get_cached_block_single_group(block_hash: BlockHash, group_id: int) -> KVCacheBlock | None` that returns the block (or None) directly without allocating the outer `list` wrapper. Callers in `single_type_kv_cache_manager.py` (lines 734, 754, 946, 1190, 1321, 1335) that already know `len(kv_cache_group_ids) == 1` (the common attention-only case) invoke this fast path — they then wrap the single result in a tuple/list only when needed for uniform downstream handling, or better, extend the caller loop to consume the scalar directly. Also expose `BlockHashToBlockMap.get_one_block_by_key_bytes(key_bytes) -> KVCacheBlock | None` bypassing NewType wrapping.

Correctness oracle: all four map operations (get/contain/insert/pop) preserve prior semantics for singleton, promotion (singleton→duplicate on second insert of same hash), demotion (duplicate→singleton when only one id remains — either add this simplification or keep the duplicate dict with one entry to match existing pop-and-restore behavior; keep the existing behavior to minimize test churn), and the multi-group all-hit/all-miss contract of `get_cached_block`. Validate against existing `tests/v1/core/test_kv_cache_utils.py` and block-pool duplicate-insertion + eviction tests referenced in `evolve_rationale`.

Expected impact: prefix-cache probes dominate TTFT for agentic histories with tens of thousands of cached prefix tokens; removing per-probe list allocation, per-group `to_bytes` cost, and the isinstance dispatch is a direct wall-clock reduction on median TTFT with no algorithmic change and no accuracy risk.

**Novelty rationale.**

The candidate lists zero deep_research_proposals, so this proposal cannot overlap any prior deep_research suggestion. The idea combines three concrete, mutually-reinforcing micro-optimizations (split-storage to remove isinstance dispatch, precomputed 4-byte group-id suffixes to eliminate per-probe `int.to_bytes` calls, and a scalar single-group fast path avoiding the outer `list` allocation) targeting exactly the innermost hot loop called during long-history prefix probes — the workload named in the caller context.

---

### 2. Add a direct representative-block index for hash probes
- **Agent:** codex

**Detailed description.**

Augment `BlockHashToBlockMap` in `vllm/v1/core/block_pool.py` with a separate representative lookup table, for example `_primary: dict[BlockHashWithGroupId, KVCacheBlock]`, that always points to one currently cached block for each hash/group key. Then make `get_one_block()` a single `return self._primary.get(key)` lookup, while keeping the existing singleton-or-duplicate ownership structure for `contain()`, `insert()`, and `pop()` if desired. On first insert, set both `_cache[key]` and `_primary[key]`. On duplicate insert, leave `_primary[key]` unchanged unless the key was absent. On pop, if the removed block is the current primary, either delete `_primary[key]` when no duplicates remain or replace it with `next(iter(remaining.values()))` only on that eviction path. This moves the duplicate-dict iterator cost and type-dispatch cost out of the hot prefix-probe path and into comparatively colder insert/evict paths, while preserving the current “return any matching block” semantics. Add focused tests around representative replacement after popping the primary from a duplicate entry, popping a non-primary duplicate, and full removal so `get_cached_block()` continues to return `None` only when any requested group is missing.

**Novelty rationale.**

There are no listed deep_research_proposals. Agent A proposed split singleton/duplicate maps, precomputed group-id suffixes, and a single-group scalar fast path. This proposal is different: it adds a direct representative-block index so the hot `get_one_block` path is always one dictionary lookup, including duplicate-hash cases, and pushes representative maintenance to insert/pop paths rather than relying on split-map probing or `next(iter(...))` during lookups.

---
