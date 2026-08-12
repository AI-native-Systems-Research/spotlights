# MooncakeStoreWorker.lookup

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py) (lines 1760–1863)
- **Symbol:** `MooncakeStoreWorker.lookup`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_distributed_kv_transfer-0017`

## Description
Mooncake store prefix lookup expands each candidate block hash across KV groups and rank namespaces, sends one batch_is_exist request, reduces results to an exists_set, and asks the coordinator for the longest usable hit.

## Current approach
Candidate keys are materialized as Python strings with PoolKey.build_key_string inside nested loops over groups, chunks, and key prefixes. Result reduction builds a Python set of present (group, hash) pairs, then may call coord.find_longest_cache_hit twice when the hit reaches the request end.

## Estimated impact explanation
Lookup latency contributes directly to TTFT for Mooncake store prefix hits; long multi-turn prompts multiply groups, chunks, and rank namespaces, making Python key expansion a visible CPU cost.

## Evolve rationale
This is the central MooncakeStoreConnector lookup before scheduler hit reporting. Reducing key-string materialization, caching group/hash namespace expansions per request, using compact binary keys, or combining existence reduction with longest-hit search are contained algorithmic changes preserving returned hit_length. Oracle: tests/v1/kv_connector/unit/test_mooncake_store_connector.py and test_mooncake_store_scheduler.py assert hit lengths, async lookup behavior, request metadata, and load/store correctness.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Skip the second find_longest_cache_hit when the initial hit already spans the request; precompute per-group key-prefix byte templates and drop the exists_set to per-group frozenset[bytes] for O(1) membership without tuple hashing
- **Agent:** claude

**Detailed description.**

Two contained changes to MooncakeStoreWorker.lookup (worker.py:1760-1863) that preserve returned hit_length semantics:

(1) Eliminate the redundant second `coord.find_longest_cache_hit` call on a full-length hit. Today, when `hit_length >= num_tokens`, the code recomputes the entire hit against the same `ExternalCachedBlockPool` with `max_length = align_lookup_length(num_tokens - 1)`. Because `find_longest_cache_hit` is monotone in `max_length` under a fixed `cached_block_pool` (it walks block hashes in order until either a miss or the max is hit), the answer for the shorter length equals `min(hit_length, usable_length)` truncated down to the nearest hash-block boundary — which is exactly what `align_lookup_length(num_tokens - 1)` computes. Replace the second call with: `hit_length = min(hit_length, self.coord.align_lookup_length(num_tokens - 1))` after asserting the pool has no partial-hash / SWA/Mamba-only quirks that would make the truncation invalid; where partial-hash-hits or eagle apply, keep the current re-walk path as a fallback guarded by a single flag on the coordinator. This saves one full tree walk over all groups and block hashes on every request whose full prompt is already in the store — the common case for multi-turn agentic workloads where prior turns' blocks are near-guaranteed hits.

(2) Replace `exists_set: set[tuple[int, bytes]]` with a per-group `list[frozenset[bytes]]` (indexed by `g_idx`) built in a single vectorized pass over `res`. Change `ExternalCachedBlockPool` (coordinator.py:28) to accept either the existing set or the new per-group frozensets and, in `get_cached_block`, do `all(h in exists_by_group[g] for g in group_ids)` instead of the current `(g, h) in self._exists`, avoiding one tuple allocation and hash per (group, hash) probe inside the hot `_find_hit_blocks` loop. Also precompute `self._lookup_prefix_counts: tuple[int, ...]` at `_init_lookup_key_prefixes` time so the reduction loop uses a local int per group instead of `len(self._lookup_key_prefixes[g_idx])` on every iteration, and use slice-sum (`sum(res[pos:pos+count]) == count`) instead of the Python-level `all(... for j in range(count))` generator to let CPython evaluate the reduction in C.

Both edits are purely internal to worker.py/coordinator.py, do not change the batch_is_exist wire format or PoolKey layout, and are covered by tests/v1/kv_connector/unit/test_mooncake_store_connector.py hit-length assertions.

**Novelty rationale.**

The candidate has no listed deep_research_proposals, but the evolve_rationale sketches three directions: reducing key-string materialization, caching group/hash namespace expansions per request, and combining existence reduction with longest-hit search. This proposal is orthogonal: it targets the redundant second `find_longest_cache_hit` call on a full-length hit — a control-flow optimization that avoids re-walking the entire hit tree rather than reducing key materialization or fusing reduction with search — and swaps the (int, bytes)-tuple set for per-group frozensets keyed on raw hash bytes, cutting tuple allocation and hashing inside the coordinator's `get_cached_block` hot loop. Neither the extra-call elimination nor the per-group frozenset restructuring of `ExternalCachedBlockPool` is mentioned in the evolve_rationale, and both target CPU cost specifically at the coordinator boundary, not at PoolKey.build_key_string.

---

### 2. Add a bounded positive-existence cache to skip repeated Mooncake lookups for already-confirmed prefix blocks
- **Agent:** codex

**Detailed description.**

In MooncakeStoreWorker.lookup (worker.py:1760-1863), add a small bounded LRU/ordered-set cache of confirmed-present logical entries keyed by (group_index, hash_bytes). During candidate construction, check this cache before expanding a (group, hash) across every rank namespace: cached positives can be inserted directly into the present set used by ExternalCachedBlockPool and omitted from candidate_keys/candidate_meta, so batch_is_exist only contains previously unseen blocks. After batch_is_exist returns, add a (group, hash) entry to the cache only when all of that group's lookup namespaces returned present. Do not cache misses, because a concurrent or later store may make them valid. Clear the cache in the RESET_MSG/remove_all path and on worker close/reset lifecycle, and optionally seed it from successful local save_put/save_exists results when the same worker just confirmed or wrote a complete group/hash entry. This targets the multi-turn agentic workload directly: successive turns resend a long, mostly identical prefix, so the second and later lookups avoid both Python key expansion and remote existence checks for the stable prefix while still querying new tail blocks and preserving hit_length semantics.

**Novelty rationale.**

There are no deep_research_proposals. The candidate rationale mentions per-request key expansion and per-request caching directions, but this proposal is cross-request positive memoization of remote existence results, with explicit miss non-caching and reset invalidation. Agent A's proposal focuses on eliminating a second coordinator walk, precomputing prefix counts/templates, and changing the in-memory exists representation for one lookup. It does not propose remembering confirmed Mooncake keys across lookups or skipping batch_is_exist entries for repeated multi-turn prefixes.

---
