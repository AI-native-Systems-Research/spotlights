# BlockPool.get_new_blocks

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/block_pool.py`](vllm/v1/core/block_pool.py) (lines 647–677)
- **Symbol:** `BlockPool.get_new_blocks`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_core-0004`

## Description
Allocates fresh KV cache blocks from the free queue, evicts stale prefix-cache metadata when needed, increments ref counts, and records allocation metrics.

## Current approach
Calls FreeKVCacheBlockQueue.popleft_n, then loops per returned block. The caching branch calls _maybe_evict_cached_block, asserts ref_cnt is zero, increments it, and emits metrics; the no-cache branch duplicates the ref-count and metrics loop without eviction.

## Estimated impact explanation
Runs on the scheduler path for prefill chunks and decode steps that cross block boundaries. Per-block savings accumulate under high-concurrency agent workloads and reduce TPOT scheduler overhead.

## Evolve rationale
Every slot allocation funnels through this method. Headroom in bulk eviction, batched metrics emission, specialized allocation branches, and tighter integration with popleft_n. Correctness oracle: block-pool tests can assert free count decreases by num_blocks, every returned block has ref_cnt == 1 and no stale hash after eviction, and evicted hashes exactly match the popped blocks' previous hashes.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Fast-path skip eviction for never-cached blocks and batch KV BlockRemoved events by group in get_new_blocks
- **Agent:** claude

**Detailed description.**

Rework `BlockPool.get_new_blocks` (vllm/v1/core/block_pool.py:647-677) so the post-popleft_n loop does three things differently, all still in a single pass:

1. Fast-path uncached blocks. Currently, in the `enable_caching` branch every popped block unconditionally goes through `_maybe_evict_cached_block`, which calls `metrics_collector.on_block_evicted` (dict pop) and then `_remove_cached_block_hashes` (two more dict pops). For freshly-allocated blocks that were never a full cache entry, `block.block_hash is None` and `block.block_id not in self.cached_block_hashes_by_block`. Gate the eviction call on `if block.block_hash is not None or block.block_id in self.cached_block_hashes_by_block:` before entering `_maybe_evict_cached_block`. This turns the common decode-step allocation of never-cached blocks into a pure ref_cnt bump plus (optional) metrics call, eliminating ~3 dict operations per block. Under the multi-turn agentic workload targeted by the caller, decode steps that spill into new blocks are frequent and mostly touch never-cached slots, so this shaves per-step scheduler overhead directly on the TPOT critical path.

2. Hoist the `metrics_collector` None-check. Both branches of the loop test `if self.metrics_collector:` per block. Bind `metrics_collector = self.metrics_collector` once outside the loop and either duplicate the loop body under a single top-level `if metrics_collector is not None` or capture `on_alloc = metrics_collector.on_block_allocated if metrics_collector else None`. Metrics collection is typically disabled, so this removes a per-block Python attribute load + truthiness check on the hot path.

3. Batch `BlockRemoved` KV events by group_idx. Today `_maybe_evict_cached_block` -> `_emit_block_removed_events` appends one `BlockRemoved(block_hashes=[h], group_idx=g)` per evicted hash. `BlockRemoved` already carries `block_hashes: list[ExternalBlockHash]` and a single `group_idx`. Accumulate evicted hashes locally (e.g. `defaultdict(list)` keyed by `get_group_id(h)`) across the whole `get_new_blocks` call and, at the end, emit one `BlockRemoved` per group. This changes O(num_evicted_hashes) queue appends and event objects into O(num_distinct_groups) — typically 1 for single-group deployments — which cuts allocation pressure and downstream consumer wake-ups when many stale prefix entries are evicted in one bulk request admission.

To enable (3) without regressing behavior, split `_maybe_evict_cached_block` into a `_evict_cached_block_collect(block, evicted_by_group)` helper used only from `get_new_blocks`, leaving the existing method intact for other callers (line 506, line 291). Keep the `assert block.ref_cnt == 0` invariant. Correctness oracle from the candidate holds unchanged: free count decreases by `num_blocks`, every returned block has `ref_cnt == 1` and no stale hash, and the union of evicted hashes across the batched events exactly matches the popped blocks' previous hashes (verifiable by collecting all `BlockRemoved.block_hashes` and comparing to the pre-eviction hash multiset).

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, so nothing overlaps by construction. Beyond that, the concrete mechanisms here are specific and not already implied by the candidate's own evolve_rationale: (a) using `block.block_hash is None and block_id not in cached_block_hashes_by_block` as a two-attribute gate to skip `_maybe_evict_cached_block` entirely for never-cached blocks (avoiding the metrics dict pop, not just batching it); (b) hoisting `self.metrics_collector` capture and dispatch out of the loop; and (c) coalescing `BlockRemoved` KV events by exploiting the fact that the event schema already accepts a `list[ExternalBlockHash]` per `group_idx` — the current code fans them out one-per-hash even though they naturally batch. The rationale mentions 'batched metrics emission' generically, but the concrete win on the outbound KV event queue (which feeds external consumers like KV-transfer / P2P sidecars) is a distinct integration surface not captured by 'metrics'.

---

### 2. Fuse free-queue pop with allocation initialization
- **Agent:** codex

**Detailed description.**

Add a BlockPool-specific bulk-pop path that initializes allocation metadata while `FreeKVCacheBlockQueue.popleft_n` is already walking the linked list, then have `BlockPool.get_new_blocks` use it. Concretely, introduce a helper such as `FreeKVCacheBlockQueue.popleft_n_for_allocation(n, cached_hashes_by_block=None)` that unlinks each block, appends it to the return list, asserts or validates `ref_cnt == 0`, sets `ref_cnt = 1`, and, when caching is enabled, records only blocks that actually have prefix-cache metadata (`block.block_hash is not None` or `block.block_id in cached_hashes_by_block`) into a small `maybe_cached_blocks` list. `get_new_blocks` would then evict only that smaller list and emit allocation metrics only if a collector exists. This removes the current second full pass over every allocated block in the common no-metrics/no-stale-cache case while keeping the same public return value and free-count semantics. Tests should cover `num_blocks == 0`, non-caching allocation, caching allocation with all uncached blocks, and mixed cached/uncached pops, asserting returned blocks have `ref_cnt == 1`, stale hashes are removed only for the recorded cached blocks, and `get_num_free_blocks()` drops by `num_blocks`.

**Novelty rationale.**

There are no deep_research_proposals listed. This is distinct from Claude's proposal because it is not just adding a per-block gate before `_maybe_evict_cached_block`, hoisting the metrics collector check, or batching `BlockRemoved` events. The new idea is to move the allocation-state work into the free-queue traversal itself so `get_new_blocks` no longer performs a second unconditional pass over all returned blocks; eviction becomes a follow-up over only the subset identified during the pop.

---
