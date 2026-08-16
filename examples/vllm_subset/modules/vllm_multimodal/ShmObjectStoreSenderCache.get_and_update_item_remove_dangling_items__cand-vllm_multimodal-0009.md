# ShmObjectStoreSenderCache.get_and_update_item/remove_dangling_items

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/cache.py`](vllm/multimodal/cache.py) (lines 488–565)
- **Symbol:** `ShmObjectStoreSenderCache.get_and_update_item/remove_dangling_items`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0009`

## Description
Handles sender-side shared-memory processor-cache hits and misses, including opportunistic pruning of P0 prompt-update metadata.

## Current approach
On misses, it prunes when len(_p0_cache) is at least 2 * len(_shm_cache.key_index). remove_dangling_items then builds a set from all P0 keys, subtracts shared-memory keys, and deletes dangling entries one by one.

## Estimated impact explanation
This path runs on every IPC processor-cache miss and may trigger O(N) work. Smoothing prune cost reduces TTFT outliers and can improve median latency under long-running agentic cache churn.

## Evolve rationale
The concrete policy is the hard-coded 2x prune threshold and full-scan set-difference in remove_dangling_items. Incremental pruning, batched pruning, or an adaptive threshold can trade metadata footprint against latency spikes while preserving eventual removal of dangling keys. Correctness oracle: tests/multimodal/test_cache.py; cache hit/eviction invariants and the eventual subset relation between P0 metadata and shared-memory keys must hold.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Amortize dangling-key pruning via a bounded incremental scan cursor
- **Agent:** claude

**Detailed description.**

Replace the all-at-once O(N) full-scan pruning in `ShmObjectStoreSenderCache.remove_dangling_items` (vllm/multimodal/cache.py:560-565) and its coarse 2x trigger in `get_and_update_item` (line 509) with a small bounded incremental sweep that runs on every miss. Concretely: (1) add a persistent iterator/cursor `self._prune_cursor` over `self._p0_cache` keys (using an `iter()` handle that is refreshed to `iter(self._p0_cache)` when exhausted or when a `RuntimeError`/`StopIteration` is raised due to mutation); (2) on each cache miss (right after `self._shm_cache.put` succeeds), advance the cursor by a small bounded budget `K` (e.g. `K = max(8, len(self._p0_cache) // 64)`), and for each visited key check `key not in self._shm_cache.key_index` — if dangling, `del self._p0_cache[key]`; (3) keep the existing 2x threshold as a safety fallback but only as a hard bound (e.g. trigger a slightly larger sweep of `4 * K` when hit) rather than a full scan. Because `_shm_cache.key_index` is a dict, membership testing is O(1), so each miss does O(K) work instead of an occasional O(N) burst. This keeps the eventual-subset invariant (any dangling key is discovered within `ceil(len(_p0_cache) / K)` misses) that `tests/multimodal/test_cache.py` relies on, while removing the latency spike that today lands on the unlucky request that crosses the 2x threshold — the request most likely to be a user-facing TTFT sample under multi-turn agentic churn. Add a small unit test that fills `_p0_cache` with N synthetic keys, evicts a random subset from `_shm_cache.key_index`, and asserts they are all removed within N/K misses.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any concrete design is novel by construction. Beyond that, this proposal is specifically an incremental-cursor sweep with per-miss bounded work — it is neither the batched full-scan nor the adaptive-threshold variants named in the candidate's evolve_rationale as generic directions, and it preserves the exact eventual-removal invariant the existing correctness oracle checks.

---

### 2. Prune P0 metadata from explicit shared-memory eviction results
- **Agent:** codex

**Detailed description.**

Change the sender cache path so `_p0_cache` cleanup is driven by the exact keys evicted during `self._shm_cache.put(mm_hash, item)`, instead of periodically rediscovering dangling keys by scanning metadata. Concretely, extend `SingleWriterShmObjectStorage.free_unused()`/`put()` to expose the list of freed keys, for example by returning `(address, monotonic_id, evicted_keys)` or by storing a `last_evicted_keys` result on successful eviction. Then in `ShmObjectStoreSenderCache.get_and_update_item` (vllm/multimodal/cache.py:505-513), immediately `pop` each evicted key from `self._p0_cache` after `put` succeeds and before adding the new `mm_hash`. With that exact eviction signal, the 2x threshold and `remove_dangling_items()` full set-difference scan become unnecessary except possibly as a debug/test-only consistency repair. Add a focused test in `tests/multimodal/test_cache.py` that fills the SHM cache, inserts items that force eviction, and asserts `_p0_cache.keys()` matches `_shm_cache.key_index.keys()` after each successful put. This should reduce median TTFT/TPOT under multi-turn churn by removing both the occasional O(N) pruning burst and the steady growth of stale prompt-update metadata between bursts.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. This does not duplicate Agent A's bounded incremental scan: instead of amortizing discovery of dangling keys with a cursor and per-miss budget, it avoids discovery entirely by propagating the authoritative eviction list from the shared-memory store to the sender cache and deleting P0 metadata synchronously for those exact keys.

---
