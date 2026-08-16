# BaseMultiModalReceiverCache.get_and_update_features

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/cache.py`](vllm/multimodal/cache.py) (lines 589–608)
- **Symbol:** `BaseMultiModalReceiverCache.get_and_update_features`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0023`

## Description
Updates engine-side multimodal features with cached encoder outputs while preserving receiver cache eviction order.

## Current approach
Runs two full loops over mm_features: the first recomputes each cache key and touches the receiver cache, and the second recomputes the same key and calls get_and_update_item to mutate feature.data.

## Estimated impact explanation
This path runs for multimodal encoder cache interaction in the engine. Reducing duplicate cache work shortens TTFT on cache hits and misses, especially when agentic batches carry many multimodal features.

## Evolve rationale
The concrete constructs are the two for feature in mm_features loops and repeated cache_key = feature.mm_hash or feature.identifier. Precomputing keys once, batching touch/update operations, or specializing duplicate keys can reduce Python and cache lookup overhead while preserving the anti-eviction ordering invariant. Correctness oracle: tests/multimodal/test_cache.py and encoder-cache tests; cache hit behavior, eviction protection, and updated feature.data values must remain identical.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Deduplicate mm_features by cache_key to collapse redundant touch/get_and_update_item calls
- **Agent:** claude

**Detailed description.**

Rewrite BaseMultiModalReceiverCache.get_and_update_features (vllm/multimodal/cache.py:589-608) so it performs the touch/update work once per unique cache_key rather than once per feature.

Concrete change:

1. Single pass to build a keys list and a groups dict:
   - keys: list[str] preserving first-occurrence order of each unique cache_key (feature.mm_hash or feature.identifier).
   - groups: dict[str, list[int]] mapping cache_key -> list of indices into mm_features that share it.
   - Also stash the resolved cache_key per feature in a local list to avoid the third `feature.mm_hash or feature.identifier` recomputation.

2. First loop iterates `keys` (unique) and calls self.touch_receiver_cache_item(cache_key, mm_features[groups[cache_key][0]].data) exactly once per unique key. The touch-before-update invariant is preserved because every unique key is touched before any fetch.

3. Second loop iterates `keys` (unique), calls resolved = self.get_and_update_item(mm_features[groups[cache_key][0]].data, cache_key) once, then assigns feature.data = resolved for every index in groups[cache_key]. For the shm receiver, only one MultiModalKwargsItem is returned from a single shm read; identical references are shared among duplicate features (this matches current behavior — the shm path already returns the same underlying object).

Why this helps median TTFT in multi-turn agentic workloads:
- Agentic sessions frequently resubmit the same images/audio across turns and often within one scheduled batch. Duplicates in a batch today trigger N touches and N shm reads on the P1 receiver; this collapses them to 1 each.
- Even for the LRU-backed MultiModalReceiverCache, each self._cache.touch/self._cache.get on OrderedDict-based LRUs is a dict lookup plus a move_to_end; deduplication cuts both Python overhead and cache-key hashing (which for mm_hash strings is not free at agentic scales).
- Precomputing the cache_key removes the second/third `feature.mm_hash or feature.identifier` short-circuit expression per feature per call — small but on the TTFT-critical path.

Ordering invariant is preserved: the docstring only requires that all identifiers be touched before any update, which is exactly what unique-key touch-then-update-in-key-order does. LRU recency for duplicate keys is identical to the status quo (a duplicate touch after the first is a no-op with respect to LRU ordering).

Correctness oracle: tests/multimodal/test_cache.py (cache hit behavior, eviction protection) plus an added test that passes an mm_features list with two features sharing the same mm_hash and asserts (a) touch_receiver_cache_item is called exactly once for that key, (b) get_and_update_item is called exactly once, and (c) both features' .data end up equal to the cached value.

**Novelty rationale.**

There are no existing deep_research_proposals listed for this candidate, so nothing to overlap with. Relative to the candidate's own evolve_rationale (which mentions precomputing keys / batching / 'specializing duplicate keys' as generic directions), this proposal picks the specific angle that pays off most in multi-turn agentic workloads — batch-level deduplication by cache_key — and specifies the exact data structures, ordering-invariant preservation argument, and the concrete win on the ShmObjectStoreReceiverCache shm-read path (not just the OrderedDict LRU path). It also identifies a third redundant recomputation of `feature.mm_hash or feature.identifier` in the touch call (via feature.data lookup only, not the key), which the candidate's description does not enumerate.

---
