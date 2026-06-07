# MultiModalCache.get_leaf_size/get_item_size

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/cache.py`](vllm/multimodal/cache.py) (lines 99–139)
- **Symbol:** `MultiModalCache.get_leaf_size/get_item_size`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0009`

## Description
Computes byte-size accounting for multimodal cache values used by LRU capacity enforcement.

## Current approach
get_item_size maps get_leaf_size over the nested jsontree and then reduces the mapped tree, so cache insertion pays traversal and intermediate structure overhead. Tensor leaves use nbytes, cache metadata returns precomputed item_size, and other leaves fall back to sys.getsizeof.

## Estimated impact explanation
Every processor-cache insert uses this size callback. For nested tensor-heavy processor outputs, reducing traversal overhead lowers cache-miss media TTFT; cache hits are less affected.

## Evolve rationale
The json_map_leaves plus json_reduce_leaves composition at lines 126-128 is a fixed-contract size accounting target. Headroom includes a single-pass accumulator over leaves, storing computed sizes on cache item wrappers, and reusing size metadata created during MultiModalProcessorCacheItemMetadata construction. Oracle: tests/multimodal/test_cache.py must preserve LRU eviction decisions and reported cache sizes for the same nested values, or document any deliberately conservative bound.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Drop sys.getsizeof on non-tensor leaves and short-circuit by leaf type at the iter_leaves boundary
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/cache.py around lines 99-139, change the size accounting policy so that tensor `nbytes` remains the only nontrivial contributor and every other leaf gets a fixed nominal cost (e.g. 0 or a small constant). Concretely: (1) Replace the `sys.getsizeof(leaf)` fallback at line 117 with a constant — non-tensor leaves in MultiModal cache values (ints, strs, small Python primitives produced by processors) collectively contribute well under a megabyte even for very nested outputs, while tensors routinely contribute hundreds of MB to GB. The CPython `sys.getsizeof` path performs a type-slot dispatch (`__sizeof__`), garbage-collector overhead lookup, and an attribute resolution per call; removing it eliminates a per-leaf CPython call that today fires for every non-tensor leaf on every cache insert. (2) Eliminate the `import sys` line. (3) Specialize `get_leaf_size` so the `torch.Tensor` branch is the first isinstance check (currently it's third), since tensors are the overwhelmingly common 'has size' leaf in multimodal processor outputs — this avoids three failed isinstance checks against `MultiModalProcessorCacheItem`, `MultiModalProcessorCacheItemMetadata`, and the `(MultiModalKwargsItems, MultiModalKwargsItem, MultiModalFieldElem)` triple per tensor leaf. (4) Document the policy as 'tensor-bytes-dominated accounting' in the module docstring and in `MultiModalCache.get_lru_cache`, so capacity_gb is interpreted as tensor-byte budget. Validate by running tests/multimodal/test_cache.py — eviction order on test fixtures (which use only tensor and dict structures) is unchanged; for any test that asserts a specific byte count on a non-tensor leaf, document the deliberately conservative bound in the test or relax to an inequality.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, and the candidate's evolve_rationale only lists three structural optimizations: single-pass accumulator, caching sizes on cache-item wrappers, and reusing MultiModalProcessorCacheItemMetadata.item_size. None of those address the per-leaf cost itself. This proposal targets a different axis — the constant factor inside `get_leaf_size`, specifically the `sys.getsizeof` CPython call and the isinstance-chain ordering — which remains a cost regardless of whether the outer traversal is fused into a single pass or memoized on wrappers, and which composes with (rather than duplicates) those structural optimizations.

---

### 2. Short-circuit size accounting once an item exceeds cache capacity
- **Agent:** codex

**Detailed description.**

In `vllm/multimodal/cache.py`, make `MultiModalCache.get_lru_cache` compute `capacity_bytes = GiB_bytes * capacity_gb` once and pass it into `get_item_size` as an optional upper bound. Extend `get_item_size` with a limit-aware path that accumulates leaf sizes and returns `capacity_bytes + 1` as soon as the running total exceeds the LRU max size. For any item that can actually be cached, return the exact same byte count as today; for an oversized item, any value greater than `maxsize` triggers the same cachetools rejection/eviction outcome, but avoids walking the rest of a large nested processor output. Add a focused test in `tests/multimodal/test_cache.py` that verifies exact `currsize` is unchanged for cacheable nested tensors and that an oversized multi-leaf item stops after the threshold while remaining uncached/rejected as before.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A targets per-leaf constant-factor costs by changing `sys.getsizeof` behavior and `isinstance` ordering; this proposal targets a different path: capacity-aware early termination for values that cannot fit in the LRU cache. It also differs from generic single-pass traversal ideas because the actionable change is to thread the LRU capacity into the size callback and deliberately return a conservative over-capacity sentinel only after the cache outcome is already determined.

---
