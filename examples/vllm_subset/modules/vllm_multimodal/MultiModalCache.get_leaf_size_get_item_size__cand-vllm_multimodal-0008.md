# MultiModalCache.get_leaf_size/get_item_size

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/cache.py`](vllm/multimodal/cache.py) (lines 98–139)
- **Symbol:** `MultiModalCache.get_leaf_size/get_item_size`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0008`

## Description
Computes byte-size accounting for nested multimodal cache values used as LRUCache getsizeof callbacks.

## Current approach
get_item_size builds a parallel tree with json_map_leaves(cls.get_leaf_size, value), then walks that tree again with json_reduce_leaves(operator.add, ...). Tensor leaves use nbytes; other leaves fall back to sys.getsizeof.

## Estimated impact explanation
Size accounting runs on multimodal cache inserts and evictions. Reducing this cost shortens the cache-miss ingress path, which affects TTFT for new media in multi-turn workloads.

## Evolve rationale
The optimization unit is the two-pass json_map_leaves/json_reduce_leaves size calculation. A single accumulating traversal can avoid intermediate tree allocation and halve structure walking while preserving the eviction accounting contract. Correctness oracle: tests/multimodal/test_cache.py; cache eviction behavior and reported item sizes for representative values must remain equivalent.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace json_map_leaves with a type-dispatched iterative size accumulator specialized for MultiModalCacheValue
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/cache.py:98-139, rewrite MultiModalCache.get_item_size to bypass the generic json_map_leaves/json_reduce_leaves machinery entirely and instead run a single explicit stack-based traversal specialized to the concrete MultiModalCacheValue variants. Concretely: (1) use `type(x) is torch.Tensor` identity checks (not isinstance, which walks the MRO for every leaf and dominates cost when the tree has thousands of tensor leaves for image/video features) as the first branch, returning `x.nbytes` inline; (2) inline the container dispatch with an explicit `if type(x) is dict: stack.extend(x.values())`, `elif type(x) is list or type(x) is tuple: stack.extend(x)` ladder, avoiding the generic `is_list_of`/`Mapping` protocol checks that json_utils performs per node; (3) handle MultiModalProcessorCacheItem/Metadata/KwargsItems/KwargsItem/FieldElem as terminal special cases at the top of the loop, preserving the existing recursive-into-.data / .item / .item_size semantics; (4) accumulate into a single int rather than allocating the parallel `json_map_leaves` tree of ints and then reducing it. Keep the `debug` branch that calls `json_count_leaves` unchanged so logging output is byte-identical. The correctness oracle in tests/multimodal/test_cache.py (which asserts item_size values and LRUCache eviction behavior for representative nested structures) continues to pass because the traversal enumerates exactly the same leaves in a different order but sums to the same total. Expected TTFT benefit: the size-accounting call sits on the multimodal cache insert path (MultiModalProcessorCacheItem.__init__ at line 83), which fires on every new media item ingested for a fresh turn; in multi-turn agentic workloads with many small tool-call/screenshot images, this per-insert overhead is on the critical path to first token.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, so any concrete proposal is trivially non-duplicative. Beyond that, the candidate's own evolve_rationale contemplates only the generic 'fuse the two-pass json_map/json_reduce into one accumulating traversal' idea. This proposal goes further along a distinct axis: it replaces the json_utils generic tree machinery with a type-dispatched fast path (identity-based `type(x) is torch.Tensor` checks and inlined container dispatch), which attacks the per-leaf isinstance/Protocol-check overhead — a source of cost the candidate rationale does not identify and that the naive single-pass fusion would preserve.

---

### 2. Bypass generic sizing for cache wrappers with stored item_size
- **Agent:** codex

**Detailed description.**

In vllm/multimodal/cache.py:98-139, add an O(1) fast path at the start of MultiModalCache.get_item_size for values whose cache size is already known, especially MultiModalProcessorCacheItemMetadata.item_size. Optionally mirror the same pattern for MultiModalProcessorCacheItem by computing self.item_size once in its __init__ and having get_item_size return that field for wrapper instances. This keeps raw MultiModalKwargsItem / Mapping values on the existing tree-sizing path, but avoids invoking json_map_leaves/json_reduce_leaves when LRUCache calls getsizeof on wrapper objects that exist specifically to carry cache entries. Add/adjust tests in tests/multimodal/test_cache.py to assert wrapper, metadata, and raw data currsize remain identical for the representative nested tensor cases.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. This is distinct from Agent A's proposal, which replaces the generic tree walk with a specialized iterative traversal for all values. This proposal instead targets wrapper-level memoization and LRU getsizeof fast paths, so common cache entry objects can avoid any traversal call after their size has been computed, independent of whether the remaining raw-value traversal is generic or iterative.

---
