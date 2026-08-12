# BlockHashListWithBlockSize

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/kv_cache_utils.py`](vllm/v1/core/kv_cache_utils.py) (lines 2229–2299)
- **Symbol:** `BlockHashListWithBlockSize`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0016`

## Description
Lazy view that maps hash-block-granularity BlockHash lists to larger target block sizes for heterogeneous KV cache groups.

## Current approach
Stores the original list and scale_factor. __getitem__, slicing, and __iter__ compute (idx + 1) * scale_factor - 1 in Python for every access; _get_value_at is dispatched per element.

## Estimated impact explanation
Hybrid models with mismatched block sizes can perform hundreds of hash accesses per admission. Reducing view overhead lowers prefix-lookup latency and median TTFT.

## Evolve rationale
resolve_block_hashes returns this view inside prefix-cache finders when group block sizes differ; every element access is a Python attribute walk plus index compute. Headroom in materializing bounded slices once, specializing iteration for common scale factors, or using a lower-overhead stride view. Correctness oracle: length, indexing, slicing, and iteration must match a materialized reference list for arbitrary scale factors and hash lists.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Back BlockHashListWithBlockSize by a precomputed strided slice instead of computing (idx+1)*scale-1 per access
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/kv_cache_utils.py at BlockHashListWithBlockSize (lines 2229-2299), replace the per-access index arithmetic (`self.block_hashes[(idx + 1) * self.scale_factor - 1]`) with a one-shot strided slice materialized in `__init__`. Concretely: in `__init__`, after asserting divisibility, compute `self._view = block_hashes[self.scale_factor - 1 :: self.scale_factor]` — this is a single CPython list_slice call implemented in C and produces exactly the target-granularity sequence. Then simplify: `__len__` returns `len(self._view)`; `_get_value_at(i)` becomes `self._view[i]`; `__getitem__(slice)` becomes `self._view[idx]` (delegating slice semantics to the underlying list); `__iter__` becomes `iter(self._view)`. Optionally keep `self.block_hashes` around only if any caller still needs the fine-grained list (grep shows only length/index/iter usage from block_pool.py and single_type_kv_cache_manager.py, which drive tight per-block loops during prefix-cache admission — e.g. the `for i in range(max_num_blocks - 1, -1, -1): block_hashes[i]` scan in single_type_kv_cache_manager.py:945-947 and the loops around lines 1189/1320/1336). This removes one Python-level multiply, add, subtract, attribute-load of `self.block_hashes`/`self.scale_factor`, and a method-dispatch per access. For a 4096-token request with hash_block_size=16 and target_block_size=32, we get 128 target blocks per admission and per-lookup, and the search-from-right loops call `__getitem__` up to that many times per admission per group; on multi-turn agentic workloads this fires on every turn's prefill and dominates prefix-lookup Python overhead. Memory cost is one list of Python references per view instance — bounded by number of target blocks (typically <= a few hundred), so it is negligible next to KV-cache metadata already allocated. Correctness: a stride slice `L[s-1::s]` yields exactly `L[s-1], L[2s-1], L[3s-1], ...` which matches the current `_get_value_at` formula, so length/indexing/slicing/iteration match a materialized reference list.

**Novelty rationale.**

There are no deep_research_proposals listed on this candidate, so any concrete optimization is novel. Specifically, this proposal uses a single C-level list slice with a stride (not a Python comprehension or manual `range`-based materialization), which is distinct from the generic 'materialize once' framing in evolve_rationale — it avoids allocating an intermediate index range and any Python-level per-element callback, and it keeps the same public surface while collapsing `_get_value_at` to a single C-level index.

---
