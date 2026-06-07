# BlockHashListWithBlockSize.__getitem__ / _get_value_at

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/kv_cache_utils.py`](vllm/v1/core/kv_cache_utils.py) (lines 2106–2123)
- **Symbol:** `BlockHashListWithBlockSize.__getitem__ / _get_value_at`
- **Kind:** method
- **Estimated impact:** low
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0013`

## Description
Lazily converts hash_block_size-granularity block hashes to target_block_size-granularity hashes by joining scale_factor consecutive BlockHash bytes on every index, slice, or iteration access.

## Current approach
Every access computes base/end, slices self.block_hashes, calls b''.join, and wraps the result as BlockHash. There is no memoization or pre-materialized converted list.

## Estimated impact explanation
This is a targeted allocation micro-optimization on hybrid prefix-cache paths. It can reduce TTFT for long hybrid prompts but is smaller than eliminating whole lookup passes.

## Evolve rationale
The concrete construct is _get_value_at and the slice path in __getitem__. Hybrid prefix-cache lookup and cache_full_blocks can read the same converted indices repeatedly when KV groups use larger block sizes than hash_block_size. Headroom includes per-instance memoization, eager materialization when many accesses are expected, or passing converted arrays through the coordinator. Correctness oracles include tests/v1/core/test_kv_cache_utils.py and tests/v1/core/test_prefix_caching.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Intern converted block hashes in a bounded process-wide LRU keyed on the tuple of source hashes
- **Agent:** claude

**Detailed description.**

In `BlockHashListWithBlockSize._get_value_at` (vllm/v1/core/kv_cache_utils.py:2120-2123), replace the unconditional `BlockHash(b''.join(self.block_hashes[base:end]))` with a lookup into a module-level bounded LRU (e.g., functools.lru_cache or a small OrderedDict guarded by a lock) keyed on the tuple `self.block_hashes[base:end]`. Because `BlockHash` is `bytes`, the source slice is hashable and deterministic for a given token sequence, so repeated requests in a multi-turn agentic workload that re-hash the same growing conversation prefix will hit the cache and skip the byte-concat allocation entirely. Concretely: (1) add a small helper `_intern_joined_hash(parts: tuple[BlockHash, ...]) -> BlockHash` decorated with `functools.lru_cache(maxsize=N)` (sized to expected unique-block working set, e.g., a few × max_model_len/min_block_size); (2) call it from `_get_value_at` and from the slice/iter paths in `__getitem__`/`__iter__` (where you can pass already-built sub-tuples in one batch to amortize the dict probe). For the iteration path, build the slice once via `itertools.islice` over `self.block_hashes` and feed batched tuples to `_intern_joined_hash`, so the hot loop in `_KVCacheCoordinator.find_longest_cache_hit`'s fixed-point iteration (vllm/v1/core/kv_cache_coordinator.py:510-515) becomes a sequence of dict-probe-then-return rather than repeated `bytes.join` allocations. Validate via `tests/v1/core/test_kv_cache_utils.py` and `tests/v1/core/test_prefix_caching.py`, plus a microbench that exercises hybrid (full + sliding) attention with `target_block_size = 4 * hash_block_size` over repeated turns of the same prefix to confirm a measurable drop in `b''.join` time and TTFT. Add an explicit cache_clear hook called from `EngineCore` shutdown so the LRU does not retain hashes across engine restarts in tests.

**Novelty rationale.**

The candidate has no existing deep_research_proposals, and this idea is materially different from the three items mentioned in the candidate's evolve_rationale: (a) per-instance memoization is *intra-request* and rebuilds on every new `BlockHashListWithBlockSize` (one is constructed per request in `_get_block_hashes`); (b) eager materialization shifts cost rather than deduplicating it across requests; (c) passing converted arrays through the coordinator is a plumbing change that still recomputes per request. A process-wide LRU keyed on source-hash tuples uniquely exploits the determinism of `BlockHash` bytes across requests, which is the dominant access pattern for the stated multi-turn agentic workload where the same conversation prefix is re-hashed every turn.

---

### 2. Use a packed hash byte buffer for fixed-offset conversion
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/kv_cache_utils.py`, change `BlockHashListWithBlockSize` so `_get_value_at` does not slice `self.block_hashes` and run `b''.join(...)` for every converted index. Add a lazily-built per-wrapper packed byte buffer, for example `self._packed_hash_bytes = b''.join(self.block_hashes)`, plus the source list length used to build it. Since `BlockHash` values are fixed-width byte digests, `_get_value_at` can compute `byte_base = idx * self.scale_factor * hash_nbytes` and return `BlockHash(self._packed_hash_bytes[byte_base:byte_base + target_hash_nbytes])`. Rebuild the packed buffer if `len(self.block_hashes)` changes so the wrapper preserves the current live-list behavior. Also route contiguous slices and `__iter__` through the same packed-buffer stride so left-to-right and right-to-left prefix-cache scans avoid repeated Python list-slice allocation and N-way joins. Validate with `tests/v1/core/test_kv_cache_utils.py` equivalence tests for indexing, slicing, iteration, and source-list growth, plus the existing prefix-caching tests and a small hybrid block-size microbench comparing scale factors 2/4/8.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. This is not Agent A's process-wide LRU interning idea: it does not share state across requests, does not key on tuples, and does not retain converted hashes after the wrapper dies. It improves first-touch and cache-miss paths by changing the local representation from repeated list-slice plus join to one packed source buffer with fixed-offset byte slicing, so it targets overhead that an LRU would still pay on misses or cold prefixes.

---
