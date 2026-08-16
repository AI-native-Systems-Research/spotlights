# FreeKVCacheBlockQueue

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/kv_cache_utils.py`](vllm/v1/core/kv_cache_utils.py) (lines 184–427)
- **Symbol:** `FreeKVCacheBlockQueue`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0008`

## Description
Intrusive doubly linked list used as the free KV cache block queue and eviction-order structure.

## Current approach
Maintains prev_free_block and next_free_block pointers directly on KVCacheBlock objects with fake head/tail sentinels. popleft, popleft_n, remove, append, append_n, prepend_n, and iteration are implemented with Python attribute writes and manual num_free_blocks accounting.

## Estimated impact explanation
Touched on every block allocation, free, and cache-hit touch. The per-operation cost is bounded, but reducing it lowers scheduler overhead and TPOT at high concurrency.

## Evolve rationale
The class exists specifically to reduce Python overhead of the free queue while keeping O(1) middle removal. Headroom remains in a native intrusive list, a ring buffer with lazy deletion, or a queue representation that reduces per-op Python attribute writes. Correctness oracle: existing queue tests plus randomized operation sequences where get_all_free_blocks order, num_free_blocks, and bidirectional links match a reference model.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace pointer-per-KVCacheBlock intrusive list with sidecar integer arrays keyed by block_id
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/kv_cache_utils.py at FreeKVCacheBlockQueue (lines 184-427), replace the doubly-linked list currently stored on KVCacheBlock via prev_free_block / next_free_block object references with two flat integer sidecar arrays owned by FreeKVCacheBlockQueue: prev_idx and next_idx (each of length num_gpu_blocks + 2, using either array.array('l', ...) or a numpy int32/int64 buffer). Reserve two sentinel indices for the fake head and fake tail. All list operations become C-level integer stores into these arrays instead of Python attribute writes on KVCacheBlock instances, which (a) eliminates the slot-descriptor call path per write, (b) avoids INCREF/DECREF on KVCacheBlock references stored inside prev/next links, and (c) makes popleft_n / append_n / prepend_n implementable as a single tight loop over contiguous integer memory — optionally further vectorized via numpy fancy-indexing for the bulk-eviction and bulk-free paths used when a request completes and its blocks are returned in reversed order. Concretely: (1) drop prev_free_block and next_free_block from the KVCacheBlock @dataclass(slots=True) definition; (2) in FreeKVCacheBlockQueue.__init__, allocate prev_idx and next_idx from a list[KVCacheBlock] plus keep a self.blocks: list[KVCacheBlock] table for O(1) block_id -> object lookup on popleft; (3) rewrite popleft/popleft_n/append/append_n/prepend_n/remove/iter_blocks_after/get_all_free_blocks as index arithmetic on prev_idx/next_idx with sentinel checks against the head/tail indices; (4) preserve the existing invariants and error semantics (RuntimeError on invalid remove, out-of-sync num_free_blocks assertion in popleft). Correctness is validated against the existing queue tests plus a randomized-operation oracle that mirrors the current linked-list to a reference deque-with-index model, checking get_all_free_blocks order, num_free_blocks, and bidirectional link consistency after each op. Expected wins on the multi-turn agentic TTFT/TPOT path: cache-hit reuse (which calls remove() from the middle of the free list) drops from 4 Python attribute writes + 4 attribute reads + 2 object refcount touches to 4 integer array stores; per-step bulk operations at request boundaries (append_n reversed for eviction ordering) collapse to a single C loop; and the KVCacheBlock object itself shrinks by two slot pointers, reducing per-block memory and cache footprint at large num_gpu_blocks.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete proposal is not literally covered. Beyond that, the candidate's own evolve_rationale only sketches abstract directions ('native intrusive list, ring buffer with lazy deletion, or a queue representation that reduces per-op Python attribute writes'). This proposal is specifically the third direction made concrete via a representation that exploits two already-present invariants — dense integer block_id in [0, num_gpu_blocks) and KVCacheBlock's slots=True dataclass — to move link storage off the block objects entirely into typed integer buffers. That is materially different from a native C-extension intrusive list (which keeps the same per-object pointer layout in C) and from a ring buffer with lazy deletion (which changes eviction semantics and would need O(1) tombstone tracking); it also opens the door to numpy-vectorized bulk operations for the append_n / popleft_n hot paths, which none of the sketched directions individually implies.

---

### 2. Add batched removal for prefix-cache touch runs
- **Agent:** codex

**Detailed description.**

Extend `FreeKVCacheBlockQueue` in `vllm/v1/core/kv_cache_utils.py` with a `remove_n(blocks: Sequence[KVCacheBlock])` API and route `BlockPool.touch()` through it when multiple cache-hit blocks are touched together. The implementation should coalesce adjacent blocks that are already consecutive in the free-list order, including the reversed order produced when a request's blocks are returned for eviction, and splice each run with one predecessor-to-successor relink instead of calling `remove()` once per block. It would still clear `prev_free_block`/`next_free_block` on every removed block and decrement `num_free_blocks` by the number actually removed, preserving the current invalid-block error behavior for blocks whose links are missing. Add focused tests that build queues with several adjacent and non-adjacent cached blocks, call `remove_n()` in forward, reversed, and mixed orders, and assert the remaining eviction order, `num_free_blocks`, and invalid-link handling. This targets multi-turn agentic workloads where prefix-cache hits often touch several blocks from the same prior request, turning repeated middle removals from roughly four queue relinks per block into one splice per contiguous run plus per-block invalidation.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes changing the queue representation to sidecar integer arrays and rewriting all operations around block IDs. This proposal keeps the current intrusive `KVCacheBlock` links and instead adds a higher-level batched operation that exploits the caller's multi-block touch pattern; it reduces work by coalescing adjacent removals rather than by changing pointer storage or using typed arrays.

---
