# HF3FSKVConnector.get_num_new_matched_tokens

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py) (lines 695–756)
- **Symbol:** `HF3FSKVConnector.get_num_new_matched_tokens`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0014`

## Description
Scheduler-side HF3FS cache-hit lookup that decides how many prompt tokens can be loaded from disk-backed KV storage instead of recomputed.

## Current approach
The method aligns `len(prompt_token_ids) - 1` to the block size, generates block hashes for the entire candidate prefix, calls `self._metadata_client.batch_key_exists(block_hashes)`, then scans the returned booleans to the first miss and stores a `LoadBlockInfo`. It does this as one full-prefix metadata query with no chunking, incremental reuse across repeated turns, negative-cache shortcut, or defer semantics.

## Estimated impact explanation
This method runs before a request can be scheduled to load HF3FS KV pages. In multi-turn agentic prompts, full-prefix hash generation and metadata lookup can dominate cache-hit admission; reducing it directly lowers median TTFT for disk-backed prefix reuse.

## Evolve rationale
The optimization unit is the `_generate_block_hashes` plus `batch_key_exists` plus first-miss scan in `get_num_new_matched_tokens`. It can be evolved with rolling/incremental block-hash state, chunked metadata lookups that stop on the first miss, or local positive/negative existence caches while keeping the same prefix-hit contract. Correctness oracle: for mocked `batch_key_exists` responses, the returned `(new_hit_tokens, async)` and stored `LoadBlockInfo(num_computed_blocks, num_blocks_to_load)` must match the current maximal-prefix semantics for full hits, partial hits, misses, and `num_computed_tokens` boundaries; end-to-end HF3FS tests must preserve generated output equality.

## Deep research proposals

### 1. Add frequency-admitted local existence cache for HF3FS prefix block hashes
- **Finding:** `find-0019` — *[STORE] feat: Frequency admission + LRU lock optimization for local hot cache*
- **Source URL:** <https://github.com/kvcache-ai/Mooncake/pull/1596>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a small local positive/negative existence cache in front of `self._metadata_client.batch_key_exists` inside `HF3FSKVConnector.get_num_new_matched_tokens` (vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:695-756). Before issuing the full-prefix `batch_key_exists` RPC, look up each block hash from `_generate_block_hashes` in the local cache; for hashes with a known recent existence answer, skip the remote query and only batch-query the unknown tail. Gate insertion into the local cache with a Count-Min Sketch frequency admission filter (default K=2 as in the Mooncake PR), so block hashes only get promoted into the local hot cache after they have been observed in scheduler hit-lookups at least K times. Use a shared/read lock on the lookup path and defer LRU recency updates (e.g., per-shard timestamp ring) to keep the read path contention-free under concurrent scheduling. Preserve the existing maximal-prefix-hit contract: results from the local cache must agree with `batch_key_exists` for any tested key, and the first-miss scan plus `LoadBlockInfo(num_computed_blocks, num_blocks_to_load)` semantics remain unchanged. On any negative entry encountered during the scan, behave exactly as if `batch_key_exists` reported a miss at that position.

**Proposal rationale.**

The candidate's evolve_rationale explicitly lists 'local positive/negative existence caches' as a viable evolution and identifies the metadata RPC and full-prefix scan as the dominant cost for multi-turn agentic prefixes. The Mooncake PR's Count-Min Sketch admission directly addresses the failure mode that would otherwise wreck such a local cache in this workload: one-shot prefixes from non-repeating turns evicting the genuinely reused agent prefixes that drive TTFT. Frequency admission keeps the local cache populated with repeatedly-touched prefixes (exactly the multi-turn pattern in the caller context), while shared locks + deferred LRU touches keep the read path cheap when many concurrent requests hit `get_num_new_matched_tokens`. Together this can short-circuit `batch_key_exists` on warm prefixes and lower the median TTFT for disk-backed prefix reuse without changing the hit contract.

---

### 2. Chunked prefix existence lookup with early-exit on first miss in HF3FSKVConnector.get_num_new_matched_tokens
- **Finding:** `find-0022` — *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*
- **Source URL:** <https://www.microsoft.com/en-us/research/publication/sarathi-efficient-llm-inference-by-piggybacking-decodes-with-chunked-prefills/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor HF3FSKVConnector.get_num_new_matched_tokens at vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:695-756 so the prefix existence probe is performed in fixed-size chunks rather than as a single full-prefix batch_key_exists call. Concretely: (1) in _generate_block_hashes, lazily yield block hashes in chunks of size CHUNK (e.g. 16-64 blocks) instead of materializing the full list up-front; (2) iterate chunks calling self._metadata_client.batch_key_exists(chunk_hashes), scanning returned booleans for the first False and breaking out of the loop immediately when any miss is found; (3) accumulate hit counts across chunks and only build the final LoadBlockInfo(num_computed_blocks, num_blocks_to_load) once iteration terminates. Preserve the existing alignment of len(prompt_token_ids)-1 to block_size and the same maximal-prefix semantics so that for full hits, partial hits, and clean misses the returned (new_hit_tokens, async) and stored LoadBlockInfo are byte-identical to today. Optionally allow the chunk size to be tunable via existing HF3FS config and ensure that when a chunk fully hits, the next chunk's hash generation can overlap with the current chunk's metadata RPC.

**Proposal rationale.**

SARATHI's core insight - splitting a large piece of prefill work into equal-sized chunks so downstream work can begin without waiting for the entire prefill to complete - transfers naturally to the scheduler-side cache-hit probe described in the candidate. Today the method computes hashes for the whole candidate prefix and issues one batch_key_exists round-trip even when the very first block is a miss; in multi-turn agentic workloads (the stated caller workload) where prefixes are long but may diverge early, this wastes both hash computation and metadata-server work. Applying SARATHI-style chunking lets the lookup stop on the first miss block, bounds worst-case hash + RPC cost to one chunk past the actual hit boundary, and (per the candidate's evolve_rationale, which explicitly lists 'chunked metadata lookups that stop on the first miss' as a permissible evolution) keeps the prefix-hit contract intact - directly attacking the median-TTFT objective for disk-backed prefix reuse.

---

## Agent proposals

### 1. Memoize block-hash chain across multi-turn requests in HF3FSKVConnector
- **Agent:** claude

**Detailed description.**

Add a content-keyed memo of `_generate_block_hashes` results inside `HF3FSKVConnector.get_num_new_matched_tokens` (vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:695-756) so that the rolling block-hash chain itself is reused across turns of the same conversation, attacking hash-computation cost rather than just metadata-RPC cost. Concretely: (1) maintain a small bounded trie/dict keyed by parent-block-hash-plus-block-token-tuple so each (parent_hash, block_tokens) pair maps to the deterministic child block hash; (2) refactor `_generate_block_hashes` to walk the prompt block-by-block, looking up each (parent_hash, block_tokens) in the memo before recomputing the chained hash, and inserting freshly-computed pairs back; (3) since the canonical hash is deterministic across turns of an agentic conversation (turn N+1 = turn N + assistant response + new user turn), the entire turn-N prefix's block hashes are served from the memo and only the suffix blocks unique to turn N+1 incur fresh hash work. Preserve existing alignment of `len(prompt_token_ids)-1` to block_size and the maximal-prefix-hit contract; the memo is purely a function memoization of an already deterministic computation, so `batch_key_exists` and the first-miss scan still run on the full block-hash list and `LoadBlockInfo(num_computed_blocks, num_blocks_to_load)` is byte-identical. Bound the memo with a small LRU on (parent_hash, block_tokens) pairs and keep the data structure local to the connector instance.

**Novelty rationale.**

Neither existing proposal addresses block-hash *computation* cost. find-0019 caches `batch_key_exists` answers (a metadata-RPC cache) and find-0022 chunks the metadata RPC with early-exit; both still recompute the full chained block-hash sequence from scratch on every call via `_generate_block_hashes`. The candidate's evolve_rationale explicitly lists 'rolling/incremental block-hash state' as a distinct evolution from chunked lookups and local existence caches. In multi-turn agentic workloads the dominant repeated work for long prefixes is the chained hash itself (each block's hash depends on its parent's hash), and this proposal memoizes exactly that deterministic chain, complementing rather than duplicating either RPC-side optimization.

---

### 2. Defer HF3FS prefix lookup off the scheduler thread
- **Agent:** codex

**Detailed description.**

Change `HF3FSKVConnector.get_num_new_matched_tokens` in `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:695-756` to use the KVConnectorBase `None` defer contract instead of blocking the scheduler on `_generate_block_hashes` plus `batch_key_exists`. Add scheduler-side state fields for a pending lookup future and its request signature, submit the existing full-prefix hash generation and `self._metadata_client.batch_key_exists(block_hashes)` work to a small scheduler-role executor, and return `(None, False)` while the future is pending so the scheduler can run other ready requests. On a later call, if the future is done and the signature still matches the same prompt length / `num_computed_tokens` / aligned check length, consume the result and run the current first-miss scan and `LoadBlockInfo(num_computed_blocks, num_blocks_to_load)` construction unchanged. If the future raises, clear it and fall back to the current no-hit behavior. Clean pending futures on request finish and shut down the executor in `close()`.

**Novelty rationale.**

This is not an existence-answer cache like find-0019, not a chunked metadata probe with early exit like find-0022, and not block-hash memoization like agent A's proposal. It preserves the same full-prefix lookup result and maximal-prefix semantics, but changes when the expensive work runs: the scheduler thread stops waiting on HF3FS metadata latency, which directly targets TPOT stalls for already-active requests while the cache-hit decision for the new request is still pending.

---
