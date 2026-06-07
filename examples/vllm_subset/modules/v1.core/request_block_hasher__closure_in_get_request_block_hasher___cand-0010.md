# request_block_hasher (closure in get_request_block_hasher)

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/kv_cache_utils.py`](vllm/v1/core/kv_cache_utils.py) (lines 643–684)
- **Symbol:** `request_block_hasher (closure in get_request_block_hasher)`
- **Kind:** function
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0010`

## Description
Closure that computes new full-block hashes for a request by iterating token blocks, generating multimodal/LoRA/prompt-embed extra keys, slicing all_token_ids, and calling hash_block_tokens for each block.

## Current approach
Pure Python while loop. Each block slices request.all_token_ids, scans or advances extra-key state, converts token IDs to a tuple in hash_block_tokens, calls the configured hash function, and appends one BlockHash.

## Estimated impact explanation
Long agentic prompts generate many block hashes before admission. Reducing CPU hashing overhead improves TTFT, with larger gains for long shared prompts and multimodal extra-key paths.

## Evolve rationale
The specific construct is the request_block_hasher while loop plus hash_block_tokens call. Block hashing is prefill critical path work for every newly completed block. Headroom includes native-compiling or vectorizing block slicing, avoiding tuple construction for stable contiguous token buffers, batching extra-key generation for multimodal features, and rolling/incremental hash state while preserving byte-for-byte hash stability. Correctness oracles include tests/v1/core/test_kv_cache_utils.py and tests/v1/core/test_prefix_caching.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Memoize block hashes across requests via a process-wide (parent_hash, token_tuple, extra_keys) → BlockHash LRU cache
- **Agent:** claude

**Detailed description.**

Wrap the `hash_block_tokens` call inside `request_block_hasher` (vllm/v1/core/kv_cache_utils.py:670-680) with a process-wide LRU cache keyed by `(prev_block_hash_value, block_tokens_tuple, extra_keys)` and valued by the resulting `BlockHash`. Concretely: in `get_request_block_hasher`, allocate a bounded `OrderedDict`-backed cache (size tunable via env var, e.g. 64K entries) closed over by `request_block_hasher`. For each candidate full block, materialize the tokens once into a tuple (also eliminating the redundant `tuple(curr_block_token_ids)` rebuild inside `hash_block_tokens` — pass a tuple in directly and short-circuit there), build the cache key, and on a hit reuse the stored `BlockHash`; on a miss, call `hash_block_tokens` and insert. Because the key (parent_hash, tokens, extras) deterministically identifies a block's hash inputs, the cache returns *bit-identical* `BlockHash` values — preserving the byte-for-byte hash stability that the prefix cache, KV connectors, and existing oracles in tests/v1/core/test_kv_cache_utils.py and tests/v1/core/test_prefix_caching.py require. The chained `prev_block_hash_value = block_hash` step still runs, so subsequent blocks in a different request see the same parent and continue hitting the cache for shared prefix tokens. Multi-turn agentic workloads — where each turn appends to an increasingly long shared prompt — benefit the most: turn N+1's first thousands of blocks are identical (parent, tokens, extras) to turn N's, so SHA256 work for the entire shared prefix is eliminated, dropping admission CPU from O(prompt_len) to O(new_tokens_since_last_turn) per request. Validation: extend tests/v1/core/test_kv_cache_utils.py with a test that hashes the same token sequence across two distinct `Request` objects and asserts (a) both produce identical `block_hashes` lists, and (b) the second pass triggers cache hits (via a counter on the underlying hash function); confirm tests/v1/core/test_prefix_caching.py still passes unchanged. Rollout knob: a config flag (`enable_block_hash_memo`, default on when prefix caching is enabled) so it can be disabled for debugging without altering hash semantics.

**Novelty rationale.**

The candidate has no listed deep_research_proposals, and its evolve_rationale enumerates only intra-request optimizations — native-compiling/vectorizing block slicing, avoiding tuple construction, batching extra-key generation, and rolling/incremental hash state within a single request. This proposal is on a different axis: cross-request memoization of block hash *outputs*. It exploits the observation that in multi-turn agentic chat, successive requests share long deterministic prefixes whose (parent_hash, tokens, extras) tuples are identical, so the entire SHA256 chain for the shared prefix collapses to dictionary lookups across requests — a saving the listed rationale headroom does not capture, since each `request_block_hasher` invocation today restarts from `len(request.block_hashes) == 0` for a fresh `Request` and recomputes the full prefix from scratch.

---

### 2. Cache prompt-embed bytes once for block extra-key hashing
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/kv_cache_utils.py`, optimize the prompt-embeds path used by `request_block_hasher`: change `_gen_prompt_embeds_extra_hash_keys` so the first prompt-embed block materializes a per-request contiguous CPU byte view of `request.prompt_embeds`, then hashes byte slices for each `(start_token_idx, end_token_idx)` instead of slicing the tensor and calling `tensor_data(...)` for every block. Keep the existing `_prompt_embeds_per_block_hashes` digest cache, but populate misses from `byte_view[start * row_nbytes : end * row_nbytes]`, where `row_nbytes = hidden_size * element_size`. The block hash input tuple stays exactly `(parent_hash, token_tuple, extra_keys)`, so prefix-cache keys remain byte-for-byte compatible with existing tests; only the way the prompt-embed digest is produced changes. Add coverage in `tests/v1/core/test_kv_cache_utils.py` for contiguous and non-contiguous prompt embeddings by comparing against the current `hashlib.sha256(tensor_data(prompt_embeds[start:end])).digest()` oracle, and include a small regression counter or benchmark showing only one full tensor normalization per request rather than one tensor slice conversion per full block.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A proposes cross-request memoization of complete block-hash outputs keyed by parent hash, token tuple, and extra keys. This proposal is different: it does not cache or reuse `BlockHash` results across requests, and it does not change token hashing. It targets the intra-request cost of constructing prompt-embed extra keys, specifically the repeated tensor slicing and `tensor_data` conversion inside the current hasher path for every full block.

---
