# get_request_block_hasher.request_block_hasher

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/kv_cache_utils.py`](vllm/v1/core/kv_cache_utils.py) (lines 685–726)
- **Symbol:** `get_request_block_hasher.request_block_hasher`
- **Kind:** function
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_core-0007`

## Description
Request-local closure that computes chained block hashes for newly completed hash-size token blocks.

## Current approach
Runs a pure-Python while loop over new full blocks, computes multimodal/LoRA/embedding extra keys, slices token_ids per block, calls hash_block_tokens, appends the result, and threads the parent hash forward.

## Estimated impact explanation
Agent prompts can contain tens of thousands of tokens, producing hundreds or thousands of hashes before scheduling. Reducing this CPU loop directly lowers median TTFT for long-context turns.

## Evolve rationale
Long prompts and multi-turn history extensions spend synchronous admission time in this loop. Headroom in batching extra-key-free hash computation, memoizing extra-key scans across blocks of the same request, reducing token slicing, or native/vectorized hash construction while preserving chained semantics. Correctness oracle: for arbitrary Request inputs and hash_block_size, the evolved hasher must produce byte-identical BlockHash lists to the current implementation.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Precompute extra-keys fast path and lift per-request invariants out of the block loop
- **Agent:** claude

**Detailed description.**

In `get_request_block_hasher` at vllm/v1/core/kv_cache_utils.py:685-726, close over a set of per-request booleans computed once when the closure is first invoked (or on first block-loop entry) and use them to specialize the hot loop:

1. `has_mm = bool(getattr(request, 'mm_features', None))` (mirroring what `_gen_mm_extra_hash_keys` checks).
2. `has_lora = request.lora_request is not None` and, if so, snapshot `lora_name = request.lora_request.lora_name` once.
3. `has_embeds = request.prompt_embeds is not None`.
4. `cache_salt = request.cache_salt if start_token_idx == 0 else None` (only relevant on the very first block of the request).

If `not (has_mm or has_lora or has_embeds or cache_salt)` — the overwhelmingly common text-only, non-LoRA multi-turn agent case — take a fast path that skips `generate_block_hash_extra_keys` entirely and calls `hash_block_tokens(..., extra_keys=None)` directly. This eliminates four helper calls, four list allocations, one list concatenation and one `tuple(...)` per block, plus the `curr_mm_idx` bookkeeping. If any source of extra keys is present, keep the existing behavior but still avoid re-fetching `lora_request.lora_name` per block by referencing the snapshotted values.

Additionally: pull `hash_block_size`, `caching_hash_fn`, `request.all_token_ids`, and `request.num_tokens` into local variables above the loop so the tight Python loop resolves names as LOAD_FAST rather than repeated LOAD_ATTR/LOAD_GLOBAL. Where `all_token_ids` supports it, iterate a running pointer with `list(all_token_ids[start:end])` only inside the loop body (unchanged semantics) — do not attempt a global vectorization that would change hash-input types.

Correctness oracle: for arbitrary Request inputs and any hash_block_size, this must produce a byte-identical `list[BlockHash]` to the current implementation. `hash_block_tokens` currently passes `extra_keys=None` when `generate_block_hash_extra_keys` returns `(None, ...)`, and the fast path reproduces exactly that pre-tuple shape (`(parent, tuple(tokens), None)`), so hashes remain bit-identical. A unit test can fuzz random text-only, LoRA, MM, prompt-embeds, and cache-salt requests and assert equality against the original hasher.

Expected impact on the multi-turn agentic workload: for text-only turns (the dominant caller pattern of `get_request_block_hasher`), synchronous admission time on long histories drops proportionally to the eliminated per-block overhead, lowering median TTFT without touching hash semantics or requiring native code.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, and the `evolve_rationale` only gestures broadly at 'batching extra-key-free hash computation', 'memoizing extra-key scans', 'reducing token slicing', or 'native/vectorized hash construction'. This proposal is a concrete, minimally invasive Python-only specialization: it decides once per request whether any extra-key source exists and branches to a hash_block_tokens-only fast path, additionally lifting attribute lookups and closure state into locals. It does not require native code, does not memoize (which is unsafe given cache_salt/mm variability), does not vectorize, and does not change token slicing semantics — so it is distinct from every rationale bullet while still targeting the same hot loop.

---

### 2. Add an LRU cache for completed block hash tuples
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/kv_cache_utils.py` around `hash_block_tokens` and the `request_block_hasher` loop, introduce a bounded module-level cached helper that hashes the already-normalized key `(hash_function, parent_block_hash_or_NONE_HASH, curr_block_token_ids_tuple, extra_keys)`. Keep `hash_block_tokens` as the public wrapper: resolve `None` parents to `NONE_HASH`, convert `curr_block_token_ids` to a tuple exactly as today, then call the cached helper. This makes the existing docstring claim true and preserves byte-identical results because the serialized object passed to `hash_function` remains the same `(parent, tuple(tokens), extra_keys)` tuple. Use a conservative bounded `functools.lru_cache` size and include the hash function object in the cache key so `sha256`, `xxhash`, and CBOR variants cannot collide. This targets repeated long prefixes across requests and multi-turn agent workloads with shared system/tool prompts: once one request has paid to serialize and hash a block chain segment, later requests with the same parent hash, token block, and extra keys can reuse the digest during synchronous admission. Add focused tests that call `hash_block_tokens` repeatedly for identical and differing parent/token/extra-key inputs, assert equality with the uncached hash-function result, and assert different hash functions do not share cached values.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A's proposal specializes the per-request loop by skipping extra-key generation on the text-only fast path and lifting per-request invariants into locals; it does not cache the actual block hash computation across requests or make the existing `hash_block_tokens` LRU-cache comment true. This proposal is a separate cross-request reuse mechanism for identical chained block-hash inputs, not a per-block extra-key fast path, attribute lookup reduction, or local loop specialization.

---
