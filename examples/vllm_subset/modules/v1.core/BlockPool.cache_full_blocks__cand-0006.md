# BlockPool.cache_full_blocks

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/block_pool.py`](vllm/v1/core/block_pool.py) (lines 211–320)
- **Symbol:** `BlockPool.cache_full_blocks`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0006`

## Description
Caches newly full KV blocks by assigning block hashes, inserting them into cached_block_hash_to_block, and optionally emitting a BlockStored event with token IDs and per-block extra keys.

## Current approach
Slices new_full_blocks and block_hashes, then loops over every new block for null-block checks, hash assignment, and dict insertion. When KV events are enabled, it performs a second per-block pass to compute extra_keys via generate_block_hash_extra_keys before appending one BlockStored event.

## Estimated impact explanation
The cost scales with full blocks per prefill chunk. Reducing per-block Python overhead improves TTFT for long prompts and bursty agentic admissions.

## Evolve rationale
The code construct is the per-block insertion loop plus the KV-events extra_keys loop. It is on the prefill and chunk-completion path for prefix caching. Headroom includes combining event metadata generation with insertion, avoiding repeated BlockHashListWithBlockSize joins, carrying a monotonic mm_feature cursor through both paths, and specializing the no-events fast path. Correctness oracles include tests/v1/core/test_prefix_caching.py and tests/v1/core/test_kv_cache_utils.py, including BlockStored event tests and null-block prefix-cache cases.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Memoize per-block extra_keys at hash-time and reuse for BlockStored, eliminating the second pass entirely
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/block_pool.py:cache_full_blocks (lines 211-320), the second loop at lines 292-302 calls generate_block_hash_extra_keys once per new block solely to populate the BlockStored event's extra_keys_list. This is pure recomputation: vllm/v1/core/kv_cache_utils.py:get_request_block_hasher (lines 670-672) already calls generate_block_hash_extra_keys for every block when computing request.block_hashes, then discards the result after folding it into hash_block_tokens. The redundant work scales with mm_positions length and runs on every prefill/chunk completion call — a measurable Python overhead for multi-turn agentic workloads that mix images, tool tokens, LoRA adapters, or cache_salt.

Proposed change: (1) Modify request_block_hasher to additionally produce a parallel list of extra_keys when KV-cache events are enabled (gate via a flag passed into get_request_block_hasher, or always produce it — it's already computed). Store this parallel list on the Request object (e.g., request.block_hash_extra_keys) growing in lockstep with request.block_hashes. (2) In cache_full_blocks, replace the entire second pass with a slice: extra_keys_list = request.block_hash_extra_keys[num_cached_blocks:num_full_blocks], then filter null blocks while building it in the SAME iteration that does hash assignment and dict insertion (the loop at lines 258-274). For the block_size != hash_block_size case, apply the same windowing the BlockHashListWithBlockSize wrapper applies, but to the cached extra_keys (re-aggregating tuples across hash_block_size sub-blocks). (3) Keep the enable_kv_cache_events == False fast path as the single existing loop without any extra_keys work or maybe_convert_block_hash branch — the boolean test stays out of the hot loop because it gates the parallel list construction, not per-iteration logic. (4) Update tests/v1/core/test_kv_cache_utils.py and tests/v1/core/test_prefix_caching.py to assert that BlockStored.extra_keys equals the cached values for null-block, mm, cache_salt, prompt-embeds, LoRA, and mixed cases, including cross-group block_size scaling.

Net effect on the candidate: removes one O(num_full_blocks) loop and one O(num_full_blocks * mm_positions) scan from the prefill cache-completion path; lifts the work to a place where it was already being performed; reduces TTFT proportional to chunk size and mm-input density.

**Novelty rationale.**

The candidate has no listed deep_research_proposals, but the evolve_rationale frames the headroom strictly inside cache_full_blocks (combining the two passes, threading mm_idx through them, BlockHashListWithBlockSize avoidance, no-events fast path). All four ideas treat extra_keys as something cache_full_blocks must produce. This proposal identifies a different mechanism: the producer-side path in get_request_block_hasher already computes the exact extra_keys tuple per block as a hash input and discards it. The win is not 'merge the loops' but 'remove one loop entirely by caching the producer's value' — which requires touching kv_cache_utils.py and the Request schema, files outside the candidate's scope, and produces a single source of truth for extra_keys (cached hash and BlockStored payload provably consistent) rather than two parallel computations that must agree by construction.

---

### 2. Bound hash access to the newly cached block window
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/block_pool.py:cache_full_blocks`, replace the unbounded `new_block_hashes = block_hashes[num_cached_blocks:]` slice with windowed/indexed access over `range(num_cached_blocks, num_full_blocks)`. The current code only caches `blocks[num_cached_blocks:num_full_blocks]`, but it copies every remaining request hash, and for `BlockHashListWithBlockSize` it eagerly concatenates all remaining converted hashes even when the current prefill chunk commits only a small prefix. A concrete change is to loop by absolute block index, fetch `blk = blocks[i]` and `block_hash = block_hashes[i]` only after the null-block check, then assign/insert as today. This also allows dropping the `new_full_blocks` list slice. Add a regression test with a long request and a small `num_full_blocks` window, including the `block_size != hash_block_size` wrapper case, that instruments converted-hash access and asserts cache insertion/event hashes are produced only for the current window.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A focuses on producer-side memoization of `extra_keys` to remove the BlockStored metadata recomputation pass. This proposal targets a separate cost in the main insertion path: the unbounded block-hash slice and eager `BlockHashListWithBlockSize` conversion for future blocks. It helps even when KV-cache events are disabled, does not require storing extra keys on `Request`, and is independent of Agent A's extra-key caching design.

---
