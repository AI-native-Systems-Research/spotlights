# ChunkedLocalAttentionManager.find_longest_cache_hit

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 1101–1198)
- **Symbol:** `ChunkedLocalAttentionManager.find_longest_cache_hit`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0023`

## Description
Chunked-local attention prefix-cache hit lookup that treats blocks before the local chunk as already computed and scans cache hits within the current chunk.

## Current approach
Computes the local attention chunk start, pre-fills computed block lists with null blocks up to the chunk start, then scans forward through in-window blocks with block_pool.get_cached_block until the first miss.

## Estimated impact explanation
Only chunked-local models use it, but it runs during admission and can touch many chunk-prefix blocks for long contexts. Reducing scan and null-padding overhead lowers TTFT for those workloads.

## Evolve rationale
Fixed-contract lookup path for chunked-local attention. Headroom in avoiding large null-list materialization for long contexts through a compact skipped-prefix representation and batching in-window hash probes. Correctness oracle: returned hit_length equals the skipped chunk prefix plus the contiguous cached in-window prefix, non-window positions are null blocks, and the method stops exactly at the first in-window miss.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Reuse a shared null-prefix slice and localize the in-window probe in ChunkedLocalAttentionManager.find_longest_cache_hit
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/single_type_kv_cache_manager.py:1101-1198, replace the per-admission O(local_attention_start_block_idx) null-block list materialization (line 1184-1187) with a shared, lazily-grown [null_block]*N list held on the manager instance (or a module-level cache keyed by block_pool.null_block and length). Return a slice of it per group instead of allocating a fresh list of that size on every admission. This is safe because block_pool.null_block has untracked ref_cnt (block_pool.py:188-191) and callers in kv_cache_coordinator.py treat the returned list as append-only when extending request block tables. Then localize the in-window scan (line 1188-1196): hoist block_pool.get_cached_block and kv_cache_group_ids into locals before the loop, and iterate the small in-window slice block_hashes[local_attention_start_block_idx:max_num_blocks] with a tight for-loop that short-circuits at first miss. For the common single-group case, collapse the zip/append into a single append on computed_blocks[0] to eliminate per-iteration tuple/zip overhead. Preserves the exact semantics called out in the candidate's correctness oracle: hit_length equals the skipped chunk prefix plus the contiguous cached in-window prefix, non-window positions stay null_block, and the scan stops exactly at the first in-window miss. In multi-turn agentic workloads the chunk prefix grows linearly with history length (e.g. ~7680 null refs per group per admission at 128k tokens with chunk=8192, block=16), so removing that allocation from the admission hot path directly reduces TTFT for chunked-local models (Llama-4 family) under long histories.

**Novelty rationale.**

No deep_research_proposals were listed for this candidate, so the proposal cannot overlap any. It is also distinct from the candidate's own evolve_rationale: while the candidate mentions 'compact skipped-prefix representation and batching in-window hash probes' as headroom in the abstract, it does not prescribe the concrete mechanism — a shared, lazily-grown null-block list sliced per admission (leveraging null_block's untracked ref_cnt and the append-only downstream usage), combined with hoisting of the probe callable and a single-group fast path. These two mechanisms target the two distinct costs (prefix materialization and per-probe rebinding) with a single self-contained edit to this method.

---

### 2. Short-circuit chunk-boundary hits before resolving block hashes
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/single_type_kv_cache_manager.py:1101-1198`, compute `max_num_blocks`, `local_attention_start_idx`, and `local_attention_start_block_idx` before calling `resolve_block_hashes`, then add an early return when `local_attention_start_block_idx >= max_num_blocks`. In that case every complete block up to `max_length` is outside the current local-attention chunk and must be treated as already computed with `block_pool.null_block`, so the method can return the null-block prefix and `max_num_blocks * kv_cache_spec.block_size` without resolving hashes or probing the prefix cache. This covers exact chunk-boundary admissions described in the method docstring, plus short prompts with no complete in-window block, and avoids unnecessary hash-list conversion/materialization on a hot admission path for long multi-turn histories.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes reducing null-prefix allocation and tightening the in-window probe loop, but still assumes the method proceeds through hash resolution. This proposal is a separate control-flow optimization: detect the no-probe case before `resolve_block_hashes` and skip hash resolution/cache lookup entirely when the full hit consists only of skipped local-attention blocks.

---
