# MambaManager.find_longest_cache_hit

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 1280–1356)
- **Symbol:** `MambaManager.find_longest_cache_hit`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0019`

## Description
Mamba prefix-cache hit lookup using fine-grained partial-state lookup or a right-to-left block scan for the deepest reusable recurrent state.

## Current approach
Resolves block hashes to Mamba block size, allocates per-group computed block lists, scans partial hash units or full blocks from right to left, pads skipped positions with null blocks, and returns the first deepest hit.

## Estimated impact explanation
Impact is model-family scoped, but Mamba/hybrid agent workloads depend on this path to reuse long recurrent-state prefixes. Faster deepest-hit lookup reduces TTFT on follow-up turns.

## Evolve rationale
Hybrid-Mamba workloads call this during admission and fixed-point reconciliation. Headroom in indexed lookup of deepest cached boundary, batched get_cached_block probes, avoiding repeated null-list extension, or a native scan over hash keys. Correctness oracle: returned hit_length is the deepest cached alignment-valid Mamba boundary at or below max_length, and returned block lists contain null padding before exactly the matched cached block for each group.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Retention-aware reverse candidate scan for MambaManager.find_longest_cache_hit
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/single_type_kv_cache_manager.py (MambaManager.find_longest_cache_hit, lines 1280-1356), replace the unconditional right-to-left scan over every block/partial-unit index with a retention-aware iteration that only probes indices which can legally be cached. Concretely: (1) When MambaSpec exposes retention_interval > 0 (dense mask absent), precompute the set of candidate block indices as `{k*per_segment - 1 for k in 1..max_num_blocks//per_segment}` plus any indices covering `reachable_boundaries` (mirroring the mask built in reachable_block_mask at lines 1358+); iterate this small sorted list in descending order and break on the first `get_cached_block` hit. This turns an O(max_num_blocks) hash-map probe loop into O(max_num_blocks / retention_interval * block_size) probes — for the typical agentic-turn case (retention_interval covering ~64 blocks) this is a ~64x reduction in dict lookups per admission. (2) Cache the resolved candidate-index list on MambaSpec (or memoize on kv_cache_spec keyed by (block_size, alignment_tokens, retention_interval, tuple(reachable_boundaries))) so subsequent turns of a multi-turn conversation don't recompute it. (3) Replace `computed.extend([block_pool.null_block] * block_idx)` on lines 1326 and 1351 with a single `computed += (block_pool.null_block,) * block_idx` from a per-BlockPool cached tuple of null_blocks sized to the largest seen prefix — avoids reallocating a Python list of thousands of null_block references on every deep hit (long agentic sessions have block_idx in the hundreds+). (4) For the fine-grained partial-unit branch (lines 1310-1330), similarly restrict the scanned `fine_idx` range to alignment-valid boundary positions when retention is sparse — the current loop from `max_num_partial_units - 1` down to 0 probes O(N/alignment_tokens) hashes even though only boundary-aligned indices could match. Correctness oracle is preserved: the deepest cached alignment-valid boundary at or below max_length is exactly the max element of the reverse-iterated candidate list that hits, and null-padding count `block_idx` remains identical to what the current code produces.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete proposal is novel. The specific insight — that MambaManager already has a well-defined `reachable_block_mask` (lines 1358-1449) declaring which indices are cacheable under sparse retention, but `find_longest_cache_hit` ignores this and blindly probes every index — is not stated in the evolve_rationale (which only lists 'indexed lookup of deepest cached boundary' and 'batched get_cached_block probes' as generic headroom). Tying the two together (using retention parameters to derive a probe candidate list, memoized per spec) is the concrete mechanism, and the null_block tuple-caching cleanup is an orthogonal Python-level win for the deep-hit branch that the evolve_rationale's 'avoiding repeated null-list extension' hint gestures at but does not specify.

---

### 2. Memoize repeated Mamba prefix-hit searches per cache epoch
- **Agent:** codex

**Detailed description.**

Add a small per-BlockPool or per-manager lookup memo for MambaManager.find_longest_cache_hit in vllm/v1/core/single_type_kv_cache_manager.py:1280-1356, keyed by the immutable search shape rather than the whole hash list: (cache_epoch, id or stable identity of block_hashes, max_length rounded to the searched unit, tuple(kv_cache_group_ids), kv_cache_spec.block_size, alignment_tokens, fine_grained). Store the winning unit index and cached block tuple, not the mutable computed_blocks lists; on a memo hit, rebuild the returned per-group lists with the same null padding and cached block. Increment cache_epoch on cache-affecting BlockPool operations used by this path, such as cache_full_blocks, move_block_hashes, free/evict, and partial-tail cache insertion, so misses and hits are not reused across cache mutations. This targets the admission/fixed-point reconciliation pattern where the same request prefix can be queried multiple times in one scheduler step or across immediately repeated scheduler passes before the cache changes, turning duplicate O(num_blocks) or O(num_partial_units) reverse scans into O(num_groups + padding) reconstruction.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes reducing the number of indices probed inside a single scan using retention-aware candidate lists and optimizing null padding allocation; this proposal is different because it eliminates repeated identical scans across calls by caching the final search result behind a cache-mutation epoch. It does not depend on sparse retention and remains useful for dense Mamba caching and fine-grained partial-state lookup when the scheduler asks the same longest-hit question more than once.

---
