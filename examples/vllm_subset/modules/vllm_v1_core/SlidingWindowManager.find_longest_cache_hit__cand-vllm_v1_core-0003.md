# SlidingWindowManager.find_longest_cache_hit

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 897–993)
- **Symbol:** `SlidingWindowManager.find_longest_cache_hit`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0003`

## Description
Sliding-window prefix-cache lookup that searches right-to-left for enough consecutive cached blocks to satisfy the attention window.

## Current approach
Allocates null-filled per-group lists of size max_num_blocks, scans candidate blocks in reverse via block_pool.get_cached_block, resets num_contiguous_blocks on each miss, trims trailing blocks after a valid run, and applies EAGLE/alignment trimming. An in-source TODO notes that misses could skip by sliding_window_contiguous_blocks.

## Estimated impact explanation
Applies only to sliding-window models, but those pay this scan at admission. Cold or low-hit first turns are common, so reducing reverse-scan work moves TTFT for SWA and SWA-hybrid workloads.

## Evolve rationale
The TODO identifies an asymptotic win for low-hit-rate scans: O(max_num_blocks / K + K) instead of O(max_num_blocks). Additional headroom exists in avoiding full-length null preallocation and batching cache probes. Correctness oracle: existing SWA prefix-cache tests plus invariants that returned non-null blocks match their hashes, null padding is preserved, and hit_length is alignment-valid after EAGLE trimming.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Workload-aware start-index hint short-circuits reverse SWA prefix-cache scan
- **Agent:** claude

**Detailed description.**

In `SlidingWindowManager.find_longest_cache_hit` (vllm/v1/core/single_type_kv_cache_manager.py:897-993), add an optional per-parent-request hint (e.g. the block index of the last successful SWA cache-hit tail, or the block-hash of that tail) that is probed *before* the existing right-to-left scan from `max_num_blocks - 1`. For multi-turn agentic workloads, the new turn's prompt is the prior turn's context plus a small delta, so the K = `sliding_window_contiguous_blocks` blocks needed for a hit sit at approximately `prev_hit_end + delta_blocks`. The change: (1) plumb a `start_hint_idx: int | None` parameter (or read it from the caller-tracked parent-request state) through to `find_longest_cache_hit`; (2) if the hint is present and `block_pool.get_cached_block(block_hashes[hint_idx], kv_cache_group_ids)` returns a match, seed `num_contiguous_blocks` at that position and probe leftward until K contiguous blocks are gathered (or a miss forces fallback); (3) on any hint miss/inconsistency, fall through to the existing (or K-stride-optimized) reverse loop with zero correctness impact. The hint is a soft optimization — the existing hash-based `get_cached_block` remains the correctness oracle: any stale hint (evicted block, divergent sibling turn, or hash mismatch) is silently rejected and the current code path executes. Correctness is preserved by the existing SWA prefix-cache tests plus a new two-turn test asserting (a) the probe count collapses to O(K) on the common repeat-turn path, and (b) an evicted-hint scenario returns the same hit blocks as the unhinted path. The full-length `[null_block] * max_num_blocks` preallocation can additionally be deferred until a hit is confirmed at the hint position, avoiding N-sized null fills on the hot path.

**Novelty rationale.**

The candidate lists no deep_research_proposals. The in-source TODO the candidate references targets *algorithmic* worst-case improvement — reducing the reverse scan from O(max_num_blocks) to O(max_num_blocks/K + K) via K-stride skipping on miss — which is a general-purpose asymptotic win independent of workload. This proposal is orthogonal and workload-aware: it exploits the multi-turn agentic locality named in the caller's workload hints to collapse the common repeat-turn case to O(K) with a single seed probe, before any reverse scan begins. The K-stride skip alone still starts at index max_num_blocks-1 and never uses cross-request temporal locality; the hint alone still relies on the existing reverse fallback for misses. The two changes compose (hint hits → O(K); hint misses → K-stride reverse scan) and neither subsumes the other.

---

### 2. Scan only alignment-eligible SWA tail blocks before validating the run
- **Agent:** codex

**Detailed description.**

In `SlidingWindowManager.find_longest_cache_hit` (`vllm/v1/core/single_type_kv_cache_manager.py:897-993`), split the reverse lookup into a tail-candidate scan and a contiguous-run validation. When `alignment_tokens` is larger than `block_size`, the current loop still calls `block_pool.get_cached_block` for every block and only discards an unaligned first hit after the lookup. Instead, compute the valid tail indices up front from the existing condition: without EAGLE, `(tail_idx + 1) * block_size` must be divisible by `alignment_tokens`; with EAGLE, `tail_idx * block_size` must be divisible by `alignment_tokens`. Iterate those eligible tail indices right-to-left, probe the tail block once, and only if it is cached probe the preceding `sliding_window_contiguous_blocks - 1` blocks to validate the required run. Return the same null-padded prefix shape and keep the existing final EAGLE pop/re-alignment logic. Add a focused test that instruments `BlockPool.get_cached_block` for a hybrid/alignment case and asserts that unaligned tail indices are not probed while the returned hit length and block identities match the current exhaustive scan.

**Novelty rationale.**

There are no deep_research_proposals to duplicate. Agent A's proposal uses cross-request locality via a per-parent start hint and falls back to the existing scan; this proposal is intra-call and stateless, exploiting the already-required alignment predicate to avoid cache probes that can never produce a legal SWA hit. It is also distinct from the candidate's TODO about skipping by `sliding_window_contiguous_blocks` on misses: this skips alignment-ineligible tail positions regardless of hit rate and is most valuable for hybrid models where `alignment_tokens` is multiple blocks.

---
