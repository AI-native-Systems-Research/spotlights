# SlidingWindowManager.find_longest_cache_hit

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 512–604)
- **Symbol:** `SlidingWindowManager.find_longest_cache_hit`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0014`

## Description
Sliding-window prefix-cache lookup that scans block hashes from right to left, tracks contiguous cached blocks inside the sliding window, materializes null-block-padded computed_blocks, enforces alignment, and handles EAGLE last-block pop.

## Current approach
Reverse linear scan over max_num_blocks with block_pool.get_cached_block on each candidate. A TODO at lines 543-547 notes that miss handling could reduce complexity from O(max_num_blocks) to roughly O(max_num_blocks / sliding_window_contiguous_blocks + sliding_window_contiguous_blocks).

## Estimated impact explanation
This primarily moves TTFT for SWA/hybrid agentic prompts. The gain can be large for long prompts with sparse hits, but the affected model class is narrower than the general scheduler path.

## Evolve rationale
The optimization unit is the reverse for-loop and contiguous-window matching policy. Sliding-window and hybrid models pay this during prefix-cache admission, and low/mixed cache-hit scenarios currently scan many impossible starting positions. Headroom includes jumping left by sliding_window_contiguous_blocks after a miss, precomputing aligned candidate positions, and avoiding full null-list materialization until a match boundary is known. Correctness oracles include tests/v1/core/test_single_type_kv_cache_manager.py and tests/v1/core/test_prefix_caching.py, which cover sliding-window cache hits, null blocks, alignment, and admission caps.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Rewrite SlidingWindowManager.find_longest_cache_hit as miss-jumping, alignment-indexed, lazy-materialized scan
- **Agent:** claude

**Detailed description.**

Restructure the right-to-left scan in single_type_kv_cache_manager.py:512-604 to combine three coordinated changes that resolve the TODO at lines 543-547 while also eliminating its known interactions with alignment and output materialization:

1) Miss-jumping with an alignment-feasible candidate set. Precompute, once, the set of starting indices i in [0, max_num_blocks) for which a contiguous run ending at i is alignment-feasible (i.e. (i+1)*block_size % alignment_tokens == 0, or i*block_size % alignment_tokens == 0 when use_eagle). Walk i from max_num_blocks-1 downward; on a miss, jump i -= sliding_window_contiguous_blocks rather than i -= 1, then snap i to the next lower alignment-feasible index. This realizes the O(max_num_blocks / sliding_window_contiguous_blocks + sliding_window_contiguous_blocks) target the TODO calls out, while preserving the current alignment skip semantics at lines 563-566 without re-running the modulus check inside the inner loop. On a hit at the pivot, fall back to a left-stepping verifier that walks contiguously to confirm the full window of length sliding_window_contiguous_blocks (or stops at the leftmost contiguous block, mirroring the existing 'partial' branch at lines 581-591).

2) Lazy output materialization. Do not pre-allocate `[block_pool.null_block] * max_num_blocks` per group at lines 549-552. Instead, first determine the matched range [start, end] (or the leading partial-hit length if no full window is found), then build each computed_blocks[g] as `[null_block] * start + matched_blocks_g` of exact final length. This avoids two O(max_num_blocks) costs that are paid per call regardless of hit rate: (a) constructing the null-padded list and (b) trimming with `del computed[i+num_contiguous_blocks:]` at line 576 or `del computed[num_contiguous_blocks:]` at line 585.

3) EAGLE last-block elision in the search bound. Because the use_eagle path at lines 592-594 always pops the rightmost matched block, exclude index max_num_blocks-1 from being treated as a window-completing position when use_eagle is true: start the scan at max_num_blocks-2 for the purpose of forming the final returned window, while still allowing it to count toward contiguity. This keeps the existing semantics (same final list length and content) but avoids one wasted get_cached_block call per call when the last block is uncached, and one wasted append+pop cycle when it is cached.

Keep the public signature, return shape, and observable semantics identical so tests/v1/core/test_single_type_kv_cache_manager.py and tests/v1/core/test_prefix_caching.py continue to pin behavior. Add micro-benchmarks in those test files (or in tests/v1/core/) covering: full miss over a 4k-block prompt, low-hit-rate (single isolated cached block), partial-window leading hit, full-window middle hit, and EAGLE on/off, asserting result equality against a reference loop implementation kept as an oracle.

**Novelty rationale.**

The candidate has no existing deep_research_proposals listed, so any concrete proposal is novel by definition. Beyond that, the evolve_rationale only sketches three independent headroom hints (jump-on-miss, precomputed aligned positions, avoid null materialization) without resolving how they interact. This proposal turns them into a single coordinated rewrite: the alignment-feasible candidate set is what makes miss-jumping safe (a naive jump-by-window step would skip past alignment-feasible windows when block_size != alignment_tokens), and lazy materialization changes the loop's output contract so the trimming branches at lines 576 and 585 disappear rather than just shrink. It also adds a third element the rationale does not mention — eliding the always-popped EAGLE last block from the scan bound — and pins the rewrite to the existing test oracles to preserve correctness.

---

### 2. Add a short-prompt prefix fast path before the sliding-window scan
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/single_type_kv_cache_manager.py:512-604`, add an early branch after computing `max_num_blocks` and `sliding_window_contiguous_blocks`: when `max_num_blocks <= sliding_window_contiguous_blocks`, a non-prefix sliding-window hit is impossible because there are not enough complete blocks to form a cached window after any skipped/null prefix. In that case, use a FullAttention-style left-to-right prefix lookup: append cached blocks until the first miss, then apply the existing EAGLE pop and alignment trimming rules before returning. This avoids the current right-to-left scan over the whole prompt and the null-padded list allocation for requests that are at or below the sliding-window dependency depth, where the only valid result is an ordinary cached prefix. Add focused tests covering all-hit, first-block miss with later hits, partial prefix hit, `alignment_tokens != block_size`, and EAGLE enabled, comparing results to the existing behavior and optionally asserting reduced `get_cached_block` calls for early-miss cases.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A proposes a general rewrite of the viable sliding-window scan using miss-jumping, alignment-indexed candidates, lazy materialization, and EAGLE search-bound elision. This proposal is a separate semantic fast path for the case where no sliding-window skip/null match can exist at all; it changes which algorithm is used before entering the scan, rather than optimizing the scan’s candidate progression or materialization strategy.

---
