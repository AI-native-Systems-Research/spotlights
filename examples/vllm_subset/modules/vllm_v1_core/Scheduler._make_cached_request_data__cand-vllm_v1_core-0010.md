# Scheduler._make_cached_request_data

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 1405–1462)
- **Symbol:** `Scheduler._make_cached_request_data`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0010`

## Description
Builds CachedRequestData for scheduled running and resumed requests, including token slices, block IDs, computed counts, and output-token counts.

## Current approach
Iterates over itertools.chain(running_reqs, resumed_reqs), appends to several parallel lists, conditionally copies token IDs for PP/non-async, tracks resumed ids in a set, and calls get_block_ids for each request.

## Estimated impact explanation
Cost scales with the number of scheduled requests every step. Reducing packing overhead lowers the TPOT floor at high concurrency.

## Evolve rationale
Runs once per non-empty scheduler step after scheduling decisions. Headroom in pre-sizing outputs, separating PP and non-PP paths, reducing per-request get_block_ids allocations, and packing data with fewer parallel-list appends. Correctness oracle: byte-for-byte equivalent CachedRequestData for arbitrary running/resumed/token-map inputs, covered by scheduler tests plus a focused unit test.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Pre-size list buffers and split PP/non-PP loops in _make_cached_request_data
- **Agent:** claude

**Detailed description.**

Refactor Scheduler._make_cached_request_data at vllm/v1/core/sched/scheduler.py:1405-1462 to eliminate repeated per-request Python overhead by (1) pre-sizing all six output buffers to len(running_reqs)+len(resumed_reqs) via [None]*N assignment rather than repeated .append(), and (2) hoisting the two branch-invariant booleans `send_pp_tokens = self.use_pp and not self.scheduler_config.async_scheduling` and `track_all_tokens = not self.use_v2_model_runner` out of the loop, then dispatching to two specialized loops so the hot inner path contains zero per-iteration flag tests. Additionally, replace the `resumed_req_ids = set(); if idx >= num_running_reqs: resumed_req_ids.add(req_id)` pattern with a single post-loop construction `resumed_req_ids = {r.request_id for r in resumed_reqs}` (or reuse the already-materialized slice of req_ids), removing the per-iteration index compare and set.add call from every running request. Bind `req_to_new_blocks.__getitem__` and each list's `__setitem__` to locals to skip attribute lookups. Preserve byte-for-byte equivalence of the returned CachedRequestData (same ordering: running then resumed; same None handling for new_block_ids via get_block_ids(allow_none=True); same conditional population of new_token_ids and all_token_ids). Validate with existing scheduler tests plus a targeted unit test that constructs mixed running/resumed batches with and without use_pp/async_scheduling/use_v2_model_runner and asserts field-by-field equality against the pre-refactor implementation.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any concrete optimization is novel by construction. Specifically, pre-sizing lists + hoisting invariant branch predicates into two specialized loops + reconstructing resumed_req_ids from the input list rather than tracking during iteration are three distinct micro-optimizations not enumerated anywhere on the candidate.

---

### 2. Avoid serializing empty KV block updates for cache-hit steps
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/sched/scheduler.py:1405-1462`, special-case requests whose `req_to_new_blocks[req_id]` contains no newly allocated blocks and append `None` to `new_block_ids` without calling `KVCacheBlocks.get_block_ids(allow_none=True)`. This keeps the existing `CachedRequestData.new_block_ids` contract, because `get_block_ids(allow_none=True)` already represents absent per-layer block updates as `None`, but avoids constructing the per-layer tuple/list structure for the common multi-turn path where resumed or running requests may advance using already-computed/cache-hit tokens before any fresh KV allocation is needed. Implement this behind a small `KVCacheBlocks` predicate or length/emptiness check if one already exists, and add a focused scheduler unit test that compares the old and new output for mixed running/resumed requests where at least one request has no new blocks and another has real new blocks.

**Novelty rationale.**

There are no deep_research proposals listed. Agent A covers list pre-sizing, branch hoisting/specialized loops, resumed-id construction, and local binding of setters/getters, but it still calls `get_block_ids(allow_none=True)` for every request. This proposal targets the separate allocation/conversion cost of `get_block_ids` by skipping it only when the new-block payload is empty.

---
