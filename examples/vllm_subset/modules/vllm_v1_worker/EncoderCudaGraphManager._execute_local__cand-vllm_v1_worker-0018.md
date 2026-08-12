# EncoderCudaGraphManager._execute_local

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/encoder_cudagraph.py`](vllm/v1/worker/encoder_cudagraph.py) (lines 360–451)
- **Symbol:** `EncoderCudaGraphManager._execute_local`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0018`

## Description
Greedy multimodal item packer that batches encoder items under max_batch_size and per-path token budgets before graph replay or eager fallback.

## Current approach
Sorts items by output tokens and first-fits them until max batch size or max path budgets are reached, without budget-cliff awareness or lookahead.

## Estimated impact explanation
Better packing can turn near-budget eager batches into graph hits, reducing multimodal TTFT for mixed image/video agent turns.

## Evolve rationale
The sorted_indices first-fit loop and path_budgets selection are the concrete policy. Correctness is anchored by return ordering, graph hit/miss accounting, and DP gather/reordering tests around encoder cudagraph execution.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Make encoder cudagraph packing bucket-aware to avoid budget-cliff eager fallbacks
- **Agent:** claude

**Detailed description.**

Replace the ascending-by-output-tokens first-fit loop in `EncoderCudaGraphManager._execute_local` (vllm/v1/worker/encoder_cudagraph.py:360-451) with a bucket-aware packer that treats the discrete `path_token_budgets` buckets — not `max_path_budgets` — as the packing target. Concretely: (a) sort `sorted_indices` in descending order of output tokens so the largest item anchors each batch (largest-first bin-packing typically dominates first-fit-ascending when bin sizes are discrete); (b) when evaluating whether an item `fits` at line 397, also compute what the resulting `current_tokens[path]` would be per path and reject the add if it would push any path from a hit-eligible sum (≤ some bucket) to strictly greater than `max_path_budgets[path]` OR would force jumping to the next-larger bucket while a different still-open item could be placed in a smaller bucket-tight batch — i.e., prefer closing the batch on a bucket boundary. Add a small lookahead of one item: if the currently-considered item would cross a bucket boundary but the next item in `sorted_indices` also fits and lands the sum back on/under the same bucket, keep going; otherwise close the batch. Keep the existing `append_current_batch` / `_find_smallest_fitting_budget_given_tokens` machinery unchanged so return ordering, `graph_hits`/`graph_misses` accounting, and the DP gather path in `_dp_shard` remain identical — only the pre-batching selection order and close condition change. Gate behind a manager-level flag (default on) so the old policy is one line away for regressions. Validate with existing encoder cudagraph tests that assert output ordering, plus a new microbenchmark under `benchmarks/kernels/` that measures graph hit rate on a mixed image/video multi-turn workload approximating the caller's agentic scenario.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, and the candidate's own `evolve_rationale` explicitly calls out that the current policy lacks 'budget-cliff awareness or lookahead' — this proposal directly targets both by (a) switching sort direction to largest-first, (b) making the fit test bucket-boundary-aware rather than only max-budget-aware, and (c) adding a bounded single-item lookahead. None of these three levers are implied by the candidate description; the candidate only names the problem.

---

### 2. Add per-modality graph-path lanes before greedy packing
- **Agent:** codex

**Detailed description.**

In `EncoderCudaGraphManager._execute_local` (`vllm/v1/worker/encoder_cudagraph.py:360-451`), split the candidate item list into independent packing lanes keyed by the eventual encoder cudagraph path identity, such as encoder model/path plus item modality or feature-shape class already used to choose `path_token_budgets`, and run the existing greedy packer within each lane before concatenating the resulting batches in original request order for execution/result assembly. The concrete intent is to stop a high-token video item and several low-token image items from sharing one provisional `current_batch` whose aggregate per-path tokens force an eager fallback, when the same items could produce one video graph hit and one image graph hit if packed separately. Keep `_find_smallest_fitting_budget_given_tokens`, graph hit/miss accounting, and return reordering unchanged; only the candidate grouping before the current `sorted_indices` loop changes. Add a targeted unit test with mixed multimodal items from at least two graph paths showing that the existing cross-path greedy batch misses a graph budget while lane-local batches hit available budgets, and assert output ordering remains stable.

**Novelty rationale.**

There are no deep_research_proposals. Agent A proposes changing the item order, adding bucket-boundary awareness, and one-item lookahead within a single greedy packing loop. This proposal is different: it introduces a pre-packing partition by graph-path/modality lane so unrelated encoder paths do not consume one another's batch slots or force a shared eager fallback. It can be combined with A's bucket-aware policy but does not duplicate its sort-direction, bucket-cliff, or lookahead mechanics.

---
