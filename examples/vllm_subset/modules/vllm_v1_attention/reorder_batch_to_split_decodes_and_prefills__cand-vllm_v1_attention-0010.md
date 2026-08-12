# reorder_batch_to_split_decodes_and_prefills

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/utils.py`](vllm/v1/attention/backends/utils.py) (lines 736–813)
- **Symbol:** `reorder_batch_to_split_decodes_and_prefills`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0010`

## Description
Reorders InputBatch state into contiguous decode, short-extend, long-extend, and pure-prefill regions.

## Current approach
Builds numpy masks and a permutation, then follows cycles with input_batch.swap_states one pair at a time. decode_threshold is externally supplied and there is no local cost model for whether reordering is worthwhile on a given step.

## Estimated impact explanation
Agentic workloads frequently alternate decode and short tool-response extends. Faster reorder work reduces scheduler-side per-step overhead and can improve median TPOT.

## Evolve rationale
The swap_states loop and permutation construction are owned code with observable output: final InputBatch ordering and scheduler state. Existing worker tests plus permutation equality property tests provide the oracle for batched or vectorized implementations.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Fast-path for already-ordered agentic batches + two-pointer minimum-swap reorder
- **Agent:** claude

**Detailed description.**

Restructure `reorder_batch_to_split_decodes_and_prefills` in vllm/v1/attention/backends/utils.py:736-813 with two changes tuned for multi-turn agentic workloads, both hidden behind the existing signature and return contract.

1) Add a cheap monotonicity fast-path *before* building the target permutation. Encode the region label per request as a single uint8 via a vectorized expression over `has_context`, `is_below_threshold`, `done_prefilling` (no boolean masks, no scatter). Then check `np.all(req_regions[1:] >= req_regions[:-1])`. When the labels are already non-decreasing, return `False` immediately. In steady-state agentic decode with an occasional short-extend appended by the scheduler at the tail, this is the dominant case, and today the function still pays for `np.repeat`, an `argsort`, a Python dict, and the cycle-follow loop even when no swap is required. Also collapse the four `int(mask.sum())` reductions into one `np.bincount(req_regions, minlength=4)` call so region counts are produced in a single pass over the label array.

2) Replace the argsort + `src_dest_map` cycle walk (lines 797-811) with a two-pointer minimum-swap loop that only touches out-of-place slots. Compute `misplaced = req_regions != target_regions` once, then walk the misplaced indices with two cursors: for each `i` where `req_regions[i] > target_regions[i]` (label "too large for its slot"), advance a second cursor `j` over misplaced indices where `req_regions[j] < target_regions[j]` and `req_regions[j] == target_regions[i]`, and emit exactly one `input_batch.swap_states(i, j)`. Because there are exactly four labels and the target is contiguous by label, four size-4 head/tail cursors over the misplaced-index array produce the minimum swap count in a single linear pass, with no argsort, no `src_dest_map`, and no `while src != dst` re-lookup. Every `swap_states` call is expensive (block_table row swap, ~15 CPU-array pair swaps, plus a `MoveDirectionality.SWAP` event appended to `batch_update_builder.moved` — see vllm/v1/worker/gpu_input_batch.py:586-701), so cutting Python-side overhead per swap and guaranteeing minimum swap count is a direct win on scheduler-side per-step latency.

Both changes preserve the function's observable output (final `InputBatch` ordering and the sequence of `swap_states` calls, up to swap ordering within a cycle) and are covered by the existing permutation-equality tests referenced in the candidate's evolve_rationale. No new configuration and no cost model — the fast-path is essentially free (one vectorized comparison) and the two-pointer path is unconditionally cheaper than the current cycle walk.

**Novelty rationale.**

The candidate lists zero existing deep_research_proposals, so any concrete proposal is trivially novel by that bar. Substantively, this proposal targets two specific micro-inefficiencies in the current implementation that the candidate's own `current_approach` description does not call out: (a) the absence of an early-exit for already-monotonic label sequences, which is the dominant shape in multi-turn agentic decode and directly serves the caller's stated goal of reducing median TPOT, and (b) the algorithmic swap-count property of the current cycle walk, which this proposal replaces with a minimum-swap two-pointer scheme grounded in the fact that there are only four contiguous target regions. Neither observation is reducible to a generic 'vectorize the permutation' suggestion.

---

### 2. Add a conservative no-reorder cost gate for tiny mixed batches
- **Agent:** codex

**Detailed description.**

Add a local cost gate inside `reorder_batch_to_split_decodes_and_prefills` in `vllm/v1/attention/backends/utils.py:736-813` that skips physical reordering when the current step is too small for `swap_states` overhead to plausibly pay back. Compute cheap region counts first, then return `False` without building the full permutation or swapping when `num_reqs` is below a conservative threshold and the batch has only a tiny decode/extend mix, which is common in multi-turn agentic serving. Keep the existing behavior for larger or heavily interleaved batches. Validate the threshold with scheduler-side microbenchmarks comparing reorder overhead versus backend benefit across small batch sizes, and add tests asserting the skipped branch preserves batch state and scheduler accounting.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A covers an already-ordered fast path and a cheaper minimum-swap implementation once reordering is chosen, and explicitly avoids adding a cost model. This proposal targets the separate candidate gap: deciding locally when reordering is not worth doing at all for a given step.

---
