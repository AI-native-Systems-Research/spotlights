# reorder_batch_to_split_decodes_and_prefills

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/backends/utils.py`](vllm/v1/attention/backends/utils.py) (lines 595–672)
- **Symbol:** `reorder_batch_to_split_decodes_and_prefills`
- **Kind:** function
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0005`

## Description
CPU-side batch reordering routine that groups requests into decode, short_extend, long_extend, and prefill regions before attention metadata is built.

## Current approach
Builds NumPy masks, computes target regions with repeat, finds mismatches, stable-sorts by current region, builds a Python src-to-dst dict, then performs pairwise input_batch.swap_states calls in a Python while loop with chained dict mutation.

## Estimated impact explanation
At high QPS or for small decode batches, per-step host overhead is visible in TPOT. Replacing chained pairwise swaps with a cheaper permutation reduces scheduler-side latency, though GPU kernels still dominate large batches.

## Evolve rationale
This runs on the host every scheduler step before GPU attention work. The grouping post-condition is local and testable, and the implementation has Python-loop and permutation headroom: a single permutation gather or vectorized batched swap could keep the same contract. Correctness oracle: tests/v1/attention/test_batch_reordering.py checks ordering and request-state preservation, and backend tests catch ordering mismatches.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace pairwise swap loop with cycle-rotation reorder to cut row-copy traffic ~3×
- **Agent:** claude

**Detailed description.**

In `reorder_batch_to_split_decodes_and_prefills` (vllm/v1/attention/backends/utils.py:595-672), the final loop walks chained `src_dest_map` entries and calls `input_batch.swap_states(src, dst)` once per hop. Each `swap_states` call (vllm/v1/worker/gpu_input_batch.py:567) performs a full save/copy of `token_ids_cpu[:, :max_active_token_count]` plus two row assignments — i.e. ~3 row-sized memcopies per swap. Following a permutation cycle of length k via chained pairwise swaps therefore costs 3·(k−1) row copies, on top of (k−1) Python-level dispatches into `swap_states` and its embedding/prompt_embeds bookkeeping.

Replace the loop with an explicit cycle decomposition that performs each cycle as a single rotation: (1) compute the permutation `perm[dst] = src` from the existing `src_dest_map`; (2) walk it once with a `visited` mask; (3) for every cycle of length ≥ 2, snapshot the first slot's per-request state once (req_id, output/spec token ids, scalar counters, `token_ids_cpu` row, `is_token_ids` row, and prompt-embeds entry), shift the remaining k−1 slots in place, then write the snapshot into the final slot. This reduces row-sized memcopies from 3·(k−1) to k+1 per cycle and halves Python dispatch overhead, while preserving the exact post-condition that the existing test (`tests/v1/attention/test_batch_reordering.py`) checks. To keep the change localized, expose the cycle-rotation primitive as a new `InputBatch.apply_permutation(perm: np.ndarray)` (alongside the unchanged `swap_states`) so TPU/GPU input-batch variants can implement the optimal in-place rotation for the arrays they actually carry, and have the reorder helper call it once instead of looping. As a side benefit, the helper sheds its mutating-dict-while-iterating pattern, which is currently a subtle correctness hazard if `swap_states` ever raises mid-loop.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete proposal is novel by construction. Beyond that, this idea is more specific than the rationale's hand-wave at a 'vectorized batched swap': it identifies the dominant cost (the per-swap `token_ids_cpu` row copy in `swap_states`), quantifies the 3·(k−1) → k+1 reduction via cycle rotation, and proposes a minimal API addition (`apply_permutation`) that keeps semantics, tests, and the TPU/GPU split intact rather than rewriting the helper to a full fancy-indexed gather (which would force every per-request structure into vectorizable form, including the Python dicts for prompt embeds).

---

### 2. Build the reorder plan with a linear stable-partition fast path
- **Agent:** codex

**Detailed description.**

In `reorder_batch_to_split_decodes_and_prefills`, replace the `target_regions = np.repeat(...)`, `needs_swap`, and `np.argsort(req_regions[needs_swap], kind="stable")` planning block with a four-bucket stable partition plan. After `req_regions` is computed, first check `np.all(req_regions[:-1] <= req_regions[1:])` and return `False` for the common already-grouped path without allocating `target_regions` or a mismatch mask. When reordering is needed, build `order` directly as `concat(flatnonzero(is_decode), flatnonzero(is_short_extend), flatnonzero(is_long_extend), flatnonzero(is_pure_prefill))`, where `order[dst] = src`, then construct the existing `src_dest_map` only for `src != dst` entries. This keeps the current stable ordering contract and can feed either the current `swap_states` loop or a later batched permutation primitive, while removing an O(n log n) sort and one full-length repeated target array from every reorder decision. Add a small randomized/property-style test in `tests/v1/attention/test_batch_reordering.py` that compares the new plan against the old stable region order for random mixes of the four request classes, plus an assertion that already-grouped batches report `modified=False`.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A's proposal targets the state-application phase, replacing chained pairwise swaps with cycle rotation and a new `InputBatch.apply_permutation` API. This proposal targets the earlier permutation-planning phase inside the same helper: no-op detection and linear stable bucket construction instead of `np.repeat` plus stable `argsort`. It is orthogonal and remains useful whether the final permutation is applied by the current swap loop or by Agent A's proposed rotation primitive.

---
