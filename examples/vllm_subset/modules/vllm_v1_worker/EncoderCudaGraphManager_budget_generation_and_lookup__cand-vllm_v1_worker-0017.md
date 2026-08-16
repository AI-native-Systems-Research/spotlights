# EncoderCudaGraphManager budget generation and lookup

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/encoder_cudagraph.py`](vllm/v1/worker/encoder_cudagraph.py) (lines 190–296)
- **Symbol:** `EncoderCudaGraphManager budget generation and lookup`
- **Kind:** config_block
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0017`

## Description
Generates encoder CUDA graph token budgets and finds the smallest captured budget for a multimodal batch.

## Current approach
Uses a power-of-two budget ladder plus max_budget and performs a linear smallest-fitting lookup over captured budgets.

## Estimated impact explanation
Multimodal agent TTFT depends on encoder graph hits; denser or workload-adaptive budgets can reduce eager fallback and padding waste for screenshot/image turns.

## Evolve rationale
The budget ladder in _generate_budgets and lookup in _find_smallest_fitting_budget_given_tokens define graph hit/miss and padding policy. Encoder CUDA graph tests and graph_hits/graph_misses accounting validate capture and replay behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Add workload-adaptive encoder budget ladder driven by observed token histograms
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/encoder_cudagraph.py (lines 190-296), replace the static power-of-2 ladder from `_generate_budgets` and the linear `_find_smallest_fitting_budget_given_tokens` with a two-tier scheme tuned for multi-turn agentic workloads:

1. Capture-time ladder: keep `_generate_budgets` producing an initial power-of-2 seed ladder, but additionally accept a hint list of frequently-seen token counts from prior runs (e.g., typical screenshot / thumbnail / image-tile encoder token counts for the model's mm-processor). The union of {power-of-2 seed, hinted sizes, max_budget} is deduplicated, sorted, and capped by a `max_num_encoder_budgets` config knob so total captured graphs remain bounded. This yields dense buckets exactly where multimodal agents cluster (e.g., 1568, 2352, 4096) and avoids the sparse 2048 → 4096 jump that today forces ~2x padding for a 2100-token batch.

2. Runtime histogram + rebalance: inside `EncoderCudaGraphManager`, maintain a lightweight ring-buffer histogram of `total_tokens` values passed to `_find_smallest_fitting_budget_given_tokens` (and a counter of eager fallbacks when the lookup returns None). Expose `get_recommended_budgets()` that reports the top-K modes plus max, so operators can feed them back as hints on the next server restart. Also emit these into the existing graph_hits / graph_misses accounting so the effect is measurable.

3. Lookup micro-opt: replace the linear scan in `_find_smallest_fitting_budget_given_tokens` with `bisect_left` over the pre-sorted `self.token_budgets`, cached per path. This is O(log N) instead of O(N) and matters once budgets is a longer, denser list, and it also removes one Python-level branch per multimodal step which shows up on the TTFT critical path for image-heavy first turns.

Only `_generate_budgets`, `_find_smallest_fitting_budget_given_tokens`, `__init__`, and a small histogram helper need to change; the capture loop and BudgetGraphMetadata structure are untouched. Validation: extend the existing encoder CUDA graph tests to assert (a) hinted sizes are captured, (b) `bisect_left`-based lookup matches the current linear result on random inputs, (c) graph_hits increases and graph_misses decreases on a synthetic agentic trace with clustered image sizes; then run `vllm bench serve` on a multi-turn multimodal workload and report TTFT/TPOT deltas in the PR.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate (the list is empty), so any concrete change is novel by construction. Beyond that, this proposal is specific to the two functions named in the candidate (`_generate_budgets`, `_find_smallest_fitting_budget_given_tokens`) and combines three distinct, actionable ideas — hint-driven dense buckets, runtime histogram feedback exposed via existing graph_hits/graph_misses counters, and a bisect-based lookup — that together target the agentic-workload TTFT concern in the caller context rather than generic capture tuning.

---

### 2. Add a padding-efficiency cutoff before replaying encoder graphs
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/encoder_cudagraph.py` around `_find_smallest_fitting_budget_given_tokens`, add a small policy gate that refuses a captured budget when the fitted graph would be too under-utilized, causing the caller to use the existing eager fallback instead. Concretely, after finding the smallest `budget >= total_tokens`, compare `total_tokens / budget` against a configurable threshold such as `encoder_cudagraph_min_budget_utilization` or a model-provided default in `EncoderCudaGraphConfig`. Return `None` when utilization is below the threshold, and count/log these separately from true no-budget misses so operators can tell capacity misses from padding-waste fallbacks. This targets cases where the current ladder technically hits a graph but replays a much larger encoder shape than the batch needs, which can hurt TTFT/TPOT more than eager execution for small or irregular multimodal turns. Validation should extend encoder CUDA graph tests with inputs just below and above the cutoff to assert the selected eager-vs-graph path, plus a synthetic sparse workload showing reduced padded token work without changing outputs.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes denser, workload-adaptive budgets, histogram feedback, and a bisect lookup so more requests fit better buckets. This proposal is different: it keeps the budget ladder unchanged and adds a runtime utilization guard to deliberately avoid graph replay when the smallest available captured budget would waste too much padded encoder work.

---
