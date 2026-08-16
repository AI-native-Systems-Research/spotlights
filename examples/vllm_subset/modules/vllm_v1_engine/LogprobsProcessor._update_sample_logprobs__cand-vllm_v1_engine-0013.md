# LogprobsProcessor._update_sample_logprobs

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/logprobs.py`](vllm/v1/engine/logprobs.py) (lines 69–119)
- **Symbol:** `LogprobsProcessor._update_sample_logprobs`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0013`

## Description
Sample-logprobs converter that appends generated-token logprob alternatives and cumulative logprob state for each produced token position.

## Current approach
For each generated logprob position, converts rank/logprobs/token_ids tensors or arrays to Python lists, optionally tokenizes ids, recomputes sampled context ids, verifies UTF-8 correction, updates cumulative_logprob, and appends one Logprob container.

## Estimated impact explanation
When logprobs are enabled for agentic telemetry or ranking, this work is paid on every generated token; reducing Python conversion and tokenizer overhead can lower median TPOT for those requests.

## Evolve rationale
This is a per-output-token path whenever sample logprobs are requested, and speculative decoding can pass multiple positions per step. Optimization units include batched token conversion across positions, avoiding repeated context extraction, reducing .tolist() churn, and specializing tokenizer-disabled or flat-logprobs cases. Correctness oracle: tests/v1/engine/test_logprobs_processor.py and tests/v1/engine/test_output_processor.py cover sample logprobs, cumulative logprob, decoded-token correction, and RequestOutput parity.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Batch `.tolist()` across all sample-logprob positions and cache sampled-context IDs incrementally
- **Agent:** claude

**Detailed description.**

In `vllm/v1/engine/logprobs.py::LogprobsProcessor._update_sample_logprobs` (lines 69-119), replace the per-position `.tolist()` triple (`rank_np.tolist()`, `logprobs_np.tolist()`, `token_ids_np.tolist()`) with a single batched conversion of the outer numpy arrays once at the top of the method, then slice/index into the resulting Python lists per position. Concretely: after unpacking `token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists`, call `.tolist()` once on each of `token_ids_lst`, `logprobs_lst`, `ranks_lst` (which are already `np.ndarray` produced by `LogprobsTensors.tolists()`), producing nested Python lists of shape `[num_positions][max_num_logprobs+1]` (and a flat rank list). This collapses 3*N array->list conversions plus per-row numpy dispatch into 3 conversions total per engine step, which materially reduces Python/CPython overhead in the spec-decoding hot path (multi-position outer loop) and in dense agentic telemetry runs. Second, add an incrementally maintained bounded deque (e.g. `_recent_sampled_ids: collections.deque[int]` with `maxlen=4`) on `LogprobsProcessor`, initialized in `from_new_request`. On each appended position, push `token_ids[0]` (the sampled token, which is always index 0 as noted at line 108) onto the deque. Change the call at line 100 to read `context_token_ids = list(self._recent_sampled_ids)` instead of `self._get_sampled_context_ids(self.logprobs)`; this avoids re-scanning `self.logprobs` (either doing per-position `FlatLogprobs.start_indices/end_indices` lookups or `next(iter(entry))` on `list[dict]`) once per generated token. Keep `_get_sampled_context_ids` as-is for the prompt-logprob path (`_update_prompt_logprobs`) so behavior there is unchanged, or optionally maintain a parallel bounded deque for the prompt path too if desired. Correctness is preserved because: (i) the batched `.tolist()` produces identical Python values, just amortized; (ii) the deque holds exactly the same 'first entry per recent position' the current helper returns, and it is updated after each `append_logprobs_for_next_position` call, matching the invariant on lines 208-247. The existing oracles `tests/v1/engine/test_logprobs_processor.py` and `tests/v1/engine/test_output_processor.py` (sample logprobs, cumulative logprob, UTF-8 correction, RequestOutput parity) directly verify the invariants this change must preserve. Estimated wins are concentrated in (a) multi-position steps from spec decoding, where the outer loop runs multiple times per engine step, and (b) requests where logprobs are enabled for agentic ranking/telemetry, contributing to median TPOT on multi-turn agentic workloads.

**Novelty rationale.**

The candidate has no existing deep_research_proposals listed, so no prior proposal overlap exists. The proposal targets two specific optimization units called out in `evolve_rationale` ('batched token conversion across positions' and 'avoiding repeated context extraction') with a concrete implementation strategy: a single top-of-method batched `.tolist()` per outer numpy array (replacing 3*N per-position calls) plus a bounded `deque` attribute for O(1) recent-sampled-id lookup that eliminates the per-position `_get_sampled_context_ids` walk. Neither the batched conversion pattern nor the incremental context ring-buffer is present in the current code path, and both are directly load-bearing for the multi-position spec-decoding loop that motivates this candidate.

---

### 2. Cache per-request decoded token strings for sample logprobs
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/logprobs.py::LogprobsProcessor._update_sample_logprobs`, add a small per-request token-string cache on `LogprobsProcessor` and route sample-logprob detokenization through it. Concretely, initialize something like `_decoded_token_cache: dict[int, str]` in `from_new_request` when a tokenizer is present. Replace the direct `convert_ids_list_to_tokens(self.tokenizer, token_ids)` call with a helper that finds token IDs missing from the cache, calls `convert_ids_list_to_tokens` once for those misses, stores the results keyed by token ID, and then returns `[cache[token_id] for token_id in token_ids]` for the current position. Keep `_verify_tokens` unchanged and run it on a fresh list so UTF-8 correction remains context-sensitive and does not mutate cached base decodings. This targets the common logprobs workload where high-probability alternatives recur across adjacent generated positions and across multi-turn agentic requests, avoiding repeated tokenizer `decode([tid])` / `convert_ids_to_tokens` work for the same vocabulary IDs while preserving the existing output shape, cumulative logprob behavior, and byte-fallback correction semantics. Tests should extend `tests/v1/engine/test_logprobs_processor.py` with a tokenizer stub that repeats alternative token IDs across multiple positions and verifies identical emitted `Logprob.decoded_token` values plus a reduced number of underlying decode calls.

**Novelty rationale.**

There are no deep_research_proposals listed for this candidate. Agent A's proposal covers batching numpy `.tolist()` conversions and replacing repeated sampled-context scans with an incremental deque. This proposal is different: it focuses on tokenizer work by memoizing the stable token-id-to-string conversion across positions, while explicitly leaving context-dependent UTF-8 correction outside the cache. It does not rely on batched outer-array conversion or cached sampled-context IDs.

---
