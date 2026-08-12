# LogprobsProcessor._update_prompt_logprobs

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/logprobs.py`](vllm/v1/engine/logprobs.py) (lines 121–187)
- **Symbol:** `LogprobsProcessor._update_prompt_logprobs`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0008`

## Description
Prompt-logprobs converter that turns prompt logprob tensors into per-position Logprob containers.

## Current approach
Flattens token ids, converts all ids to tokens, Pythonizes ranks/logprobs/token_ids with .tolist(), then loops over prompt positions to slice decoded tokens, recompute sampled context ids, verify UTF-8, and append per-position logprobs.

## Estimated impact explanation
Prompt-logprobs users pay this on every new turn; reducing per-position Python work can lower median TTFT for prefill-heavy multi-turn traffic.

## Evolve rationale
Runs during prefill when prompt_logprobs is enabled, with loop trip count proportional to prompt length. The monotonic context and flat decoded-token layout create opportunities to avoid repeated context extraction, reduce slicing/allocation, or batch UTF-8 verification. Correctness oracle: tests/v1/engine/test_logprobs_processor.py and tests/v1/engine/test_output_processor.py assert prompt_logprobs shape, sampled-token placement, decoded token correction, and RequestOutput parity.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Fast-path prompt logprobs when no UTF-8 replacement chars are present
- **Agent:** claude

**Detailed description.**

In vllm/v1/engine/logprobs.py::LogprobsProcessor._update_prompt_logprobs (lines 121-187), add a single O(num_prompt_tokens * num_logprobs) scan over all_decoded_tokens right after it is materialized (~line 150) to check whether ANY string ends with '�' (the replacement character). If none do — the overwhelmingly common case for typical prompts — bypass the per-position UTF-8 correction machinery entirely: skip the per-iteration call to self._get_sampled_context_ids(self.prompt_logprobs) (which currently repeats work proportional to len(self.prompt_logprobs) * num_prompt_tokens across the loop) and skip the per-position self._verify_tokens call, feeding the pre-sliced all_decoded_tokens[offset:offset_end] directly to append_logprobs_for_next_position. Only take the slow path — the existing loop body with _get_sampled_context_ids + _verify_tokens — when the initial scan detects at least one '�'. Concretely: (1) compute `needs_correction = self.tokenizer is not None and any(t.endswith('�') for t in all_decoded_tokens)` immediately after building all_decoded_tokens; (2) if False, run a tight loop that only slices decoded tokens and appends; (3) if True, keep current behavior but hoist context extraction outside the loop when possible (context grows monotonically as we append, so it can be maintained incrementally rather than recomputed via _get_sampled_context_ids each iteration). The fast path also lets us avoid the `token_ids_list[pos]` per-position indexing for the tokens argument to _verify_tokens (unused when we skip verification). Correctness is preserved because _verify_tokens is a no-op when no decoded string ends with '�' (see _verify_tokens body: the `if text.endswith('�')` guard means the correction map is empty and the list is returned unchanged). Validate against tests/v1/engine/test_logprobs_processor.py and tests/v1/engine/test_output_processor.py; add a targeted micro-benchmark under benchmarks/ that measures _update_prompt_logprobs on a 4K-token prompt with num_prompt_logprobs=5 to demonstrate the wall-clock win on prefill-heavy multi-turn traffic.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any concrete change is novel by construction. Beyond that, this proposal targets a specific observation not stated in the candidate's evolve_rationale: the _get_sampled_context_ids call inside the loop performs work proportional to accumulated prompt_logprobs length on every iteration, but its result is only consumed by _verify_tokens, which itself is a no-op unless a decoded token contains U+FFFD. Turning the near-always-no-op path into a branchless fast path — gated by one linear scan — is a targeted algorithmic change distinct from the generic 'batch UTF-8 verification / reduce slicing' framing in the candidate description.

---

### 2. Trim prompt logprob rows before Python conversion
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/logprobs.py::LogprobsProcessor._update_prompt_logprobs` (lines 121-187), compute the effective retained row width before detokenization and `.tolist()` conversion, then slice `token_ids` and `logprobs` to that width. For example, derive `effective_num_logprobs = num_logprobs if self.num_prompt_logprobs == -1 else min(num_logprobs, self.num_prompt_logprobs + 1)` immediately after reading `logprobs.shape`, and use `token_ids[:, :effective_num_logprobs]` / `logprobs[:, :effective_num_logprobs]` for `flatten().tolist()`, `logprobs.tolist()`, `token_ids.tolist()`, offsets, and per-position loops. This avoids detokenizing and Pythonizing padded/sentinel columns that `append_logprobs_for_next_position` will discard via its rank iterator anyway, mirroring the existing sampled-logprobs truncation invariant. Add a focused prompt-logprobs test with rows wider than `num_prompt_logprobs + 1` and sentinel trailing token IDs/logprobs to assert those trailing entries are neither surfaced nor detokenized.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A targets UTF-8 correction and repeated sampled-context extraction, with a fast path based on absence of replacement characters. This proposal is independent: it reduces the amount of tensor data converted to Python and detokenized before either the fast or slow UTF-8 path runs, specifically for padded prompt-logprob columns that are already semantically ignored.

---
