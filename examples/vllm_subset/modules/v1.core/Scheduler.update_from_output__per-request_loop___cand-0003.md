# Scheduler.update_from_output (per-request loop)

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 1333–1474)
- **Symbol:** `Scheduler.update_from_output (per-request loop)`
- **Kind:** loop
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0003`

## Description
Hot loop over scheduler_output.num_scheduled_tokens that processes ModelRunnerOutput: spec-decode acceptance accounting, encoder-input freeing, stop checks, structured-output grammar advancement, logprob slicing, NaN bookkeeping, and EngineCoreOutput construction.

## Current approach
Branch-heavy Python loop with repeated dict lookups and per-request attribute access. The code comment immediately above the loop flags the up-to-1K-request loop as a performance bottleneck.

## Estimated impact explanation
The loop is O(num_scheduled_requests) every step. Lowering its per-request CPU overhead directly reduces scheduler-side step latency and median TPOT for 1K-request agentic decode mixes.

## Evolve rationale
This construct is explicitly on the engine critical path after every model step. With high concurrency, interpreted per-request overhead contributes directly to TPOT. Headroom includes hoisting invariant lookups, batching spec-decode accounting, avoiding unnecessary output construction, replacing repeated dict probes with aligned iteration over model_runner_output.req_id_to_index, and reducing per-request list slicing. Correctness oracles include tests/v1/core/test_scheduler.py, tests/v1/core/test_async_scheduler.py, and tests/v1/core/test_output.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Split update_from_output into fast-path / slow-path loops dispatched by a precomputed per-request feature mask
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/sched/scheduler.py:1333-1474, replace the single branch-heavy loop with a dispatching scheme driven by a per-request feature bitmask cached on `Request` (e.g. `request._hot_flags`). The mask is set once at request admission and updated only when a request's feature set actually changes (e.g. structured output attached, encoder inputs released, pooling enabled). Bits cover the rarely-enabled features: HAS_GRAMMAR, HAS_ENCODER_INPUTS, HAS_PROMPT_LOGPROBS, HAS_SAMPLE_LOGPROBS, HAS_POOLING, NEEDS_SPEC_ACCOUNTING, NAN_TRACKING. Then in `update_from_output`, partition `num_scheduled_tokens` into two iterables: a 'vanilla decode' subset (mask == 0 plus generated_token_ids of length 1, no stop hit, RUNNING status) and a 'rich' subset (everything else). The vanilla loop is a tight body that only does: (1) lookup request, (2) `_update_request_with_output`, (3) early-exit if `stopped`, (4) construct `EngineCoreOutput` with all the rare fields hard-coded to None/[] and only the four hot fields populated. The rich loop keeps today's logic. Crucially, also gate the per-request `structured_output_manager.should_advance(request)` and `prompt_logprobs_dict.get(req_id)` / `num_nans_in_logits` probes behind the mask so they never run for vanilla requests — these dict probes are the dominant non-essential cost in decode-heavy 1K-request mixes. To avoid a second pass over `num_scheduled_tokens.items()`, dispatch inline: for each `(req_id, n)`, read `request._hot_flags`; if zero and len(generated_token_ids)==1, take fast path; else fall through to the existing path. Separate stop checks remain in the rich path; `_handle_stopped_request` and `_free_request` are unaffected. Correctness oracles unchanged: tests/v1/core/test_scheduler.py, tests/v1/core/test_async_scheduler.py, tests/v1/core/test_output.py (the partition is purely a code-organization change preserving observable order because `num_scheduled_tokens` insertion order is honored within each request).

**Novelty rationale.**

The candidate has no existing deep_research_proposals. The candidate's own `evolve_rationale` lists hoisting invariants, batching spec-decode accounting, aligned iteration over `req_id_to_index`, and reducing slicing — all generic Python micro-optimizations applied uniformly to every request. This proposal is structurally different: it introduces a precomputed per-request feature mask maintained outside the hot loop and uses it to dispatch into two loop bodies, eliminating the branch and dict-probe cost entirely for the dominant 'vanilla decode' shape rather than just shaving cycles off each branch. The mask-on-Request approach also amortizes the cost across the request's lifetime instead of recomputing per-step, which the rationale's bullet list does not contemplate.

---

### 2. Specialize request update and stop checks for single-token decode
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/sched/scheduler.py` within `Scheduler.update_from_output`, replace the unconditional `_update_request_with_output(request, new_token_ids)` call with a length-1 specialization for the dominant decode case. When `len(new_token_ids) == 1`, append that token directly to the `Request` and run a helper that mirrors `check_stop` using the known `token_id` instead of entering the generic `for enumerate(...)` path, rereading `request.output_token_ids[-1]`, and carrying list-trimming machinery that only matters for multi-token spec-decode outputs. Keep the existing `_update_request_with_output` path unchanged for `len(new_token_ids) > 1` so speculative decoding still trims rejected tokens correctly. To avoid duplicated stop logic, add a small shared helper in the scheduler utils layer, e.g. `check_stop_after_token(request, token_id, max_model_len)`, and have the existing `check_stop(request, max_model_len)` delegate to it. Validate EOS, custom stop token, min_tokens, max_tokens/model length cap, repetition detection, and multi-token trimming behavior with focused scheduler/output tests.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A proposes a per-request feature-mask dispatch that skips rare branches and dict probes by splitting vanilla and rich request handling. This proposal targets a different cost center that remains inside either loop shape: the token append plus stop-check path itself. It does not introduce feature masks, partition requests, or change optional-field probing; it removes avoidable loop/slice/tail-lookup overhead for the common one-token decode output while preserving the existing multi-token slow path.

---
