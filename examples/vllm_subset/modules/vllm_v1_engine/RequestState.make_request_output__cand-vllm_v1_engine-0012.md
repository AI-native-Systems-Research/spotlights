# RequestState.make_request_output

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/output_processor.py`](vllm/v1/engine/output_processor.py) (lines 276–340)
- **Symbol:** `RequestState.make_request_output`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0012`

## Description
Per-request RequestOutput factory that applies FINAL_ONLY and stream_interval gating, handles pooling vs completion output, and aggregates parallel-sampling children.

## Current approach
Branches on finish state, output kind, stream_interval, pooling output, and parent request state; may slice detokenizer.output_token_ids for DELTA mode and allocates CompletionOutput/RequestOutput objects on each emitted step.

## Estimated impact explanation
This is a per-request allocation and branching hotspot on the frontend path; reducing its constant factors can lower median TPOT, especially with many short concurrent generations.

## Evolve rationale
Called once per active request per output-processing step. Candidate improvements include specializing common streaming modes, reducing repeated prompt/logprob handling in _new_request_output, object/allocation reuse, and adaptive stream_interval behavior. Correctness oracle: tests/v1/engine/test_output_processor.py and tests/v1/engine/test_parallel_sampling.py cover stream_interval, DELTA/FINAL_ONLY, pooling, parent aggregation, and finished flags.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Hoist FINAL_ONLY / stream_interval gating out of make_request_output into a cached fast-reject predicate
- **Agent:** claude

**Detailed description.**

At vllm/v1/engine/output_processor.py:276-340, RequestState.make_request_output is called once per active request per output-processing step (see the call site at line 668 in _process_engine_outputs). On the dominant non-streaming path (RequestOutputKind.FINAL_ONLY, which is what OpenAI-compatible non-stream requests and most multi-turn agentic tool calls use), every non-finish invocation walks: attribute lookup for self.output_kind, equality check against FINAL_ONLY, branch, and function return None. The stream_interval>1 case additionally reads self.detokenizer.num_output_tokens() and self.sent_tokens_offset each step even when no emission is due. All of this is pure overhead: for a 200-token generation with FINAL_ONLY there are 199 useless calls per request. Concrete change: (1) at RequestState construction (from_new_request, ~line 253), precompute self._emits_intermediate: bool = (output_kind != FINAL_ONLY) and self._stream_interval_gt1: bool = (stream_interval > 1), and a bound-method-free fast-path flag self._emit_only_on_finish = final_only and stream_interval <= 1. (2) Introduce a tiny inline predicate RequestState.should_emit(self, finished: bool) -> bool that returns finished or not self._emit_only_on_finish for the common case, and only falls into the current stream_interval math when self._stream_interval_gt1 is true. (3) In OutputProcessor._process_engine_outputs (around line 668) and in the abort path (around line 502), call `if not req_state.should_emit(finish_reason is not None): continue` before calling make_request_output, so the full method body — including local variable binding for final_only, the assert on detokenizer, and the attribute chain on self.detokenizer — is skipped entirely for the ~99% of steps that emit nothing. (4) Inside make_request_output itself, replace the recomputation of `final_only = self.output_kind == RequestOutputKind.FINAL_ONLY` with the precomputed self._emit_only_on_finish so it stays consistent. Correctness is preserved because should_emit is a strict subset of the existing gating logic (same predicates, precomputed). Coverage: tests/v1/engine/test_output_processor.py already exercises stream_interval, DELTA/FINAL_ONLY, and finished flags, and tests/v1/engine/test_parallel_sampling.py covers parent aggregation — both still pass since parent_req and DELTA slicing branches are only reached when should_emit already returned true. Expected impact on TPOT: per-step per-request overhead drops from a Python method call + several attribute lookups to a single bool AND-load, which on 100+ concurrent short generations (typical agentic tool-loops with 20-50 output tokens each) removes a measurable slice of frontend serialization time on the output processor.

**Novelty rationale.**

The candidate lists zero existing deep_research_proposals, so any concrete change is novel by construction. Beyond that, this proposal is specifically about lifting the fast-reject predicate out of make_request_output into the caller loop and precomputing the mode flags at RequestState construction — a call-site refactor, not a change to CompletionOutput/RequestOutput allocation, object pooling, adaptive stream_interval, or _new_request_output prompt/logprob handling (the four directions the candidate's evolve_rationale explicitly enumerates). It targets the currently-uncounted overhead of the 'return None' branch itself, which the evolve_rationale doesn't mention.

---

### 2. Defer text decoding for FINAL_ONLY requests with no stop strings
- **Agent:** codex

**Detailed description.**

For `vllm/v1/engine/output_processor.py:276-340`, add a `RequestState` flag computed in `from_new_request` for the common non-streaming case: `output_kind == RequestOutputKind.FINAL_ONLY`, detokenization enabled, and no string stop criteria that require incremental stop checks. In `_process_engine_outputs`, when this flag is set and the request is not finished, append `new_token_ids` to the detokenizer/token-id state without calling the expensive incremental decode path or `get_next_output_text`; only decode the accumulated output once when `finish_reason` is present and `make_request_output` will actually emit. This likely needs a small detokenizer API such as `append_token_ids_only(new_token_ids)` plus `finalize_output_text()` so the existing final `CompletionOutput` fields remain identical. Keep the existing path for streaming, DELTA/CUMULATIVE intermediate outputs, requests with stop strings, and cases where stop detection depends on decoded text. Add/extend `tests/v1/engine/test_output_processor.py` coverage to assert final text/token IDs match the current behavior for FINAL_ONLY without stop strings, while stop-string cases still use incremental decoding.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A proposes skipping calls into `make_request_output` when no output should be emitted, but it leaves the earlier per-step detokenizer update and stop-check work intact. This proposal targets a different hotspot in the same output-processing path: avoid building decoded text on every token for FINAL_ONLY requests that cannot need intermediate text, then decode once at the only emitted output.

---
