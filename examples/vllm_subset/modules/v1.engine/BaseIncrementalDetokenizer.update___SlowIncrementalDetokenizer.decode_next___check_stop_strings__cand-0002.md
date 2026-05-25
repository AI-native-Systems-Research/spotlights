# BaseIncrementalDetokenizer.update / SlowIncrementalDetokenizer.decode_next / check_stop_strings

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/detokenizer.py`](vllm/v1/engine/detokenizer.py) (lines 95–339)
- **Symbol:** `BaseIncrementalDetokenizer.update / SlowIncrementalDetokenizer.decode_next / check_stop_strings`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0002`

## Description
Per-step incremental detokenization and stop-string detection. update() walks new token ids one by one, delegates to fast or slow decode_next, grows output_text, and scans newly exposed suffixes for configured stop strings.

## Current approach
Pure-Python token loop with per-token string concatenation, a per-token min_tokens branch, per-token slow detokenize_incrementally fallback, and check_stop_strings iterating every stop string with str.find over a suffix window.

## Estimated impact explanation
This frontend path runs for every generated token that needs text output. Under high request fan-out, shaving per-token Python detokenization and stop-check overhead directly reduces median TPOT and output-handler backlog.

## Evolve rationale
The optimization unit is the token loop at lines 117-122 and stop-string scan at lines 322-338. Headroom: batched decode where the tokenizer backend permits, list-append plus join for decoded fragments, hoisting min_tokens handling out of the per-token branch, and replacing repeated str.find with a rolling or Aho-Corasick matcher when many stops are configured. Correctness oracle: tests/v1/engine/test_output_processor.py and tests/v1/engine/test_fast_incdec_prefix_err.py must produce byte-identical output_text, DELTA chunks, and matched stop reasons across a streamed token corpus.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Skip stop-string scanning via a per-request token-id prefilter
- **Agent:** claude

**Detailed description.**

In vllm/v1/engine/detokenizer.py at BaseIncrementalDetokenizer.update (lines 95-142), the `check_stop_strings` call (lines 130-136) runs every step that produces text, and inside (lines 322-338) iterates each configured stop string with a Python-level `str.find` over a suffix of the growing `output_text`. For agentic workloads, `self.stop` typically contains several tool/turn delimiters (e.g. '</tool_call>', '<|im_end|>', 'Observation:'), and the vast majority of tokens are not even candidates to complete any of them. Add a lightweight pre-scan that elides the suffix search on those steps.

Concrete change:
1. In `BaseIncrementalDetokenizer.__init__` (and in the Fast/Slow subclasses where the tokenizer is in scope), when `self.stop` is non-empty, build a `frozenset[int]` of candidate 'stop-tail token ids'. Compute it by tokenizing each stop string with the request's tokenizer using `add_special_tokens=False` and collecting (a) the last token id of that encoding and (b) any single-token encoding where applicable. If any stop string fails to encode cleanly (encoding empty, contains UNK, or its surface form re-tokenizes differently when glued to a context probe), set `self._stop_prefilter = None` to disable the optimization for that request and fall through to the existing path.
2. Store the result as `self._stop_tail_token_ids: frozenset[int] | None` on the detokenizer.
3. In `update()` (lines 117-136), while looping `new_token_ids`, also track `tail_hit = any(tid in self._stop_tail_token_ids for tid in new_token_ids)` (or compute it once via `not self._stop_tail_token_ids.isdisjoint(new_token_ids)` before the loop). Only call `check_stop_strings` when `self._stop_prefilter is None or tail_hit` is true. When the prefilter is active and no token in this batch is a tail candidate, skip the scan entirely — `output_text` already contains the buffered region of length `stop_buffer_length`, so a stop string can only complete on a step that contains one of the tail tokens.
4. Keep the existing `min_tokens` gate (line 130) unchanged; it composes naturally with the prefilter.

Correctness: when `_stop_prefilter` is None (any stop string failed the clean-encoding probe), behavior is byte-identical to today. When active, the invariant 'stop string completes ⇒ its final token id is one of the tail candidates' must hold for the request's tokenizer. The probe in step 1 verifies this by re-encoding each stop string against a small suffix of `output_text` at construction and rejecting strings whose tail-token identity is context-dependent. tests/v1/engine/test_output_processor.py and tests/v1/engine/test_fast_incdec_prefix_err.py continue to pass; add a parametric test that randomizes whether the prefilter is enabled and asserts identical `output_text`, DELTA chunks, and matched stop reasons.

Why this helps median TPOT for agentic workloads: it converts the per-step cost of stop-string detection from O(num_stops × suffix_window) Python string scans into a single set-disjoint check on a tiny `new_token_ids` list (typically 1-4 ids). On the steady-state token stream where no delimiter is forming, the scan is fully skipped.

**Novelty rationale.**

The candidate has no existing deep_research_proposals. The evolve_rationale enumerates batched decode, list-append+join for the output buffer, hoisting min_tokens, and replacing str.find with a rolling/Aho-Corasick matcher. This proposal is a different layer: rather than making the byte-level scan faster, it eliminates the scan on the dominant fraction of steps using a token-id-level set membership test computed once at request construction. It is complementary to (not subsumed by) an Aho-Corasick replacement, and it does not appear in the rationale list.

---

### 2. Defer detokenization for requests without stop strings
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/detokenizer.py`, add a demand-driven path for requests where `self.stop` is empty. `BaseIncrementalDetokenizer.update()` currently decodes every new token and grows `output_text` even when no stop-string check is possible and the output processor may not emit text yet, such as FINAL_ONLY requests or stream-interval throttled requests. Introduce `_pending_decode_token_ids` plus an eager/deferred flag. On the deferred path, `update()` should only extend `self.token_ids`, append text-eligible ids to `_pending_decode_token_ids`, respect `stop_terminated and not include_stop_str_in_output` by recording but not queuing the final stop token, and return `None`. Add `_flush_pending_decode()` and call it at the start of `get_next_output_text()` so pending ids are fed through the existing `decode_next()` logic in order before full/delta slicing. Keep the current eager path whenever stop strings are configured so stop-string aborts and `stop_buffer_length` semantics are unchanged. To make this safe for slow tokenizers, first decouple `SlowIncrementalDetokenizer.decode_next()` from `self.token_ids[-1]` by passing the supplied `next_token_id` into `detokenize_incrementally()` via a one-id list or a small helper; after initialization, that helper only needs the new id plus `self.tokens`, `prefix_offset`, and `read_offset`. Add tests with a dummy counting detokenizer to assert no decode calls occur before `get_next_output_text()` on stop-free requests, plus output-processor coverage for FINAL_ONLY, stream interval, and stop-token termination where token ids still include the skipped stop token while text excludes it.

**Novelty rationale.**

The candidate has no deep_research_proposals, and the listed evolve_rationale focuses on making the existing per-token work faster: batched decode, list-append/join, min_tokens hoisting, and faster stop-string search. This proposal changes when decoding happens for stop-free requests and can initially reuse the existing per-token decoder unchanged. It is also distinct from Agent A's token-id stop prefilter, which only skips `check_stop_strings` for requests that do have stop strings; this proposal applies when there are no stop strings and defers the whole detokenization step until text is actually requested.

---
