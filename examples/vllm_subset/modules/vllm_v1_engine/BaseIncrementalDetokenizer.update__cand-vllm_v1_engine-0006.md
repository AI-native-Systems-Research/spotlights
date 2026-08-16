# BaseIncrementalDetokenizer.update

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/detokenizer.py`](vllm/v1/engine/detokenizer.py) (lines 96–143)
- **Symbol:** `BaseIncrementalDetokenizer.update`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_engine-0006`

## Description
Per-request incremental detokenization and stop-check driver for newly generated token ids.

## Current approach
Loops in Python over new_token_ids, appends each id, calls decode_next per token, concatenates into output_text, adjusts min_tokens stop-check offset, then calls check_stop_strings once for the newly appended text.

## Estimated impact explanation
For agentic workloads with many short turns, detokenization is a repeated frontend CPU cost; reducing per-token Python decoding and string concatenation can lower median TPOT.

## Evolve rationale
This method is called for each active generated request on each engine step. Alternatives include buffered string assembly, batched decode for multi-token speculative outputs, specialized no-stop/no-min_tokens paths, and incremental stop-state integration. Correctness oracle: detokenizer tests and tests/v1/engine/test_output_processor.py should preserve emitted text, min_tokens behavior, include_stop_str_in_output semantics, and stop-string truncation.

## Deep research proposals

### 1. Use per-request Aho-Corasick automaton for incremental stop-string matching
- **Finding:** `find-vllm_v1_engine-0010` — *Efficient string matching*
- **Source URL:** <https://cir.nii.ac.jp/crid/1364233268843300096>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/engine/detokenizer.py, replace the linear per-stop-string scan currently performed by check_stop_strings (lines 310-349, invoked from BaseIncrementalDetokenizer.update at lines 129-142) with a persistent Aho-Corasick automaton constructed once per request from the params.stop list (build alongside self.stop_buffer_length in __init__, lines 78-90). Persist the automaton's current node/state on the RequestState across update() calls so that only the newly appended text produced by the decode_next loop (lines 118-120) is fed into the automaton, preserving cross-chunk matches that span token boundaries. On each update(), after the detokenization loop, advance the automaton over self.output_text[stop_check_offset:] (the same slice check_stop_strings is currently given), and report the earliest-completing match, with tie-break by stop-list order to match the existing semantics documented at lines 325-329. Reuse the existing truncation logic (lines 138-141) using the match's end offset. The min_tokens gating at line 131 remains unchanged; the automaton state is simply not consulted for a stop result until num_output_tokens() > min_tokens. When self.stop is empty, skip automaton construction entirely and preserve the current fast path.

**Proposal rationale.**

check_stop_strings iterates over every configured stop string and calls str.find on a growing tail window on every engine step, so its cost per update grows with the number of stop strings and repeats work across chunks. The Aho-Corasick construction described in the finding processes text in a single pass over a combined pattern automaton and, when its state is retained between chunks, avoids re-examining the boundary window on each call. Because BaseIncrementalDetokenizer.update runs for every active request on every engine step and is called out in the candidate's evolve_rationale as an incremental stop-state integration opportunity affecting median TPOT for multi-turn agentic workloads (which frequently configure several stop strings such as role/turn markers), this is a direct, transferable application of the finding to the exact hot path named in the candidate.

---

### 2. Batch DecodeStream over new_token_ids in FastIncrementalDetokenizer for speculative multi-token steps
- **Finding:** `find-vllm_v1_engine-0011` — *Decoders*
- **Source URL:** <https://huggingface.co/docs/tokenizers/main/en/api/decoders>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/v1/engine/detokenizer.py:96-143` (`BaseIncrementalDetokenizer.update`), the per-token Python loop (`for new_token_id in new_token_ids: self.token_ids.append(...); self.output_text += self.decode_next(new_token_id)`) drives one FFI hop per token via `FastIncrementalDetokenizer.decode_next` → `self.stream.step(self.tokenizer, next_token_id)` (lines 211-222, 224-248). The HuggingFace `tokenizers` `DecodeStream.step` API accepts `id: int | List[int]`, so the loop can be replaced with a single native call that consumes the full `new_token_ids` slice (after the `stop_terminated`/`include_stop_str_in_output` head-trim at lines 108-114) and returns the aggregated decoded string in one hop, preserving UTF-8 buffering inside the stream. Concretely: add a `decode_batch(new_token_ids: list[int]) -> str` (or overload of `decode_next`) on the detokenizer ABC; implement it in `FastIncrementalDetokenizer` by calling `self.stream.step(self.tokenizer, new_token_ids)` inside `_protected_step`-equivalent error handling (retrying per-token on `INVALID_PREFIX_ERR_MSG`/`OverflowError`/`TypeError` to preserve the stream-reset recovery in lines 232-247); leave `SlowIncrementalDetokenizer.decode_next` per-token (it maintains running `prefix_offset`/`read_offset` state that isn't batchable through `detokenize_incrementally`). In `update`, replace the loop body with `self.token_ids.extend(new_token_ids); self.output_text += self.decode_batch(new_token_ids)` on the fast path, and recompute the `min_tokens` `stop_check_offset` after the batch: when `self.min_tokens` is set and `len(self.token_ids)` crossed `self.min_tokens` inside the batch, either fall back to per-token decode for just this step (cheap: `min_tokens` only fires early in a request) or use the pre-batch `output_text` length plus a proportional cut-point. Keep `spaces_between_special_tokens=False` behavior (lines 194-222, which inspects each `next_token_id` for `added_token_ids` and can substitute a raw token string) on the per-token path — either by falling back when that flag is active or by checking membership against `new_token_ids` up-front and batching contiguous non-special runs. `check_stop_strings` at lines 129-142 is already called once per step and already handles multi-char growth (via `new_char_count`), so it needs no change.

**Proposal rationale.**

The candidate's `evolve_rationale` explicitly names "batched decode for multi-token speculative outputs" as a promising alternative and calls out per-token Python decoding plus string concatenation as the dominant frontend CPU cost that hurts median TPOT on multi-turn agentic workloads. The finding cites the documented `DecodeStream` API surface (`id: int | List[int]`) that makes exactly this batching safe — UTF-8 buffering stays inside the Rust stream, so streamed text is bit-identical to the per-token path. This directly closes the gap for speculative decoding / MTP steps (where `new_token_ids` routinely holds several tokens) by replacing N Python→Rust FFI hops with 1, without touching the correctness oracles (`tests/v1/engine/test_output_processor.py`, detokenizer tests) beyond preserving their invariants around `min_tokens`, `include_stop_str_in_output`, and stop-string truncation, all of which remain driven by the same offset math around `output_text`.

---

## Agent proposals

### 1. Amortize output_text growth via chunk buffer with bounded stop-check tail
- **Agent:** claude

**Detailed description.**

In vllm/v1/engine/detokenizer.py `BaseIncrementalDetokenizer`, replace the immutable growing `self.output_text: str` with a chunk-log plus a bounded rolling tail, so per-step cost becomes O(new-chars) instead of O(total-text). Concretely: in `__init__` (lines 91-94) replace `self.output_text = ""` with `self._chunks: list[str] = []`, `self._output_len: int = 0`, and `self._tail: str = ""` sized to `tail_cap = (max((len(s) for s in self.stop), default=0)) + EXPECTED_MAX_CHARS_PER_STEP - 1` (a small constant, e.g. 64, is sufficient because `check_stop_strings` at line 340 only reads `output_text[1 - new_char_count - stop_string_len:]`). Expose `output_text` as a `@property` that returns a cached `"".join(self._chunks)`, invalidated on each `update`, so external readers and the truncation code path still work unchanged. In `update` (lines 116-142) replace the loop body `self.output_text += self.decode_next(new_token_id)` with `chunk = self.decode_next(new_token_id); if chunk: self._chunks.append(chunk); self._output_len += len(chunk); self._tail = (self._tail + chunk)[-tail_cap:]`, and derive `stop_check_offset` from `self._output_len` rather than `len(self.output_text)`. Pass the bounded `self._tail` (plus a shifted `new_char_count` measured against the tail window) into `check_stop_strings` — the existing `output_text.find(stop_str, 1 - new_char_count - stop_string_len)` slice math is already a bounded-window form and yields identical results provided `tail_cap >= max_stop_len + new_char_count - 1`. When `check_stop_strings` returns a truncation (line 141), pop chunks from the end while `self._output_len > truncate_to`, slice the final popped chunk once to the exact cutoff, push it back, and invalidate the cached materialized string. Update `get_next_output_text` (lines 149-165): in delta mode, join only the suffix chunks after `self._last_output_text_offset` (track a `(chunk_idx, intra_chunk_offset)` cursor so the join is O(delta)); in non-delta mode, return the cached materialization — this is exercised only when the consumer's `output_kind != DELTA` or when `finished` is true (see output_processor.py:400), so the hot streaming path never pays for the full join. Preserve bit-identical text on all correctness oracles in tests/v1/engine/test_output_processor.py (delta parity, `include_stop_str_in_output`, `min_tokens`, stop-string truncation).

**Novelty rationale.**

Neither listed deep_research_proposal addresses Python string-growth cost. Proposal 1 (Aho-Corasick automaton) changes how stop-string matches are computed but still consumes `self.output_text` and still performs the same `self.output_text += ...` per token. Proposal 2 (batched `DecodeStream.step(list[int])`) reduces FFI hops per step but still assembles the result into the same growing immutable `self.output_text` via `+=`. Both therefore continue to pay O(total-generated-length) per engine step for string reallocation, which dominates on long agentic outputs (code, JSON tool results). This proposal targets exactly that O(n²)-over-a-request cost by replacing the growing `str` with a chunk log plus a bounded tail window sized to what `check_stop_strings` actually reads, and it composes cleanly with both listed proposals (the Aho-Corasick state consumes the same bounded tail; the batched decode simply produces one chunk per step instead of many).

---

### 2. Add a no-stop fast path for requests without stop strings
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/detokenizer.py`, specialize `BaseIncrementalDetokenizer.update` for the common case where `self.stop` is empty. After the existing `new_token_ids` and `stop_terminated` handling, branch before computing `stop_check_offset`: if there are no stop strings, append/decode the new ids and return `None` without reading `len(self.output_text)` before the loop, running the per-token `min_tokens` branch, or calling the stop-check block. Keep the `stop_terminated and not include_stop_str_in_output` head-trim behavior exactly as-is, since token-level stop ids still need to be withheld from detokenization. For a slightly cleaner shape, factor the current stop-aware path into `_update_with_stop_strings` and make the no-stop path the first hot branch in `update`; this preserves existing behavior while removing stop-string bookkeeping from requests that only rely on EOS or token ids.

**Novelty rationale.**

The Aho-Corasick proposal optimizes matching when stop strings exist, but it does not remove the stop-check offset and `min_tokens` bookkeeping for requests with no stop strings. The batched DecodeStream proposal reduces tokenizer FFI hops, but still leaves the same generic stop-aware control flow around every update. Agent A's chunk-buffer proposal targets immutable string growth and bounded stop tails, not the branch and length-check overhead in the no-stop case. This proposal is a narrower fast path for `self.stop == []`, which is independently composable with all three existing ideas.

---
