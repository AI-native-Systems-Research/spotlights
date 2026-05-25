# _iter_placeholders

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/processing/processor.py`](vllm/multimodal/processing/processor.py) (lines 865–931)
- **Symbol:** `_iter_placeholders`
- **Kind:** function
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0006`

## Description
Scans tokenized prompts and yields PlaceholderFeaturesInfo spans for multimodal placeholder content.

## Current approach
For each prompt start position, it loops over the next expected item for each modality and over that item's candidate updates. For every candidate it obtains content_tokens_full through _seq2tokens, computes a slice, and compares prompt[start_idx:end_idx_full] to the candidate tokens. On mismatch it advances by one token.

## Estimated impact explanation
Placeholder localization runs for each multimodal prefill and scans the full prompt. Long agentic prompts with many media placeholders amplify this pure-Python cost, so a linear multi-pattern scan can directly reduce media TTFT.

## Evolve rationale
The nested start_idx/modality/update scan and slice comparison at lines 888-905 are the optimization unit. Headroom includes hoisting all candidate token sequences out of the scan, indexing candidates by first token, using an Aho-Corasick automaton across all candidate sequences, and replacing slice equality with allocation-free matching. Oracle: find_mm_placeholders behavior in tests/multimodal/test_processing.py must return identical PlaceholderFeaturesInfo lists, preserving modality priority, item_idx advancement, start_idx, tokens, and is_embed masks.

## Deep research proposals

### 1. Replace nested slice-compare scan in _iter_placeholders with a pyahocorasick automaton
- **Finding:** `find-0011` — *pyahocorasick — ahocorasick documentation*
- **Source URL:** <https://pyahocorasick.readthedocs.io/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/multimodal/processing/processor.py at _iter_placeholders (lines 865-931), eliminate the per-position triple-nested loop (start_idx × modality × candidate update) and the per-step prompt[start_idx:end_idx_full] == content_tokens_full slice equality. Build a pyahocorasick.Automaton once per call (or cache it on the MultiModalPromptUpdates / processor) keyed by every distinct candidate content_tokens_full sequence. Because the automaton operates on bytes/strings, encode each token-id sequence to bytes via a fixed-width packing (e.g. struct.pack with a width chosen from tokenizer vocab size, or '!I' for 4-byte ids) and pack the prompt the same way; store the (modality, update_index_within_modality, content_tokens_full, is_embed) tuple as the value for each pattern. Call automaton.make_automaton(), then iterate automaton.iter(packed_prompt) once to obtain all (end_byte_offset, payload) hits in linear scan time. Convert each hit's end_byte_offset back to a token start_idx and walk the hits in ascending start_idx, applying the existing priority rules: maintain item_idx_by_modality and only accept a hit whose modality matches one of the next-pending updates (modality earlier in mm_prompt_updates wins ties at the same start_idx), then jump start_idx past the matched span (non-overlapping). Hoist content_tokens_full and is_embed lookups out of the hot loop so _seq2tokens is invoked once per candidate while building the automaton, not per prompt position. Preserve current yield order, item_idx advancement, modality priority, start_idx, tokens, and is_embed semantics so find_mm_placeholders results match exactly. Fall back to the current implementation when pyahocorasick is unavailable, when the prompt is short, or when only one candidate exists (build/setup not worth it).

**Proposal rationale.**

The candidate's hot path is exactly the problem pyahocorasick is designed for: locate occurrences of many fixed sequences inside a longer sequence in a single linear pass. The current code is O(prompt_len × Σ candidates × pattern_len) due to per-position slice equality, which the evolve_rationale explicitly flags as the optimization unit and which directly inflates media TTFT for the multi-turn agentic workload with many placeholders described in the caller context. The finding's quoted technique — 'find multiple key strings occurrences at once in some input text' — maps onto scanning for all placeholder token sequences in O(prompt_len + total_pattern_len + matches), allocation-free per step, and removes redundant _seq2tokens calls. The library is mature, pure-C-extension, and importable as an optional dependency, so the change is concrete and bounded to this function while preserving the priority/advancement invariants required by tests/multimodal/test_processing.py.

---

## Agent proposals

### 1. Replace nested slice scan with packed-bytes memmem cursor over only currently-pending candidates
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/processing/processor.py at _iter_placeholders (lines 865-931), keep the priority/advancement semantics but replace the per-position triple-nested slice scan with a stdlib-only, dependency-free leftmost-match cursor that exploits CPython's C-level memmem (Two-Way) inside `bytes.find`. Concretely: (1) Pack `prompt` once to bytes via `array.array('I', prompt).tobytes()` (4 bytes per token-id, big enough for any tokenizer vocab), and maintain a parallel `byte_start = start_idx * 4`. (2) When (re)entering the loop for a modality whose `item_idx_by_modality[modality]` advanced, pack each `update.content.full` (lazily via `_seq2tokens`) for that item's update list to bytes once and cache per (modality, item_idx, update_index); also lift `is_embed` resolution out of the hot loop into the cache entry. (3) Maintain a tiny dict `next_hit: dict[(modality, update_index), int]` of the next match's byte offset for every currently-pending candidate (one item per modality at a time), initialized via `packed_prompt.find(packed_content, byte_start)`. (4) On each iteration, pick the candidate with the smallest `next_hit` value that is `>= byte_start`, breaking ties using the existing modality order in `mm_prompt_updates` (and update order within a modality). Yield the corresponding `PlaceholderFeaturesInfo`, then set `byte_start = hit + len(packed_content)` (i.e. `start_idx = end_idx_full`), increment `item_idx_by_modality[modality]`, and rebuild that modality's cache entries for the next item; for the other modalities' candidates whose cached `next_hit < byte_start`, refresh in place via `packed_prompt.find(packed_content, byte_start)`. (5) Stop when `_all_items_found` or all `next_hit` are -1. (6) Skip the optimization (fall back to the current code) when prompt is short (e.g. < 512 tokens), when there is exactly one pending candidate of length <= 4 tokens (slice equality is already cheap), or when any tokenizer vocab exceeds 32-bit ids. This preserves the exact yield order, item_idx advancement, modality-priority tie-breaking, start_idx, tokens, and is_embed fields required by find_mm_placeholders and tests/multimodal/test_processing.py, but reduces total work from O(prompt_len × Σ_pending candidates × pattern_len) Python-level slice equality to O(Σ_pending candidates × prompt_len) C-level SIMD-accelerated memmem invocations — and in practice far less, because each candidate's cursor advances monotonically and is only re-searched when surpassed.

**Novelty rationale.**

The existing deep_research_proposal (find-0011) builds a pyahocorasick.Automaton over every candidate sequence and scans the packed prompt once. This proposal is fundamentally different in three ways: (a) it adds no third-party dependency — it uses only `array` and `bytes.find`, leveraging CPython's existing C-level Two-Way/memmem search; (b) it tracks only the currently-pending candidates (bounded by |modalities| × updates-per-item, typically 1-3) via a per-candidate next-hit cursor instead of compiling a full multi-pattern automaton over the Cartesian product of all items × updates, which avoids automaton build/teardown cost on every call and avoids the awkwardness of filtering automaton hits to enforce item_idx and modality-priority rules post-hoc; (c) cursors advance monotonically and are only refreshed when surpassed by the chosen match, so the amortized work for typical multi-turn agentic prompts (few placeholders separated by long spans) is closer to O(matches × prompt_len_remaining) rather than a full linear pass per call. None of these characteristics are described in find-0011, which centers on `pyahocorasick.Automaton`, `make_automaton`, and `automaton.iter`.

---

### 2. Scan only token positions that can start the next placeholder
- **Agent:** codex

**Detailed description.**

In `vllm/multimodal/processing/processor.py::_iter_placeholders` (lines 865-931), replace the token-by-token `start_idx += 1` scan with a first-token position index. At function entry, materialize each non-empty candidate `content.full` once into a small per-modality/per-item/update entry containing `tokens`, `length`, and `first_token`, while leaving `content.is_embed` lazy until a match is actually yielded. Build `positions_by_token` in one pass over `prompt` for the union of all candidate first tokens. During the loop, look only at the earliest indexed position at or after the current `start_idx` among the first tokens for currently pending modality items; if none exists, stop. At that candidate start, preserve the existing modality order and update order by iterating `mm_prompt_updates` exactly as today, but skip entries whose `first_token` differs from `prompt[start_idx]`. Verify the full candidate with an allocation-free token-by-token comparison instead of `prompt[start_idx:end_idx_full]` slicing. On match, yield the same `PlaceholderFeaturesInfo`, advance `start_idx` to `end_idx_full`, increment that modality's `item_idx`, and continue; on no match, advance the cursor for that first token past the rejected position. Add focused coverage to `tests/multimodal/test_processing.py::test_find_mm_placeholders` for shared first tokens, equal placeholder tokens across modalities, lower-priority single-token candidates, and long non-placeholder prefixes to ensure identical yield order and `start_idx` behavior.

**Novelty rationale.**

This is not the deep-research pyahocorasick proposal: it does not introduce an automaton, pack token IDs into bytes, scan all full patterns, or filter automaton hits after the fact. It is also not Agent A's packed-`bytes.find` cursor: it does not search for full candidate byte strings or maintain memmem cursors per candidate. The distinct idea is a lightweight prompt-side inverted index over candidate first tokens, using the existing left-to-right priority loop only at positions that could possibly match the currently pending placeholders. That keeps setup cost low, avoids a new dependency and byte packing, and targets the common long-agentic-prompt case where almost every token cannot be the beginning of any multimodal placeholder.

---
