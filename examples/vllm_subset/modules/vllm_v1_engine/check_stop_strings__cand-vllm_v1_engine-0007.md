# check_stop_strings

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/detokenizer.py`](vllm/v1/engine/detokenizer.py) (lines 310–362)
- **Symbol:** `check_stop_strings`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0007`

## Description
Stop-string matcher that finds the earliest stop string completing in the newly generated suffix and returns the truncation offset.

## Current approach
For each stop string, calls output_text.find with a negative start bound, then selects the match with the smallest end offset and stop-list-order tie behavior.

## Estimated impact explanation
Agentic clients often use multiple chat/tool delimiters; reducing O(num_stops * suffix_window) scans lowers frontend CPU and median TPOT when stop strings are configured.

## Evolve rationale
The function has a stable contract and runs whenever a request has stop strings after new text is decoded. Candidate replacements include a single-stop fast path, per-request trie/Aho-Corasick state, rolling suffix matching, or byte-level scanning while preserving earliest-completion semantics. Correctness oracle: detokenizer and output_processor stop-string tests should match str.find behavior, truncation offsets, include_in_output behavior, and tie-breaking.

## Deep research proposals

### 1. Replace per-stop str.find loop with per-request Aho-Corasick automaton retaining state across decoded chunks
- **Finding:** `find-vllm_v1_engine-0010` — *Efficient string matching*
- **Source URL:** <https://cir.nii.ac.jp/crid/1364233268843300096>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Change check_stop_strings in vllm/v1/engine/detokenizer.py:310-362 from an O(num_stops * suffix_window) loop of output_text.find calls into a single-pass multi-pattern match driven by an Aho-Corasick automaton. Build the automaton once per request from the stop list (cached alongside the IncrementalDetokenizer / request state, keyed by the tuple of stop strings so identical stop sets can share a compiled automaton) and persist its current DFA state between decoded chunks. On each detokenization step, feed only the newly appended characters (plus a max(len(stop_str)) - 1 carryover from the prior suffix so cross-chunk matches are not missed) through the automaton in a single pass, recording each pattern-completion event with its end offset in output_text. Preserve current semantics exactly: (1) select the stop string whose completion end offset is smallest within the region newly exposed by new_char_count (matching the existing `find` start bound of `1 - new_char_count - stop_string_len`), (2) break ties by stop-list order (assign each pattern its list index and prefer the smaller index on equal end offsets), and (3) return the same (stop_string, offset) tuple with the include_in_output truncation logic unchanged. Keep the existing fast exits for empty new_char_count or empty stop. For the common single-stop case, retain a fast path using str.find to avoid automaton overhead. Correctness must be verified against the existing detokenizer and output_processor stop-string tests, including tie-breaking, include_in_output behavior, and multi-token speculative-decoding chunks.

**Proposal rationale.**

The candidate explicitly calls out per-request trie/Aho-Corasick state as a target replacement and identifies O(num_stops * suffix_window) scans as the cost driver, which matters for agentic clients configuring many chat/tool delimiters (the stated caller workload). The Aho-Corasick finding directly addresses this gap: it gives earliest-completion multi-pattern matching in a single pass over new text with cost independent of the number of stop strings, and it naturally supports the incremental cross-chunk state the candidate mentions (rolling suffix matching). This is a concrete, transferable algorithmic substitution that preserves the stable contract (earliest completion, stop-list tie-break, include_in_output truncation) while removing the per-stop scan factor that the estimated-impact rationale flags as the source of frontend CPU and median TPOT cost.

---

## Agent proposals

### 1. Stateless bounded-tail scan with first-char bucket + shingle prefilter for check_stop_strings
- **Agent:** claude

**Detailed description.**

Refactor vllm/v1/engine/detokenizer.py:310-362 (check_stop_strings) to eliminate the per-stop str.find loop without introducing any per-request persistent state. Concretely: (1) At function entry, compute max_len = max(len(s) for s in stop) and slice a bounded tail window tail = output_text[-(new_char_count + max_len - 1):] together with its absolute offset base = len(output_text) - len(tail); this is the only region str.find would ever match in, and constrains all subsequent scanning to O(new_char_count + max_len) bytes regardless of prior output length. (2) Precompute (once, memoized on the id() of the stop list plus its length as a cheap invalidation key on a small module-level LRU of size ~64) two structures: a dict mapping each stop string's first character to a list of (stop_index_in_list, stop_str) entries, and a set of two-character shingles taken from stops of length >= 2. Stops of length 1 fall through to the first-char bucket only. (3) Scan tail with a single Python-level for i, ch in enumerate(tail) loop that (a) checks ch in first_char_buckets, (b) if len >= 2 checks tail[i:i+2] in shingle_set, and (c) only then iterates the small candidate bucket to test tail.startswith(stop_str, i). Record each match as (base + i + len(stop_str), stop_list_index, stop_str). (4) Preserve the exact contract: prefer the smallest end offset; on tie, prefer the smaller stop-list index. Short-circuit the outer loop once i > current_best_end - base - min_stop_len, since no later position can complete earlier. Retain the existing fast exit for empty new_char_count or stop, and keep a straight str.find fast path when len(stop) == 1. (5) The include_in_output truncation branch and the -1 sentinel for no-truncation are unchanged. Correctness is verified against the existing detokenizer and output_processor stop-string tests, including tie-breaking, include_in_output=True/False, and multi-token speculative-decoding chunks where new_char_count > 1.

**Novelty rationale.**

The listed deep_research_proposal replaces the loop with a per-request Aho-Corasick automaton that caches compiled state keyed by the tuple of stop strings and persists DFA state across decoded chunks with a max_len - 1 carryover. This proposal is stateless: it holds no per-request DFA, no cross-chunk carry, and no tuple-keyed compilation cache; it derives its work bound from the already-available (new_char_count, max_stop_len) pair by slicing a tail window, and uses a first-character bucket plus optional two-character shingle prefilter over that window rather than an automaton. It also avoids Aho-Corasick's dependency (either a new pyahocorasick dep or a hand-rolled Python-level DFA whose per-character constants can rival str.find on short inputs) and its cache-invalidation edge cases when stop lists mutate or when identical stop sets are used across many short-lived requests. The earliest-end + stop-list-order tie-breaking short-circuit on the scanned position is an optimization the AC finding does not describe. This is a distinct algorithmic substitution operating on the same candidate lines and preserving the same contract.

---

### 2. Use a cached literal-regex matcher for multi-stop lists
- **Agent:** codex

**Detailed description.**

Add a multi-stop fast path in `vllm/v1/engine/detokenizer.py:310-362` that compiles the request stop list into one escaped `re` pattern and reuses it while preserving the current result contract. For `len(stop) == 1`, keep the existing `str.find` path. For larger lists, build a small cached matcher keyed by `tuple(stop)` that stores a compiled alternation of escaped stop strings plus metadata mapping literal strings to their first stop-list index. At runtime, slice the same bounded search region implied by the current negative `find` start, run `finditer` over that region, and for each regex match compute `(absolute_end, first_stop_index, absolute_start, stop_str)`. Select the smallest `absolute_end`, breaking ties by `first_stop_index`, then return the same truncation offset logic for `include_in_output`. The implementation should escape all stops with `re.escape`, handle duplicate stop strings by retaining the first index, and include regression tests for overlapping stops such as `['abc', 'bc']`, same-end ties such as `['bc', 'abc']` on output ending in `abc`, duplicate stops, and speculative chunks where `new_char_count > 1`.

**Novelty rationale.**

The deep research proposal uses a per-request Aho-Corasick automaton with persistent DFA state across decoded chunks. Agent A proposes a Python-level bounded-tail scanner with first-character buckets and shingle prefilters. This proposal is neither persistent automaton nor manual bucket scanning: it delegates literal multi-pattern search to CPython's regex engine, amortizes compilation with a tuple-keyed cache, and keeps only lightweight match selection logic in Python. It targets the same per-stop `str.find` cost while using a distinct implementation strategy and different correctness hazards to test, especially regex escaping, duplicates, and earliest-end selection rather than earliest-start regex behavior.

---
