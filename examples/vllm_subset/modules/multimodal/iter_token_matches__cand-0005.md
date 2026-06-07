# iter_token_matches

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/processing/processor.py`](vllm/multimodal/processing/processor.py) (lines 619–645)
- **Symbol:** `iter_token_matches`
- **Kind:** function
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0005`

## Description
Yields every non-overlapping occurrence of a token-id sequence inside a prompt token-id list.

## Current approach
A naive sliding window compares token_ids[start_idx:end_idx] to match_ids at each candidate start. Each failed probe allocates a fresh Python list slice and advances by one token.

## Estimated impact explanation
This is used while applying multimodal prompt updates across the full prompt. Avoiding per-position slice allocation reduces pure-Python prefill work for long prompts with multiple image/audio/video placeholders, moving media TTFT more than steady-state TPOT.

## Evolve rationale
The slice-equality probe at line 639 is a fixed-contract algorithmic target. Headroom includes a single-token fast path using list.index, allocation-free element comparison, or KMP/Boyer-Moore-style skipping for multi-token targets while preserving non-overlap semantics. Oracle: tests/multimodal/test_processing.py for iter_token_matches and replace_token_matches must yield exactly the same (start_idx, end_idx) pairs, including empty-pattern and overlap exclusion behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Anchor-skip scanning via list.index for the first match token, then tail-compare without slicing
- **Agent:** claude

**Detailed description.**

Rewrite the body of iter_token_matches at vllm/multimodal/processing/processor.py:619-645 to exploit the fact that multimodal placeholder token IDs (the typical match_ids[0]) are rare reserved tokens in real prompts. Instead of advancing by one and slicing on every step, use CPython's C-implemented list.index to jump directly to the next occurrence of match_ids[0], then verify the remaining match_len-1 tokens with an allocation-free element-wise loop:

  if match_len == 0: return
  first = match_ids[0]
  prompt_len = len(token_ids)
  last_start = prompt_len - match_len
  while start_idx <= last_start:
      try:
          start_idx = token_ids.index(first, start_idx, last_start + 1)
      except ValueError:
          return
      # tail-compare without slicing
      i = 1
      while i < match_len and token_ids[start_idx + i] == match_ids[i]:
          i += 1
      if i == match_len:
          end_idx = start_idx + match_len
          yield _TokenMatch(start_idx=start_idx, end_idx=end_idx)
          start_idx = end_idx  # preserve non-overlap semantics
      else:
          start_idx += 1

For match_len == 1, the inner while is skipped and the loop reduces to the single-token list.index fast path naturally. Non-overlap semantics, empty-pattern early return, and start_idx semantics are preserved bit-for-bit, so tests/multimodal/test_processing.py for iter_token_matches and replace_token_matches will yield identical (start_idx, end_idx) pairs. The win is two-fold: (1) eliminating the per-position list slice allocation at line 639 (each failed probe currently allocates a fresh list of length match_len plus a tuple-equivalent comparison through Python bytecode); (2) replacing the outer per-position bytecode loop with a C-speed scan for the first byte, which is overwhelmingly the dominant cost when placeholders are sparse (typical case: a few image/video placeholder runs in a multi-thousand-token prompt). This directly attacks media TTFT in the multi-turn agentic workload, where _apply_prompt_updates iterates over potentially many PromptUpdate targets per request.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate (the field is empty). The evolve_rationale on the candidate itself enumerates three generic directions (single-token list.index, allocation-free element comparison, KMP/Boyer-Moore), but does not specify the hybrid: use list.index as an *anchor jump* for the first token of a multi-token pattern, then do an allocation-free tail compare. That hybrid is strictly different from the single-token-only list.index path and from KMP/BM (which require precomputing failure/shift tables and add complexity that is unjustified given that placeholder tokens are typically rare and patterns are short). It is also more concrete than 'allocation-free element comparison' alone, since it specifies the C-level skip mechanism that yields the largest constant-factor win in this candidate's actual workload.

---

### 2. Add a homogeneous-token run fast path
- **Agent:** codex

**Detailed description.**

In vllm/multimodal/processing/processor.py:619-645, add a special path before the general scanner for match_ids that are made of the same token repeated, e.g. match_len > 1 and match_ids.count(match_ids[0]) == match_len. For that case, scan token_ids as contiguous runs of that token and yield non-overlapping chunks of length match_len from each run: find the next run_start at or after start_idx, advance run_end while token_ids[run_end] equals the repeated token, then yield _TokenMatch(run_start, run_start + match_len), _TokenMatch(run_start + match_len, run_start + 2 * match_len), etc. until fewer than match_len tokens remain in the run, then continue after run_end. This preserves empty-pattern behavior, start_idx filtering, and overlap exclusion exactly for homogeneous patterns, while avoiding per-candidate slice allocation and avoiding repeated tail verification inside long placeholder runs. Add focused tests where start_idx falls inside a repeated-token run, and keep the existing iter_token_matches and replace_token_matches parametrizations as the oracle.

**Novelty rationale.**

There are no deep_research_proposals listed for this candidate. Agent A's proposal is a general first-token list.index anchor with an allocation-free tail compare for every candidate occurrence. This proposal is narrower and different: it detects the repeated-token target shape and consumes whole contiguous runs by length, producing all non-overlapping matches from run arithmetic without comparing the tail for each candidate. It specifically targets patterns like repeated image placeholder token sequences, which Agent A's tail-compare loop still processes one candidate match at a time.

---
