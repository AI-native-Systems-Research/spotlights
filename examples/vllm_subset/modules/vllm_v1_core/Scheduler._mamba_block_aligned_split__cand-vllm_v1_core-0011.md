# Scheduler._mamba_block_aligned_split

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 349–424)
- **Symbol:** `Scheduler._mamba_block_aligned_split`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0011`

## Description
Mamba align-mode chunk-boundary heuristic that clips prefill chunks to cacheable recurrent-state boundaries.

## Current approach
Computes start/end, last cacheable block boundary with EAGLE back-off, prompt-tail hash boundary, next block boundary, and shared-prefix boundary, then chooses the earliest mandatory stop strictly inside the proposed chunk.

## Estimated impact explanation
Scoped to Mamba align-mode models, but those workloads can gain or lose substantial reuse from chunk boundaries. Better choices reduce repeated prefill work and improve TTFT/TPOT for hybrid-Mamba agent turns.

## Evolve rationale
This heuristic trades immediate prefill throughput against future Mamba state reuse. Headroom exists in adaptive boundary selection using prompt length, connector activity, EAGLE status, or shared-prefix frequency. Correctness oracle: existing Mamba-align tests plus invariants that the returned token count is non-negative, no larger than the input chunk, and ends either at the original end or one of the enumerated legal boundaries.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Cache last-turn Mamba boundary decisions per request to skip re-scanning shared prefixes
- **Agent:** claude

**Detailed description.**

In Scheduler._mamba_block_aligned_split (vllm/v1/core/sched/scheduler.py:349-424), the routine recomputes the last cacheable block boundary, prompt-tail hash boundary, next block boundary, and shared-prefix boundary on every prefill chunk decision. In a multi-turn agentic workload the vast majority of the prompt is a stable shared prefix from prior turns whose boundary structure is invariant across chunks within a single scheduling pass over a request. Attach a small per-Request memoization on the request object (or in the Scheduler's per-request scheduling state) that caches (a) the highest already-computed cacheable-block boundary <= num_computed_tokens+chunk_end and (b) the shared-prefix boundary length, invalidated only when new blocks are cached or the shared-prefix set changes. Then _mamba_block_aligned_split becomes O(1) in the common path: it consults the cache, and only falls back to the full boundary scan when the chunk crosses beyond the cached horizon or the invalidation flag fires. Additionally, when the request is a continuation of a previous turn (num_computed_tokens > 0 at scheduling start and the prompt shares the previous turn's hash chain), skip the prompt-tail hash boundary computation entirely for the segment already covered by the previous turn's cached decision. This preserves all existing correctness invariants (returned token count non-negative, <= input chunk, ends at original end or one of the enumerated legal boundaries) because the cache only stores boundaries that the current code path would itself have chosen.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete proposal is novel by construction. Substantively, this proposal targets scheduler-side per-scheduling-pass overhead reduction via memoization of boundary computations across chunks of the same request in multi-turn workloads, rather than changing the boundary-selection policy itself. It is complementary to (and does not overlap with) heuristic-tuning ideas mentioned in the evolve_rationale such as adaptive selection using prompt length, connector activity, EAGLE status, or shared-prefix frequency.

---

### 2. Gate shared-prefix stops on actual reuse demand
- **Agent:** codex

**Detailed description.**

In `Scheduler._mamba_block_aligned_split` (`vllm/v1/core/sched/scheduler.py:349-424`), split the stop set into correctness-required boundaries (`next_block_boundary`, `last_cache_position`, and `tail_boundary`) versus the opportunistic Marconi `shared_prefix_boundary`. Keep the required stops exactly as-is, but only include the shared-prefix stop when there is evidence that another active or waiting request can reuse that boundary soon, or when the saved suffix is large enough to justify the extra scheduler step. For the multi-turn agentic case, this avoids clipping a request at a one-off shared-prefix junction that no sibling will consume, reducing avoidable prefill fragmentation and median TTFT/TPOT while preserving the Mamba alignment invariants for all mandatory cacheable-state boundaries. Add targeted tests that verify the shared-prefix boundary is skipped when it is the only stop and no reuse candidate is present, retained when a sibling/reuse signal exists, and never bypasses block, EAGLE, or partial-tail stops.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposed memoizing already-computed boundary decisions to reduce scheduler-side scanning overhead; this proposal changes the boundary-selection policy itself by treating shared-prefix clipping as conditional on expected reuse demand. It does not cache computations and targets a different cost: unnecessary extra prefill chunks when shared-prefix materialization has no near-term consumer.

---
