# Scheduler.schedule (RUNNING-phase loop)

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 388–556)
- **Symbol:** `Scheduler.schedule (RUNNING-phase loop)`
- **Kind:** loop
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0001`

## Description
Per-step loop over self.running that computes each request's num_new_tokens, clamps it by long_prefill_token_threshold, token_budget, and max_model_len, schedules encoder inputs, applies Mamba block alignment, calls allocate_slots, and preempts another running request on allocation failure.

## Current approach
FIFO traversal with greedy budget consumption. FCFS preemption pops the tail of self.running; PRIORITY preemption scans self.running with max(key=(priority, arrival_time)) and removes the victim. The split of token budget across running requests is an implicit side effect of traversal order.

## Estimated impact explanation
This loop decides which active requests decode on every step. Better budget and preemption heuristics can reduce decode stalls, preemption-rate spikes, and TPOT tail latency in high-concurrency agentic workloads.

## Evolve rationale
This loop is the decode-side scheduler policy for every engine step. Multi-turn agentic traffic has many short decode steps, so victim selection, traversal order, and the spec/lookahead clamp directly control preemption churn and median TPOT. Headroom exists in smarter victim scoring, explicit running-vs-waiting budget reservations, and adaptive lookahead/chunk clamps. Correctness oracles include tests/v1/core/test_scheduler.py, tests/v1/core/test_priority_scheduler_random.py, and tests/v1/core/test_prefix_caching.py; SchedulerStats exposes preemption signals.

## Deep research proposals

### 1. Order RUNNING-phase traversal and preemption victim selection by decode-deadline laxity
- **Finding:** `find-0006` — *Optimal Scheduling Algorithms for LLM Inference: Theory and Practice*
- **Source URL:** <https://arxiv.org/abs/2508.01002>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/core/sched/scheduler.py:388-556, augment the RUNNING-phase loop with a per-request laxity score derived from a target TBT/TPOT deadline minus elapsed time since the request's last produced decode token. Maintain a last-decode timestamp (or step index) on each Request and a configurable per-request TBT target. Before iterating self.running for token-budget allocation, compute laxity for every running request and visit them in ascending laxity order (most-urgent-first) so requests at risk of missing their TBT deadline get their num_new_tokens out of token_budget before slack requests do; this changes only traversal order, leaving the long_prefill_token_threshold/token_budget/max_model_len clamps, encoder scheduling, Mamba block alignment, and allocate_slots calls unchanged. Apply the same laxity signal to victim selection on allocate_slots failure: replace the FCFS tail-pop with 'pop the running request with the largest laxity' (i.e., farthest from its TBT deadline), and in PRIORITY mode break ties on (priority, arrival_time) using laxity so the most-slack running request is preempted. Keep the laxity computation behind a scheduler-config flag so the existing FIFO/priority oracles (tests/v1/core/test_scheduler.py, tests/v1/core/test_priority_scheduler_random.py, tests/v1/core/test_prefix_caching.py) remain the default and continue to pass.

**Proposal rationale.**

The RUNNING-phase loop currently has no signal about which decode is closest to stalling: token_budget is split by FIFO order, and preemption victims are chosen by tail position or static (priority, arrival_time) - neither reflects time-since-last-token, which is exactly what determines TPOT/TBT tail latency. SLAI's evidence that real-time decode deadline measurements (TBT-aware prioritization) reduce decode-deadline misses transfers directly: laxity-ordered traversal favors requests about to violate TPOT, and laxity-based victim selection avoids preempting a request that is already close to its deadline. For the stated multi-turn agentic objective (median TPOT, median TTFT), this targets the candidate's identified headroom in 'smarter victim scoring' and 'explicit running-vs-waiting budget reservations' without redesigning allocate_slots or the encoder/Mamba paths.

---

### 2. Reserve decode budget before prefill chunks in RUNNING loop to enforce stall-free scheduling
- **Finding:** `find-0007` — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- **Source URL:** <https://arxiv.org/abs/2403.02310>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify the RUNNING-phase loop in vllm/v1/core/sched/scheduler.py:388-556 so that token-budget consumption is split into two explicit passes over self.running, mirroring Sarathi-Serve's stall-free schedule construction. Pass 1 walks self.running and admits the per-request decode tokens (typically num_new_tokens == 1, plus speculative/lookahead tokens) for every request whose unfinished work is purely decode, deducting from token_budget first. Pass 2 then walks the remaining requests (those still in chunked-prefill state, i.e. num_computed_tokens < num_prompt_tokens) and admits prefill chunks sized adaptively from the residual budget instead of the static long_prefill_token_threshold: chunk_size = min(long_prefill_token_threshold, remaining_token_budget // max(1, num_prefill_running), max_model_len_headroom). The Mamba alignment, encoder-input scheduling, and allocate_slots / preemption paths remain in pass 2; preemption on allocation failure still falls back to the existing FCFS-tail / PRIORITY-max victim selection. Token accounting (token_budget) and the construction of scheduled_running_reqs / num_scheduled_tokens stay structurally the same — only the order in which budget is consumed changes, plus the chunk-size formula. No public API changes; correctness is gated by tests/v1/core/test_scheduler.py, test_priority_scheduler_random.py, and test_prefix_caching.py, and SchedulerStats already exposes preemption counters needed to confirm the change does not regress preemption rate.

**Proposal rationale.**

Today the RUNNING loop consumes token_budget in a single FIFO pass, so a request that happens to be early in self.running and is mid-chunked-prefill can absorb most of the step's budget and starve later decodes — exactly the head-of-line stall Sarathi-Serve identifies. The candidate's evolve_rationale explicitly calls out 'explicit running-vs-waiting budget reservations' and 'adaptive lookahead/chunk clamps' as headroom, and the caller objective is median TPOT under multi-turn agentic traffic where short decodes are frequent and sensitive to any prefill-induced stall. Sarathi-Serve's stall-free schedule — admit decodes first, then size prefill chunks from leftover budget — is a concrete, transferable rule that directly attacks this gap without changing preemption semantics or admission control elsewhere in the scheduler. The finding contributes the specific construction (decode-first two-pass, residual-budget chunk sizing) rather than a generic 'do better' observation, which is what makes it actionable on this exact loop.

---

### 3. Retention-priority-aware victim selection in RUNNING-phase preemption
- **Finding:** `find-0008` — *Introducing New KV Cache Reuse Optimizations in NVIDIA TensorRT-LLM*
- **Source URL:** <https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Augment the preemption path inside Scheduler.schedule's RUNNING-phase loop (vllm/v1/core/sched/scheduler.py:388-556) so that, when allocate_slots fails and a victim must be chosen, the victim score incorporates a retention-priority signal modeled on TensorRT-LLM's priority-based KV eviction. Concretely: (1) attach a lightweight retention hint to each Request capturing the expected reuse value of its prefix (e.g. system-prompt block range, agent role prefix, media-derived prefix, or 'active multi-turn session') with an optional TTL; (2) in FCFS mode, instead of unconditionally popping the tail of self.running, pop the lowest-retention tail-region candidate (fall back to tail when ties); (3) in PRIORITY mode, change the max(key=...) selector from (priority, arrival_time) to (priority, -retention_score, arrival_time) so high-retention requests are preempted last; (4) thread the same retention hint into the existing block-pool eviction order so freed blocks from low-retention prefixes are reused before high-retention ones. The hint is metadata-only and defaults to neutral, preserving today's behavior when callers don't set it. Validation extends tests/v1/core/test_scheduler.py and tests/v1/core/test_priority_scheduler_random.py with cases where retention-tagged requests survive allocation pressure, and uses SchedulerStats preemption counters to confirm reduced churn on multi-turn agentic traces.

**Proposal rationale.**

The candidate's preemption step is the precise place where vLLM decides whose KV cache to free, and its current victim rules (FCFS tail / PRIORITY by (priority, arrival_time)) are blind to how costly recomputing a victim's prefix will be. The finding contributes a concrete, transferable idea: let upstream knowledge about prefix reuse (system prompts, agent role prompts, media prefixes, active sessions) bias eviction away from blocks likely to be hit again. In a multi-turn agentic workload that's exactly the gap — preempting a long shared system-prompt-bearing request inflates median TPOT when it's later resumed and its prefix must be repopulated, while preempting a short single-turn tail is nearly free. Importing TensorRT-LLM's retention-priority concept at the request/victim-selection layer (rather than only at block-pool eviction) directly targets the candidate's headroom around 'smarter victim scoring' called out in evolve_rationale.

---

## Agent proposals

### 1. Adapt per-request speculative lookahead clamp from rolling acceptance rate before token-budget allocation
- **Agent:** claude

**Detailed description.**

In the RUNNING-phase loop of Scheduler.schedule (vllm/v1/core/sched/scheduler.py:388-556), make the speculative/lookahead component of num_new_tokens vary per-request based on its observed speculation acceptance rate, instead of using a single global num_lookahead_slots/num_speculative_tokens. Concretely: (1) maintain a rolling acceptance-rate estimator on each Request (EMA of accepted draft tokens / proposed draft tokens, updated when the engine records spec results — the signal already exists in spec_decode bookkeeping); (2) just before computing num_new_tokens for a running request, derive an effective_lookahead = round(base_lookahead * f(ema_acceptance)) where f is monotone increasing and bounded (e.g. clamp to [0, base_lookahead], with f(0)=0 and f(1)=1, and a small floor like 1 to allow recovery once a request "warms up"); (3) use effective_lookahead in the existing num_new_tokens computation that feeds the long_prefill_token_threshold / token_budget / max_model_len clamps and allocate_slots — no change to encoder scheduling, Mamba alignment, or the preemption fallback. This directly shrinks budget consumed by requests whose drafts are routinely rejected (so their speculation was net waste of token_budget against other running decodes) and preserves it for requests where drafts pay off. Default behavior (acceptance unknown / spec-decode disabled) reduces to current static lookahead, keeping tests/v1/core/test_scheduler.py, test_priority_scheduler_random.py, and test_prefix_caching.py oracles intact; new coverage can assert that a request stuck with 0% acceptance stops consuming lookahead budget and that one with 100% acceptance retains the full clamp. SchedulerStats can be extended with mean effective_lookahead to confirm the adaptation is happening without regressing preemption-rate counters.

**Novelty rationale.**

None of the three existing deep_research_proposals touch the speculative/lookahead component of num_new_tokens. find-0006 reorders traversal by TBT laxity, find-0007 splits the loop into decode-first then adaptive prefill chunk sizing (its chunk_size formula targets prefill chunks, not the per-request lookahead clamp), and find-0008 modifies victim selection with a retention-priority signal. The candidate's evolve_rationale explicitly lists "adaptive lookahead/chunk clamps" as headroom; the chunk side is partly addressed by find-0007, but the lookahead/spec side — which can dominate token_budget when speculation is enabled and acceptance is variable across multi-turn agent turns — is untouched. Tying the clamp to a per-request rolling acceptance rate (rather than a global static number) is a distinct, transferable mechanism with its own oracle (acceptance EMA) and is orthogonal to laxity ordering, decode-first passes, and retention-aware preemption.

---

### 2. Retry allocation after reclaiming skipped KV blocks before preemption
- **Agent:** codex

**Detailed description.**

In `vllm/v1/core/sched/scheduler.py:388-556`, add a one-shot reclaim path inside the RUNNING-phase `allocate_slots` failure branch, before FCFS/PRIORITY victim selection. Today `allocate_slots()` reclaims skipped sliding-window/local-attention blocks only for the request being allocated; other RUNNING requests may already have obsolete KV blocks that will never be attended to but are not freed until they are visited. On allocation failure, run a helper over non-inflight RUNNING requests, excluding requests with async output placeholders, and call `kv_cache_manager.remove_skipped_blocks(req.request_id, req.num_computed_tokens)`. If the free-block count increases, retry the same `allocate_slots(...)` once with unchanged `num_new_tokens`, lookahead, encoder scheduling, and Mamba split. Only fall through to the existing preemption code if the retry still fails. Add a scheduler test with a sliding-window KV setup where reclaimable skipped blocks avoid a preemption, and verify `preempted_req_ids` / SchedulerStats preemption counters stay at zero for that step.

**Novelty rationale.**

This is not covered by find-0006, which changes traversal and victim choice by decode-deadline laxity; find-0007, which reserves token budget for decodes and sizes prefill chunks from residual token budget; or find-0008, which changes victim scoring with retention priority. It also does not overlap with Agent A's adaptive speculative lookahead clamp. The proposal leaves ordering, token-budget policy, lookahead sizing, and victim selection unchanged; it adds a memory-reclamation retry that can avoid preemption when KV pressure is caused by stale skipped blocks rather than truly live cache demand.

---
