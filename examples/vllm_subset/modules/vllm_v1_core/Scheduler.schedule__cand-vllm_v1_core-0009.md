# Scheduler.schedule

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 426–1248)
- **Symbol:** `Scheduler.schedule`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_core-0009`

## Description
Main scheduler step that schedules running requests, admits waiting requests, handles preemption, prefix-cache lookup, connector loads, encoder budgets, Mamba alignment, and SchedulerOutput construction.

## Current approach
Large Python method with separate running and waiting loops, many per-request branches, inline admission/preemption bookkeeping, and a PRIORITY preemption path that uses max(self.running) plus self.running.index on every allocation failure.

## Estimated impact explanation
Every TPOT measurement includes this method. At high concurrency, Python scheduler overhead can dominate the CPU side of a decode step, so per-request reductions move median TPOT directly.

## Evolve rationale
Engine per-step scheduling critical path. Headroom in preemption-victim data structures, batched slot allocation, caching per-request scheduling decisions within a step, and separating cold connector paths from hot decode paths. Correctness oracle: tests/v1/core/sched scheduler suites for FCFS/PRIORITY, preemption, chunked prefill, async KV loads, spec decode, and prefix caching; SchedulerOutput equivalence for the same request stream is the core invariant.

## Deep research proposals

### 1. Reorder waiting queue by cached-prefix length before waiting loop in Scheduler.schedule
- **Finding:** `find-vllm_v1_core-0002` — *[RFC]: Cache-affinity-aware request ordering for the V1 scheduler*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/42185>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the waiting-queue admission phase of Scheduler.schedule (vllm/v1/core/sched/scheduler.py, roughly lines 678-1017) so that, before entering the `while (self.waiting or self.skipped_waiting) and token_budget > 0:` loop, the scheduler performs a cache-affinity pre-pass over the current waiting queue and reorders it by cached-prefix length. Concretely: (1) For each waiting request, cheaply estimate the cached-prefix hit length by querying `self.kv_cache_manager.get_computed_blocks` (or a lookup-only variant that does not touch refcounts) — reusing the same code path already used inline in the loop (lines ~744-761) — so the estimate reflects the actual local prefix cache. (2) Bucket requests by hit length (e.g., in block-sized buckets) rather than sorting on raw token counts, so small variations do not thrash ordering. (3) Preserve priority classes: only reorder within a single SchedulingPolicy priority tier (never elevate a lower-priority request above a higher-priority one when policy is PRIORITY); under FCFS, keep arrival-time as the tie-breaker inside each cache-length bucket. (4) Add sticky handling for previously preempted requests and for entries in `self.skipped_waiting`, so their positions are not disturbed by cache-affinity reordering (they already carry earned state or blocked status via `_is_blocked_waiting_status`). (5) Enforce a starvation deadline: any request whose head-of-queue wait exceeds a configurable threshold bypasses the cache-affinity reordering and is admitted in original arrival order. (6) Cache the per-request hit length inside a per-step dict keyed by request_id so the subsequent inline `get_computed_blocks` / connector lookup inside the loop reuses the value instead of redoing it. The change confines itself to the waiting-queue admission block of Scheduler.schedule and to the `create_request_queue` / peek/pop machinery already used there; the running-request loop, preemption path, encoder/mamba/spec-decode branches, and SchedulerOutput construction remain unchanged.

**Proposal rationale.**

The candidate explicitly identifies prefix-cache lookup as part of the waiting-loop hot path and lists median TTFT and multi-turn agentic workloads as the caller objective/workload hint — exactly the regime where cache-affinity ordering pays off. The finding contributes a concrete, transferable mechanism (bucketed reordering by cached-prefix length with priority preservation, sticky preempted handling, and a starvation deadline) that directly targets the same code region (Scheduler.schedule waiting loop) without altering the running-loop, preemption, encoder, or SchedulerOutput semantics. It addresses a gap the current implementation has: FCFS/PRIORITY peek order is oblivious to which waiting continuations still have warm KV blocks, so multi-turn continuations can be admitted after cold prefills have already evicted their prefix blocks. Because the same `get_computed_blocks` call the scheduler will make inline anyway is reused for the pre-pass and its result cached, the pre-pass is close to zero net cost while giving the admission phase a meaningfully better order for the stated workload. Correctness invariants named in evolve_rationale (priority-class order, preemption/spec/prefix-cache behavior, SchedulerOutput equivalence for the same stream) are preserved by the priority-tier restriction, the sticky handling for preempted/skipped requests, and the starvation deadline.

---

### 2. Bias prefix-cache eviction using conversational-continuation signals for multi-turn scheduling
- **Finding:** `find-vllm_v1_core-0005` — *Learned Prefix Caching for Efficient LLM Inference*
- **Source URL:** <https://papers.neurips.cc/paper_files/paper/2025/hash/414f642a1ea9350006669774cba9bcd4-Abstract-Conference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/core/sched/scheduler.py Scheduler.schedule (lines 426-1248), the prefix-cache lookup and connector-load path currently relies on the underlying KV-cache manager's default (LRU-style) eviction when prefix blocks must be reclaimed for admitted waiting requests. Extend the scheduler's interaction with the prefix cache so that, when a finished or paused request's blocks are returned to the free pool, the scheduler attaches a lightweight 'continuation likelihood' hint derived from cheap conversational-content features (e.g., request metadata already available on Request: number of prior turns / prompt-token count, presence of an assistant turn tail, time since last activity via the arrival/finish timestamps the scheduler already tracks, and whether the request belongs to a session/conversation id if exposed). The KV-cache manager's eviction victim selection is then biased by this score in addition to recency, so that blocks belonging to paused multi-turn sessions predicted to continue are retained longer than blocks from one-shot completions of similar age. Concretely: (1) compute the score inline in the finished-request cleanup branch of schedule() before releasing blocks, (2) pass it through to KVCacheManager.free()/touch() as an eviction-priority hint, (3) have the block pool use score-then-LRU ordering when selecting victims. No change to SchedulerOutput shape; correctness oracle in tests/v1/core/sched is preserved because behavior is identical when scores are equal (falls back to LRU).

**Proposal rationale.**

The candidate explicitly handles prefix-cache lookup for admitted waiting requests on the hot path, and the caller context targets a multi-turn agentic workload optimizing median TTFT. Prefix-cache hits are the single largest TTFT lever for multi-turn traffic: on a miss the waiting-loop path pays full prompt prefill, on a hit it is skipped. Pure LRU eviction is a known weak point exactly here — a paused conversational turn looks 'old' between user messages even though it is highly likely to resume, so LRU evicts precisely the blocks that would have produced the next hit. The finding provides a concrete, transferable mechanism (lightweight conversational features + last-access to predict continuation, bias eviction) that plugs into the existing prefix-cache/connector interaction the scheduler already orchestrates, without changing scheduler semantics for FCFS/PRIORITY, preemption, chunked prefill, or spec decode. It does not restate the current approach (which is recency-only) and it targets a measurable gap (multi-turn cache-hit rate → TTFT).

---

### 3. Adopt Sarathi decode-maximal batching in Scheduler.schedule
- **Finding:** `find-vllm_v1_core-0011` — *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*
- **Source URL:** <https://arxiv.gg/abs/2308.16369>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Restructure Scheduler.schedule (vllm/v1/core/sched/scheduler.py:426-1248) so each iteration is composed as: (1) at most one bounded prefill chunk drawn from the waiting queue, then (2) fill the remaining token budget with decodes from self.running. Concretely, split the current interleaved running/waiting loops into a decode-first pass that admits every eligible running request under the per-step token budget, followed by a single prefill-chunk admission step whose chunk size is capped by a Sarathi-style uniform chunk target (derived from long_prefill_token_threshold / max_num_batched_tokens and the number of decodes already scheduled). Preserve existing prefix-cache lookup, connector load, encoder budget, and Mamba alignment logic, but move the prefill-chunk sizing decision into a small helper that enforces the uniform-work invariant per step. Keep FCFS/PRIORITY ordering intact for waiting-queue selection; only the batch composition policy changes.

**Proposal rationale.**

The candidate's evolve_rationale explicitly targets TPOT under high concurrency, and the caller objective is median TTFT + TPOT on a multi-turn agentic workload where decode steps dominate but new turns constantly inject prefills. Sarathi's decode-maximal + bounded-chunk policy directly addresses TPOT spikes caused by large prefill chunks stealing decode slots, while the uniform-chunk cap bounds worst-case TTFT contribution per step. vLLM already has chunked-prefill machinery in this method, so the finding contributes a concrete composition policy (one prefill chunk + max decodes, uniform chunk size) rather than new infrastructure, making it a transferable, low-risk tuning change for this exact scheduler path.

---

### 4. Emit upcoming KV block prefetch plan from Scheduler.schedule for L2 prefetching
- **Finding:** `find-vllm_v1_core-0013` — *Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching*
- **Source URL:** <https://ojs.aaai.org/index.php/AAAI/article/view/39224>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend Scheduler.schedule in vllm/v1/core/sched/scheduler.py (lines 426-1248) to produce, alongside the existing SchedulerOutput, a lightweight per-request prefetch plan: an ordered list of upcoming KV block IDs (and their access order) that the runtime will read during the next one or two attention steps. The scheduler already knows, per running request, the current block table and the number of tokens to be computed this step (num_scheduled_tokens, chunked-prefill boundaries, and speculative-decode draft length); it can trivially derive the next K block IDs each running request will touch. Add a new field on SchedulerOutput (e.g. kv_prefetch_plan: dict[req_id, list[block_id]]) populated in the running-loop section that already walks self.running and computes token counts, so no extra pass over requests is needed. For waiting/admitted requests, emit the first few prompt blocks from the just-allocated block table. Keep the plan bounded (e.g. next N=1-2 blocks per request) to cap Python overhead, and make emission gated by a config flag so the hot path is unchanged when the runtime side is not compiled in. The kernel-side consumer of this plan is out of scope for this candidate; the scheduler change is purely additive metadata construction inside the existing per-request loop.

**Proposal rationale.**

The finding's central requirement on the scheduler side is exactly a stable, ordered list of upcoming KV block IDs so the runtime can issue asynchronous L2 prefetches during compute. Scheduler.schedule is the natural and only place in vllm/v1/core with authoritative per-step knowledge of (a) which requests will run, (b) their current block tables via KVCacheManager, and (c) how many tokens (and thus how many blocks) each will consume this step, including chunked prefill and spec-decode drafts. The candidate description explicitly calls out that the running/waiting loops already visit every scheduled request and construct SchedulerOutput, so attaching a bounded prefetch plan there is O(1) extra work per request and does not disturb the FCFS/PRIORITY correctness invariants tested by tests/v1/core/sched. For the multi-turn agentic workload in the caller context, decode-heavy steps have many small block reads per step, which is precisely where L2-resident KV prefetching can shave median TPOT without changing any scheduling decisions.

---

## Agent proposals

### 1. Maintain a priority-ordered index over self.running for O(log n) PRIORITY preemption victim selection in Scheduler.schedule
- **Agent:** claude

**Detailed description.**

In vllm/v1/core/sched/scheduler.py Scheduler.schedule (lines 426-1248), replace the O(n) PRIORITY preemption victim selection at lines ~577-590 — `preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))` followed by `victim_index = self.running.index(preempted_req)` — with an incrementally-maintained ordered index over the running set. Concretely: (1) Add a per-scheduler auxiliary structure `self._running_priority_index` (a heap keyed by `(-priority, -arrival_time, insertion_counter)` mapped to request, with a lazy-deletion set of tombstoned request_ids, or equivalently a `SortedList` from `sortedcontainers` keyed on the same tuple) that mirrors `self.running`. It is populated only when `self.policy == SchedulingPolicy.PRIORITY`; FCFS is unaffected. (2) Update the index at every mutation site of `self.running`: `self.running.append` (running-loop admission at line ~1050 and waiting-loop admission), `del self.running[victim_index]` inside the preemption block, `self.running.pop()` in the FCFS branch, and any request-finish/removal path that already exists (`update_from_output` finished handling, `_preempt_request` companions). Also maintain a `dict[str, int]` mapping `request_id -> current index in self.running` so `victim_index` retrieval after selection is O(1) rather than O(n) via `list.index`; keep it in sync on append/pop/del (a small `_running_id_to_index` map with recomputation on the tail region after in-place delete, or a doubly-linked-list backing — pick whichever costs less given the observed running-length distribution). (3) In the running-loop preemption branch, replace `max(...)` with a single peek at the top of the priority index, and lazily discard tombstoned entries; replace `self.running.index(preempted_req)` with the `_running_id_to_index` lookup. (4) Enforce equivalence with the existing tiebreaker: `max` with `key=(r.priority, r.arrival_time)` picks highest `(priority, arrival_time)` — the index must produce exactly the same request under ties (same policy semantics as `max` for stable ties: the last-encountered element wins), so serialize the tiebreaker with an insertion counter so the reconstructed order is deterministic and matches current behavior for the test oracles in tests/v1/core/sched. (5) Confine the change to the PRIORITY policy code paths and the small set of `self.running` mutation sites; leave FCFS `self.running.pop()` unchanged, and do not alter SchedulerOutput shape, waiting-queue ordering, chunked-prefill, spec-decode, or connector/encoder paths. Under a preemption cascade in a single `allocate_slots` failure loop, this converts the current O(n) per-failure step into O(log n), and in the pathological case of k preemptions per admission it converts O(k * n) into O(k * log n).

**Novelty rationale.**

The candidate's `evolve_rationale` explicitly names the PRIORITY preemption path — `max(self.running) plus self.running.index on every allocation failure` — as headroom, and lists 'preemption-victim data structures' as a targeted lever. None of the four existing deep_research_proposals touches this: find-0002 reorders the WAITING queue by cached-prefix length (admission side, not running-set data structure); find-0005 biases KV cache eviction with a continuation-likelihood hint (KV manager internals, not `self.running` structure); find-0011 restructures batch composition into a Sarathi decode-maximal policy (loop composition, uses the existing `max/index` calls unchanged); find-0013 emits an additive KV block prefetch plan (metadata-only, no mutation-path change). This proposal is orthogonal to all four and directly attacks the O(n) → O(log n) opportunity called out in the candidate itself, preserving the exact tiebreaker semantics needed to keep SchedulerOutput equivalent on the tests/v1/core/sched suites.

---
