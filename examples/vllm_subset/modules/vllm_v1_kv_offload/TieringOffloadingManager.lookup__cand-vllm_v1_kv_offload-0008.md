# TieringOffloadingManager.lookup

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/tiering/manager.py`](vllm/v1/kv_offload/tiering/manager.py) (lines 311–380)
- **Symbol:** `TieringOffloadingManager.lookup`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_kv_offload-0008`

## Description
Per-block tiered lookup checks the primary tier, then scans secondary tiers and initiates promotion on the first secondary hit.

## Current approach
Processes finished jobs once per step, probes the primary tier, then iterates secondary_tiers in configuration order with load-tier filters. It short-circuits on the first HIT and tracks any RETRY, with no hit-rate-aware ordering or batched secondary probing.

## Estimated impact explanation
Secondary lookup latency contributes directly to TTFT before promotions can be scheduled. Reordering or batching probes can reduce synchronous lookup delay for multi-turn agentic traffic that frequently hits non-primary tiers.

## Evolve rationale
Secondary tier probe order, retry handling, and per-block versus batched lookup are routing heuristics with clear latency signals. Correctness oracle: LookupResult semantics, existing tiering tests, and preservation of the set of promotions as keys proven present in an allowed tier.

## Deep research proposals

### 1. Batch secondary-tier lookups per request and offload probing to a background I/O path
- **Finding:** `find-vllm_v1_kv_offload-0001` — *Serving Agentic Workloads at Scale with vLLM x Mooncake*
- **Source URL:** <https://vllm.ai/blog/2026-05-06-mooncake-store>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor TieringOffloadingManager.lookup (vllm/v1/kv_offload/tiering/manager.py:311-380) so that secondary-tier probing is performed in batches for a request's block set rather than one key at a time on the critical path. Introduce a batched entry point (e.g., `lookup_batch(keys, req_context)`) that groups the request's remaining block hashes and, for each secondary tier, issues a single multi-key probe (falling back to a loop internally for tiers that lack native batch support). Move the synchronous probe work off the scheduler-visible path by delegating tier probes to a dedicated background I/O worker/thread that returns futures, mirroring Mooncake's separation of scheduler-side lookup from worker-side asynchronous data movement. The primary-tier check and the LookupResult semantics (HIT / HIT_PENDING / RETRY / MISS) for each key remain unchanged; only the aggregation and dispatch of secondary probes change. The set of promotions initiated on a HIT in an allowed tier must be preserved exactly to satisfy the correctness oracle listed in evolve_rationale.

**Proposal rationale.**

The candidate's current_approach explicitly notes per-block sequential probing of secondary_tiers with no batched secondary probing, and estimated_impact_explanation identifies synchronous secondary-lookup latency as a direct contributor to TTFT for multi-turn agentic traffic — exactly the workload class the Mooncake post targets. The finding contributes two concrete, transferable ideas to this gap: (1) scheduler-side block lookup performed as a batched operation over a request's blocks, and (2) running the I/O-bound probe/transfer work on a dedicated background thread so it does not block the scheduler. Applying these to `lookup` addresses the identified latency signal without changing LookupResult semantics or the promotion set, which is the stated correctness constraint.

---

### 2. Batch SSD-tier lookups into object-granularity probes with async promotions
- **Finding:** `find-vllm_v1_kv_offload-0006` — *Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving*
- **Source URL:** <https://arxiv.org/abs/2605.03375>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Change TieringOffloadingManager.lookup (vllm/v1/kv_offload/tiering/manager.py:311-380) from a strictly per-block secondary-tier scan into a medium-aware path that coalesces contiguous block lookups against SSD-backed secondary tiers into a single object-granularity probe. Concretely: (1) Introduce a `SecondaryTierManager` capability flag indicating object-granularity lookup/promotion (populated for SSD tiers). (2) At the caller/tier-orchestration level, group consecutive same-request blocks whose keys map to the same SSD object/chunk and issue one batched probe per object rather than N per-block `tier.lookup` calls in the loop at lines 356-373. (3) On a batched HIT, initiate a single asynchronous chunked promotion covering all blocks in the object (analogue of the existing `_initiate_promotion`, but keyed on the object) so the per-block promotion attempts do not each independently contend for primary capacity. (4) Preserve current DRAM/local secondary-tier semantics unchanged: the per-block path at lines 356-373 remains for tiers without the object-granularity capability, and the ordering (primary first, then secondary_tiers in configuration order) is unchanged. (5) Keep LookupResult contract intact — batched HIT returning RETRY (promotion started) or MISS (primary full) for every block in the object, consistent with the existing return semantics.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names 'per-block versus batched lookup' as a routing heuristic with clear latency signals, and the lookup loop at lines 356-373 does exactly one probe and (on hit) one promotion per block. Tutti's central claim is that for SSD-backed KV caches this per-block pattern degenerates into fragmented tiny random I/O and GPU stalls, and that larger KV-cache object abstractions with asynchronous bulk transfers reduce that overhead. Because secondary-lookup latency is accumulated synchronously into `sync_lookup_delay` (line 375) and contributes directly to TTFT before promotions can even be scheduled, collapsing per-block SSD probes/promotions into per-object ones addresses the caller's TTFT objective on the exact code path the candidate identifies. The proposal is medium-gated so DRAM/local tiers keep today's fine-grained semantics, avoiding regressions on tiers where per-block is already appropriate.

---

### 3. Add anticipatory secondary-to-primary promotion to hide lookup-path latency
- **Finding:** `find-vllm_v1_kv_offload-0011` — *ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs*
- **Source URL:** <https://www.usenix.org/conference/osdi26/presentation/liu-guangda>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend TieringOffloadingManager.lookup (vllm/v1/kv_offload/tiering/manager.py:311-380) with an ECHO-style anticipatory promotion path so secondary-to-primary loads can start before a block becomes the synchronous critical-path miss. Concretely: when lookup() serves a primary HIT for block N of a request, opportunistically peek ahead at the request's next K block keys (already known from the request's block hash sequence) and, for any that would currently miss the primary tier but HIT in an allowed secondary tier, call the existing _initiate_promotion() path to enqueue their promotion asynchronously. The synchronous lookup contract (HIT / HIT_PENDING / RETRY / MISS) and the set of keys that eventually get promoted are unchanged — the only difference is that by the time the request's per-block lookup reaches those keys, the primary tier probe already returns HIT or HIT_PENDING rather than driving a fresh secondary probe + promotion. Gate the look-ahead by primary-tier free capacity and by _transfer_jobs pressure to preserve the current 'primary full → MISS' semantics, and reuse load_tier_filter.allows() so the prefetch respects the same tier policy the reactive path uses. Keep the per-block iteration order in secondary_tiers unchanged; the win comes from moving the promotion earlier in wall-clock time, not from reordering tiers.

**Proposal rationale.**

The candidate's current lookup is strictly demand-driven: a secondary-tier HIT only starts a promotion at the moment the request has already stalled on that block, so secondary-probe latency and the promotion transfer time both sit on the TTFT/TPOT critical path — exactly the exposed stall the evolve_rationale flags. ECHO's contribution is the observation that in LLM serving many future KV accesses are predictable (intra-query for decoding, inter-query for prefill), which lets a scheduler overlap recall with other in-flight work. Multi-turn agentic traffic in the caller context has strong inter-turn reuse of prefix blocks, so the request's upcoming block-key sequence is a cheap, high-quality predictor. Wiring that predictor into lookup() to pre-issue promotions through the existing _initiate_promotion path addresses the specific gap called out in current_approach (no anticipatory scheduling, per-block probing only) while leaving LookupResult semantics and existing tiering tests as the correctness oracle.

---

## Agent proposals

### 1. Adaptive tier-order permutation via online hit-rate bandit in lookup()
- **Agent:** claude

**Detailed description.**

Modify TieringOffloadingManager.lookup (vllm/v1/kv_offload/tiering/manager.py:311-380) so the secondary-tier probe order is not fixed to configuration order but is dynamically permuted per-workload based on observed per-tier hit rate and per-tier probe latency. Concretely: (1) Add lightweight per-tier counters (hits, misses, cumulative probe latency in nanoseconds, EWMA-decayed with a decay factor per N lookups) maintained inside TieringOffloadingManager. Update them at the existing HIT/MISS/RETRY branches inside the loop at lines 356-373 without changing LookupResult semantics. (2) Compute an expected-cost score per tier as `expected_latency = probe_latency_ewma + (1 - hit_rate_ewma) * next_tier_expected_cost`, i.e. a simple recursion over the current permutation, and periodically (every K lookups, e.g. K=256) re-sort a cached `_ordered_secondary_tiers` list by ascending expected cost. Only tiers already allowed by the current `load_tier_filter` participate in the ordering; the filter still gates which tiers are probed for a given block. (3) Use an epsilon-greedy schedule (small epsilon, e.g. 0.05) that occasionally probes tiers in configuration order to keep counters fresh for tiers that would otherwise be starved once reordered to the back. (4) Preserve the exact HIT / HIT_PENDING / RETRY / MISS contract and preserve which tier ultimately gets a promotion — reordering only affects the order in which the loop encounters HITs, not the set of tiers eligible for promotion for a given block. Since the loop short-circuits on the first HIT, the promoted tier is still the highest-priority allowed tier that has the block, given the new permutation. (5) Expose the ordering as an observable metric so operators can see which tier the manager has learned to probe first for the current workload.

**Novelty rationale.**

The three listed deep_research_proposals all preserve the fixed configuration-order iteration of secondary_tiers (Finding-0001 explicitly batches across a request's blocks but keeps the per-tier loop; Finding-0006 coalesces per-block probes into per-object probes for SSD tiers but does not touch tier order; Finding-0011 pre-issues promotions along the request's future block sequence but explicitly states 'Keep the per-block iteration order in secondary_tiers unchanged'). The evolve_rationale explicitly names 'secondary tier probe order' and 'hit-rate-aware ordering' as unaddressed routing heuristics — none of the existing proposals target order-of-probes as the optimization axis. An online, per-workload hit-rate/latency bandit over tier permutation is orthogonal to batching, medium-aware coalescing, and anticipatory prefetch, and would compose with any of them. For multi-turn agentic workloads where prefix reuse concentrates hits in a specific tier (e.g. a warm DRAM secondary or a session-affine RDMA tier), learning to probe that tier first reduces mean synchronous lookup delay (the `sync_lookup_delay` accumulated at line 375) by up to (N-1)/N of a probe's latency when the winning tier is currently last in configuration order — a distinct latency lever from the three existing proposals.

---

### 2. Add retry-state memoization to avoid repeated secondary rescans
- **Agent:** codex

**Detailed description.**

Change `TieringOffloadingManager.lookup` in `vllm/v1/kv_offload/tiering/manager.py:311-380` to remember per-request/per-key secondary tiers that returned `LookupResult.RETRY` and avoid re-running the full secondary-tier loop on every scheduler retry while that tier is still busy. Concretely: add a small `_secondary_lookup_retries` map keyed by `(req_context.req_id, key)` that stores the retrying tier(s), the applicable load-tier filter identity or allowed `(medium, locality)` set, and a monotonic timestamp. When lookup sees a primary MISS and finds a live retry record, first re-probe only those retrying tiers that are still allowed and not `exclude_tier`; if one HITs, call the existing `_initiate_promotion()` path, and if it still RETRYs, return `RETRY` without scanning unrelated lower-priority tiers. Fall back to the normal full scan when the retry record is absent, stale, disallowed by the current filter, or the retrying tier reports MISS. Clear entries when a promotion is initiated, when the request finishes, and on `reset_cache`; also expire them after a short bounded TTL to preserve correctness if a tier’s RETRY was transient. This preserves the `HIT` / `HIT_PENDING` / `RETRY` / `MISS` contract while reducing repeated synchronous secondary probes for the same blocked block across scheduler iterations.

**Novelty rationale.**

The deep research proposals cover request-level batching/background probing, SSD object-granularity probing/promotion, and anticipatory promotion of future blocks. Agent A covers adaptive ordering based on hit rate and latency. This proposal targets the separate retry-handling gap called out in the candidate: repeated `RETRY` outcomes currently only set `any_retry` for the current call and are forgotten, so the next lookup can pay the same secondary scan again before reaching the same busy tier. Memoizing retry state is not batching, not prefetching, not medium-specific coalescing, and not reordering tiers by learned hit rate; it reduces TTFT/TPOT stalls by eliminating redundant synchronous rescans while a known secondary lookup/operation is still unresolved.

---
