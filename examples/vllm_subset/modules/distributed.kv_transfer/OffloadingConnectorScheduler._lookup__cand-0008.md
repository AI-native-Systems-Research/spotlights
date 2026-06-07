# OffloadingConnectorScheduler._lookup

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`](vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) (lines 305–441)
- **Symbol:** `OffloadingConnectorScheduler._lookup`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0008`

## Description
Scheduler-side offload lookup convergence loop that determines how many tokens beyond the GPU prefix cache can be loaded from the offloaded cache.

## Current approach
The method scans full-attention groups with prefix lookup and sliding-window groups with suffix lookup, slices `offload_keys` per group, may re-run group scans when a tighter hit size invalidates earlier results, and can return `None` to defer scheduling when the offload manager lookup is pending or when needed blocks are already being loaded.

## Estimated impact explanation
This method runs during every new-request scheduling decision with offloading enabled. It directly moves median TTFT by deciding whether a multi-turn prefix waits for offload lookup/load, uses already-loading blocks, or recomputes locally.

## Evolve rationale
The ordering, slicing, re-scan, defer, and `_blocks_being_loaded` delay rules are owned scheduling heuristics. They can be evolved with memoized group results, batched manager lookups, early convergence checks, or smarter duplicate-load suppression. Correctness oracle: for a fixed `RequestOffloadState` and mocked `OffloadingManager.lookup`, the returned token count or `None` must match the intended hit/miss/defer semantics; `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` exercises prefix, sliding-window, defer, and concurrent lookup cases.

## Deep research proposals

### 1. Partially admit requests during offload lookup/load instead of full defer
- **Finding:** `find-0005` — *Disaggregated Serving — TensorRT LLM*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0rc4/features/disagg-serving.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py at OffloadingConnectorScheduler._lookup (lines 305-441), change the two defer paths — (a) when OffloadingManager.lookup is still pending and (b) when needed blocks are in `_blocks_being_loaded` — so they no longer always return `None`. Instead, return the largest token count that is already satisfiable from the GPU prefix cache (and from any offload blocks whose loads have already completed), letting the engine begin prefill/decoding for that prefix while the outstanding offload transfer continues asynchronously. Reserve a `None` defer only for the case where even the GPU-cached prefix is shorter than what a recompute would be (i.e., overlap is strictly worse). For the `_blocks_being_loaded` case specifically, track per-block load-completion futures and admit up to the last contiguous completed boundary rather than blocking the whole request. Keep the existing per-group ordering/slicing/re-scan convergence logic intact; this change only adjusts the terminal action when the loop would otherwise return `None`.

**Proposal rationale.**

The finding's core technique is overlapping KV-cache transmission with computation so that in-flight transfers do not stall forward progress. The candidate's current defer semantics (returning `None` while offload lookup is pending or while blocks are still loading) is precisely the stall this overlap pattern targets: the request waits an entire scheduling step rather than making partial progress on a prefix that is already resident on GPU. Because the candidate sits on every new-request scheduling decision and directly gates median TTFT in multi-turn agentic workloads (where most prefixes already hit the GPU prefix cache), converting `None` defers into partial admissions plausibly reduces TTFT without changing the correctness oracle for the hit/miss path — defers become a strict subset, gated by an explicit overlap-vs-recompute check. This is a concrete, transferable mechanism rather than a restatement of the current heuristic, which today only chooses between full admit and full defer.

---

### 2. Coalesce concurrent offload loads instead of deferring duplicates
- **Finding:** `find-0008` — *Disaggregated Serving | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify `OffloadingConnectorScheduler._lookup` (vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:305-441), specifically the `_blocks_being_loaded` branch at lines 408-432, so that when a new request's hit range overlaps blocks already being loaded by an in-flight transfer, the scheduler does NOT return `None`. Instead, treat the overlapping blocks as an effective hit and let the request be admitted with a readiness handle tied to the existing transfer's completion (e.g., reuse / track the per-block in-flight future rather than blocking the entire scheduling decision). Concretely: (1) change `_blocks_being_loaded` from a set into a mapping `block_key -> in_flight_load_id` (or extend it to expose readiness); (2) in `_lookup`, when overlap is detected, count those blocks toward `num_hit_tokens` and record their pending load IDs on the `RequestOffloadState` so `get_num_new_matched_tokens` / connector load issuance can subscribe to the existing transfer instead of issuing a duplicate copy; (3) only fall back to the current `return None` deferral for cases where overlap is partial in a way the connector cannot represent (e.g., sliding-window groups whose suffix-window slice would still need a fresh lookup). The first defer case (`defer_lookup` from a pending manager lookup at lines 401-406) is preserved unchanged because it represents missing information, not a duplicate transfer.

**Proposal rationale.**

The candidate's second defer branch synchronously delays admission of any request that shares a prefix with an in-flight load — directly hurting median TTFT in multi-turn agentic workloads, where successive turns by the same session share long prefixes and are precisely the requests most likely to collide on `_blocks_being_loaded`. Dynamo's design point that 'KV transfer is non-blocking, allowing GPU forward passes to continue serving other requests during the transfer' transfers cleanly to this scheduling heuristic: rather than waiting for the in-flight transfer to finish before this request can even be scheduled, the scheduler can admit the request now, treat the in-flight blocks as effectively-arriving hits, and let GPU work proceed in parallel with the (already paid) transfer. This addresses a concrete, identifiable gap (duplicate-load suppression already named in the candidate's evolve_rationale) with a transferable mechanism (extend transfer/readiness metadata to support earlier admission) drawn from the finding.

---

### 3. Chunk-aligned incremental hit reporting in offload lookup
- **Finding:** `find-0022` — *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*
- **Source URL:** <https://www.microsoft.com/en-us/research/publication/sarathi-efficient-llm-inference-by-piggybacking-decodes-with-chunked-prefills/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify OffloadingConnectorScheduler._lookup in vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py (lines 305-441) so the convergence loop returns chunk-aligned hit counts incrementally rather than a single all-or-nothing token count. Concretely: (1) align the prefix/suffix scan boundaries to a fixed chunk size (e.g. block-multiple chunks), (2) when the OffloadingManager.lookup result for an early chunk is ready while later chunks are still pending, allow the scheduler to commit the early chunks for loading and continue the convergence loop on the tail rather than returning None (defer) for the entire request, (3) keep the existing 'blocks already being loaded' suppression but apply it per-chunk so newly-confirmed chunks start streaming immediately, (4) preserve the current correctness oracle by ensuring the cumulative committed hit equals what the existing loop would have produced once all chunks resolve. Re-scan logic when a tighter hit invalidates earlier results becomes per-chunk truncation rather than full re-scan. The signature change is internal: the public return remains a token count / None at the request level, but the method internally batches chunk-level decisions.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out batched manager lookups, smarter defer rules, and duplicate-load suppression as evolvable heuristics, and its estimated_impact identifies median TTFT on multi-turn prefixes as the primary lever. SARATHI's core insight - that splitting a long prefill into uniform chunks lets downstream work start before the whole prefill is done - transfers cleanly to the lookup/load pipeline: today _lookup can defer (return None) when manager state is pending, which stalls TTFT for the entire long prefix; chunk-aligned incremental commits let the front of a multi-turn prefix begin loading from offload while the tail's lookup still converges. This addresses the specific TTFT-on-long-prefix gap without changing the correctness contract exercised by tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py.

---

## Agent proposals

### 1. Memoize and batch OffloadingManager.lookup across convergence iterations
- **Agent:** claude

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py, modify OffloadingConnectorScheduler._lookup (lines 305-441) and its helpers _maximal_prefix_lookup (lines 244-261) and _sliding_window_lookup (lines 263-287) so that all per-block backend probes are issued at most once per scheduling decision. Concretely: (1) before entering the `while lookup_groups` loop at line 324, build the union of candidate offload keys across every group's currently-feasible slice (`offload_keys[start_block_idx : cdiv(max_hit_size_tokens, offloaded_block_size)]`) and issue a single batched `self.manager.lookup(keys, req_context)` call (extending the OffloadingManager interface with a vectorized lookup that returns one bool/None per key, falling back to per-key calls when a backend hasn't implemented it); (2) cache the resolved {hit | miss | pending} result per OffloadKey in a dict scoped to this `_lookup` call, and have _maximal_prefix_lookup / _sliding_window_lookup consult that dict instead of calling `self.manager.lookup` directly; (3) when the convergence loop re-runs all groups (the `defer_lookup` re-scan at line 391-392 and the sliding-window re-scan at line 396), reuse the cached results so each rescan iteration is pure index arithmetic with zero backend round-trips; (4) only issue an additional backend probe if a re-scan tightening exposes a key range not covered by the initial batched call (rare — only happens when sliding-window suffix slicing references a key beyond the original union). The public return contract (token count or None for hit/miss/defer) is unchanged, so the existing test_scheduler.py oracle continues to apply.

**Novelty rationale.**

The three existing deep_research_proposals all change the *semantics* of the terminal action when the loop would have stalled: find-0005 converts None defers into partial admissions, find-0008 coalesces concurrent loads by treating in-flight blocks as hits, and find-0022 chunk-aligns incremental hit commits. None of them touch the cost of the convergence loop itself. This proposal leaves admit/defer semantics identical and instead attacks a different bottleneck explicitly named in the candidate's evolve_rationale ('memoized group results, batched manager lookups'): the per-block, per-iteration backend probes performed by _maximal_prefix_lookup and _sliding_window_lookup, which are repeated wholesale every time `lookup_groups = self._lookup_groups` triggers a re-scan. By batching the union of candidate keys up-front and memoizing within a single _lookup call, the scheduler avoids the O(num_groups × num_blocks × num_rescans) backend calls that lengthen each scheduling step's critical path — directly compressing the synchronous portion of TTFT in multi-turn agentic workloads where long shared prefixes maximize both block count and re-scan probability. This is orthogonal to and stackable with all three existing proposals.

---

### 2. Return concrete sliding-window misses when pending keys are irrelevant
- **Agent:** codex

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`, tighten the `_lookup` pending-defer behavior by changing `_sliding_window_lookup` to compute whether `None` results can actually change the returned suffix hit boundary. Materialize the same right-to-left lookup outcomes for the sliced `offload_keys`, then compute two local scan results: one treating pending entries as misses and one treating them as hits. Return the concrete block count when both scans agree, and return `None` only when a pending key can change the suffix-window end index or prefix-fallback count. This keeps `_lookup`'s scalar return contract unchanged but avoids deferring a request when pending sliding-window probes are separated by known misses and cannot form a usable window. Add focused tests beside `TestSlidingWindowLookup`, e.g. a window size of 2 over `[miss, pending, miss]` returns `0` instead of `None`, while existing cases where pending can extend or create the rightmost window still defer.

**Novelty rationale.**

This is not the partial-admission idea in find-0005, the in-flight-load coalescing idea in find-0008, or the chunk-level incremental reporting in find-0022: it never admits a partial uncertain prefix and does not add readiness handles or chunk commits. It also does not overlap agent A's batching/memoization proposal, which reduces backend round-trips but preserves the current interpretation of `None`. This proposal changes only the exactness of the sliding-window lookup oracle so `_lookup` returns `None` for pending manager state only when that pending state can affect the final hit count.

---
