# disagg.PrefixBasedPDDecider non-cached-token gate

[← pkg_epp.framework.plugins.scheduling](../pkg_epp.framework.plugins.scheduling.md)

- **File:** [`pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go`](pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go) (lines 23–145)
- **Symbol:** `disagg.PrefixBasedPDDecider non-cached-token gate`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0005`

## Description
Prefix-based P/D decider that determines whether to run a separate prefill stage from the uncached suffix length.

## Current approach
NonCachedTokens is a static integer threshold. Disaggregation is disabled at 0; otherwise the decider estimates input token count, reads the decode endpoint's matched KV blocks, computes inputTokens-hitPrefixTokens, and runs prefill only when that uncached suffix meets the threshold.

## Estimated impact explanation
The decision trades dedicated prefill speed against KV transfer overhead. Multi-turn agentic prompts often sit near this boundary, so a better gate reduces median TTFT without sending short suffixes through expensive disaggregation.

## Evolve rationale
The binary static threshold is the owned admission policy for disaggregated prefill. Ratio-based gates, load-aware gates, transfer-cost-aware thresholds, or hysteresis can preserve deterministic bool output while changing the decision boundary. Correctness oracle: prefix_based_pd_decider_test.go covers disabled, below-threshold, above-threshold, missing data, and token-count behavior.

## Deep research proposals

### 1. Add a load-aware term to the disaggregation gate, co-optimizing prefix reuse and decode-endpoint load
- **Finding:** `find-0001` — *Preble: Efficient Distributed Prompt Scheduling for LLM Serving*
- **Source URL:** <https://proceedings.iclr.cc/paper_files/paper/2025/file/5bc342f48de8264779952fac378f96dc-Paper-Conference.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the gate at pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go:95-145 so the disaggregate decision is a function of both the uncached-suffix length and the candidate decode endpoint's current computation load, instead of only comparing nonCachedTokens to a static threshold. Concretely, after computing nonCachedTokens at line 133, also read a load signal already exposed by the endpoint (e.g., a queue/utilization/KV-pressure attribute via endpoint.Get analogous to how PrefixCacheMatchInfoDataKey is read at line 119) and combine the two into a single admission decision. Two preserving-bool variants worth prototyping: (a) a load-scaled threshold where the effective NonCachedTokens grows when the decode endpoint is lightly loaded (keep work co-located with the cached prefix) and shrinks when it is heavily loaded (offload prefill earlier to relieve the hot endpoint); (b) a small score `s = w_suffix * normalize(nonCachedTokens) - w_load * normalize(load)` admitted when s > 0. Keep the early-exit semantics of lines 101-116 unchanged, keep PrefixBasedPDDeciderConfig backward-compatible by adding optional weight/threshold fields with defaults that reduce to today's behavior, and extend prefix_based_pd_decider_test.go with cases that vary the load signal across the existing disabled / below-threshold / above-threshold / missing-data axes so the correctness oracle still gates the change.

**Proposal rationale.**

The candidate's evolve_rationale explicitly invites load-aware gates as a preserving-bool evolution of the static threshold, and Preble's E2 contribution is precisely the missing input: a scheduler that co-optimizes KV-prefix reuse with computation load-balancing. Today the decider sees one half of that picture (prefix match via PrefixCacheMatchInfo) and is blind to the other half (endpoint load), so under multi-turn agentic traffic with bursty hot prefixes the binary gate either keeps sending uncached suffixes to a decode endpoint that is already saturated or forces disaggregation when decode has spare capacity and the KV transfer cost dominates. Importing Preble's co-optimization principle into this gate addresses the impact lever called out in estimated_impact_explanation - the prompts sitting near the boundary - and targets the caller objective of reducing median TTFT/TPOT without changing the plugin's external contract.

---

### 2. Use request hints (expected output length, priority) to modulate the disaggregation threshold
- **Finding:** `find-0002` — *Agent Hints*
- **Source URL:** <https://docs.nvidia.com/dynamo/user-guides/agents/agent-hints>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend disagg.PrefixBasedPDDecider in pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go (lines 23-145) so that the static NonCachedTokens gate is replaced by an effective threshold derived from per-request hints, in the spirit of Dynamo agent_hints. Concretely, when the request carries a hint such as expected_output_length or priority (parsed from the body/header at request-control time and surfaced through the existing scheduling context the decider already reads from), compute effectiveThreshold = NonCachedTokens * f(hints): scale it down when expected_output_length is large (the prefill transfer cost amortizes over many decode tokens, so disaggregation pays off at shorter uncached suffixes), and up when expected_output_length is small or when priority indicates an interactive agent turn that should avoid the extra hop. Keep the bool return contract unchanged: ShouldDoPrefill still returns inputTokens - hitPrefixTokens >= effectiveThreshold, with the existing fallbacks for missing token counts and missing decode KV info. Preserve the disabled-at-zero shortcut. Update prefix_based_pd_decider_test.go's correctness oracle by adding cases where the same uncached suffix length produces different decisions under different hint values, while leaving the existing disabled / below / above / missing-data cases intact.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out transfer-cost-aware thresholds as a sanctioned evolution, and the finding contributes the concrete mechanism: a typed request-level hint surface (expected output length, priority) that scheduler-side components can consume. Expected output length is the missing variable in the current gate: a long uncached suffix is only worth disaggregating if the decode phase will be long enough to amortize the KV transfer, which is exactly what agent_hints expose. Priority maps to the multi-turn agentic workload in the caller context, where penalizing short interactive turns by routing them through prefill hurts median TTFT. The finding addresses a real gap (the gate is blind to per-request economics) rather than restating the existing static-threshold policy.

---

### 3. Add load-imbalance hysteresis gate to PrefixBasedPDDecider
- **Finding:** `find-0003` — *sglang/sgl-model-gateway/src/policies/cache_aware.rs at main · sgl-project/sglang · GitHub*
- **Source URL:** <https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Augment PrefixBasedPDDecider.disaggregate (pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go:95-145) so the binary disaggregation decision considers prefill-pool load in addition to the static NonCachedTokens threshold, mirroring SGLang's two-mode policy. Extend PrefixBasedPDDeciderConfig with hysteresis parameters (absolute load gap and relative imbalance ratio, both with sane defaults); plumb prefill-pool queue/load metrics through the existing endpoint or scheduling handle; and add a check after the current uncached-suffix computation: when the prefill pool is imbalanced beyond the configured absolute and relative thresholds, return false even if nonCachedTokens >= NonCachedTokens (route the request to decode-only / skip prefill); when the pool is balanced, fall back to the current prefix-based decision. Keep the function signature, default behavior (when load metrics are unavailable or hysteresis params are zero), and existing prefix_based_pd_decider_test.go cases unchanged so the disabled / below-threshold / above-threshold / missing-data oracles still pass; add new tests covering the imbalanced-but-above-threshold and balanced-but-above-threshold cases. The dual-threshold (absolute + relative) shape is taken directly from cache_aware.rs: absolute prevents thrashing at low load, relative scales with pool size.

**Proposal rationale.**

The candidate's evolve_rationale explicitly invites load-aware gates and hysteresis as evolutions of the static threshold, and the finding supplies a concrete, well-tested instantiation of exactly that pattern: switch strategies based on combined absolute + relative load-imbalance thresholds. For multi-turn agentic workloads (the stated workload), prefill affinity to specific pods is precisely the regime where cache-affinity creates queue hotspots that hurt TPOT, the failure mode SGLang's hysteresis was designed to avoid. Transferring the dual-threshold switch to the P/D decider preserves TTFT wins from prefix locality on balanced loads while shedding disaggregation overhead onto the decode pod when the prefill pool is hot, matching the caller's TTFT and TPOT objectives without changing the deterministic bool contract.

---

### 4. Add load-imbalance override to PrefixBasedPDDecider gate
- **Finding:** `find-0005` — *routingalgorithms package - github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms - Go Packages*
- **Source URL:** <https://pkg.go.dev/github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Augment disagg.PrefixBasedPDDecider in pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go (lines 23-145) so that the existing static NonCachedTokens threshold is wrapped by a load-aware override modeled on AIBrix's check_load_imbalance step. Before computing inputTokens-hitPrefixTokens against the threshold, sample running-request (or queued-token) counts across the candidate prefill and decode pools and compute their mean and standard deviation. When stddev/mean exceeds a configurable cap, fall back to a least-loaded behavior: short-circuit disaggregation off (route through decode-only) when the prefill pool is the imbalanced/overloaded side, and conversely lower the effective threshold (or force disagg=true above a smaller suffix) when the decode pool is overloaded relative to prefill. When load is balanced, retain the current uncached-suffix comparison verbatim. Keep the deterministic bool output and the existing disabled-at-0 behavior. Mirror the test structure in prefix_based_pd_decider_test.go to add cases for balanced load (current behavior preserved), prefill-side imbalance (disagg suppressed despite long suffix), and decode-side imbalance (disagg engaged below the static threshold).

**Proposal rationale.**

The candidate explicitly invites load-aware gates as an evolution of the static NonCachedTokens threshold, and AIBrix's documented algorithm contributes a concrete, transferable mechanism: a mean/stddev-based check_load_imbalance test that overrides prefix-driven routing when pools are unbalanced. Translating that override into the P/D decider directly addresses the median-TTFT objective for multi-turn agentic workloads, where prompts cluster near the threshold and dispatching to an overloaded prefill pool inflates queueing time more than the KV transfer it saves. The finding adds a specific, bounded statistic (stddev/mean cap) that is missing from the current binary policy without changing the decider's interface.

---

### 5. Augment PrefixBasedPDDecider with per-phase load awareness inspired by DistServe
- **Finding:** `find-0007` — *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*
- **Source URL:** <https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the gate at pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go:95-145 so the bool decision is not solely a static NonCachedTokens cutoff. Keep the existing prompt-shape signal (inputTokens - hitPrefixTokens) as the primary trigger, but combine it with a per-phase load signal observed via the endpoint attributes already available at line 119 (e.g., decode endpoint queue depth / running KV utilization, and a comparable signal for prefill-capable endpoints). Concretely: introduce optional configuration fields alongside NonCachedTokens (e.g., PrefillLoadCeiling, DecodeInterferenceFloor) and gate disaggregation only when the uncached suffix exceeds the threshold AND prefill pool load is below its ceiling AND decode pool load suggests prefill-decode interference would meaningfully hurt TPOT. The function still returns bool so the existing oracle in prefix_based_pd_decider_test.go remains valid; new branches require new test cases for the load-aware paths. NonCachedTokens=0 continues to disable; with the new fields unset, behavior is identical to today (backward compatible).

**Proposal rationale.**

DistServe's central claim is that prefill-decode co-location causes interference that hurts TTFT/TPOT, and that goodput-optimal serving co-optimizes phase-specific resources against SLOs. The candidate's evolve_rationale explicitly invites load-aware and transfer-cost-aware gates, and the workload (multi-turn agentic) frequently produces prompts hovering near the static NonCachedTokens boundary where the binary cutoff is least informative. DistServe motivates replacing the static cutoff with a decision that incorporates per-phase load and interference risk: when decode is congested, even moderately-sized uncached suffixes justify offloading prefill to cut TPOT; when prefill is saturated, short-suffix prompts already over the static threshold should stay co-located to avoid queueing and KV-transfer overhead. This directly addresses the candidate's identified gap (a single shape-only threshold) using a transferable insight from the finding (phase-aware, SLO-oriented scheduling).

---

### 6. Replace static non-cached-token gate with a transfer-cost / SLO-aware disaggregation score
- **Finding:** `find-0008` — *Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot*
- **Source URL:** <https://www.usenix.org/system/files/fast25-qin.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go (lines 95-145), evolve the binary `nonCachedTokens >= threshold` admission gate into a Mooncake-style decision that weighs the expected cost of disaggregated prefill against the cost of running prefill on the decode endpoint. Concretely, after computing `nonCachedTokens = inputTokens - hitPrefixTokens` (line 133), instead of comparing only against `d.config.NonCachedTokens`, compute a small benefit/cost expression: estimated prefill speedup on a dedicated prefill engine (~ proportional to nonCachedTokens) versus estimated KV-transfer overhead (~ proportional to hitPrefixTokens + nonCachedTokens, i.e., the KV blocks that must be moved to the decode endpoint) and any SLO headroom signal already exposed on the decode endpoint (e.g., queue depth / running requests via attributes already read on `endpoint`). Disaggregate only when expected_prefill_savings > expected_transfer_cost by a configurable margin, otherwise keep prefill local. The output stays a deterministic bool so the existing oracle in `prefix_based_pd_decider_test.go` still applies; new coefficients (transfer cost per token, SLO weight, hysteresis margin) are added to `PrefixBasedPDDeciderConfig` alongside the existing `NonCachedTokens` (which can be retained as a hard floor / disable flag at line 101). No new files or symbols outside this decider are introduced.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out transfer-cost-aware thresholds and SLO/load-aware gates as the targeted axis of change, and the Mooncake finding contributes exactly that idea: a KVCache-centric scheduler that scores prefill/decode placement jointly by transfer cost, cache residency, and SLO headroom rather than by a single suffix-length threshold. For the multi-turn agentic workload in the caller context, requests frequently sit near the static threshold with a large already-cached prefix on the decode side; under the current gate, even short uncached suffixes can trigger disaggregation whose KV-transfer cost erases the prefill speedup, hurting median TTFT. Importing Mooncake's cost-aware comparison into this decider preserves the existing bool contract and test oracle while changing the decision boundary in a way that is plausibly more accurate for this workload.

---

### 7. Make PrefixBasedPDDecider gate state-aware using decode-endpoint pressure
- **Finding:** `find-0010` — *SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling*
- **Source URL:** <https://proceedings.mlsys.org/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the binary gate in disaggregate() at pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go:95-145 so the threshold check at line 139 also incorporates SOLA-style state signals from the decode endpoint, while keeping the bool return contract intact. Concretely: (1) Read system-state attributes already exposed on the endpoint alongside the prefix-cache match info read at line 119 (e.g., running/queued decode requests, KV utilization) using the same endpoint.Get pattern. (2) Combine the existing nonCachedTokens >= threshold check with a decode-pressure modifier so the effective threshold is lowered when the chosen decode endpoint is under high load (route the suffix to dedicated prefill earlier to protect TTFT) and raised when the decode endpoint is idle (avoid paying KV-transfer cost for marginal suffixes). (3) Optionally read a request-level signal from the InferenceRequest (input token count is already computed at line 108; priority/context-length classification can be derived from it) so long contexts disaggregate more aggressively. Configuration grows from a single int to a small struct with a base NonCachedTokens plus optional pressure coefficients, defaulting to current behavior when coefficients are zero. Existing tests in prefix_based_pd_decider_test.go remain the correctness oracle for the disabled / missing-data / static-threshold cases; new tests cover high-pressure-lowers-threshold and low-pressure-raises-threshold.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out load-aware gates and transfer-cost-aware thresholds as evolutions of the static integer. SOLA's central claim is that scheduling decisions should react to both request-level state (context length, priority) and system-level state (decode pressure, cache pressure) rather than a fixed weight. The current decider already inspects per-endpoint state (prefix cache match info) but ignores live decode load when deciding whether disaggregation pays off, which is exactly the gap on multi-turn agentic workloads where suffixes sit near the boundary. Importing SOLA's state-aware framing into this gate plausibly reduces median TTFT (by routing to dedicated prefill earlier when decode is saturated) and median TPOT (by skipping disaggregation overhead when decode has headroom), matching the caller's stated objective.

---

### 8. Make disaggregation gate decode-load-aware to exploit chunked-prefill capacity
- **Finding:** `find-0011` — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- **Source URL:** <https://www.usenix.org/system/files/osdi24-agrawal.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In disaggregate() at pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go:95-145, replace the lone static NonCachedTokens comparison with a two-factor gate that combines the uncached suffix length with a decode-side stall signal read from the same endpoint. Concretely: extend PrefixBasedPDDeciderConfig with a second threshold (e.g. DecodeLoadFactor or ActiveDecodesThreshold) and, after computing nonCachedTokens at line 133, read the decode endpoint's active-decode-batch / queue-depth attribute (the same endpoint.Get() pattern already used for PrefixCacheMatchInfoDataKey) and only return true when nonCachedTokens >= NonCachedTokens AND the decode endpoint is currently saturated enough that injecting this prefill would stall ongoing decodes. When the decode endpoint has chunked-prefill capacity (low active-decode pressure), keep prefill local even for long uncached suffixes, since Sarathi-style stall-free scheduling can absorb it without inflating TPOT and avoids the KV-transfer cost of disaggregation. The bool contract and existing test cases (disabled, missing data, token-count) are preserved; new cases would cover the load-aware branch.

**Proposal rationale.**

The candidate's evolve_rationale explicitly invites load-aware gates as an alternative to a binary static threshold, and the caller's stated objective is reducing median TPOT under a multi-turn agentic workload. Sarathi-Serve's central finding is that long prefills are the dominant source of decode stalls (and therefore TPOT inflation), and that chunked-prefill + stall-free scheduling lets prefills coexist with decodes without pausing them. That directly reframes when disaggregation pays: dedicated prefill is worthwhile when the decode endpoint would otherwise stall on the prefill, and wasteful (KV-transfer overhead, no TTFT win) when the decode endpoint can interleave it. Gating on a decode-load signal alongside the existing suffix-length threshold transfers this insight into the candidate's exact decision boundary while keeping the deterministic bool output and existing oracle tests intact.

---

### 9. Gate disaggregation on decode endpoint's chunked-prefill capacity and token budget
- **Finding:** `find-0012` — *Optimization and Tuning — vLLM*
- **Source URL:** <https://docs.vllm.ai/en/v0.8.5/performance/optimization.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Augment disagg.PrefixBasedPDDecider in pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go (lines 23-145) so that the decision is not solely a static NonCachedTokens threshold on inputTokens-hitPrefixTokens. Inspect the decode endpoint's serving policy and current load: when the decode pod has chunked-prefill enabled with available token budget, it can absorb the uncached suffix into the next decode batch under vLLM's decode-prioritized policy, so the decider should raise the effective threshold (or return false) and skip disaggregation. When chunked prefill is disabled or the decode endpoint's pending-decode batch is already saturating the token budget, retain or lower the threshold to push prefill to a dedicated prefill endpoint. Concretely, extend the per-endpoint metrics consumed alongside matched KV blocks to include a chunked-prefill-enabled flag and a token-budget-headroom signal, and combine them with the existing uncached-suffix length to produce the boolean output. Keep the deterministic bool contract that prefix_based_pd_decider_test.go enforces; add cases that fix the new metric inputs.

**Proposal rationale.**

The candidate's evolve_rationale explicitly invites load-aware and transfer-cost-aware variants of the gate. vLLM's documented chunked-prefill policy (decode requests batched first, prefill chunks scheduled under remaining token budget) is the missing signal: a decode endpoint running this policy with budget headroom is cheap to co-locate prefill on, while one without it pays full prefill latency that disaggregation was designed to avoid. Folding that endpoint capability into the decider directly targets the multi-turn agentic workload hint, where suffix lengths sit near the boundary and median TTFT/TPOT are sensitive to whether short uncached suffixes are forced through KV transfer.

---

### 10. Tier-aware uncached-suffix gate using hierarchical KV cache locality
- **Finding:** `find-0016` — *SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends*
- **Source URL:** <https://www.lmsys.org/blog/2025-09-10-sglang-hicache/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Evolve the disagg.PrefixBasedPDDecider gate in pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go (lines 23-145) from a single static NonCachedTokens count into a transfer-cost-aware decision that weights the prefix hit by where the matched KV actually resides. Instead of computing a raw uncachedSuffix = inputTokens - hitPrefixTokens and comparing to one integer threshold, treat the prefix-hit signal as a tier vector (e.g. GPU-resident, CPU-resident, remote/disk). Effective uncached cost becomes uncachedSuffix + sum_i(transferCost_i * tokensInTier_i), and the decider runs disaggregated prefill only when that effective cost crosses a configurable threshold. Concretely: extend the input the decider reads from the decode endpoint so that matched KV blocks are reported per tier (today only a flat hitPrefixTokens count is consumed), keep the bool output unchanged so the existing prefix_based_pd_decider_test.go contract still applies, and add per-tier weight knobs alongside the existing NonCachedTokens field. Sessions whose prefixes are hot in the decode endpoint's GPU stay co-located even with a moderately long suffix; sessions whose prefix has spilled to a slower tier disaggregate sooner because the effective uncached work is larger.

**Proposal rationale.**

The candidate's evolve_rationale explicitly lists transfer-cost-aware thresholds as an in-bounds evolution of this gate, and the finding supplies the missing concrete signal: HiCache's cache controller treats KV cache as a hierarchy across GPU/CPU/disk/remote tiers where hits at different tiers carry materially different transfer costs. Today the decider conflates all prefix hits, so a long-prefix multi-turn agent whose KV has been demoted to CPU/remote looks identical to one still warm in GPU, even though the disaggregation tradeoff is very different. Weighting prefix hits by tier directly addresses that gap, preserves the deterministic bool contract covered by the existing test file, and targets the multi-turn agentic workload in the caller context where prefixes routinely sit near the disagg boundary and tier placement dominates median TTFT.

---

## Agent proposals

### 1. Self-tuning NonCachedTokens via online outcome feedback (EWMA bandit)
- **Agent:** claude

**Detailed description.**

Replace the static d.config.NonCachedTokens at pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go:139 with an adaptively-tuned effective threshold that the decider learns online from the actual outcomes of past decisions, while keeping the bool contract and the disabled-at-0 short-circuit at lines 101-104 unchanged. Concretely, treat NonCachedTokens as the seed/center of a bounded interval [Tmin, Tmax] (configurable; defaults reduce to today's behavior). For each call where ShouldDoPrefill returns, record (nonCachedTokens, decision) on the InferenceRequest's scheduling context so a downstream observer (the same package's request lifecycle that already sees TTFT and prefill timing for the resulting placement) can publish an outcome event back to the decider: observed TTFT and an attribution of prefill latency vs. KV-transfer latency. The decider keeps a small per-model (or per-(model, decode-pool) keyed) EWMA of two quantities at the threshold boundary: (a) mean TTFT for disagg=true on requests with nonCachedTokens in a narrow band around the current threshold, and (b) mean TTFT for disagg=false in the same band. After enough samples in a band, it nudges the effective threshold up or down by a bounded step (e.g., epsilon-greedy explores +/-step every N decisions) toward the regime that minimizes observed TTFT, clipped to [Tmin, Tmax] and gated on a minimum sample count to avoid early thrash. The output of disaggregate() remains a deterministic bool given current state, but state is updated out-of-band on outcome arrival rather than on each decision. Existing tests in prefix_based_pd_decider_test.go remain valid since with no outcome feedback the effective threshold equals the seed; new tests inject synthetic outcomes and assert that the threshold drifts toward the better arm. Configuration grows by an EWMA half-life, exploration rate, and bounds, all with defaults that disable adaptation when zero.

**Novelty rationale.**

Every listed deep_research_proposal (find-0001, -0002, -0003, -0005, -0007, -0008, -0010, -0011, -0012, -0016) chooses a different *instantaneous* signal to combine with the static threshold: decode/prefill pool load, request hints, mean/stddev imbalance, phase-specific load, transfer cost vs. suffix length, decode chunked-prefill capacity, or KV cache tier. None of them use observed *outcomes* (post-hoc TTFT/prefill-attributed latency) to close a feedback loop on the threshold itself. This proposal is orthogonal: it does not introduce a new instantaneous signal but instead lets the threshold self-calibrate from realized latency, which can compose with any of the existing proposals later. It also targets a failure mode none of them address - operator mis-tuning of the static seed - by replacing tuning-by-hand with bounded online adaptation while preserving the deterministic bool contract and existing test oracle.

---

### 2. Add session-sticky deadband around the token threshold
- **Agent:** codex

**Detailed description.**

Extend `PrefixBasedPDDeciderConfig` in `pkg/epp/framework/plugins/scheduling/profilehandler/disagg/prefix_based_pd_decider.go` with optional `decisionDeadbandTokens` and bounded `stickySessionCacheSize` fields. With both unset, keep today's `nonCachedTokens >= NonCachedTokens` behavior. When enabled, after line 133 computes `nonCachedTokens`, apply a Schmitt-trigger-style band around `NonCachedTokens`: values below `NonCachedTokens-decisionDeadbandTokens` return false, values above `NonCachedTokens+decisionDeadbandTokens` return true, and values inside the band reuse the last decision for the same session/model/decode endpoint. Key the small LRU from `session.ReadSessionID(request)` when available, or a non-default `request.FairnessID`; do not apply stickiness when no real session key exists. Add tests showing existing cases unchanged, same-session decisions are reused inside the band, different sessions do not inherit state, and values outside the band override stale state.

**Novelty rationale.**

The listed deep-research proposals change the instantaneous decision inputs: load, request hints, load imbalance, transfer cost, chunked-prefill capacity, or KV tier. The one hysteresis proposal is load-imbalance hysteresis, not per-session routing hysteresis around the token boundary. Agent A's proposal adapts the global threshold from observed TTFT outcomes; it does not preserve routing continuity for a single multi-turn agent session. This proposal targets boundary flapping caused by repeated agent turns hovering near `NonCachedTokens`, using existing session identity surfaces rather than new load, cost, hint, tier, or outcome signals.

---
