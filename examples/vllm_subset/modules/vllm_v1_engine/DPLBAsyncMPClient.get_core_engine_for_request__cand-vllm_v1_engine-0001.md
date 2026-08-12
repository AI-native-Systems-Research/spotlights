# DPLBAsyncMPClient.get_core_engine_for_request

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/core_client.py`](vllm/v1/engine/core_client.py) (lines 1471–1522)
- **Symbol:** `DPLBAsyncMPClient.get_core_engine_for_request`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_engine-0001`

## Description
Per-request data-parallel load balancer that selects one engine core from the locally managed DP engines.

## Current approach
Scans every engine on each request, scoring each as max(client_count * local_inflight, waiting + running) plus waiting * 6.0 * max(0, kv_cache_usage - 0.5) when there are queued requests. Ties are broken by a rotating scan start, and the stats snapshot can be stale between coordinator updates.

## Estimated impact explanation
Routing choice directly controls per-engine queue depth and KV pressure; under multi-turn agentic bursts, avoiding stale or overloaded cores can lower median TTFT without changing model kernels.

## Evolve rationale
This is a request-admission hot path with an explicit TODO for P2C at larger DP sizes and several owned policy knobs: scan-vs-sample selection, stale-snapshot handling, local-inflight weighting, the 0.5 KV-pressure knee, the 6.0 slope, and tie-breaking. Correctness oracle: tests/v1/engine/test_engine_core_client.py can assert requests route only to admissible non-shutdown engines, data_parallel_rank overrides are honored, and bursts spread without losing requests.

## Deep research proposals

### 1. Adopt Power-of-Two Choices with weighted least-request for DP admission scoring
- **Finding:** `find-vllm_v1_engine-0003` — *Supported load balancers*
- **Source URL:** <https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/load_balancers.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the full O(N) engine scan in DPLBAsyncMPClient.get_core_engine_for_request (vllm/v1/engine/core_client.py:1471-1522) with a hybrid selection policy: retain the existing full scan for small DP sizes (e.g., N below a threshold like 8), and switch to Power-of-Two Choices (P2C) for larger DP deployments, matching the existing TODO. In the P2C path, sample two admissible non-shutdown engines uniformly at random from the local DP set, score each using an active-request bias formula inspired by Envoy's weighted least-request variant, and pick the lower-scoring engine. Replace the hand-tuned fixed slope (waiting * 6.0 * max(0, kv_cache_usage - 0.5)) with an active-request-weighted form aligned with Envoy's approach, where the KV-pressure term acts as a weight modifier over active_requests = waiting + running rather than as a separately calibrated linear penalty. Preserve current tie-breaking semantics (rotating scan start) only in the full-scan branch; in the P2C branch, resample on ties. Keep the data_parallel_rank override path unchanged, and continue to skip shutdown engines. Update tests/v1/engine/test_engine_core_client.py assertions to cover: (1) P2C activates above the threshold and full scan below, (2) requests never route to shutdown engines, (3) data_parallel_rank overrides remain honored, and (4) bursts of requests spread across engines without loss.

**Proposal rationale.**

The candidate explicitly flags a TODO for P2C at larger DP sizes and calls out several owned policy knobs (scan-vs-sample selection, the 0.5 KV knee, the 6.0 slope, tie-breaking) that lack principled grounding. The Envoy load balancers doc directly describes P2C as an O(1) algorithm picking the host with fewest active requests, and its weighted least-request section gives a concrete active-request bias formula. This maps cleanly onto the candidate's gap: it reduces the per-request scan cost at high DP (helping TTFT under multi-turn agentic bursts where admission latency matters), and replaces hand-tuned slopes with a documented weighted formulation, without changing model kernels or the coordinator snapshot mechanism. The finding is transferable because DP engines are the exact analog of Envoy's upstream hosts, and active-request count maps onto (waiting + running) already tracked in the stats snapshot.

---

### 2. Add prefix-reuse-aware multiplicative routing to DPLB engine selection
- **Finding:** `find-vllm_v1_engine-0004` — *[Feature] RFC: Paper Reproduction of LMetric Multiplication Scheduling in SGLang Gateway*
- **Source URL:** <https://github.com/sgl-project/sglang/issues/29573>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/engine/core_client.py:1471-1522 (DPLBAsyncMPClient.get_core_engine_for_request), replace the purely additive load score with a multiplicative composition that jointly reflects estimated new prefill work after prefix reuse and current engine load, in the shape of SGLang's LMetric formula: score = new_prefill_work_estimate(engine, request) * (load_estimate(engine) + 1), pick the argmin, keep the existing rotating tie-break and in-flight bookkeeping. Concretely: (1) Compute a per-engine new_prefill_work_estimate for the incoming request. Reuse the request's token count as the upper bound and subtract a per-engine expected prefix-cache hit length; a cheap first cut is a shared token-prefix affinity signal (e.g. hash of the first K tokens routed to the same engine within a short TTL window, as the DP LB already owns per-engine bookkeeping via engine_inflight and reqs_in_flight). When no affinity information exists, degrade gracefully to request.num_prompt_tokens for every engine, making the multiplicative term a constant that cancels out and preserving today's load-only behavior. (2) Keep the current load_estimate as it is defined at lines 1487-1504: max(client_count * inflight, waiting + running), still additively lifted by the waiting * 6.0 * max(0, kv_cache_usage - 0.5) KV-pressure penalty; feed that value as load_estimate(engine) into the product (with the +1 to avoid zero-load collapse to a single winner). (3) Retain the eng_start_index rotation for ties and the local waiting += client_count increment so bursts still spread between coordinator snapshots. Gate the new prefill-work term behind a config flag (e.g. VLLM_DP_LB_PREFIX_AWARE) so operators can fall back to the current pure-load policy, and keep tests/v1/engine/test_engine_core_client.py's admissibility and data_parallel_rank-override invariants intact.

**Proposal rationale.**

The candidate's evolve_rationale explicitly lists the scan-vs-sample choice, the 0.5 KV knee, the 6.0 slope, and tie-breaking as owned knobs, and the caller objective is reducing median TTFT/TPOT under a multi-turn agentic workload — exactly the regime where repeated prefixes make prefix-cache locality a first-order TTFT lever. The current policy has no per-engine prefix-reuse signal at all: two engines with identical (waiting, running, inflight, kv_cache_usage) score identically even if one already has the request's conversation prefix cached, so bursts of multi-turn requests spread round-robin and pay full prefill cost on cold engines. SGLang's LMetric formula supplies a concrete, well-scoped transfer: a single multiplicative score that trades cache locality against load without a brittle threshold switch, which is a strictly richer generalization of the additive load-only score in this candidate. Multiplication makes the two signals scale together (a lightly loaded engine with a strong prefix hit dominates over a similarly loaded engine with a cold cache, and vice versa) instead of adding fixed weights that require re-tuning per DP size and workload. The proposal preserves the candidate's admissibility, rank-override, and burst-spreading invariants because the multiplicative form collapses to today's load-only ordering whenever the prefix-work estimate is uniform across engines.

---

### 3. Add prefix-aware three-tier DP routing (affinity → cache-util → P2C) in DPLBAsyncMPClient
- **Finding:** `find-vllm_v1_engine-0005` — *Prefix-aware routing*
- **Source URL:** <https://docs.ray.io/en/latest/serve/llm/user-guides/prefix-aware-routing.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend DPLBAsyncMPClient.get_core_engine_for_request in vllm/v1/engine/core_client.py (lines 1471-1522) to use a three-tier routing decision inspired by Ray Serve's prefix-aware routing: (1) maintain an approximate prefix tree/trie on the client that maps recent request prompt prefixes (hashed n-grams or token-block hashes computed from EngineCoreRequest.prompt_token_ids) to the engine(s) that most recently served matching prefixes, with per-entry TTL and bounded size; (2) at admission, compute a match score against the trie — if a strong prefix match exists AND queue imbalance across engines is small (e.g., max(waiting+running) - min(waiting+running) below a threshold derived from client_count), pick the affine engine to exploit KV prefix-cache reuse; (3) else, if the match is weak, fall back to a low-cache-utilization tier that picks the engine with the smallest kv_cache_usage from the lb_engines snapshot; (4) else, under load imbalance, apply Power-of-Two-Choices (P2C) using the same score function currently in place (max(client_count * inflight, waiting + running) plus the 6.0 * max(0, kv_cache_usage - 0.5) * waiting term), which also resolves the existing 'TODO use P2C alg for larger DP sizes'. Preserve the existing invariants: honor request.data_parallel_rank overrides and get_late_interaction_engine_index() first (unchanged), keep the local engine_inflight floor and the +client_count local-waiting bump on the chosen engine, and continue rotating eng_start_index for tie-breaking within the P2C tier. Add a small router-side prefix bookkeeping structure on DPLBAsyncMPClient (initialized alongside reqs_in_flight/engine_inflight) and update it on admission and on finished_requests (via process_engine_outputs) so evicted sessions release affinity. Expose the imbalance threshold, the prefix-match strength cutoff, and the trie size/TTL as configurable knobs with conservative defaults that reduce to today's behavior when the trie is empty.

**Proposal rationale.**

The candidate is exactly the DP request-admission hot path the finding targets: it already scans engines and scores them by queue/KV pressure, and it explicitly flags P2C as future work. The Ray Serve prefix-aware routing technique adds a concrete, transferable idea the current code lacks — routing to the engine likely to have the relevant KV prefix cached — while still deferring to load-based routing under imbalance. Under the stated caller objective (reduce median TTFT/TPOT) and the multi-turn agentic workload hint, shared system prompts and session history make prefix affinity a direct lever on TTFT: hitting a cached prefix skips prefill work on the chosen engine. Layering affinity above the existing score (rather than replacing it) preserves the correctness oracle in tests/v1/engine/test_engine_core_client.py (admissible non-shutdown engines only, data_parallel_rank overrides honored, bursts spread without loss), because tiers 2 and 3 fall back to today's behavior whenever prefix signal is weak or engines are imbalanced. It also converts the existing P2C TODO into the imbalance-tier of the same policy, addressing an owned knob the evolve_rationale already calls out.

---

### 4. Score DP engines on separate prefill and decode token dimensions
- **Finding:** `find-vllm_v1_engine-0006` — *Scheduler*
- **Source URL:** <https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/03-scheduler.md>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/engine/core_client.py DPLBAsyncMPClient.get_core_engine_for_request (lines 1471-1522), replace the single-scalar engine score with a two-dimensional shape-aware score that separates input/prefill load from output/decode load. Concretely: (1) extend the coordinator stats payload consumed at lines 1394-1404 so each entry in self.lb_engines carries prefill-side load (waiting requests plus queued/uncomputed prompt tokens or a running prefill indicator) and decode-side load (running requests currently in the decode phase, optionally weighted by kv_cache_usage) alongside the existing kv_cache_usage; (2) in the scan at lines 1483-1507, compute score_prefill = max(client_count * local_inflight_prefill, waiting + prefill_running) and score_decode = max(client_count * local_inflight_decode, decode_running), then combine as score = max(score_prefill, alpha * score_decode) (alpha tuned so a decode-saturated engine still repels new prefills without over-penalizing engines that are simply serving healthy steady-state decodes); (3) retain the current KV-pressure penalty on waiting (line 1504) but apply it only to score_prefill since that is what stalls TTFT; (4) keep the rotating tie-break at line 1516 unchanged. Local per-engine inflight tracking (self.engine_inflight, engine_inflight) is updated at request admission (line 1521) and completion (process_engine_outputs, line 1542) — split it into two counters keyed by whether the request is still in its prefill phase, transitioning on the first decode output observed in process_engine_outputs.

**Proposal rationale.**

The finding's supporting evidence explicitly frames shape-aware DP routing as balancing input/prefill and output/decode token load as separate dimensions rather than a single scalar. The candidate's current_approach collapses both into `waiting + running` plus a KV-pressure term, so an engine loaded with long decodes is scored identically to one with a fat prefill queue. Under the caller's multi-turn agentic workload, requests alternate between short prompts and multi-token decodes; a shape-aware score lets new admissions avoid prefill-saturated cores (lowering median TTFT) and decode-saturated cores (lowering median TPOT) without changing model kernels. This directly addresses the candidate's `evolve_rationale` gap on the local-inflight weighting and the 0.5/6.0 KV knee knobs by giving those knobs distinct roles per phase. The LPM-fallback half of the finding is not applied because this candidate is a client-side load balancer, not a scheduler with prefix-priority computation.

---

### 5. Smooth per-engine load with short/long EWMA divergence to detect queueing trend between coordinator updates
- **Finding:** `find-vllm_v1_engine-0015` — *GitHub - Netflix/concurrency-limits*
- **Source URL:** <https://github.com/Netflix/concurrency-limits>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In DPLBAsyncMPClient.get_core_engine_for_request (vllm/v1/engine/core_client.py:1471-1522), replace the raw use of the coordinator's (waiting, running, kv_cache_usage) snapshot with per-engine short-window and long-window EWMAs of each of these signals (or of the derived score), updated on every stats refresh and on every local admission (which already bumps waiting by client_count at line 1510). Compute a `trend = short_ewma - long_ewma` per engine as an early indicator of queue buildup between the ~100ms coordinator updates, and fold it into the current score: e.g. `score = max(client_count * inflight, short_ewma_waiting + short_ewma_running) + waiting_penalty * (1 + max(0, trend))`, so an engine whose short-window load is rising above its longer-term baseline is deprioritized before the next coordinator snapshot arrives. The two EWMA half-lives should straddle the coordinator interval (e.g. ~20-50ms short vs ~500ms-1s long) so the divergence signal fires when a burst is landing but has not yet been reported. Keep the existing rotating tie-breaker and the 0.5 KV-pressure knee; the change is limited to how the load signal fed into the score is smoothed and how a rising-trend penalty is added.

**Proposal rationale.**

The candidate explicitly calls out stale-snapshot handling and local-inflight weighting as owned policy knobs, and its stats snapshot can be up to ~100ms stale between coordinator updates - exactly the regime where Netflix concurrency-limits' Gradient2 uses divergence between a short and long EWMA to identify a queueing trend rather than waiting for a fixed reporting interval. Applying that idea gives the load balancer an earlier, lower-variance signal of rising per-engine congestion under bursty multi-turn agentic traffic, addressing the stated caller objective of lowering median TTFT without changing model kernels. It is transferable (a well-known adaptive-limit technique), concrete (well-defined change to the score computation on the existing snapshot fields), and scoped to the same hot path and knobs the candidate already owns.

---

## Agent proposals

### 1. Make DP admission request-size aware via a projected-KV-headroom penalty
- **Agent:** claude

**Detailed description.**

Extend the scoring loop in DPLBAsyncMPClient.get_core_engine_for_request (vllm/v1/engine/core_client.py:1483-1507) so the incoming request's own token footprint participates in the score, turning admission into best-fit bin packing against per-engine KV capacity. Concretely: (1) At entry, derive req_tokens = len(request.prompt_token_ids) (already on EngineCoreRequest) and req_out = min(request.sampling_params.max_tokens or DEFAULT_OUT, cap), then req_cost = req_tokens + req_out, treated as the request's KV footprint in tokens; the cap prevents an unbounded max_tokens from dominating. (2) Add a per-engine kv_capacity_tokens field to the coordinator stats payload that already flows into self.lb_engines at lines 1394-1404, populated once at startup from the engine's num_gpu_blocks * block_size so no per-step update is needed. (3) Inside the scan at lines 1487-1504 compute projected_usage = kv_cache_usage + req_cost / kv_capacity_tokens and add a size_penalty = beta * max(0, projected_usage - 1.0) ** 2 with a small beta (e.g. 4.0), so the term is zero unless this specific request would push the chosen engine past full KV and grows quadratically once it would; engines whose free headroom fits req_cost are preferred, engines that would overflow are penalized in proportion to the overflow. (4) Add the penalty to the existing score without changing the max(client_count * inflight, waiting + running) core or the current waiting * 6.0 * max(0, kv_cache_usage - 0.5) term; the rotating tie-break at line 1516 and the local waiting += client_count bump at line 1510 are unchanged. (5) Guard behind a config flag VLLM_DP_LB_SIZE_AWARE (default off) so scoring is bit-identical to today when disabled; when kv_capacity_tokens is missing from the snapshot (older coordinator), skip the penalty and degrade to current behavior. (6) Preserve every invariant tests/v1/engine/test_engine_core_client.py already asserts: shutdown engines remain excluded, data_parallel_rank and get_late_interaction_engine_index() overrides bypass scoring first, and bursts still spread because the local waiting += client_count bookkeeping is untouched.

**Novelty rationale.**

The five existing deep_research_proposals all treat the incoming request as unit weight and score engines purely from engine-side signals: 0003 (P2C + weighted least-request) changes how many engines are sampled and reweights the same (waiting, running, inflight) fields; 0004 (multiplicative prefix-reuse) changes the prefill-work factor via cache-hit estimation but does not use raw request size or per-engine KV capacity; 0005 (three-tier prefix affinity) still uses the current scalar score in its load tier; 0006 (prefill/decode split) splits the engine score into two dimensions but a 200-token and a 32k-token admission still contribute equally to each; 0015 (short/long EWMA trend) smooths historical snapshot signals and cannot distinguish a small vs large admission arriving at the same instant. This proposal introduces two orthogonal inputs to the score - the request's own req_tokens + req_out footprint and the engine's kv_capacity_tokens - and combines them as a best-fit-decreasing bin-packing penalty, an angle none of the listed proposals touch and one that composes cleanly on top of any of them.

---

### 2. Add deterministic request-affinity stickiness with bounded spillover
- **Agent:** codex

**Detailed description.**

Extend `DPLBAsyncMPClient.get_core_engine_for_request` in `vllm/v1/engine/core_client.py:1471-1522` with an optional per-request/session stickiness layer before the load-score scan: derive a stable affinity key from request metadata that identifies the conversation/session when available (falling back to a normalized request group identifier only when it is stable across turns, never the unique request id), remember `affinity_key -> engine_index` in a bounded LRU/TTL map on the client, and route subsequent turns to the same non-shutdown engine if its current score is within a configurable spillover factor of the best available engine, e.g. `affine_score <= best_score + max(1, gamma * client_count)`. If the affine engine is shutdown, missing from the local DP set, or exceeds the spillover threshold, fall back to today's full scan and update the affinity map to the chosen engine. Keep `data_parallel_rank` and `get_late_interaction_engine_index()` overrides ahead of this logic, preserve the existing scoring formula and local `waiting += client_count` bump, and gate the behavior behind a conservative config flag such as `VLLM_DP_LB_STICKY_SESSIONS` defaulting off. Add tests in `tests/v1/engine/test_engine_core_client.py` that verify repeated affinity keys stay on the same admissible engine under modest imbalance, spill over when that engine is clearly overloaded, and clear/reassign when the remembered engine is shutdown.

**Novelty rationale.**

The deep-research prefix proposals (0004 and 0005) optimize for token-prefix cache reuse by estimating prefix hits or maintaining prefix tries; this proposal is not prefix matching and does not inspect token content. It uses a stable request/session key to preserve multi-turn locality with bounded spillover, which is cheaper and targets conversational state affinity even when prefixes are transformed, truncated, or otherwise hard to match. Proposal 0003 is about P2C sampling, 0006 splits prefill/decode load dimensions, and 0015 smooths stale load snapshots; none introduce a persistent affinity map keyed by session identity. Agent A's proposal is request-size/KV-headroom bin packing, which is orthogonal because it scores each admission by token footprint and capacity rather than preserving routing continuity across turns.

---
