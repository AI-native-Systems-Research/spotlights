# DPLBAsyncMPClient.get_core_engine_for_request

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/core_client.py`](vllm/v1/engine/core_client.py) (lines 1350–1378)
- **Symbol:** `DPLBAsyncMPClient.get_core_engine_for_request`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0001`

## Description
Routes a new DP request to one engine rank after honoring explicit data_parallel_rank and late-interaction placement. The owned load-balancing path scores each candidate engine, picks the minimum, and records the chosen engine for abort routing.

## Current approach
O(num_engines) Python scan with a hard-coded score of waiting * 4 + running, deterministic tie-breaking from eng_start_index, and an optimistic local waiting-count bump by client_count to compensate for stale coordinator stats.

## Estimated impact explanation
For DP>1 multi-turn agentic serving, admission rank directly controls queueing delay for each turn. Better load scoring can reduce median TTFT and tail variance when many short turns arrive between coordinator stats publications.

## Evolve rationale
The optimization unit is the score loop and local bump at lines 1357-1373. The scoring formula, tie-break, P2C-vs-full-scan choice, stale-count correction, and optional stickiness on a stable request-affinity key are all tunable without changing the contract. Correctness oracle: every request maps to exactly one valid EngineIdentity, reqs_in_flight records that same engine for abort routing, and DP request-output streams remain identical under tests/v1/engine/test_engine_core_client.py plus a replay of (waiting, running, chosen_rank) traces.

## Deep research proposals

### 1. Add intercept-aware engine affinity for resumed agentic turns in DP load balancing
- **Finding:** `find-0001` — *InferCept: Efficient Intercept Support for Augmented Large Language Model Inference*
- **Source URL:** <https://proceedings.mlr.press/v235/abhyankar24a.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend DPLBAsyncMPClient.get_core_engine_for_request (vllm/v1/engine/core_client.py:1350-1378) to recognize a stable per-conversation affinity key for multi-turn agentic requests and route follow-up turns back to the engine that holds their retained KV/request state, rather than scoring them as fresh admissions. Concretely: (1) accept (or derive from request metadata) an affinity token such as parent_request_id / conversation_id; (2) before running the existing waiting*4 + running scan over candidate engines, consult a bounded affinity map maintained alongside reqs_in_flight that records the engine which last served that affinity key; (3) if the recorded engine is still in the candidate set and its score is within a small slack of the minimum, short-circuit and return it; otherwise fall through to the existing scoring loop and update the affinity map with the chosen rank. The optimistic local waiting-count bump and tie-break logic stay intact for fresh requests. The contract is preserved: every request still maps to exactly one valid EngineIdentity, reqs_in_flight still records the chosen engine for abort routing, and behavior for non-agentic requests (no affinity key) is unchanged.

**Proposal rationale.**

InferCept's core insight is that agentic tool-boundary turns should resume against retained request/KV state instead of being re-prefilled as new requests. In a DP deployment that benefit is only realized if the resumed turn lands on the engine rank that still holds that state; the current pure load-score routing can scatter successive turns of the same conversation across engines, forcing prefill re-materialization and inflating TTFT — exactly the metric the caller wants to reduce. Adding an affinity short-circuit at lines 1357-1373 is within the candidate's declared tunable scope (the rationale explicitly lists 'optional stickiness on a stable request-affinity key') and addresses the gap that pure min-score scoring is oblivious to per-conversation locality value. The slack threshold keeps worst-case imbalance bounded so median TTFT/TPOT improve on multi-turn workloads without regressing single-turn admission behavior.

---

### 2. Add tunable TTFT/ITL-weighted, KV-overlap-aware scoring to DP engine selection
- **Finding:** `find-0006` — *Configuration and Tuning | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.nvidia.com/dynamo/latest/components/router/configuration-and-tuning>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the fixed `waiting * 4 + running` score in `DPLBAsyncMPClient.get_core_engine_for_request` (vllm/v1/engine/core_client.py:1357-1373) with a parameterized cost function inspired by NVIDIA Dynamo's router: `score = w_waiting * waiting_tokens + w_running * running_tokens - w_overlap * kv_overlap_blocks`, where weights are exposed as configuration knobs (defaults preserve current behavior with `w_waiting=4`, `w_running=1`, `w_overlap=0`). Two augmentations: (1) prompt-side load accounting — bump the local stale-stats correction by an estimate of the new request's prompt length (or token count) rather than a flat `client_count`, so newly admitted long prompts immediately raise their engine's score for subsequent picks; (2) optional WSPT-style ordering — when multiple engines tie on the primary score, break ties by shortest expected processing time using the existing per-engine waiting/running counts instead of `eng_start_index`. Keep the same EngineIdentity contract and `reqs_in_flight` recording so abort routing and the test in tests/v1/engine/test_engine_core_client.py remain valid. KV-overlap input is optional; when unavailable, the term collapses to zero and the policy degrades to the current count-based scan.

**Proposal rationale.**

The candidate explicitly calls out the scoring formula, stale-count correction, and tie-break as tunable without contract changes, and the caller objective targets median TTFT/TPOT for multi-turn agentic workloads. The Dynamo finding contributes three transferable ideas that map directly onto those tunables: a TTFT-vs-ITL weighted score, prompt-side load accounting that anticipates the new request's cost (addressing the same staleness gap the current `+ client_count` bump tries to patch), and WSPT queue ordering for average-TTFT improvement. Multi-turn agentic traffic has high prefix reuse, so even a zero-default KV-overlap term creates a clean extension point for future overlap-aware routing without altering current behavior. The change stays inside the score loop the candidate flagged for evolution and preserves the existing correctness oracle.

---

### 3. Add prefix-cache-aware affinity term to DP engine scoring
- **Finding:** `find-0007` — *Prefix-Cache Aware Routing*
- **Source URL:** <https://github.com/cncf/testing-llm-d/blob/main/docs/architecture/advanced/kv-management/prefix-cache-aware-routing.md>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the per-engine scoring loop in DPLBAsyncMPClient.get_core_engine_for_request at vllm/v1/engine/core_client.py:1350-1378 so that engine selection is biased toward replicas likely to already hold the request's KV prefix, in addition to the current load metric. Concretely: (1) Derive a lightweight prefix-affinity key for the incoming request from the leading tokens of its prompt (or from a caller-supplied affinity key, e.g., a session/conversation id for multi-turn agentic flows). The simplest approximation is a rolling hash over a fixed-size prefix window of the prompt token ids, matching the rolling-hash-chain technique called out in the finding. (2) Maintain a small per-client lookup that maps recent prefix-hash buckets to the engine rank previously chosen for that bucket (bounded LRU keyed by hash, with TTL or size cap so it stays O(1) memory per active session). This complements the existing reqs_in_flight bookkeeping without changing its contract. (3) Inside the score loop at lines 1357-1373, compute score = waiting_with_bump * 4 + running - prefix_hit_bonus, where prefix_hit_bonus is a tunable constant applied only to the engine recorded for this request's prefix-hash bucket (zero for all others). Keep the deterministic eng_start_index tie-break and the client_count optimistic bump on the local waiting counter unchanged. (4) After the chosen engine is committed (reqs_in_flight[request_id] = chosen_engine), update the prefix-hash -> engine map so subsequent turns of the same conversation are sticky to that engine while load remains comparable. The bonus magnitude is the only new hyperparameter and can be tuned (or set to 0 to recover current behavior) without changing the method's contract: every request still maps to exactly one valid EngineIdentity, abort routing via reqs_in_flight is preserved, and the existing tests in tests/v1/engine/test_engine_core_client.py continue to hold for non-prefix-keyed traces. Optionally expose a precise variant later by consuming KV-event indexing if the engine publishes block-level cache state, but the rolling-hash approximation is sufficient as a first step and matches the finding's "approximate implementation" path.

**Proposal rationale.**

The candidate's current score (waiting * 4 + running) ignores cache locality entirely, so in a multi-turn agentic workload the second turn of a conversation is just as likely to land on a cold engine as a warm one. Cold landings force prefix recomputation and directly inflate TTFT, which is exactly the metric the caller objective targets. The finding describes a transferable, low-overhead mechanism (rolling-hash prefix chain) that fits naturally as an additive bias inside the existing O(num_engines) scoring loop, preserves the deterministic tie-break and stale-count compensation, and only kicks in when a recognizable prefix recurs - so worst-case behavior degrades to the current load-only policy. The evolve_rationale already calls out "optional stickiness on a stable request-affinity key" as a tunable axis for this exact scoring path, making the finding a direct fit rather than topically adjacent.

---

### 4. Add prefix-cache-aware term to DP engine selection score
- **Finding:** `find-0009` — *Preble: Efficient Distributed Prompt Scheduling for LLM Serving*
- **Source URL:** <https://arxiv.org/abs/2407.00023>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify the per-engine scoring loop in DPLBAsyncMPClient.get_core_engine_for_request (vllm/v1/engine/core_client.py:1357-1373) so the score co-optimizes load and KV/prefix-cache locality instead of using waiting*4 + running alone. Concretely: derive a stable affinity key for the incoming request (e.g., hash of the leading prompt-token prefix or, for multi-turn agentic traffic, a session/conversation id when available on the request) and combine a cache-hit estimate per engine with the existing load term, e.g. score = alpha*(waiting + client_count_bump)*4 + beta*running - gamma*prefix_hit_score(engine, key). The prefix_hit_score can be a lightweight last-N affinity table maintained on the client (engine-of-last-request keyed by affinity key, decayed by time/turn count) so no new RPC is required. Keep the existing deterministic tie-break from eng_start_index when scores tie, keep reqs_in_flight bookkeeping unchanged, and gate the cache-aware term behind a tunable so falling back to pure load-based routing is one config flip away. The contract (one valid EngineIdentity per request, abort routing via reqs_in_flight, identical output streams) is preserved.

**Proposal rationale.**

The candidate's evolve_rationale already calls out 'optional stickiness on a stable request-affinity key' as in-scope, and the caller context names a multi-turn agentic workload targeting median TTFT/TPOT - exactly the regime where Preble shows that load-only routing leaves prefix-cache reuse on the table. Preble's central claim ('co-optimizes KV state reuse and computation load-balancing') maps directly onto this scoring loop: the gap is that waiting*4 + running ignores which engine is most likely to already hold this request's prefix in its block cache, so multi-turn turns can be admitted to a cold engine and pay full prefill latency. Adding a cache-locality term, while keeping a load term so a hot worker doesn't get pinned, is the minimal transferable idea from the paper and is implementable inside this single function without changing the engine-side contract.

---

### 5. Add KV-cache affinity term to DP engine scorer for prefix-locality routing
- **Finding:** `find-0011` — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*
- **Source URL:** <https://madsys.cs.tsinghua.edu.cn/publication/mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend DPLBAsyncMPClient.get_core_engine_for_request (vllm/v1/engine/core_client.py:1350-1378) so the per-engine score combines the existing queue pressure (waiting*4 + running) with a KV-cache affinity term. Specifically, in the scoring loop at lines 1357-1373, derive a stable affinity key from the incoming request (e.g., a hash of the longest cacheable prefix of prompt_token_ids, or an explicit session/conversation id when the caller supplies one) and prefer an engine that is recorded as the most recent owner of that key. Maintain a small bounded LRU map { affinity_key -> engine_identity } updated whenever a request is admitted, and bias the score by subtracting a tunable affinity_bonus when an engine matches. Keep the optimistic client_count waiting bump and deterministic tie-break from eng_start_index so the routing contract (one valid EngineIdentity per request, reqs_in_flight records the same engine for abort routing, identical DP output streams under existing tests) is preserved. The affinity bonus and key derivation are tunable without touching the dispatch contract, matching the candidate's evolve_rationale.

**Proposal rationale.**

The candidate currently scores engines purely on queue depth, which ignores that in multi-turn agentic workloads each turn re-presents a long shared prefix that another engine likely already has cached. Mooncake's KV-centric scheduling argues that treating KV cache as a first-class scheduling signal (rather than just throughput/queue) is what unlocks lower TTFT for repeated-context traffic - precisely the caller objective here. A prefix/session affinity bias on the existing scorer is the minimal, transferable adaptation of that idea to vLLM's DP coordinator: it keeps the cheap O(num_engines) loop and its tie-break, but lets the load balancer exploit prefix-cache locality instead of fighting it, which is the dominant TTFT lever for multi-turn agentic serving. It directly addresses the gap that score = waiting*4 + running cannot see: an engine with slightly more queue but a hot prefix cache will often produce lower TTFT than the least-loaded cold engine.

---

### 6. Add prefix-locality affinity term to DP engine scoring
- **Finding:** `find-0012` — *MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool*
- **Source URL:** <https://arxiv.org/abs/2406.17565>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/engine/core_client.py at DPLBAsyncMPClient.get_core_engine_for_request (lines 1350-1378), extend the per-engine score loop (lines 1357-1373) to combine the existing load term (waiting * 4 + running, with the client_count optimistic bump) with a prefix-locality affinity term inspired by MemServe's global prompt-tree locality policy. Concretely: derive a stable affinity key from the incoming request's prompt prefix (e.g., a rolling hash over the first N token-block hashes, reusing the block-hashing already done for prefix caching) and maintain a lightweight per-engine recency map of recently admitted prefix-key shards. During scoring, subtract a tunable bonus from an engine's score when the request's affinity key matches that engine's recent admissions, so multi-turn agentic follow-ups are biased toward the rank that most likely still holds their KV blocks, while load-imbalance still dominates when affinity ties or is absent. Keep the existing deterministic tie-break from eng_start_index and the reqs_in_flight bookkeeping unchanged so the correctness oracle (every request maps to one valid EngineIdentity, abort routing intact, identical outputs under tests/v1/engine/test_engine_core_client.py) is preserved. The affinity map should be bounded (LRU over request-affinity keys) and updated only on the chosen engine to avoid touching the engine-side KV state from the client.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'optional stickiness on a stable request-affinity key' as a tunable knob within the scoring loop, and the caller context names a multi-turn agentic workload where successive turns share long prefixes. MemServe's contribution is precisely a global prompt-tree locality policy used to route requests so that distributed KV/prefix state is reused rather than recomputed; transposing that idea from MemServe's cross-instance setting to vLLM's intra-deployment DP rank choice gives a concrete way to lower median TTFT for follow-up turns by avoiding prefix-cache misses caused by the current purely load-based scoring. The change stays inside the score-loop optimization unit, does not alter the routing contract, and addresses the gap that load-only scoring ignores per-engine prefix-cache state — exactly the gap MemServe's locality policy targets.

---

### 7. Weight DP load score by predicted output-length ranks
- **Finding:** `find-0014` — *Efficient LLM Scheduling by Learning to Rank*
- **Source URL:** <https://openreview.net/forum?id=wlLjYl0Gi6>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/engine/core_client.py at lines 1357-1373, replace the uniform `waiting * 4 + running` score in DPLBAsyncMPClient.get_core_engine_for_request with a length-rank-weighted estimate of remaining work per engine. Concretely: (1) accept or compute a relative output-length rank hint for the incoming request and persist a lightweight per-engine aggregate of predicted ranks for in-flight (waiting + running) requests, updated when reqs_in_flight is mutated; (2) score each candidate engine as a sum of expected remaining tokens derived from predicted ranks rather than a raw count, while keeping the optimistic +client_count bump on the waiting term to compensate for stale coordinator stats; (3) preserve deterministic tie-breaking via eng_start_index when scores tie. The contract (one EngineIdentity per request, reqs_in_flight recording the chosen engine for abort routing) and the existing test surface in tests/v1/engine/test_engine_core_client.py remain unchanged, since only the scalar score function is altered. Length ranks can come from a small ranker model invoked at admission, from request metadata when callers supply hints, or from a cheap heuristic (prompt length, role, prior turn lengths in multi-turn sessions) when no learned predictor is wired up.

**Proposal rationale.**

The candidate explicitly calls out the scoring formula as a tunable knob and notes that admission rank directly controls queueing delay for each turn under DP>1 agentic workloads. The current score treats every waiting/running request as equal load, which systematically misroutes when output lengths are heterogeneous: an engine with one long-tail decode looks identical to one with several short decodes. The finding's central idea — that *relative* output-length ranks are sufficient to approximate SJF-style scheduling and are easier to predict than exact lengths — transfers naturally to cross-engine routing: weighting in-flight counts by predicted ranks yields a better proxy for remaining work per engine, which is exactly what get_core_engine_for_request is trying to estimate. This targets the stated objectives (median TTFT and TPOT) on multi-turn agentic traffic, where prior-turn signal makes rank prediction especially tractable, without changing the routing contract or coordinator protocol.

---

### 8. Add prefix-locality and output-length signals to DP engine scoring
- **Finding:** `find-0015` — *Prefix and Output Length-Aware Scheduling for Efficient Online LLM Inference*
- **Source URL:** <https://openreview.net/forum?id=DOZiCWyK0N>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend DPLBAsyncMPClient.get_core_engine_for_request at vllm/v1/engine/core_client.py:1350-1378 so the per-engine score combines three signals instead of the current `waiting * 4 + running`: (1) a prefix-locality bonus derived from a lightweight per-engine prefix-hash sketch (e.g. last-N prompt-prefix hashes or a small Bloom/HLL of recently admitted prefix block hashes maintained in reqs_in_flight bookkeeping), (2) an estimated remaining-decode-load term that sums an output-length estimate over each engine's in-flight requests (estimate sourced per-request from a cheap predictor: prior turn's decode length for the same conversation/session, or a conservative default), and (3) the existing waiting/running queue-depth term as the fallback. Concretely: keep the O(num_engines) scan and tie-break order, but replace the score with `alpha * queue_depth + beta * estimated_pending_decode_tokens - gamma * prefix_locality_match`, where prefix_locality_match is a 0/1 or small integer indicating overlap between the new request's prompt-prefix hash and the engine's recent-prefix sketch. Maintain the local waiting-count bump by client_count for staleness compensation. Coefficients alpha/beta/gamma become tunable knobs honoring the contract (single valid EngineIdentity returned, reqs_in_flight updated identically). The prefix sketch is updated in add_request/abort paths already touched here so no new RPC is required. Output-length estimates can be passed in via the request (caller already has session/turn context for agentic workloads) or defaulted to the engine's running-average decode length.

**Proposal rationale.**

The candidate's current scoring is purely queue-depth based and ignores both KV-cache reuse and future decode work, which are the two dominant TTFT/TPOT drivers in multi-turn agentic workloads with shared prompts and variable tool-result lengths. The finding directly proposes combining prefix-aware routing with output-length estimation for online LLM serving, which maps cleanly onto the optimization unit (the score loop at lines 1357-1373) without changing the routing contract. Prefix locality reduces prefill cost (TTFT) by steering turns of the same conversation back to the engine that already has the KV cache; output-length estimation prevents piling new requests onto an engine whose 'running' count is small but whose remaining decode work is large (TPOT). Both are tunable knobs the candidate's evolve_rationale explicitly invites.

---

## Agent proposals

### 1. Replace deterministic full scan with power-of-d-choices to break multi-client herding under stale stats
- **Agent:** claude

**Detailed description.**

Restructure the engine-selection loop in DPLBAsyncMPClient.get_core_engine_for_request at vllm/v1/engine/core_client.py:1357-1373 from a deterministic O(num_engines) full scan into a power-of-d-choices (P2C) sampler with per-client randomization, while leaving the score function (waiting*4 + running) and the contract (one valid EngineIdentity, reqs_in_flight bookkeeping, abort routing) unchanged. Concretely: (1) draw d (default 2, configurable) candidate engine indices uniformly at random from the eligible pool using a per-client RNG seeded from the client's identity (so different DP clients pick different subsets even when their cached coordinator stats are byte-identical); (2) compute the existing score for those d candidates only and pick the minimum; (3) keep the optimistic local waiting-count bump by client_count to compensate for stale stats, but apply it only to the sampled engines so the bump's per-client correction stays meaningful; (4) replace the deterministic eng_start_index tie-break with a coin-flip among tied sampled engines so coincident ties across clients don't all resolve to the same rank; (5) preserve a fallback to the full scan when num_engines <= d or when an explicit data_parallel_rank / late-interaction placement is requested (these paths already short-circuit above the loop, so no behavior change there). When the request's caller supplies an affinity key or data_parallel_rank, the existing pre-loop branches still take precedence; P2C only governs the unowned-load-balancing path. The change keeps the function O(d) instead of O(num_engines) and, more importantly, decorrelates concurrent admissions from independent front-end clients whose coordinator snapshots are stale by the same delta — today they all converge on the same min-score engine and the +client_count bump is a uniform coarse fix; under P2C with per-client seeds, two clients admitting in the same window almost never sample the same pair, so herding into the apparent-cheapest engine is broken without needing fresher stats. Tunable knobs: d (default 2 reproduces classical P2C), an env flag to fall back to full-scan (recovers current behavior bit-for-bit when seeded RNG is disabled), and the same client_count bump scalar.

**Novelty rationale.**

All eight existing deep_research_proposals modify the *score function* (adding affinity, prefix-locality, KV-overlap, output-length, or weighted load terms) while keeping the deterministic O(num_engines) scan and deterministic eng_start_index tie-break. None address the *structural* alternative the candidate's evolve_rationale explicitly calls out — 'P2C-vs-full-scan choice' — and none address the multi-client herding pathology where the +client_count optimistic bump is a uniform per-engine coarse correction that does not actually decorrelate concurrent admissions from different front-end clients viewing the same stale coordinator snapshot. The proposed change is orthogonal to all of them: it can compose with any of the affinity/locality/length-aware scoring ideas (P2C still works on whatever score function is plugged in), but on its own it targets a different failure mode — correlated mis-routing under stale stats — that none of the existing proposals diagnose or fix. The stale-count correction issue is the second tunable axis the candidate explicitly lists, and no existing proposal touches it beyond keeping the current flat bump.

---

### 2. Account pinned DP admissions in the local load snapshot
- **Agent:** codex

**Detailed description.**

Update `DPLBAsyncMPClient.get_core_engine_for_request` in `vllm/v1/engine/core_client.py:1350-1378` so the optimistic local waiting-count bump is applied to every admitted request, not only requests that enter the load-balancing scan. Today requests with `data_parallel_rank` or late-interaction placement bypass lines 1357-1373 and therefore do not update `self.lb_engines` until the next coordinator stats publication, so subsequent unpinned requests can be routed as if that pinned traffic never arrived. Concretely: after `eng_index` is determined by any path and before returning `chosen_engine`, increment `self.lb_engines[eng_index][0] += self.client_count` when the index is valid; keep the scoring loop unchanged, and keep `reqs_in_flight[request_id] = chosen_engine` unchanged. Add focused tests where an explicitly ranked request and a late-interaction sticky request make a previously empty rank look locally busier before the next ordinary request is routed.

**Novelty rationale.**

The deep-research proposals all change the score inputs or add affinity/prefix/output-length signals for the scored path; none address the fact that explicit-rank and late-interaction short-circuit paths skip the local stale-stats correction entirely. Agent A's proposal changes the unpinned path to P2C and explicitly leaves explicit-rank and late-interaction placement as pre-loop fallbacks. This proposal targets a different failure mode: stale local accounting caused by already-pinned admissions, which can misroute later unpinned traffic even if the score function and scan/P2C choice remain unchanged.

---
