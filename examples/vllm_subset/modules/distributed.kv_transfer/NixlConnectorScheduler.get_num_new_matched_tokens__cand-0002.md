# NixlConnectorScheduler.get_num_new_matched_tokens

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py`](vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py) (lines 288–364)
- **Symbol:** `NixlConnectorScheduler.get_num_new_matched_tokens`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0002`

## Description
Scheduler-side admission decision for how many tokens can be loaded from a remote NIXL peer. The remote-decode path gates pulling on the fixed `kv_recompute_threshold` extra config, defaulting to 64 tokens.

## Current approach
The method computes a remote-token count from request KV transfer parameters and returns either `(count, True)` or `(0, False)`. For remote decode, `count < self.kv_recompute_threshold` forces local recompute; the threshold is a single static integer read during scheduler initialization with no dependence on recent transfer latency, queue depth, request shape, network RTT, or prefill compute cost.

## Estimated impact explanation
This gate directly decides recompute versus remote KV pull. In multi-turn agentic workloads, an overly low threshold wastes TTFT on network setup for tiny prefixes, while an overly high threshold recomputes cheaply transferable prefixes; adapting it can move median TTFT by avoiding the worse path per request.

## Evolve rationale
The `kv_recompute_threshold` branch is a compact routing/filtering gate. It can be evolved into a cost model, hysteretic threshold, or load-aware policy without changing the connector contract. Correctness oracle: any positive token count must be backed by valid `remote_*` parameters and remote block IDs, and existing NIXL scheduler tests can verify the returned `(num_external_tokens, async)` pairs plus end-to-end output equality for remote-prefill/decode cases.

## Deep research proposals

### 1. Lower the remote-pull threshold for agentic delta transfers backed by local prefix cache
- **Finding:** `find-0002` — *[Roadmap] Prefill-Decode Disaggregation Roadmap (2026 Q2)*
- **Source URL:** <https://github.com/sgl-project/sglang/issues/21703>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py` lines 288-364, replace the static `count < self.kv_recompute_threshold` gate in the remote-decode branch of `NixlConnectorScheduler.get_num_new_matched_tokens` with a threshold that is sensitive to whether the request looks like an agentic delta. Concretely: when the scheduler can observe (or be informed) that the local block manager / prefix cache already covers a large fraction of the request's prefix and the remote-supplied `count` is the residual delta, treat the gate as effectively zero (always pull the small delta) rather than forcing local recompute below 64. Keep the existing higher threshold as a fallback when no prefix-cache hit is observed. The connector contract (returning `(num_external_tokens, async)`) is unchanged; only the admission decision becomes a simple two-tier policy: aggressive small-delta admission when prefix coverage is high, current threshold otherwise. Existing NIXL scheduler tests still validate end-to-end output equality for remote-prefill/decode cases, plus new cases where small `count` values pass through under simulated high prefix-cache coverage.

**Proposal rationale.**

The candidate's current gate is a single static integer that does not distinguish between "64 tokens are too few to justify a network round-trip from cold" and "64 tokens are exactly the per-turn delta of a multi-turn agentic request whose long shared prefix is already in the local cache." The finding's central idea — that in agentic traffic decode can reuse shared prefixes locally and request only the delta KV — directly identifies the regime where small remote-token counts are *most* valuable, not least. Lowering or zeroing the threshold conditional on prefix-cache coverage operationalizes this insight inside the candidate's decision point without changing the transfer mechanism, addressing the gap that the static threshold currently treats every small `count` as wasteful network setup. This aligns with the caller's stated objective of reducing TTFT on multi-turn agentic workloads, where the recompute path penalizes exactly the requests the finding identifies as the hot pattern.

---

### 2. Make kv_recompute_threshold load-aware using async NIXL transfer state
- **Finding:** `find-0008` — *Disaggregated Serving | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py at NixlConnectorScheduler.get_num_new_matched_tokens (lines 288-364), replace the single static `self.kv_recompute_threshold` comparison in the remote-decode branch with a dynamically computed threshold that reflects current NIXL transfer subsystem state. Concretely: (1) expose lightweight counters from the NIXL connector (e.g., number of in-flight async pulls, recent average transfer completion latency, and queue depth of pending transfers) and read them from the scheduler; (2) compute an effective threshold `eff_threshold = base_threshold * f(load)` where `f(load)` decreases when the transfer engine is idle (so small prefixes are admitted for remote pull, exploiting the non-blocking overlap with concurrent forward passes) and increases when the engine is saturated (so small prefixes fall back to local recompute); (3) keep the existing return contract `(count, True)` / `(0, False)` and the correctness checks on `remote_*` parameters unchanged; (4) add hysteresis so the threshold does not oscillate request-to-request. Default `base_threshold` remains the existing config value, preserving current behavior when load signals are unavailable.

**Proposal rationale.**

The finding documents that KV transfer is non-blocking and overlaps with GPU forward passes serving other requests. This directly informs the admission gate at lines 288-364: the value of pulling a small remote prefix vs. recomputing depends on whether the transfer can be hidden behind concurrent decode/prefill work. A static `kv_recompute_threshold` ignores this overlap potential entirely. By turning the threshold into a function of in-flight transfer state (the same async machinery the candidate already returns `True` for), the scheduler can admit smaller remote pulls when overlap is cheap and fall back to recompute when the transfer engine is contended. This addresses the candidate's stated gap (no dependence on transfer latency, queue depth, or load) and targets the multi-turn agentic TTFT objective by reducing the cases where short reusable prefixes are needlessly recomputed.

---

### 3. Replace static kv_recompute_threshold with overlap- and load-aware cost gate
- **Finding:** `find-0013` — *Router Guide | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.dynamo.nvidia.com/dynamo/user-guides/kv-cache-aware-routing>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py:288-364`, replace the single static comparison `count < self.kv_recompute_threshold` (line 350-353) on the remote-decode branch of `NixlConnectorScheduler.get_num_new_matched_tokens` with a small cost model that mirrors Dynamo's routing logic. Compute two estimated costs per request and admit the remote pull only when transfer is cheaper:

- transfer_cost: a function of the unique remote tokens to fetch (i.e., `count` minus the portion that already overlaps local cache), an EMA of recent NIXL transfer latency per token/block, and current inflight transfer volume.
- recompute_cost: a function of `count` times an estimated per-token prefill cost (pulled from a simple EMA of recent prefill step times) plus a penalty proportional to active decode load (number of running decode requests / batch occupancy) to reflect contention added by recompute.

Factor the predicate into a private helper like `_should_pull_remote_decode(request, count, params) -> bool` that returns False when `recompute_cost <= transfer_cost`, and otherwise returns True. Keep `kv_recompute_threshold` as a hard lower bound (early-out for very small `count`) so existing tests remain valid. Update the EMAs from existing connector signals: NIXL transfer completion timestamps already flowing through the connector (see surrounding scheduler/connector state in the same module) for transfer latency, and scheduler-provided counts for active decode load. Preserve the return contract `(num_external_tokens, async)` exactly: still return `(0, False)` on rejection and `(count, True)` on admission. No changes to the connector worker or the wire protocol.

The implementation should not introduce new public configuration beyond optional EMA decay and the existing `kv_recompute_threshold` floor, and the cost weights should default to behavior equivalent to the current static threshold when no telemetry has been collected yet.

**Proposal rationale.**

The candidate's gap is precisely that the admit/reject decision is a single static integer ignoring transfer latency, queue depth, request shape, and prefill compute cost (per the candidate's `current_approach`). The Dynamo router guide describes the same decision class and prescribes combining decode load, new prefill work, and KV overlap to minimize redundant computation. That is a concrete, transferable structure for the cost model, and it directly targets the multi-turn agentic workload in the caller context where session-continuation requests have high overlap and benefit from being pulled rather than recomputed, while one-off short prefixes should be recomputed. Using the finding's three-input cost framing here is conservative (it stays within the existing return contract, preserves correctness because admitted pulls still require valid `remote_*` params and remote block IDs) and plausibly improves median TTFT/TPOT in exactly the regime the candidate's `estimated_impact_explanation` highlights.

---

### 4. Replace static kv_recompute_threshold with cache-hit/transfer-cost class gating
- **Finding:** `find-0014` — *Cache-aware prefill–decode disaggregation (CPD) for up to 40% faster long-context LLM serving*
- **Source URL:** <https://www.together.ai/blog/cache-aware-disaggregated-inference>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py at NixlConnectorScheduler.get_num_new_matched_tokens (lines 288-364), replace the single static `count < self.kv_recompute_threshold` cutoff on the remote-decode path with a cache-hit/transfer-cost classification that mirrors the warm/cold split from CPD. Concretely, compute a hit-rate proxy from the request's remote KV parameters (e.g., remote_token_count divided by total prompt length, plus the size of the contiguous remote block list) and combine it with a per-peer transfer-cost estimate (e.g., recent observed bytes/sec or RTT for that peer, or a static cost-per-block constant if no telemetry is wired) to decide between three buckets: (a) warm/large -> return `(count, True)` to pull remote KV asynchronously, (b) cold/small -> return `(0, False)` to force local recompute, (c) borderline -> fall through to the existing threshold as a tiebreaker. Keep the connector contract unchanged: any non-zero return must still be backed by valid `remote_*` parameters and remote block IDs, so the existing NIXL scheduler tests covering remote-prefill/decode `(num_external_tokens, async)` pairs and end-to-end output equality continue to apply. The threshold stays configurable as a fallback constant; the new logic adds a bucket function around it rather than replacing the field.

**Proposal rationale.**

The candidate's gap is exactly the one CPD targets: a single static cutoff cannot tell a cheaply-recomputable cold prefix from a large warm prefix that would benefit from a remote pull, so multi-turn agentic workloads pay the worse path per request. The finding contributes a concrete, transferable idea — classify requests by cache-hit-rate and transfer-cost class and route accordingly — that maps directly onto the binary gate at lines 288-364 without changing the connector contract or correctness oracle. This addresses the candidate's stated impact lever (median TTFT) by sending warm delta transfers down the async-pull path while keeping cold/small prefixes on local recompute, which is the operational core of CPD applied at the scheduler-admission layer rather than at instance routing.

---

### 5. Replace static kv_recompute_threshold with a tier-/cost-aware admission gate inspired by Mooncake's KVCache-centric scheduler
- **Finding:** `find-0015` — *Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot*
- **Source URL:** <https://www.usenix.org/system/files/fast25-qin.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

At vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py:288-364, evolve the remote-decode branch in NixlConnectorScheduler.get_num_new_matched_tokens so the recompute-vs-pull decision is driven by an explicit cost comparison instead of the fixed integer self.kv_recompute_threshold. Concretely: (1) maintain lightweight EWMA estimates inside the scheduler of (a) per-block NIXL pull latency (already observed via the connector's transfer completion path) and (b) per-token local prefill cost for the current model/shape (derivable from recent prefill durations divided by token count, similar to how scheduler stats are tracked elsewhere in v1); (2) replace `count < self.kv_recompute_threshold` with a predicate of the form `estimated_remote_pull_cost(count, queue_depth) < alpha * estimated_recompute_cost(count, request_shape)`, with hysteresis (two thresholds, or a low-pass on the decision) to avoid flapping near the boundary; (3) keep self.kv_recompute_threshold as a fallback when no measurements exist yet, and as a hard floor (never pull below N tokens to amortize NIXL handshake/RTT). Preserve the existing correctness invariants: a positive token count is still backed by valid `remote_*` parameters and remote block IDs, the (num_external_tokens, async) return contract is unchanged, and the `(0, False)` path on missing params is untouched. The change is local to this method plus a small accumulator on the scheduler instance; no connector-side / worker-side contract changes are needed, so the existing NIXL scheduler tests and end-to-end remote-prefill/decode tests remain the oracle.

**Proposal rationale.**

Mooncake explicitly frames cached-prefix reuse versus recompute as a resource/cost tradeoff across heterogeneous tiers (CPU/DRAM/SSD/NIC) rather than a fixed length cutoff, which is exactly the gap in this candidate: today the gate is a single static integer with no dependence on transfer latency, queue depth, or prefill cost. Mooncake's KVCache-centric scheduling principle transfers cleanly to this single decision point — we don't need to adopt the full disaggregated cache fabric to benefit; we just need to make the local pull-vs-recompute admission gate cost-aware. For the stated multi-turn agentic workload, prefix lengths and network conditions vary substantially per turn, so a cost-aware threshold can avoid both pathologies the candidate's evolve_rationale calls out (paying NIXL setup for tiny prefixes, and redundantly recomputing cheaply transferable ones), directly targeting median TTFT.

---

### 6. Replace static kv_recompute_threshold with bandwidth-aware adaptive gate
- **Finding:** `find-0020` — *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*
- **Source URL:** <https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py:288-364 (NixlConnectorScheduler.get_num_new_matched_tokens), replace the fixed `count < self.kv_recompute_threshold` comparison in the remote-decode branch with an adaptive threshold derived from measured KV-transfer cost and local prefill cost. Have the NIXL connector layer expose recent transfer telemetry (e.g., observed per-block transfer latency or effective endpoint bandwidth from completed pulls, plus current in-flight transfer queue depth) and compute, on each call, a break-even token count: threshold = ceil(transfer_latency(count, bandwidth, rtt) / prefill_throughput_per_token). Use hysteresis (two thresholds with a small gap) and an EWMA over recent samples to avoid flapping when transfer cost is near the break-even point. Keep `kv_recompute_threshold` as a hard floor/ceiling and as the fallback when no telemetry is yet available, preserving the existing `(num_external_tokens, async)` contract and behavior on the remote-prefill branch. Existing remote-prefill/decode equality tests still apply since the change only shifts the recompute-vs-pull decision boundary, not the correctness oracle.

**Proposal rationale.**

The candidate's gap is a static, telemetry-blind threshold that cannot tell when remote pull is actually cheaper than local recompute. DistServe explicitly argues that placement and transfer decisions for disaggregated prefill/decode should be driven by measured bandwidth so that KV communication does not regress TTFT. Applying that idea here turns a hard-coded 64-token cliff into a cost-balanced gate using the same signals (bandwidth, transfer latency) DistServe relies on, which directly targets the candidate's stated impact mechanism (avoiding the worse of {network setup for tiny prefixes, recomputing cheaply transferable prefixes}) for the multi-turn agentic workload's median TTFT.

---

### 7. Replace static kv_recompute_threshold with bandwidth-aware remote-pull admission
- **Finding:** `find-0021` — *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*
- **Source URL:** <https://arxiv.org/html/2604.15039v1>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py` lines 288-364, change `NixlConnectorScheduler.get_num_new_matched_tokens` so that the remote-decode branch (lines 325-361) no longer compares `count` against a single static `self.kv_recompute_threshold`. Instead, decide pull-vs-recompute by estimating two costs and choosing the smaller: (a) expected remote-pull cost ≈ per-request setup latency + count * per_token_bytes / observed_effective_bandwidth, and (b) expected local recompute cost ≈ count * per_token_prefill_time at the current local prefill load. Maintain a small EWMA of recently observed NIXL transfer throughput / setup latency on the scheduler (updated when the connector reports completion of prior pulls) and use it as the bandwidth term; treat `kv_recompute_threshold` as the floor of this comparison so that very small `count` values still take the local path even under perfect network conditions. Add hysteresis (e.g. require the pull-cost to beat recompute by a configurable margin) to avoid flapping near the cross-over point. Keep the existing `(num_external_tokens, async)` contract intact: return `(0, False)` whenever recompute wins, and `(count, True)` only when remote pull is predicted to reduce TTFT. The decision is local to this method, so existing remote-prefill/decode tests that assert correctness of returned token counts and end-to-end output equality continue to act as the oracle.

**Proposal rationale.**

The candidate's gap is precisely a missing cost model: `kv_recompute_threshold` is a fixed integer that ignores transfer latency, RTT, queue depth, and prefill compute cost — exactly the dimensions the PrfaaS finding identifies as critical for deciding when to offload prefill versus recompute. The paper's `bandwidth-aware scheduling` and `selective offloading` technique transfers directly to this gate: instead of a length threshold, admit a remote pull only when the bandwidth-conditioned transfer cost is expected to beat local recompute. For multi-turn agentic workloads where prefix sizes and network conditions vary across turns, this can move median TTFT by avoiding the worse path per request — the same impact mechanism the candidate describes — while leaving the connector contract and the existing correctness oracle unchanged.

---

## Agent proposals

### 1. Co-admit sub-threshold remote-decode pulls that share a peer with an already-admitted pull in the current scheduler step
- **Agent:** claude

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py` at `NixlConnectorScheduler.get_num_new_matched_tokens` (lines 288-364), change the remote-decode branch so the `count < self.kv_recompute_threshold` gate is *per-peer-amortized within a scheduler step* rather than per-request in isolation. Concretely: (1) add a tiny per-step accumulator on the scheduler instance, e.g. `self._step_pending_pull_peers: dict[PeerId, int]`, keyed by the destination peer (derivable from the request's `remote_*` KV transfer parameters / remote block IDs already used at lines 325-361) and storing the cumulative `count` admitted to that peer in the current scheduler step. (2) Reset this dict at the start of each scheduler step via the existing per-step entry point (or a one-line hook in the surrounding scheduler loop) so it never persists across steps. (3) In the gate, change the predicate to: admit if `(count + self._step_pending_pull_peers.get(peer, 0)) >= self.kv_recompute_threshold`, otherwise apply the existing `count < self.kv_recompute_threshold` rejection. On admission, increment `self._step_pending_pull_peers[peer] += count`. Intuition: NIXL session/handshake/RTT cost is largely fixed per peer; the marginal cost of pulling additional blocks on the same step's session for the same peer is dominated by transfer time, not setup. So a request whose 30-token delta would normally fall below a 64-token threshold should be admitted when another request in the same step has already justified the round-trip to that peer. (4) Keep `kv_recompute_threshold` as the seed/fallback (the first request to a peer in a step still pays the standard gate); behavior degrades cleanly to today's behavior when only one request per peer is in flight. (5) Preserve the connector contract exactly: still return `(count, True)` on admission and `(0, False)` on rejection; no worker-side or wire-protocol change. Existing NIXL scheduler tests for `(num_external_tokens, async)` and end-to-end remote-prefill/decode equality remain the oracle; new tests can simulate two same-peer sub-threshold requests in one step and assert both are admitted.

**Novelty rationale.**

All seven listed deep_research_proposals (find-0002, -0008, -0013, -0014, -0015, -0020, -0021) make the threshold adaptive based on signals computed *per request*: prefix-cache coverage of that request, bandwidth/latency EMAs, recompute-vs-transfer cost for that request's `count`, in-flight transfer queue depth, hit-rate proxy, etc. None of them reason about *other requests in the same scheduler step* sharing a peer, and none amortize the NIXL handshake/RTT across multiple sub-threshold requests via co-admission. This proposal changes the basis of the decision from per-request to per-peer-per-step, which is orthogonal to (and composable with) every telemetry-driven threshold the listed findings propose — even with a perfectly tuned per-request break-even, today every request still pays the gate alone, missing the cross-request setup amortization that is particularly large in multi-turn agentic workloads where many concurrent sessions hit overlapping peer sets each step.

---

### 2. Bypass the recompute threshold when local recompute would be chunked
- **Agent:** codex

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py` inside `NixlConnectorScheduler.get_num_new_matched_tokens`, refine the remote-decode threshold check so a sub-threshold `count` is still admitted when local recompute would be split across multiple scheduler/model steps. Concretely, before returning `(0, False)` for `count < self.kv_recompute_threshold`, read `self.vllm_config.scheduler_config.long_prefill_token_threshold`; if it is positive and smaller than `count`, return `(count, True)` because rejecting the pull would force at least two chunked prefill iterations before the request can reach decode. Keep the current threshold behavior when chunked prefill is not configured or `count` fits within one local prefill chunk. Add a focused scheduler test with `kv_recompute_threshold=64`, `long_prefill_token_threshold=16`, and `remote_num_tokens=48` asserting the remote-decode request is admitted for async load, plus a control case where `count <= long_prefill_token_threshold` is still rejected.

**Novelty rationale.**

The listed deep_research_proposals adapt the gate using prefix-cache coverage, transfer/bandwidth telemetry, queue depth, per-request cost models, and load/hysteresis. Agent A amortizes the threshold across same-peer requests in the same scheduler step. None of them uses the scheduler's own chunked-prefill policy as a discrete admission signal: even if a remote delta is below the static NIXL amortization threshold, rejecting it can add multiple scheduler iterations to TTFT when local recompute cannot be completed in one chunk. This proposal is orthogonal to telemetry and peer co-admission, and can compose with either later.

---
