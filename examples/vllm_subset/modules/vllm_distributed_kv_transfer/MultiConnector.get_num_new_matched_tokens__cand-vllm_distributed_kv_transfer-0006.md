# MultiConnector.get_num_new_matched_tokens

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py) (lines 385–404)
- **Symbol:** `MultiConnector.get_num_new_matched_tokens`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_distributed_kv_transfer-0006`

## Description
Routes prefix-hit lookup across an ordered connector list and pins the request to the first connector that reports tokens > 0.

## Current approach
First-positive-wins policy is enforced by the to_return[0] == 0 guard. The method still queries every connector to detect pending async lookups, but does not compare hit length, expected load latency, async behavior, backpressure, or tier speed before choosing a connector.

## Estimated impact explanation
The chosen connector determines where turn-2 KV is loaded from, often the dominant TTFT factor when multiple cache tiers are configured.

## Evolve rationale
Multi-turn deployments may combine GPU, CPU, offload, remote, and object-store connectors. A scored router using hit length, estimated load time, and connector load/backpressure can replace the tie-breaker while preserving update_state_after_alloc's chosen-connector contract. Oracle: tests/v1/kv_connector/unit/test_multi_connector.py and NIXL multi-connector integration tests assert delegation, chosen connector behavior, and output correctness.

## Deep research proposals

### 1. Replace first-positive-wins with a scored connector router using overlap + projected load cost
- **Finding:** `find-vllm_distributed_kv_transfer-0001` — *Routing Concepts*
- **Source URL:** <https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py at MultiConnector.get_num_new_matched_tokens (lines 385-404), replace the current to_return[0] == 0 tie-breaker with a cost-based selection loop modeled on Dynamo's router cost function. Concretely: (1) continue iterating every connector to preserve async-pending semantics (toks is None short-circuit still returns (None, False)); (2) for each connector reporting a positive hit, compute a per-connector cost that combines (a) KV overlap gain, approximated by toks (larger hit = lower cost / higher benefit), (b) projected load latency for transferring those toks from that tier (a per-connector method such as c.estimated_load_latency(request, toks) if available, otherwise a tier-weighted constant configured on the connector's KVTransferConfig), and (c) an active-load / backpressure term drawn from state the connector already tracks (in-flight loads counter, queue depth, or entries in self._requests_to_connector currently assigned to that index); (3) select the connector with the lowest cost among those with toks > 0 instead of the first, pin it via self._requests_to_connector[request.request_id] = i, and return (toks, load_async) from that chosen connector. Preserve the update_state_after_alloc contract at lines 406-416 unchanged: only the chosen index is recorded, so downstream allocation and the MultiConnector tests in tests/v1/kv_connector/unit/test_multi_connector.py continue to see exactly one chosen connector. Where a connector cannot supply latency/load estimates, fall back to a static tier-priority weight so the change degrades gracefully to today's ordered-list behavior.

**Proposal rationale.**

The candidate explicitly calls out that get_num_new_matched_tokens picks the first connector reporting any hit and never compares hit length, load latency, async behavior, or backpressure across tiers -- exactly the gap Dynamo's router closes. The finding provides a concrete, transferable formulation ('cost function combines three worker-specific cost terms' and 'router selects the lowest-cost eligible worker') that maps cleanly onto MultiConnector's ordered connector list: overlap = matched tokens, prefill/decode load = per-tier transfer latency, active-request load = in-flight loads per connector. For multi-turn agentic workloads with mixed GPU/CPU/offload/remote tiers -- the exact workload hint in the caller context -- choosing the lowest-cost tier rather than the first-hit tier is the dominant TTFT lever, matching the candidate's estimated_impact rationale. The change is localized to scoring inside the existing loop and preserves the chosen-connector contract enforced by update_state_after_alloc, so the multi-connector oracle tests remain valid.

---

### 2. Score connectors by predicted load latency instead of first-positive-wins
- **Finding:** `find-vllm_distributed_kv_transfer-0015` — *Use predicted latency-based routing with GKE Inference Gateway*
- **Source URL:** <https://docs.cloud.google.com/kubernetes-engine/docs/how-to/use-predicted-latency-based-routing?authuser=2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the `to_return[0] == 0 and toks > 0` tie-breaker in `MultiConnector.get_num_new_matched_tokens` (vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:385-404) with a scoring pass modeled on the GKE Inference Gateway / llm-d EPP scheduler. Continue querying every connector so pending async lookups still return `(None, False)`, but collect all `(i, toks, load_async)` triples that report `toks > 0` into a candidate set. Score each candidate by an estimated `time-to-serve = predicted_load_latency(connector_i, toks) - saved_prefill_time(toks)` where the predictor is an online-learned regressor (start with a simple per-connector EWMA of bytes-per-second and queue depth, keep the door open to swap in an XGBoost-style model as llm-d does). Features available at this call site: `toks` (prefix match score), per-connector rolling transfer throughput, per-connector in-flight/pending load count (already trackable via `_requests_to_connector` and existing `on_new_request`/`request_finished` hooks), and `load_async` (async loads are cheaper to hide under prefill). Pick the argmin-latency connector, pin it via `self._requests_to_connector[request.request_id] = i`, and return its `(toks, load_async)`. Preserve the current ordering-based behavior as the fallback when the predictor has insufficient samples (cold start) so existing tests in tests/v1/kv_connector/unit/test_multi_connector.py that assert first-positive delegation and the NIXL multi-connector integration tests continue to hold. Record actual per-connector load durations in `request_finished` (or an equivalent completion hook) to close the training loop.

**Proposal rationale.**

The candidate's gap is exactly the one llm-d's predicted-latency router addresses: a static heuristic (here, connector ordering) picks a source without accounting for hit length, connector speed, or backpressure, which directly hurts multi-turn TTFT when GPU/CPU/offload/remote tiers coexist. The finding contributes a concrete, transferable mechanism (online-trained latency predictor over cache utilization, queue depth, and prefix match score) that maps cleanly onto features the MultiConnector already has or can cheaply collect, and its output (a scalar latency estimate) is a drop-in replacement for the `to_return[0] == 0` tie-breaker while keeping `update_state_after_alloc`'s single-chosen-connector contract intact.

---

## Agent proposals

### 1. Speculative parallel prefetch from top-K connectors with first-complete-wins arbitration
- **Agent:** claude

**Detailed description.**

Change MultiConnector.get_num_new_matched_tokens (vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:385-404) from a single-connector pinning decision into a bounded speculative multi-source load. Today the method must commit exactly one connector before update_state_after_alloc runs, which forces any router (first-positive, cost-scored, or latency-predicted) to guess correctly up front. Instead: (1) keep the existing loop that queries every connector so pending-async semantics are preserved (toks is None still returns (None, False)); (2) among connectors reporting toks > 0, select the top-K (K=2 by default, configurable via KVTransferConfig, K=1 preserves today's behavior) using max(toks) as the primary key so the largest prefix hit is always among the speculators; (3) pin the request to the connector with the largest hit as the 'primary' via self._requests_to_connector[request.request_id] = i_primary and return its (toks, load_async), but additionally record the shadow set in a new self._speculative_loads[request.request_id] = [(i_other, toks_other, load_async_other), ...]; (4) in start_load_kv, dispatch async loads on all speculative connectors in parallel (bounded to the primary's toks so the KV block layout matches what allocation reserved), and in get_finished (or the per-connector load-completion hook) treat the first connector to signal completion for those blocks as authoritative, cancel/abort the losers via a new connector method abort_load(request_id) with a no-op default, and update self._requests_to_connector to the winning index so update_state_after_alloc's downstream bookkeeping and request_finished routing remain consistent; (5) fall back to K=1 automatically when any candidate reports load_async=False (sync tiers cannot be raced safely) or when the speculators' aggregate projected bandwidth would exceed a per-tier budget tracked on the connector. Tests in tests/v1/kv_connector/unit/test_multi_connector.py continue to see a single chosen connector at update_state_after_alloc time (the primary), and a new test asserts that when the primary stalls, the shadow load resolves the request without a retry.

**Novelty rationale.**

Both existing deep_research_proposals stay inside the 'pick one connector before allocation' contract and only change how that single pick is scored -- one uses a static cost function (overlap + tier weight + backpressure), the other uses an online-learned latency predictor. Both still commit to a single source and pay the full penalty of a mispredicted tier (cold cache miss, transient queue spike, network hiccup on a remote tier). This proposal is structurally different: it keeps the single-chosen-connector contract at the update_state_after_alloc boundary but overlaps loads from multiple tiers between get_num_new_matched_tokens and load completion, arbitrating on actual observed completion rather than any predicted score. For a multi-turn agentic workload where turn-2 TTFT is dominated by the slowest tier that happened to be picked, hedged loads are a well-established latency-reduction technique (tail-tolerant systems, BigTable/Spanner hedged reads) that neither listed proposal contemplates -- they optimize prediction accuracy; this eliminates the need to predict correctly in the first place. The mechanism (top-K speculators, first-complete-wins, abort losers, primary is the pinned index) is orthogonal to and composable with either scoring change: a better scorer picks a better primary and shadow set, but the tail-latency win comes from the racing itself.

---

### 2. Require compatible hit spans before accepting a connector match
- **Agent:** codex

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py` at `MultiConnector.get_num_new_matched_tokens` (lines 385-404), strengthen the routing decision so a connector can only be pinned if its reported match is compatible with the allocation and load contract expected by the rest of `MultiConnector`. Today the method only compares `toks > 0`, which assumes every connector's positive hit refers to the same prefix span and block alignment. Add an internal validation step that asks each connector for, or derives from the request metadata, the concrete matched block/span descriptor alongside `toks`; discard or defer candidates whose span is not a contiguous prefix ending on a loadable block boundary, whose token count would require a different allocation shape than the chosen prefix length, or whose span conflicts with an already pinned connector for the same request. If multiple connectors report the same valid span, keep the existing ordered fallback unless a separate scorer is introduced. If connectors cannot expose span details yet, add a conservative capability flag so legacy connectors continue using current behavior, while multi-tier connectors that can report spans get correctness protection. Extend `tests/v1/kv_connector/unit/test_multi_connector.py` with a case where an earlier connector reports a larger but non-prefix or misaligned match and a later connector reports a smaller valid prefix; assert that only the valid connector is recorded in `_requests_to_connector` and that `update_state_after_alloc` still delegates to exactly one connector.

**Novelty rationale.**

The deep_research proposals focus on selecting the best positive connector by cost, predicted latency, backpressure, and hit length. Agent A proposes racing top-K loads and choosing the first completion. This proposal is different: it addresses the validity of a positive hit before any scoring or hedging happens, preventing `get_num_new_matched_tokens` from pinning a connector whose reported token count cannot be safely loaded under the downstream allocation contract. It is orthogonal to cost routing and speculative loading because those strategies still need a candidate set whose hits are semantically compatible, not just numerically positive.

---
