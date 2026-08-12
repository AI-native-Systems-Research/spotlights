# NixlBaseConnectorScheduler.__init__ transfer policy defaults

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py`](vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py) (lines 70–172)
- **Symbol:** `NixlBaseConnectorScheduler.__init__ transfer policy defaults`
- **Kind:** config_block
- **Estimated impact:** high
- **Id:** `cand-vllm_distributed_kv_transfer-0005`

## Description
Initializes NIXL policy defaults for KV lease duration, heartbeat interval, recompute threshold, bidirectional transfer enablement, and decoder KV block TTL.

## Current approach
Static extra-config defaults are applied once at scheduler construction. kv_recompute_threshold is later used as a fixed token-count cutoff in pull_scheduler.py to choose remote pull versus local recompute, independent of observed RTT, bandwidth, queue depth, or prefill throughput.

## Estimated impact explanation
A wrong pull-vs-recompute decision directly adds TTFT on every candidate remote-prefill hit; short turn-2 agentic prompts make this a frequent per-turn decision.

## Evolve rationale
The pull-vs-recompute break-even depends on workload and topology, and NIXL records transfer sizes and timings through xfer_stats.record_transfer. An EWMA policy keyed by remote engine and request size could adjust kv_recompute_threshold and lease/TTL margins while preserving external scheduler contracts. Oracle: NIXL scheduler/unit tests and nixl_integration correctness tests verify hit counts, block retention, and identical generated outputs; vllm bench can compare TTFT/TPOT.

## Deep research proposals

### 1. Replace static kv_recompute_threshold with a Dynamo-style cost-scored pull-vs-recompute decision
- **Finding:** `find-vllm_distributed_kv_transfer-0001` — *Routing Concepts*
- **Source URL:** <https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py` (lines 70-172), the transfer-policy defaults treat `kv_recompute_threshold` as a single static token cutoff, and `pull_scheduler.py` consults it as a fixed boundary between remote pull and local recompute. Change this initialization block so the scheduler additionally instantiates a small per-remote-engine cost model instead of collapsing the decision to one integer. Concretely: (1) Keep the extra-config value read at line 155 as an initial bias / floor; retain `kv_lease_duration`, `decoder_kv_blocks_ttl`, and `bidirectional_kv_xfer` unchanged. (2) Add three EWMA state maps keyed by `remote_engine_id` (mirroring the existing per-engine dicts like `_heartbeat_by_engine`): estimated per-token transfer time (updated from NIXL's `xfer_stats.record_transfer` durations and sizes), estimated remote prefill queue delay (updated from heartbeat/handshake round-trip already flowing through `HeartbeatInfo`), and estimated local prefill throughput (from the scheduler's own step timings). (3) At the pull-vs-recompute call site in `pull_scheduler.py`, replace the `num_remote_tokens > kv_recompute_threshold` test with a lowest-cost selection between two options — `cost_pull = transfer_time(num_remote_tokens, engine) + queue_delay(engine)` and `cost_recompute = num_remote_tokens / local_prefill_tps` — mirroring Dynamo's structure of summing worker-specific cost terms and picking the minimum. (4) Clamp the effective threshold so it never falls below the configured floor to preserve today's conservative behavior when EWMAs are cold. This preserves the external scheduler contract, the extra-config keys, and the heartbeat/TTL machinery; only the interpretation of `kv_recompute_threshold` changes from a hard cutoff to a warm-start bias for a scored decision.

**Proposal rationale.**

The candidate's own `evolve_rationale` names the gap directly: the pull-vs-recompute break-even depends on RTT, bandwidth, queue depth, and prefill throughput, but the current defaults collapse it to one static integer. The Dynamo routing-concepts finding contributes the specific missing shape — a cost function that sums multiple worker-specific terms (KV overlap benefit, projected prefill load, active-request load) and picks the lowest-cost eligible option. That structure ports cleanly here even though vLLM's scheduler is not choosing among many workers: the two 'options' become pull-from-remote vs. recompute-locally, and the same additive cost terms (transfer time, remote queue delay, local prefill cost) apply. The finding also justifies why per-engine keying matters — different remote engines have different observed costs — which aligns with the EWMA-per-remote-engine sketch in the rationale. For the caller's multi-turn agentic workload targeting median TTFT/TPOT, a scored decision is most valuable precisely on short turn-2 prompts where the static 64-token cutoff most often mispredicts, so the finding's cost-function idea addresses the stated impact directly.

---

### 2. Replace static kv_recompute_threshold with a CacheBlend-style dynamic transfer-vs-recompute policy in NIXL scheduler
- **Finding:** `find-vllm_distributed_kv_transfer-0008` — *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*
- **Source URL:** <https://www.alphaxiv.org/abs/2405.16444>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:70-172, keep the extra-config value as an initial/default but treat kv_recompute_threshold as a dynamic quantity maintained per remote engine (and optionally per size bucket) rather than a fixed constant consumed unchanged by pull_scheduler.py. On each remote-prefill decision, use an EWMA of observed transfer time from NIXL xfer_stats.record_transfer and an EWMA of local prefill throughput (tokens/sec) to compute the break-even token count at which pulling remote KV is no longer faster than recomputing. Following CacheBlend's core insight that recompute cost is proportional to the number of selected tokens and can be hidden under storage load, the policy should (a) compare estimated pull latency against estimated recompute latency for the request's hit length, and (b) when hits exceed the break-even, also allow partial recompute of the last k tokens overlapped with pipelined KV load if the connector supports layer-wise streaming. kv_lease_duration, decoder_kv_blocks_ttl, and _heartbeat_interval defaults remain externally configurable; only the threshold used at pull-vs-recompute time becomes adaptive. Expose the current effective threshold via a read-only accessor so pull_scheduler.py's existing consumer switches over without changing the scheduler contract.

**Proposal rationale.**

The candidate's gap is that kv_recompute_threshold is a static, workload-agnostic cutoff even though NIXL already records the timings needed to know when pulling beats recomputing. CacheBlend directly addresses the same decision boundary in a RAG/multi-turn setting: it chooses a recompute ratio that can be hidden under storage load and shows the compute overhead scales with selected tokens. Its dynamic transfer-vs-recompute framing transfers cleanly to the pull-vs-recompute decision here, and its latency-hiding idea suggests when hits are marginal, a small partial recompute can be overlapped with the KV load rather than paying a full round-trip. This targets median TTFT on multi-turn agentic prompts where short turn-2 requests are exactly the regime where a wrong static threshold hurts most.

---

### 3. Learn kv_recompute_threshold from recorded NIXL transfer stats
- **Finding:** `find-vllm_distributed_kv_transfer-0015` — *Use predicted latency-based routing with GKE Inference Gateway*
- **Source URL:** <https://docs.cloud.google.com/kubernetes-engine/docs/how-to/use-predicted-latency-based-routing?authuser=2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the static kv_recompute_threshold default currently applied in NixlBaseConnectorScheduler.__init__ (vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:70-172) with an adaptive, EWMA-based estimate keyed by remote engine identity and request size buckets. Feed the estimator from NIXL's existing xfer_stats.record_transfer signal (bytes moved, elapsed time) plus lightweight local features already available to the connector scheduler (queue depth of pending pulls, observed prefix/block hit counts, and recent local prefill throughput). At the pull-vs-recompute decision site in pull_scheduler.py, compare the predicted remote-pull latency for the candidate request size against the predicted local recompute latency for the miss-suffix token count, and choose the smaller; the resulting break-even token count is what dynamically replaces the static threshold. Keep the extra-config value as an initial prior and a hard clamp so external scheduler contracts (kv_lease_duration, heartbeat, bidirectional flag, decoder TTL) remain untouched. Persist the running EWMA in-process only, one entry per (remote_engine_id, size_bucket), with a small (e.g. 8-16) fixed bucket set to bound memory.

**Proposal rationale.**

The finding describes GKE's llm-d Inference Gateway replacing static routing weights with a live-trained latency predictor over features that closely match what NIXL already records: transfer sizes/timings via record_transfer, KV cache/prefix hit signals used by the pull scheduler, and pending-transfer queue depth. That is the same class of decision the candidate makes with a static kv_recompute_threshold, so the transferable idea is 'replace fixed cutoff with an online latency estimate' rather than adopting XGBoost or the gateway itself. It directly targets the evolve_rationale's stated gap (threshold does not react to observed RTT/bandwidth/queue depth) and the caller's TTFT objective on multi-turn agentic prompts, where a wrong recompute call adds latency to every turn-2 remote-prefill hit.

---

## Agent proposals

### 1. Adapt decoder_kv_blocks_ttl and heartbeat cadence per remote engine from observed inter-turn reuse gaps
- **Agent:** claude

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:70-172, keep the extra-config reads for kv_lease_duration (line 70), kv_recompute_threshold (line 155), bidirectional_kv_xfer (line 163), and decoder_kv_blocks_ttl (line 168) as initial values but treat decoder_kv_blocks_ttl (and the derived _heartbeat_interval at line 76) as per-remote-engine quantities adapted from observed inter-turn reuse timing rather than global constants. Concretely: (1) Add a new per-engine map self._reuse_gap_ewma: dict[EngineId, float] alongside the existing self._heartbeat_by_engine (line 118) and self._heartbeat_req_engine (line 120) dicts. (2) In on_new_request (line 189), when a request arrives with do_remote_prefill and its remote_engine_id already appears in _heartbeat_req_engine or was recently torn down (track a small self._last_seen_by_engine: dict[EngineId, float] populated in _stop_heartbeat at line 225), compute the wall-clock gap between the previous teardown and the new registration and feed it into an EWMA per remote_engine_id. This gap is precisely the D-side reuse interval that decoder_kv_blocks_ttl must exceed to avoid a wasted recompute on turn-2. (3) Expose an effective_decoder_kv_blocks_ttl(remote_engine_id) accessor returning clamp(2 * gap_ewma[engine], floor=configured_value, ceiling=configured_value * K) where the configured extra-config value is retained as the floor to preserve today's conservative behavior. Emit this per-engine effective TTL in NixlHandshakePayload / KV transfer params so the D-side actually applies it when scheduling block eviction. (4) Similarly, retune _heartbeat_interval per engine to max(kv_lease_duration/6, gap_ewma[engine]/4) so heartbeats scale with observed reuse cadence rather than a fixed lease_duration // 6 (line 76) — under a bursty agent that reuses every 200ms, a 5s heartbeat is wasted work; under a slow agent that reuses every 60s, six evenly-spaced heartbeats over a 30s lease is fine. (5) Cold-start: when an engine has no observations, fall back exactly to today's constants. The pull_scheduler.py boundary at line 97 is untouched — this proposal is orthogonal to the pull-vs-recompute decision.

**Novelty rationale.**

All three listed deep_research_proposals target the same variable — kv_recompute_threshold — and the same decision — pull versus recompute for a single request. Proposal #1 (Dynamo-style cost) and #3 (llm-d Inference Gateway latency predictor) both propose EWMA cost models keyed by remote engine and request size for that binary choice; proposal #2 (CacheBlend-style) additionally suggests overlapping partial recompute with KV load but still centers on the recompute-vs-transfer boundary for one request. This proposal touches a different pair of defaults in the same __init__ block (decoder_kv_blocks_ttl and _heartbeat_interval), uses a different signal (inter-request reuse gap distribution derived from heartbeat register/unregister events, not xfer_stats.record_transfer transfer timings), and addresses a different failure mode (D-side block eviction before turn-N reuse forces a fresh remote pull or recompute regardless of what the pull-vs-recompute policy chooses). It is compatible with all three existing proposals — they could be adopted in addition, not instead — and directly targets the multi-turn agentic workload hint by making turn-N KV availability itself workload-adaptive rather than governed by a fixed 480s TTL.

---

### 2. Make bidirectional KV transfer opt-in per engine after capability handshake
- **Agent:** codex

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:70-172`, keep `bidirectional_kv_xfer` as the configured global default, but add a per-remote-engine effective flag that is initialized from the NIXL handshake payload rather than blindly applying one scheduler-wide value. The scheduler should record whether each peer advertises decoder-to-prefill KV support, compatible block layout, and enough free transfer slots; `pull_scheduler.py` and the send path should consult `effective_bidirectional_kv_xfer(remote_engine_id)` before scheduling reverse transfers. For cold or unadvertised peers, fall back to one-way transfer even if the global flag is enabled. This avoids enabling reverse KV movement on engines where it can only add control-plane work, queue contention, or failed transfer retries, while preserving the existing extra-config key as the operator's upper bound.

**Novelty rationale.**

The three deep-research proposals all focus on replacing `kv_recompute_threshold` with a dynamic pull-vs-recompute model, using transfer timings, queue depth, or local prefill throughput. Agent A's proposal focuses on adapting `decoder_kv_blocks_ttl` and `_heartbeat_interval` from inter-turn reuse gaps. This proposal targets the separate `bidirectional_kv_xfer` default in the same initialization block and addresses a different failure mode: unnecessary or unsupported reverse-direction transfers consuming NIXL scheduling capacity and adding retries. It is compatible with adaptive thresholds and adaptive TTLs because it only bounds whether the reverse transfer path is eligible per peer.

---
