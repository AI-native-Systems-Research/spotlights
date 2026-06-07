# MoRIIOConnectorWorker handshake-ready spin loops

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py) (lines 1298–1389)
- **Symbol:** `MoRIIOConnectorWorker handshake-ready spin loops`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0013`

## Description
Worker-side MoRIIO save/load handshake readiness loops that wait for remote agents to become available before issuing write or read transfers.

## Current approach
After initiating a background handshake, `save_kv_layer` and `start_load_kv` enter `while True` loops. When `_ready_requests` is empty and the `remote_engine_id` is not present in `write_ready_flags` or `load_ready_flag`, the loop immediately continues, creating a busy spin on the engine path until another thread flips readiness. When ready, each loop drains at most one queued request before breaking.

## Estimated impact explanation
First use of a MoRIIO remote peer can sit on the TTFT path, and a tight spin loop can consume CPU needed by scheduler and networking threads. Replacing it with event-driven wakeups can reduce handshake-related TTFT spikes and lower TPOT interference during multi-turn bursts that touch new peers.

## Evolve rationale
The specific constructs are the busy `while True` readiness gates and `_ready_requests.get_nowait()` calls in `save_kv_layer` and `start_load_kv`. They can be evolved into a condition/event-driven ready queue keyed by remote engine, with bounded waits and batch draining, while preserving the same transfer start conditions. Correctness oracle: with a fake handshake thread and fake `_write_blocks_for_req`/`_read_blocks_for_req`, transfers must be invoked only after the matching remote agent/flag is ready, every queued request must be invoked exactly once, and `metadata.reqs_to_send` handling must remain unchanged.

## Deep research proposals

### 1. Replace handshake spin loops with non-blocking, overlap-friendly readiness gating
- **Finding:** `find-0005` — *Disaggregated Serving — TensorRT LLM*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0rc4/features/disagg-serving.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor the worker-side handshake-ready loops in `save_kv_layer` and `start_load_kv` (vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:1298-1389) so they no longer occupy the engine thread on a busy `while True` waiting for `write_ready_flags`/`load_ready_flag` to populate. Instead: (1) keep a per-`remote_engine_id` event/condition variable (or a `queue.Queue` keyed by engine id) that the background handshake thread signals exactly once when the agent becomes ready; (2) on the engine path, do a single non-blocking check — if the matching agent is not ready and `_ready_requests` is empty, return immediately without issuing any transfer for this layer/request, leaving the pending request enqueued for the next scheduler step; (3) when the agent is ready, drain all currently-ready queued requests for that engine in one pass (batch drain rather than one-per-call) and dispatch their `_write_blocks_for_req`/`_read_blocks_for_req` invocations. This preserves the existing correctness conditions (transfers only fire after the matching remote agent/flag is ready, each queued request fires exactly once, `metadata.reqs_to_send` handling unchanged) while letting the engine return to scheduling and computing other independent requests during the handshake window. Optional refinement: dispatch the drained transfers onto the existing background transfer thread/executor so the engine call site is fully non-blocking even when readiness arrives.

**Proposal rationale.**

The TensorRT-LLM disaggregated-serving guidance frames KV transmission as something that should overlap with computation for other requests rather than stall the forward pass. The candidate's current spin-loop does the opposite: it pins the engine thread on a remote-agent readiness check, so during first-touch handshakes (a known TTFT contributor in multi-turn agentic workloads where new peers appear) no other independent request can make progress on that worker. Converting the gate into an event-driven, non-blocking check with batch draining directly applies the finding's overlap principle to this specific call site — independent requests keep flowing through `save_kv_layer`/`start_load_kv` while the slow handshake completes asynchronously, and the readiness signal triggers a single batched dispatch instead of a tight CPU-burning poll that contends with scheduler and networking threads. This addresses both the TTFT spike on first peer contact and the median TPOT interference the candidate flags.

---

### 2. Pre-warm MoRIIO remote-peer handshakes to avoid first-request readiness waits
- **Finding:** `find-0007` — *Prefill-decode disaggregation — Ray 2.55.1*
- **Source URL:** <https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Augment MoRIIOConnectorWorker so that the handshake/agent setup for each remote engine is initiated eagerly rather than lazily on the first save_kv_layer / start_load_kv call at vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:1298-1389. Concretely: (1) on worker initialization (or as soon as the set of peer engine IDs becomes known via metadata exchange / scheduler hooks), spawn the same background handshake threads that today are launched on demand, populating write_ready_flags / load_ready_flag and registering remote agents before any real request arrives; (2) expose a small pre-warm entry point (e.g., warmup_remote_peer(remote_engine_id)) that the engine can invoke during startup or when a new peer is first observed; (3) keep the existing readiness checks in save_kv_layer / start_load_kv as a correctness fallback, but in the common pre-warmed case the loops should observe the flag already set on the first iteration and proceed immediately, eliminating the busy while-True spin on the engine path. The transfer start conditions, _ready_requests semantics, and metadata.reqs_to_send handling remain unchanged; only the timing of when the handshake completes is moved off the request critical path.

**Proposal rationale.**

The finding documents that NIXL-style KV transfer connectors require a per-pair prefill/decode handshake that, by default, happens eagerly on the first request, and Ray Serve's prefill-decode pattern explicitly pre-warms the connector to remove that latency from TTFT. The MoRIIO worker has the same shape of problem: the first request to a new remote peer pays for handshake establishment plus a busy readiness spin that competes with scheduler and networking threads. Pre-warming directly attacks the root cause of the candidate's TTFT and TPOT-interference concerns (multi-turn agentic workloads frequently touch new or idle peers), and is complementary to — but stronger than — merely replacing the spin with an event wait, because in the pre-warmed common case there is nothing to wait for at all.

---

### 3. Make MoRIIO handshake-gated transfers non-blocking on the engine path
- **Finding:** `find-0008` — *Disaggregated Serving | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the busy `while True` readiness gates in `MoRIIOConnectorWorker.save_kv_layer` (vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:1318-1332) and `start_load_kv` (lines 1373-1387) with a non-blocking deferral pattern that returns control to the engine forward path immediately when the remote agent is not yet ready, instead of spinning until readiness flips. Concretely: (1) when `remote_engine_id` is missing from `write_ready_flags`/`load_ready_flag` and `_ready_requests` is empty, do not loop — record the (req_id, layer_name) pending state in a per-engine pending map keyed by `remote_engine_id` and return; (2) drive transfer start from the handshake-completion callback (the same path that currently sets `write_ready_flags`/`load_ready_flag` and pushes onto `_ready_requests`), which on flip should drain all pending entries for that engine via `_write_blocks_for_req`/`_read_blocks_for_req`; (3) replace `_ready_requests.get_nowait()` single-drain with a batched drain so every queued request is invoked exactly once; (4) keep `metadata.reqs_to_send` updates and the existing transfer-start conditions unchanged so the correctness oracle still holds. The engine’s `save_kv_layer`/`start_load_kv` thus becomes a fast, non-blocking enqueue, mirroring the disaggregated-serving principle that KV transfer must not stall forward passes for other in-flight requests.

**Proposal rationale.**

The finding’s core idea — “the KV transfer is non-blocking, allowing GPU forward passes to continue serving other requests during the transfer” — directly addresses the candidate’s gap: today the engine thread either spin-loops (`continue` with no wait) or, even with an event/condition replacement, would still block the forward path until a remote handshake lands. By deferring transfer initiation to a handshake-completion callback and returning immediately when not ready, the worker stops holding the engine on a first-touch peer, which is exactly the TTFT/TPOT pressure point flagged in the candidate (multi-turn workloads that occasionally touch new MoRIIO peers). This is a transferable design constraint from Dynamo’s disaggregated-serving doc rather than a restatement of the candidate’s existing “event-driven ready queue with bounded waits” framing, which still implicitly blocks the caller; the finding pushes the change toward true non-blocking enqueue + background-driven start.

---

## Agent proposals

### 1. Add bounded handshake timeout with circuit-breaker abort path for unready MoRIIO peers
- **Agent:** claude

**Detailed description.**

Augment the worker-side handshake-ready loops in MoRIIOConnectorWorker.save_kv_layer (vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:1318-1332) and start_load_kv (lines 1373-1387) with an explicit per-(remote_engine_id, req_id) deadline and a circuit-breaker so a wedged or unreachable peer cannot indefinitely accumulate pending KV requests. Concretely: (1) on the first time a request waits on a given remote_engine_id, record a monotonic deadline (configurable via env, e.g. VLLM_MORIIO_HANDSHAKE_TIMEOUT_S with a sane default such as 30s) keyed by (remote_engine_id, req_id); (2) if the deadline elapses without write_ready_flags/load_ready_flag being set, mark that remote_engine_id as 'unhealthy' in a small per-worker circuit-breaker map, drop the pending request from _ready_requests, surface the failure back to the engine through the existing return paths (e.g., by populating metadata.reqs_to_send with a 'failed' marker or raising a typed KVTransferUnavailable error that the engine path translates into a request abort/retry), and tear down the partially-initialized agent state so a future request can retry the handshake cleanly; (3) while a remote_engine_id is in the unhealthy state, immediately fail-fast new save_kv_layer/start_load_kv calls targeting it (with a short half-open probe window after which a single new handshake attempt is allowed); (4) keep all existing transfer-start conditions and metadata.reqs_to_send semantics for the success path unchanged so the correctness oracle (transfers fire only after readiness, each queued request invoked exactly once) still holds. Optionally emit a structured log/metric on each timeout and circuit-open event so operators can see flapping peers during multi-turn agentic runs.

**Novelty rationale.**

All three existing deep_research_proposals (find-0005, find-0007, find-0008) target the *latency/overlap* axis: they replace the busy spin with event-driven gating, pre-warm handshakes, or defer transfer start to a callback. None of them address the *liveness/failure* axis — namely, what happens when a remote peer never becomes ready. The current code (and even the proposed event-driven or callback-driven replacements) would let pending requests accumulate forever against a wedged or unreachable MoRIIO peer, which in a multi-turn agentic workload that touches many peers is a realistic source of stuck requests and unbounded queue growth that ultimately *worsens* TTFT/TPOT for all unrelated requests sharing the worker. A bounded handshake timeout plus a circuit breaker is orthogonal to (and composable with) the spin-loop refactor: it gives the engine a deterministic upper bound on handshake-induced stalls and an explicit abort path, which the existing proposals leave unspecified.

---

### 2. Deduplicate in-flight MoRIIO handshakes per remote engine
- **Agent:** codex

**Detailed description.**

Make the handshake initiation path used by `save_kv_layer` and `start_load_kv` idempotent for each canonical `remote_engine_id` while the first handshake is still running. In `moriio_connector.py`, the callers check `dp0_remote_engine_id` in `_remote_agents`, but the protected second check uses unsuffixed `remote_engine_id`, while `_remote_agents` and `_handshake_futures` are populated with DP-suffixed keys. As a result, every layer/request that reaches an unready peer can enqueue another all-DP handshake before the first one finishes. Replace this with a per-base-engine handshake group, e.g. `_pending_handshake_reqs[remote_engine_id]` plus `_handshake_group_futures[remote_engine_id]`: the first request submits the DP handshake futures and records `(req_id, meta)`; later requests for the same remote engine only append their entry. When the group future completes, set the ready flags once and enqueue all accumulated entries for transfer start, then clear the group state. Keep `_remote_agents` keyed by `get_engine_name_with_dp(...)` for established agents, but use the base `remote_engine_id` consistently for in-flight deduplication.

**Novelty rationale.**

The existing deep_research_proposals focus on how the engine waits or does not wait for readiness: event-driven gating, pre-warming, callback-driven non-blocking transfer start, and batch draining. Agent A adds bounded timeout and circuit-breaker behavior for failed peers. None of them address the current key-mismatch/idempotency problem that can submit duplicate ZMQ handshakes for the same peer while readiness is pending. This proposal targets that separate source of TTFT/TPOT pressure: avoid redundant handshake work, remote-agent registration, and single-threaded executor backlog before any readiness policy or timeout logic runs.

---
