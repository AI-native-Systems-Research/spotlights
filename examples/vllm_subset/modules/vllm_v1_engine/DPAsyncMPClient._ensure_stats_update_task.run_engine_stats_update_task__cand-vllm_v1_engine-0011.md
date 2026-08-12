# DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/core_client.py`](vllm/v1/engine/core_client.py) (lines 1305–1408)
- **Symbol:** `DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0011`

## Description
Async background stats receiver that maintains the frontend's local DP load-balancing snapshot from the coordinator XSUB socket.

## Current approach
Polls the stats and first-request sockets, drains all pending stats messages with nonblocking recv, keeps only the latest counts/wave/running tuple, slices counts to local managed ranks, and directly replaces lb_engines without smoothing or staleness metadata.

## Estimated impact explanation
Stale or noisy stats cause mis-routes that inflate per-engine queues; improving snapshot quality can reduce median TTFT tails in DP agentic bursts.

## Evolve rationale
DPLBAsyncMPClient.get_core_engine_for_request depends on this snapshot, so freshness and smoothing determine routing quality. Candidate changes include exposing snapshot age, EWMA or hysteresis for noisy counts, bounded draining, or routing-aware first-request wakeups. Correctness oracle: tests/v1/engine/test_engine_core_client.py should still show stats propagation, local-rank slicing, scale-up handling, and successful request routing.

## Deep research proposals

### 1. Split lb_engines snapshot into prefill and decode load dimensions
- **Finding:** `find-vllm_v1_engine-0006` — *Scheduler*
- **Source URL:** <https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/03-scheduler.md>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the DP load-balancing snapshot maintained by DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task (vllm/v1/engine/core_client.py:1305-1408) from a single-scalar per-engine load into two separate dimensions: (a) prefill/input-token pressure (e.g. queued prefill tokens or waiting-request count weighted by prompt length) and (b) decode/output-token pressure (e.g. currently generating requests, KV usage). Concretely: change the message payload drained from the XSUB stats socket at lines 1383-1394 so the coordinator publishes per-engine (prefill_load, decode_load, kv_cache_usage, running) tuples instead of (waiting, running, kv_cache_usage), update the slicing at lines 1401-1404 to preserve the two-dimensional structure, and expand the empty-init and scale-up padding at lines 1278 and 1356-1364 to match. The receiver task otherwise keeps its existing drain-latest semantics, first-request wakeup handling, and local-rank slicing. Downstream, DPLBAsyncMPClient.get_core_engine_for_request (lines 1471-1494+) then scores engines by a shape-aware combination — e.g. weight prefill_load when the incoming request is a fresh prompt (new turn / long input) and weight decode_load for continuing turns — instead of collapsing both into one waiting+running scalar. Existing tests in tests/v1/engine/test_engine_core_client.py must still show stats propagation, local-rank slicing, scale-up handling, and successful routing; add coverage that a prefill-heavy engine is deprioritized for new-prompt requests while remaining eligible for pure decode continuation.

**Proposal rationale.**

The finding's core transferable idea is that prefill/input and decode/output load are qualitatively different resources and should be balanced as separate dimensions rather than summed into one scalar. The candidate's snapshot today only carries an aggregate (waiting, running, kv_cache_usage) tuple, and the router collapses waiting+running further, so multi-turn agentic bursts — where new turns are prefill-heavy and ongoing turns are decode-heavy — cannot be steered away from engines that are prefill-saturated but decode-idle (or vice versa). Widening the snapshot at the receiver is a prerequisite for any shape-aware routing decision, directly addresses the median TTFT objective (TTFT is dominated by prefill queueing) without harming decode TPOT, and stays inside the candidate's file/lines/symbol. The prefix-aware LPM→FCFS fallback half of the finding is not applicable to this frontend snapshot task and is intentionally excluded.

---

### 2. Adopt short/long EWMA divergence to smooth DP load-balancing stats and detect queueing trends
- **Finding:** `find-vllm_v1_engine-0015` — *GitHub - Netflix/concurrency-limits*
- **Source URL:** <https://github.com/Netflix/concurrency-limits>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/engine/core_client.py within DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task (lines 1305-1408), replace the current direct replacement of self.lb_engines with a two-EWMA smoothing scheme inspired by Netflix concurrency-limits' Gradient2 algorithm. Maintain two exponentially weighted moving averages per local managed engine of the (waiting, running) counts drained from the coordinator XSUB stats socket: a short-window EWMA (alpha ~0.5, reacting within a handful of updates) and a long-window EWMA (alpha ~0.05, reflecting steady-state load). On each stats message received in the drain loop, update both EWMAs for every local rank slice, then publish an lb_engines snapshot whose per-engine load score is the smoothed short-window value plus a divergence term (short_ewma - long_ewma) that captures a queueing trend when the short-window count is growing faster than the long-window baseline. Also record the last-update monotonic timestamp alongside the snapshot so DPLBAsyncMPClient.get_core_engine_for_request can detect stale snapshots (e.g. when no XSUB updates have arrived for longer than a threshold) and fall back to round-robin instead of trusting outdated smoothed values. Keep the existing bounded nonblocking drain, wave/running tuple handling, local-rank slicing, and scale-up path unchanged so tests/v1/engine/test_engine_core_client.py assertions on stats propagation, local-rank slicing, and scale-up continue to hold; only the value written into lb_engines (and a companion staleness field) changes.

**Proposal rationale.**

The candidate's stated evolve_rationale explicitly calls out EWMA/hysteresis for noisy counts and exposing snapshot age as desirable directions, and notes that DPLBAsyncMPClient.get_core_engine_for_request quality depends on freshness and smoothing. The Netflix concurrency-limits Gradient2 technique provides a concrete, well-tested formulation for exactly this problem: it uses divergence between short- and long-window exponential averages to identify a queueing trend from noisy per-sample measurements, which maps directly onto the coordinator's bursty (waiting, running) updates in a multi-turn agentic workload where prefill spikes and short generations produce high-variance instantaneous counts. Applying short/long EWMA divergence to the smoothed load score, combined with a staleness timestamp, addresses the two specific gaps in the current approach (no smoothing, no staleness metadata) with a technique that has demonstrated value for adaptive routing under bursty load, plausibly reducing mis-routes that inflate per-engine queues and thus the median TTFT tail the caller wants to shrink.

---

## Agent proposals

### 1. Piggyback engine stats onto request-response path with sequence-numbered snapshots for tail freshness
- **Agent:** claude

**Detailed description.**

In vllm/v1/engine/core_client.py within DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task (lines 1305-1408), augment the coordinator XSUB pull-based snapshot with an inline push channel: have each engine attach a compact (engine_index, waiting_delta, running_delta, kv_cache_usage, monotonic_seq) header to every outbound response frame it already sends back to the frontend on the existing response path (the same sockets DPAsyncMPClient already reads for output tokens). In the stats task, alongside the existing XSUB drain at lines 1383-1394, register a callback that the response-processing coroutine invokes whenever it sees such a header, folding the delta into an authoritative per-engine (waiting, running, kv_cache_usage, seq) map keyed by monotonic_seq so late-arriving XSUB messages with older seq numbers are discarded. Slicing at lines 1401-1404 and scale-up padding at lines 1356-1364 remain intact; lb_engines is written from the max-seq view of the merged map. This eliminates the O(coordinator publish interval) staleness window on hot engines — the ones actively responding refresh their own load on every response, exactly when routing decisions matter most — while cold/idle engines still get periodic XSUB updates. Tests in tests/v1/engine/test_engine_core_client.py continue to see coordinator stats propagation and local-rank slicing; add a test that verifies a fast-responding engine's lb_engines entry reflects its response-piggybacked deltas rather than an older XSUB tuple, and that scale-up still relies on XSUB for newly added ranks that have not yet responded.

**Novelty rationale.**

The two existing deep_research_proposals both keep the same pull-based XSUB coordinator publish channel as the sole source of load information and change only what is stored (two-dimensional prefill/decode split in #0006, EWMA smoothing plus staleness timestamp in #0015). Neither changes the *transport* or *timing* of updates. This proposal adds a distinct second information path — piggybacked deltas on the existing engine-to-frontend response frames — with monotonic sequence numbers for merge ordering. The freshness improvement comes from the update arriving with the response itself (event-driven, on the hot path) rather than from smoothing noisy periodic samples (#0015) or reshaping the payload (#0006). It is orthogonal to and composable with either existing proposal (a two-dim payload or an EWMA could be layered on top), so it is not covered by them.

---

### 2. Bound stats drain work per poll to keep routing wakeups responsive
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/core_client.py`, update `DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task` so the nonblocking stats-drain loop at lines 1383-1394 consumes at most a fixed number of coordinator stats messages per poll iteration, for example `MAX_STATS_MESSAGES_PER_POLL`, while preserving the current latest-snapshot semantics. If the cap is reached and the stats socket may still contain more messages, immediately continue the outer polling loop with a zero or very small timeout so draining resumes cooperatively rather than monopolizing the async task. Keep `first_request_poll_socket` handling ahead of, or interleaved with, capped stats batches so a burst of queued XSUB stats cannot delay first-request wakeups and route selection. Add focused coverage in `tests/v1/engine/test_engine_core_client.py` using a synthetic backlog of stats messages plus a first-request notification: assert that `lb_engines` still converges to the latest stats tuple after multiple batches, and that first-request handling is observed without waiting for the entire backlog to drain in one tight loop.

**Novelty rationale.**

The existing proposals change the contents and interpretation of the snapshot: prefill/decode dimensions, EWMA plus staleness metadata, or response-path piggybacked deltas with sequence ordering. This proposal instead changes the scheduler behavior of the receiver itself: it bounds per-iteration drain work so stats bursts cannot starve first-request socket handling or other async progress. It preserves the same payload and source of truth, so it is orthogonal to smoothing, richer load dimensions, and response-piggyback freshness.

---
