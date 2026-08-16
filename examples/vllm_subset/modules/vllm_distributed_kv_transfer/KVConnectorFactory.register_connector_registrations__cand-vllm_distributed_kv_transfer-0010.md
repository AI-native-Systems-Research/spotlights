# KVConnectorFactory.register_connector registrations

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/factory.py`](vllm/distributed/kv_transfer/kv_connector/factory.py) (lines 152–242)
- **Symbol:** `KVConnectorFactory.register_connector registrations`
- **Kind:** plugin_seam
- **Estimated impact:** medium
- **Id:** `cand-vllm_distributed_kv_transfer-0010`

## Description
Registry maps external connector-name strings to module/class pairs at import time, and KVConnectorFactory.create_connector lazily instantiates the implementation selected by KVTransferConfig.

## Current approach
The interface is KVConnectorBase_V1 in vllm/distributed/kv_transfer/kv_connector/v1/base.py, reached through KVConnectorBase compatibility in vllm/distributed/kv_transfer/kv_connector/base.py. Existing implementations include NixlConnector in vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py, OffloadingConnector in vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py, MooncakeConnector in vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py, and HF3FSKVConnector in vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py. Runtime selection is KVTransferConfig.kv_connector, typically supplied through --kv-transfer-config; kv_connector_module_path can load an external connector module.

## Estimated impact explanation
The registry is not hot, but it enables low-risk replacement of connector routing, batching, or transfer scheduling logic that can move turn-2 TTFT without invasive scheduler changes.

## Evolve rationale
This is a concrete plugin seam: a new evolved connector can implement KVConnectorBase_V1, add one register_connector call or use kv_connector_module_path, and be selected by configuration for A/B evaluation without changing scheduler or worker call sites. Oracle: factory behavior plus tests/v1/kv_connector/unit/test_multi_connector.py and connector-specific integration tests exercise interface delegation and end-to-end KV correctness.

## Deep research proposals

### 1. Add a scored KV-routing MultiConnector selected via the factory registry
- **Finding:** `find-vllm_distributed_kv_transfer-0001` — *Routing Concepts*
- **Source URL:** <https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new connector implementation (e.g., ScoredRoutingConnector) that implements KVConnectorBase_V1 and is registered in vllm/distributed/kv_transfer/kv_connector/factory.py alongside the existing register_connector calls at lines 152-242, selectable through KVTransferConfig.kv_connector (no scheduler/worker call-site changes). Modeled on the existing MultiConnector in vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py, it wraps a list of child connectors but replaces the current first-hit selection in MultiConnector.get_num_new_matched_tokens (multi_connector.py:385-404, which today assigns a request to the first child with toks > 0) with a scored selection. For each child, query get_num_new_matched_tokens to obtain the reusable-KV overlap term, then combine it with two locally observable load terms: (a) an active-prefill/pending-load signal derived from outstanding async loads and in-flight save/load counters already tracked per connector, and (b) an active-request count for that child. The score is a weighted sum (weights configurable via kv_connector_extra_config) matching Dynamo's 'lowest-cost eligible worker' formulation: cost = w_load * projected_load - w_overlap * matched_tokens. The child with the minimum cost is recorded in _requests_to_connector and used by update_state_after_alloc exactly as MultiConnector does today; saves continue to fan out to all children. Because it registers through the existing plugin seam, it can be A/B evaluated against MultiConnector by flipping KVTransferConfig.kv_connector, and it composes with NixlConnector, OffloadingConnector, MooncakeConnector, and HF3FSKVConnector without touching their code.

**Proposal rationale.**

The candidate's current selection policy for multi-source KV is explicitly first-hit (multi_connector.py docstring at lines 132-135 and the get_num_new_matched_tokens loop). For a multi-turn agentic workload optimizing median TTFT, that ordering-dependent policy can route a request to a connector with a small prefix match while a later connector holds a much longer reusable prefix, or to a source whose load makes the transfer slower than recomputation. The finding contributes a concrete transferable idea — a cost function that jointly scores KV overlap and load, then picks the minimum-cost eligible source — that maps directly onto MultiConnector's per-request selection point and is naturally delivered as a new registered connector at exactly the plugin seam this candidate describes. That keeps the change contained to a new module plus one register_connector line, matches the candidate's evolve_rationale (A/B via config, no scheduler/worker changes), and targets TTFT by choosing the cache source with the best latency estimate rather than the first with any hit.

---

### 2. Power-of-two connector selection for cache-affinity load balancing
- **Finding:** `find-vllm_distributed_kv_transfer-0002` — *DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving*
- **Source URL:** <https://papers.cool/arxiv/2602.06502>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new KVConnectorBase_V1 implementation registered via KVConnectorFactory.register_connector (vllm/distributed/kv_transfer/kv_connector/factory.py:152-242) that wraps two or more underlying connector backends (e.g., multiple NixlConnector or MooncakeConnector endpoints, or heterogeneous backends). On each incoming request, the wrapper computes two independent hash functions over the prompt prefix (or request-derived key) to select two candidate backend instances, then queries live system state (in-flight requests, queue depth, or cached-prefix hit signals exposed by the wrapped connectors) and dispatches to the less-loaded of the two candidates. Selection happens inside the connector's request-routing entry point; scheduler and worker call sites are unchanged because the composite still implements KVConnectorBase_V1 and is chosen through KVTransferConfig.kv_connector. Configuration parameters (hash seeds, candidate count, load metric) are passed through KVTransferConfig.kv_connector_extra_config, and the implementation can also be loaded via kv_connector_module_path for out-of-tree evaluation. Correctness is validated against tests/v1/kv_connector/unit/test_multi_connector.py plus connector-specific integration tests.

**Proposal rationale.**

The candidate is explicitly a plugin seam whose value is enabling A/B evaluation of connector routing/scheduling changes without touching scheduler or worker code, and the caller objective is reducing median TTFT/TPOT for multi-turn agentic workloads. Multi-turn workloads produce repeated prefixes whose KV blocks live on specific backends, so pure random or pure cache-affinity routing trade off head-of-line blocking against reuse. DualMap's power-of-two-choices over independently hashed candidates directly targets that tension: it preserves prefix locality (each prompt only has two eligible backends, both stable across turns) while allowing runtime load state to shed hotspots that would otherwise inflate TTFT. The connector factory is the correct layer to implement this because it composes existing KVConnectorBase_V1 implementations without changing hot-path scheduler logic.

---

### 3. Add NIXL-optimized connector variant with batched progress and cached remote metadata
- **Finding:** `find-vllm_distributed_kv_transfer-0003` — *Enhancing Distributed Inference Performance with the NVIDIA Inference Transfer Library*
- **Source URL:** <https://developer.nvidia.com/blog/?p=113426>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Register a new KVConnectorBase_V1 implementation alongside the existing NixlConnector entry in vllm/distributed/kv_transfer/kv_connector/factory.py (register_connector at lines 152-242), selected via KVTransferConfig.kv_connector (e.g., 'NixlConnectorBatched') so it can be A/B evaluated without touching scheduler/worker call sites. The new connector reuses the KVConnectorBase_V1 interface but changes three internal behaviors on top of NIXL: (1) post transfers via NIXL's non-blocking API and coalesce progress/completion polling into a single batched call per scheduler step instead of per-request scans, driven by NIXL target-side notifications where available; (2) register KV memory as larger contiguous regions once at connector init/warmup and cache the resulting NIXL descriptors, so per-request get_num_new_matched_tokens/start_load_kv paths avoid re-registering or re-validating remote metadata for stable peers; (3) maintain a small validated-remote-agent metadata cache keyed by peer identity, invalidated only on topology change, to skip repeated handshake/metadata exchange on multi-turn requests hitting the same remote. External selection remains 'kv_connector: NixlConnectorBatched' in --kv-transfer-config, or via kv_connector_module_path for out-of-tree evaluation. Correctness is exercised by tests/v1/kv_connector/unit/test_multi_connector.py and the NIXL integration tests; the change is contained to the new connector class plus one register_connector call.

**Proposal rationale.**

The candidate is explicitly a plugin seam whose value is enabling low-risk replacement of connector routing and transfer scheduling logic to move turn-2 TTFT. The NVIDIA NIXL guidance directly targets the two dominant overheads on that path for multi-turn agentic workloads: per-step polling cost and metadata/registration churn. Batching non-blocking progress and using notifications addresses TPOT overhead accumulated per decode step when a NIXL-backed connector is active, and registering larger regions plus caching validated remote metadata attacks the TTFT tail on turn-2+ requests where the remote peer and KV layout are stable. The finding contributes concrete, transferable mechanics (non-blocking API usage, larger-region registration, notification-driven completion) rather than restating the current NixlConnector approach, and the seam is specifically designed so a variant connector implementing these mechanics can be selected by configuration alone.

---

### 4. Register async DEALER/ROUTER NIXL connector variant to pipeline handshake and metadata round-trips
- **Finding:** `find-vllm_distributed_kv_transfer-0004` — *Advanced Request-Reply Patterns*
- **Source URL:** <https://zguide.zeromq.org/docs/chapter3/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new evolved connector variant registered through the plugin seam at vllm/distributed/kv_transfer/kv_connector/factory.py:152-242 (e.g. register_connector("NixlAsyncHandshakeConnector", ...)) that reimplements the KV-transfer handshake and metadata-lookup client on top of ZeroMQ DEALER instead of REQ. Concretely, replace the REQ-based serial loop in NixlConnectorWorker.add_remote_agent (base_worker.py:615-635), which currently does `for (pp_rank, tp_rank) in product(...): sock.send(GET_META_MSG, ...); sock.recv_multipart()` through a single `zmq.REQ` socket, with a DEALER-side implementation that (1) pipelines all `GET_META_MSG` requests to every target `(remote_pp_rank, remote_rank)` immediately, tagging each frame with a correlation id, and (2) drains replies as they arrive, matching them by correlation id and updating the best-RTT/clock-offset estimator as usual. The peer side (currently `zmq.ROUTER` in nixl/base_scheduler.py:339) already speaks the ROUTER wire format and remains unchanged; only the initiator switches from lockstep REQ to async DEALER. Wire the new class through KVConnectorFactory.register_connector (or via KVTransferConfig.kv_connector_module_path for an out-of-tree variant) so scheduler and worker call sites — KVConnectorBase_V1 interface in v1/base.py and the compatibility shim in base.py — are untouched, and selection happens purely via --kv-transfer-config kv_connector=NixlAsyncHandshakeConnector for A/B evaluation.

**Proposal rationale.**

The finding directly targets a real serial-RTT hot spot exercised on the turn-2 TTFT path of multi-turn agentic workloads: NixlConnector's handshake in base_worker.py:615 walks `itertools.product(range(remote_pp_size), p_remote_ranks)` and pays one round-trip latency per remote rank through a single REQ socket (which is forced into lockstep by the REQ/REP state machine — you cannot send again until you've received). When target instance TP > local TP, or with PP > 1, this multiplies. The ZMQ guide's DEALER/ROUTER pattern removes exactly this constraint: DEALER is asynchronous, so all metadata requests can be fired first and replies drained later, collapsing N serial RTTs to ~one bounded by the slowest peer. Because this candidate is a plugin seam and the evolve_rationale explicitly calls out registering a new KVConnectorBase_V1 implementation for A/B evaluation without changing scheduler/worker code, the finding provides a concrete, transferable technique that fits the seam's shape: swap the REQ initiator for a DEALER initiator inside a new registered connector class, keep the ROUTER server-side, and gate rollout via KVTransferConfig. Expected impact is on turn-2 TTFT (handshake happens on first cross-instance fetch after prefix reuse), matching the caller's stated objective.

---

### 5. Register a batched, topology-aware scatter-gather KV connector variant
- **Finding:** `find-vllm_distributed_kv_transfer-0005` — *Transfer Engine*
- **Source URL:** <https://aionw.github.io/design/transfer-engine/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new KVConnectorBase_V1 implementation (e.g., BatchedRdmaKVConnector) and register it via KVConnectorFactory.register_connector in vllm/distributed/kv_transfer/kv_connector/factory.py (lines 152-242), alongside the existing NixlConnector and MooncakeConnector entries, so it can be selected purely via KVTransferConfig.kv_connector for A/B evaluation without touching scheduler or worker call sites. The connector packages KV block transfers as BatchTransfer-style arrays over non-contiguous source/target ranges: it (a) coalesces adjacent/near-adjacent KV blocks for the same request into a single scatter-list descriptor rather than issuing per-block operations, (b) fuses descriptors across requests that share a destination peer within the same scheduling step, and (c) applies topology-aware NIC/path selection (and large-transfer slicing across NICs) when issuing the batched transfer. Behavior is delegated through the existing KVConnectorBase_V1 interface (v1/base.py) reached via base.py compatibility, so scheduler/worker code and tests/v1/kv_connector/unit/test_multi_connector.py continue to exercise the same interface contract. The connector can optionally be composed under MultiConnector to fall back to an existing implementation for unsupported transfer shapes.

**Proposal rationale.**

In a multi-turn agentic workload, turn-2 TTFT is dominated by transferring prefix KV blocks that are logically contiguous per layer but scattered across the block-table layout, which today produces many small RDMA descriptors and per-transfer initiation overhead. The Mooncake Transfer Engine finding contributes two concrete, transferable ideas the current registered connectors do not fully exploit: (1) BatchTransfer-style scatter/gather arrays that describe many non-contiguous source/target ranges in one operation, cutting descriptor count and completion-polling overhead, and (2) topology-aware NIC/path selection with large-transfer slicing to keep multiple NICs busy for a single logical transfer. Both directly attack the initiation and serialization overhead on the KV-transfer critical path, which is what moves median TTFT for prefix-heavy multi-turn traffic. The factory registry at lines 152-242 is exactly the low-risk seam to introduce this variant: a new register_connector line (or kv_connector_module_path) makes the new implementation selectable via --kv-transfer-config without invasive changes, and MultiConnector plus the existing unit tests provide the oracle for interface fidelity.

---

### 6. Add a CacheBlend-style selective-recompute connector that overlaps KV load with partial recomputation
- **Finding:** `find-vllm_distributed_kv_transfer-0008` — *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*
- **Source URL:** <https://www.alphaxiv.org/abs/2405.16444>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new KVConnectorBase_V1 implementation (e.g., CacheBlendConnector) and register it via KVConnectorFactory.register_connector in vllm/distributed/kv_transfer/kv_connector/factory.py (lines 152-242) so it is selectable through KVTransferConfig.kv_connector (or loaded via kv_connector_module_path) without touching scheduler or worker call sites. The connector wraps an existing external KV backend (e.g., NIXL/Mooncake/HF3FS) and, on cross-turn reuse in multi-turn agentic requests, implements the CacheBlend pipeline: (a) issue layer-wise asynchronous KV fetches through the underlying transport during the connector's start_load_kv/wait_for_layer_load hooks, and (b) select a small subset of tokens with the highest expected KV deviation and mark them for recomputation on the worker, sizing the recompute ratio so its compute cost is hidden under the remaining transfer latency. A per-request threshold decides transfer-vs-recompute: if the estimated transfer time for a block exceeds estimated full recompute time (based on measured layer bandwidth and prompt length), the connector falls back to pure recompute and skips the load. The layer-pipelined load path reuses the existing KVConnectorBase_V1 layer-wise callbacks; the token-selection and recompute-ratio policy lives inside the connector so it is a drop-in A/B alternative to the current NixlConnector/OffloadingConnector/MooncakeConnector/HF3FSKVConnector registrations.

**Proposal rationale.**

The candidate is explicitly a plugin seam whose value is enabling low-risk A/B replacement of connector-side transfer scheduling logic to move turn-2 TTFT in multi-turn workloads. CacheBlend directly targets that gap: it hides KV load latency by pipelining layer-wise fetches with selective recomputation on tokens with high KV deviations, and it makes the transfer-vs-recompute choice dynamic so reuse is only taken when transfer beats recompute — the exact regime the caller (multi-turn agentic, TTFT/TPOT-sensitive) cares about. Because KVConnectorBase_V1 already exposes layer-wise load hooks and the factory selects implementations by name, the technique can be delivered as a new connector registration rather than invasive scheduler changes, matching the candidate's evolve_rationale and oracle (factory + multi_connector tests + connector integration tests).

---

### 7. Add a CacheGen-style compressed KV connector as a new plugin for slow-tier reuse
- **Finding:** `find-vllm_distributed_kv_transfer-0009` — *CacheGen*
- **Source URL:** <https://docs.lmcache.ai/dev/kv_cache_optimizations/compression/cachegen.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new KVConnectorBase_V1 implementation (e.g. `CompressedStoreConnector`) that wraps a storage/offload transport and applies CacheGen-style compact bitstream encoding on save and decoding on load. Register it in vllm/distributed/kv_transfer/kv_connector/factory.py alongside the existing entries at lines 152-242 with a single `KVConnectorFactory.register_connector("CompressedStoreConnector", "vllm.distributed.kv_transfer.kv_connector.v1.compressed_store_connector", "CompressedStoreConnector")` call, so selection is driven purely by KVTransferConfig.kv_connector without touching scheduler or worker call sites. The connector should compose with an inner slow-tier backend (HF3FSKVConnector-like storage, OffloadingConnector, or MooncakeStoreConnector) by delegating block placement/lookup while transforming payloads through an encode/decode step tuned for negligible decode overhead per the CacheGen technique. Expose a config knob to enable compression per tier (e.g. only for remote/disk, not local HBM). Validation reuses the plugin-seam oracle: tests/v1/kv_connector/unit/test_multi_connector.py for interface delegation plus a connector-specific end-to-end test that verifies KV correctness (bit-exact or within a tolerance if lossy) and measures turn-2 TTFT vs. the uncompressed baseline on a multi-turn agentic workload.

**Proposal rationale.**

The candidate is explicitly the plugin seam where new connectors are added without scheduler/worker changes, and the caller's objective is reducing median TTFT/TPOT on a multi-turn agentic workload where prior-turn KV is fetched from a slower tier on turn 2+. CacheGen directly addresses that regime: when bandwidth to remote/offload storage dominates recomputation cost, encoding KV into compact bitstreams cuts transferred bytes and thus load latency, improving turn-2 TTFT. The finding contributes a concrete, transferable mechanism (compact bitstream encode/decode with negligible decoding overhead) that maps cleanly onto a new KVConnectorBase_V1 implementation registered via one factory line, which is exactly the low-risk A/B evaluation shape the candidate is designed for. It targets a gap not covered by any currently registered connector, which move KV bytes uncompressed.

---

### 8. Register a RadixAttention-inspired prefix-tree connector for multi-turn agentic KV reuse
- **Finding:** `find-vllm_distributed_kv_transfer-0014` — *SGLang: Efficient Execution of Structured Language Model Programs*
- **Source URL:** <https://mast.stanford.edu/pubs/sglang_efficient_execution_of_structured_language_model_programs/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new KVConnectorBase_V1 implementation (e.g. RadixPrefixConnector) registered in vllm/distributed/kv_transfer/kv_connector/factory.py alongside the existing entries at lines 152-242 via a single KVConnectorFactory.register_connector call, selectable through KVTransferConfig.kv_connector without touching scheduler or worker call sites. The connector maintains a radix / prefix tree over token-id sequences of admitted requests (as in SGLang's RadixAttention), keyed to the block hashes already produced by vLLM's prefix cache, and uses it to (1) prioritize offload admission for blocks that sit on high-fan-out prefix-tree nodes shared across sibling turns and parallel calls, (2) order prefetch on request arrival by walking the longest matching prefix path so the largest reusable prefix is staged into GPU KV before the first forward pass, and (3) bias eviction on the offload tier to keep internal prefix-tree nodes with many descendants resident longer than leaf-only blocks. Internally the connector can be implemented as a thin scheduler-side wrapper that composes with the existing OffloadingConnector machinery in vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py: reuse OffloadingConnectorScheduler / OffloadingConnectorWorker for the actual block movement, and layer the radix tree as an additional ordering / scoring signal on the connector metadata that selects which blocks to load and evict. The default eviction / prefetch heuristic remains available as a fallback so behavior is unchanged when the tree is cold or the workload is single-turn.

**Proposal rationale.**

The candidate is precisely a plugin seam: KVConnectorFactory registrations at factory.py:152-242 let a new KVConnectorBase_V1 subclass replace connector routing and transfer scheduling behavior via configuration alone, exactly the low-risk A/B path the candidate's evolve_rationale calls out. The finding contributes a concrete, transferable mechanism (radix / prefix tree over token sequences to drive KV reuse ordering) that maps directly onto the workload hint 'multi-turn agentic' and the caller objective of reducing median TTFT: turn-2 TTFT is dominated by whether the shared conversation prefix is resident, and prefix-tree-aware admission / prefetch / eviction is the class of technique SGLang shows is effective for exactly this pattern of branching and repeated prefix lookups. It is not a restatement of the current approach — existing connectors (Nixl, Offloading, Mooncake, HF3FS) do not use prefix-tree structure to order transfers, so the finding adds a specific, implementable idea rather than a topical adjacency.

---

### 9. Add latency-predicting connector router to KVConnectorFactory for multi-connector selection
- **Finding:** `find-vllm_distributed_kv_transfer-0015` — *Use predicted latency-based routing with GKE Inference Gateway*
- **Source URL:** <https://docs.cloud.google.com/kubernetes-engine/docs/how-to/use-predicted-latency-based-routing?authuser=2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the plugin seam at vllm/distributed/kv_transfer/kv_connector/factory.py (KVConnectorFactory.register_connector registrations, lines 152-242) by introducing a new registered connector implementing KVConnectorBase_V1 that acts as a routing meta-connector over already-registered backends (e.g., NixlConnector, OffloadingConnector, MooncakeConnector, HF3FSKVConnector). Inspired by GKE Inference Gateway's predicted latency-based routing, this router would maintain a lightweight online-trained predictor (e.g., a small gradient-boosted or linear model) of expected KV transfer latency / TTFT contribution per underlying connector, using features already observable at the connector seam: request/prefix token length, prefix-cache match score, pending transfer queue depth per backend, recent transfer size, and rolling measured transfer/load latency. On each get_num_new_matched_tokens / build_connector_meta call, the router queries the predictor for each candidate backend, selects the one with the lowest predicted latency (falling back to a static default when predictions are cold), and dispatches KVConnectorBase_V1 calls to it. Training data comes from wrapping start_load_kv/wait_for_layer_load/save_kv_layer to record (features, observed latency) tuples; updates are done incrementally out-of-band. Selection is enabled through KVTransferConfig.kv_connector = 'PredictedLatencyRouterConnector' (or via kv_connector_module_path for out-of-tree use), so no scheduler or worker call sites change. Oracle: reuse tests/v1/kv_connector/unit/test_multi_connector.py patterns to validate delegation, and add tests that assert deterministic selection when the predictor is stubbed.

**Proposal rationale.**

The candidate is explicitly a plugin seam whose 'evolve_rationale' invites replacing connector routing/scheduling without touching scheduler/worker code. The finding contributes a concrete, transferable idea: replace static heuristics with an online-trained latency predictor using features (queue depth, cache utilization/prefix match, request size) that map directly onto signals already available inside KVConnectorBase_V1 implementations. For the stated multi-turn agentic workload, TTFT is dominated by prefix-cache hits and KV transfer costs, which vary sharply between offloading vs. RDMA-class backends like Nixl/Mooncake; picking the faster backend per request based on predicted rather than static cost is a plausible lever on median TTFT (and, via avoided recomputes, TPOT). The change is localized to a new registration plus a router class, matching the candidate's medium-impact, low-blast-radius profile.

---

## Agent proposals

### 1. Register a hedged fetch-vs-recompute racing connector for tail-TTFT bounding
- **Agent:** claude

**Detailed description.**

Add a new KVConnectorBase_V1 implementation (e.g. HedgedRaceConnector) registered via a single KVConnectorFactory.register_connector call in vllm/distributed/kv_transfer/kv_connector/factory.py alongside the existing entries at lines 152-242, selectable purely through KVTransferConfig.kv_connector (or kv_connector_module_path). The connector wraps one inner backend connector (Nixl/Mooncake/HF3FS/Offloading) chosen via kv_connector_extra_config. On each get_num_new_matched_tokens/build_connector_meta call for a request whose reusable prefix would otherwise be loaded remotely, the connector predicts the remote load latency from a rolling per-peer EWMA of recently observed transfer times (measured in wait_for_layer_load) and a per-block-count linear term, and compares it to an estimated local recompute time derived from measured prefill throughput and prefix length. When the two estimates fall within a configurable band (default: remote_p90 > local_recompute), the connector reports the matched prefix as EMPTY to the scheduler (so vLLM prefills locally on the critical path) while still issuing the remote load asynchronously in the background; whichever finishes first wins — if the remote load completes before the corresponding layer's local prefill, wait_for_layer_load returns immediately and the loaded KV supersedes the in-flight recompute for later layers; otherwise the background load is dropped in a get_finished sweep. When remote is confidently faster than recompute, the connector behaves exactly like the inner backend (reports the full match, no hedge). The hedge fraction and cancellation policy are tuned via kv_connector_extra_config; interface fidelity is validated by tests/v1/kv_connector/unit/test_multi_connector.py and connector-specific tests, plus a new test that verifies KV correctness under a mocked slow remote (recompute wins) and a fast remote (load wins).

**Novelty rationale.**

None of the 9 listed deep_research_proposals propose racing a full local recompute against a full remote KV fetch with abort-on-first-complete. Finding 0008 (CacheBlend) commits to the load and only recomputes a small selected token subset to fill deviation, sized to hide under transfer time — it does not race and does not abandon the load. Finding 0009 (CacheGen) reduces transferred bytes via compression but still commits to a single transfer path. Findings 0001/0002/0015 pick among candidate remote sources by cost/hash/predicted-latency but always take one source. Finding 0003/0004/0005 optimize NIXL/RDMA mechanics (batched progress, async handshake, scatter-gather) but not the transfer-vs-recompute race. Finding 0014 orders prefix admission via a radix tree but does not hedge. The proposed mechanism specifically bounds TTFT by min(remote_transfer, local_recompute) under tail latency variance — the dominant TTFT-tail regime for multi-turn agentic workloads on shared remote KV tiers — and is delivered as a new registered connector at exactly the plugin seam this candidate describes.

---

### 2. Register a TPOT-aware async writeback connector
- **Agent:** codex

**Detailed description.**

Add a new KVConnectorBase_V1 wrapper connector, e.g. `ThrottledWritebackConnector`, and register it in `vllm/distributed/kv_transfer/kv_connector/factory.py` alongside the existing `KVConnectorFactory.register_connector` entries. The connector delegates load-side behavior to an inner backend selected through `kv_connector_extra_config`, but changes `save_kv_layer` handling: instead of issuing remote/offload saves immediately on the decode-critical path, it enqueues per-layer KV writebacks onto a bounded background queue, coalesces adjacent blocks and same-peer writes within a scheduler step, and drains them under a configurable bandwidth / in-flight transfer budget. When the queue is saturated or decode latency rises above a rolling threshold, the connector drops or defers low-value saves first, using simple local signals such as prefix length, block reuse count if available, and whether the request is likely to continue. This keeps turn-2 reuse available for high-value conversation prefixes while preventing aggressive cache population from competing with active decode and inflating median TPOT. The change is selected purely by `KVTransferConfig.kv_connector`, with no scheduler or worker call-site changes, and can be tested by extending existing connector delegation tests plus a mocked backend test that verifies save ordering, throttling, and correctness when deferred saves complete before later reuse.

**Novelty rationale.**

The listed proposals focus on read-side source selection, load latency prediction, RDMA/NIXL mechanics, compression, prefix-tree admission, selective recompute, or fetch-vs-recompute hedging. None specifically targets the writeback/save side as a TPOT control point by decoupling `save_kv_layer` from the decode-critical path with bounded asynchronous draining and backpressure-aware dropping. Agent A's proposal races remote fetch against recompute for TTFT tail bounding; it does not address save-path contention or throttled background population. This proposal is therefore a distinct connector variant aimed at median TPOT while still preserving multi-turn TTFT benefits for high-value prefixes.

---
