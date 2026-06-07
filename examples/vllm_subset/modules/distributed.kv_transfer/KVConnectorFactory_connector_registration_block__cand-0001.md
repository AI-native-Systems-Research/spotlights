# KVConnectorFactory connector registration block

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/factory.py`](vllm/distributed/kv_transfer/kv_connector/factory.py) (lines 149–228)
- **Symbol:** `KVConnectorFactory connector registration block`
- **Kind:** plugin_seam
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0001`

## Description
Actual registration site for built-in KV transfer connectors. Each `KVConnectorFactory.register_connector(...)` call binds a public connector name to a lazy-loaded implementation class, and `create_connector` later instantiates the selected class from the configured connector name or external module override.

## Current approach
A name-keyed registry `KVConnectorFactory._registry: dict[str, Callable[[], type[KVConnectorBase]]]` in `vllm/distributed/kv_transfer/kv_connector/factory.py`. The real v1 interface is `KVConnectorBase_V1` in `vllm/distributed/kv_transfer/kv_connector/v1/base.py`; reference implementations include `NixlConnector` in `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py`, `OffloadingConnector` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`, `MooncakeConnector` in `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`, and `P2pNcclConnector` in `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`. Runtime selection is `KVTransferConfig.kv_connector`, with `KVTransferConfig.kv_connector_module_path` taking priority as an external plugin surface.

## Estimated impact explanation
Connector choice controls whether multi-turn prefixes are recomputed, loaded from local offload, or pulled from a remote prefill worker. A better connector can move median TTFT through higher remote-prefix hit rate and lower transfer setup cost, and can move median TPOT by reducing per-step save/load overhead.

## Evolve rationale
This seam isolates transport and cache policy from scheduler and model execution code. A new connector can implement the fixed `KVConnectorBase_V1` scheduler/worker contract and add one registration call here. Correctness oracle: for the same request stream, connector metadata, finished request sets, invalid block reporting, and generated token outputs must match an existing connector's behavior under unit tests in `tests/v1/kv_connector/unit/` and end-to-end remote-prefill/offload output tests, while only transport mechanics and admission policy differ.

## Deep research proposals

### 1. Register a prefix-aware PD connector that transfers only the decode-side KV delta
- **Finding:** `find-0001` — *[PD Disaggregation] Avoid Transferring Prefix Cache’s KVCache of Decode Node*
- **Source URL:** <https://github.com/sgl-project/sglang/pull/7990>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new KVConnectorBase_V1 implementation (e.g., a `PrefixAwareNixlConnector` variant alongside `NixlConnector` in `vllm/distributed/kv_transfer/kv_connector/v1/`) and add a corresponding `KVConnectorFactory.register_connector(...)` call in the registration block at `vllm/distributed/kv_transfer/kv_connector/factory.py:149-228`. The new connector implements the scheduler/worker contract such that, before pulling remote prefill KV blocks, it queries the decode node's local prefix-cache state (via the existing block-hash / block-pool plumbing already exposed to KV connectors) and computes the set of token-ranges/blocks already resident locally. The transfer request sent to the prefill worker is restricted to the complement (the suffix delta), and only those blocks are written into decode-side KV slots; locally-cached blocks are reused in place. The scheduler-side metadata, finished-request sets, and invalid-block reporting continue to match the existing connector contract so that the unit tests under `tests/v1/kv_connector/unit/` and the end-to-end remote-prefill/offload output tests still pass. Registration follows the existing lazy-import pattern used for `NixlConnector`, `OffloadingConnector`, `MooncakeConnector`, and `P2pNcclConnector`, and is selectable via `KVTransferConfig.kv_connector`.

**Proposal rationale.**

The candidate is precisely the seam where new connector strategies are bound and made selectable, and its evolve_rationale explicitly anticipates new transport/admission policies plugged in here. The finding contributes a concrete, transferable mechanism — subtract the decode node's already-resident prefix KV from the requested transfer set — that targets the same TTFT-on-multi-turn workload the caller specifies. Because multi-turn agentic traffic is dominated by repeated conversation prefixes, eliminating retransfer of those prefix blocks directly reduces transfer-setup time and bytes-on-the-wire on the critical path to first token, addressing TTFT, while leaving steady-state TPOT unaffected for connectors that previously paid the redundant transfer cost. The change is localized: it reuses the existing connector contract and plugin seam rather than altering scheduler or model-execution code, and an existing connector's outputs serve as the correctness oracle.

---

### 2. Add a delta-KV transfer connector for multi-turn agentic prefix reuse
- **Finding:** `find-0002` — *[Roadmap] Prefill-Decode Disaggregation Roadmap (2026 Q2)*
- **Source URL:** <https://github.com/sgl-project/sglang/issues/21703>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Register a new built-in connector at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 (e.g. `KVConnectorFactory.register_connector("DeltaKVConnector", "vllm.distributed.kv_transfer.kv_connector.v1.delta_kv.delta_kv_connector", "DeltaKVConnector")`) implementing `KVConnectorBase_V1`. On the decode side, the connector resolves the longest shared prefix of the incoming request against a hierarchical/local KV cache (analogous to how `OffloadingConnector` consults its offload store) and emits load metadata only for the prefix blocks it already has; it then requests from the prefill worker only the *delta* KV for tokens past that matched prefix. On the prefill side, the connector's scheduler/worker hooks restrict the saved/sent block range to the delta region (tokens beyond the decode-side matched prefix length communicated via connector metadata), reusing the existing finished-request and invalid-block reporting paths so the V1 contract is unchanged. Selection is via `KVTransferConfig.kv_connector="DeltaKVConnector"`; no other call sites in the factory or scheduler/model-runner need to change. Correctness oracle from the candidate (matching connector metadata, finished sets, invalid blocks, and generated tokens for the same request stream against e.g. `NixlConnector` on non-agentic traces, where prefix match length is 0 and the connector degenerates to full-prefix transfer) directly applies.

**Proposal rationale.**

The candidate is exactly the plugin seam the finding's technique slots into: a new transport+admission policy that keeps the V1 scheduler/worker contract fixed and is added by one `register_connector` call. The caller's objectives (median TTFT and TPOT under multi-turn agentic traffic) are precisely what delta-KV targets — per the candidate's own impact model, TTFT is dominated by remote-prefix hit rate and transfer setup, and per-turn agentic deltas are small relative to accumulated prefix, so shipping only the delta should raise effective hit rate and shrink transfer volume vs. existing connectors (`NixlConnector`, `MooncakeConnector`, `P2pNcclConnector`) that transfer the full prefix on each turn, while reducing per-step save/load work vs. `OffloadingConnector` when the prefix is already resident on the decode side. The finding contributes a concrete, transferable mechanism (decode-side hierarchical-cache prefix lookup + prefill-side delta-only send) rather than restating the candidate's current behavior.

---

### 3. Add a CacheGen-style compressing KV connector and register it at the factory seam
- **Finding:** `find-0003` — *CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving*
- **Source URL:** <https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new connector implementing KVConnectorBase_V1 (e.g. `CacheGenConnector` under `vllm/distributed/kv_transfer/kv_connector/v1/cachegen/`) that wraps an existing transport (initially mirroring `NixlConnector`'s remote-prefill path and `OffloadingConnector`'s local offload path) but applies a CacheGen-style KV-specific tensor encoder on the send side and a matching decoder on the receive side. The encoder exploits per-layer/per-channel distributional properties of K and V tensors to produce a compact bitstream; the decoder reconstructs tensors directly into the destination KV blocks with negligible overhead. Add a per-chunk adaptive compression level selector keyed off recently observed transfer bandwidth (sliding window measured in the connector's worker-side send/recv loop) so that high-bandwidth links use light compression and constrained links use heavier compression, matching the paper's bandwidth-adaptive streaming. Wire the new connector into `KVConnectorFactory` by adding a single `KVConnectorFactory.register_connector("CacheGenConnector", "vllm.distributed.kv_transfer.kv_connector.v1.cachegen.connector", "CacheGenConnector")` entry in the registration block at `vllm/distributed/kv_transfer/kv_connector/factory.py:149-228`, alongside the existing Nixl/Offloading/Mooncake/P2pNccl entries; selection remains driven by `KVTransferConfig.kv_connector` with `kv_connector_module_path` still available for out-of-tree variants. Encoder/decoder configuration (target bitrate bounds, chunk size, layer grouping) is exposed through `KVTransferConfig.kv_connector_extra_config` so the same registered class can be tuned without further factory changes. Validate against the existing oracle: connector metadata, finished/invalid-block reporting, and generated tokens must match `NixlConnector`/`OffloadingConnector` under `tests/v1/kv_connector/unit/` and the end-to-end remote-prefill/offload output tests; only transport bytes-on-the-wire and timing differ.

**Proposal rationale.**

The candidate seam exists precisely to plug in alternative transport+policy implementations behind a fixed scheduler/worker contract, and the caller's objective is to reduce median TTFT (and secondarily TPOT) on a multi-turn agentic workload where remote prefix reuse and offload reloads dominate the prefill path. CacheGen targets exactly the bottleneck this seam controls: the bytes and latency of moving KV tensors between a remote prefill worker (or offload tier) and the consuming worker. By compressing KV chunks with a KV-aware encoder and adapting the compression level to current bandwidth, the new connector can raise the effective remote-prefix hit-rate-to-TTFT conversion (more hits become cheaper than recompute) and shrink per-step save/load overhead on the offload path, both of which are called out in the candidate's `evolve_rationale` as the levers for TTFT/TPOT. The change is contained: it adds one class plus one `register_connector(...)` line in the existing registration block, reuses the v1 contract already honored by `NixlConnector` and `OffloadingConnector`, and is gated by `KVTransferConfig.kv_connector` so it does not perturb other connectors. The finding contributes a concrete, transferable mechanism (KV-specific encoder + bandwidth-adaptive chunk compression) rather than a topical restatement of the existing approach, which today moves uncompressed KV tensors.

---

### 4. Register a batch-transfer KV connector variant that packs all layers into one transfer call
- **Finding:** `find-0004` — *[PD] optimize kv cache transfer directly using batch transfer*
- **Source URL:** <https://github.com/sgl-project/sglang/pull/9149>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new KVConnectorBase_V1 implementation (e.g., a `BatchTransferNixlConnector` or a flag-gated mode on an existing transport-based connector such as `NixlConnector` in vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py) and bind it through the registration block at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 via `KVConnectorFactory.register_connector(...)`. The new connector packs the per-layer transfer descriptors (src/dst block handles, sizes, layer indices) for all model layers of a request into a single batched descriptor list and submits one batch transfer call to the underlying transport, instead of dispatching per-layer executor work in the worker-side `start_load_kv` / `save_kv_layer` / `wait_for_save` paths. Naming it as a distinct registry entry (alongside NixlConnector, OffloadingConnector, MooncakeConnector, P2pNcclConnector) keeps the scheduler/worker contract identical so that finished-request sets, invalid block reporting, and generated tokens match the reference connector under tests/v1/kv_connector/unit/ and existing remote-prefill/offload end-to-end tests; only the transport submission mechanics change.

**Proposal rationale.**

The candidate is precisely the plugin seam where alternative transport/cache-policy connectors are bound to public names; its evolve_rationale explicitly invites new KVConnectorBase_V1 implementations that differ only in transport mechanics. The finding contributes a concrete, transferable mechanic — coalescing per-layer transfer descriptors into a single batch call — drawn from an external implementation (sglang PR #9149) that reports reduced launch/control overhead for many-layer models. For the caller's multi-turn agentic workload, this targets median TTFT (remote-prefill loads on long prefixes) and median TPOT (per-step save overhead) by collapsing O(num_layers) submission work per transfer into O(1), without touching the scheduler contract — a good fit for the registration seam rather than a deeper refactor.

---

### 5. Register an overlap-scheduled KV transfer connector that pipelines transmission with compute for other requests
- **Finding:** `find-0005` — *Disaggregated Serving — TensorRT LLM*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0rc4/features/disagg-serving.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new connector implementation under vllm/distributed/kv_transfer/kv_connector/v1/ (e.g., an `OverlapConnector` or an overlap-mode variant of NixlConnector) that conforms to KVConnectorBase_V1, then register it in the registration block at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 with a new `KVConnectorFactory.register_connector(...)` call binding a public name (e.g., "OverlapConnector") to its lazy import. The connector's scheduler-side methods (request_finished, get_num_new_matched_tokens, build_connector_meta) and worker-side methods (start_load_kv, wait_for_layerwise_load, save_kv_layer, get_finished) implement the TensorRT-LLM overlap technique: KV send/receive operations for one request are issued asynchronously and explicitly tracked so that, while transfers for request A are in flight on the transport (NIXL/Mooncake/NCCL), worker forward passes can proceed for independent requests B/C whose KV state is already resident or whose transfers have completed. Concretely, the connector maintains a per-request transfer-completion set used by `get_finished` and admits requests into compute as soon as their dependent KV blocks are local, decoupling the transfer critical path from the per-step compute schedule. The change at the candidate site is one registration entry plus the lazy-loader callable; the substantive code lives in the new connector class. Selection is via KVTransferConfig.kv_connector (or kv_connector_module_path for external use). Correctness oracle from evolve_rationale holds: connector metadata, finished sets, invalid-block reports, and generated tokens must match an existing connector under tests in tests/v1/kv_connector/unit/ and remote-prefill/offload e2e output tests; only transfer admission and overlap mechanics differ.

**Proposal rationale.**

The candidate explicitly frames new transports/policies as new connectors registered at this seam, and TensorRT-LLM's documented technique targets exactly the gap that drives median TTFT and TPOT in this caller context: KV transfer time appearing on the critical path of forward passes. Multi-turn agentic workloads amplify this because remote prefill or offload hits are frequent and serialized waits on transport stall otherwise-ready requests. Overlapping transmission with compute for independent requests addresses that constraint directly without changing the scheduler/worker contract or model execution code, which is the kind of transferable idea this plugin seam is designed to absorb. The finding contributes a concrete mechanism (async-tracked transfers gated only at the per-request level) rather than restating the current approach, where existing connectors already register here but do not document this overlap admission policy at the connector boundary.

---

### 6. Add a ring-buffer staging KV connector that coalesces heterogeneous-TP slices into larger transfers
- **Finding:** `find-0006` — *[Roadmap] Prefill-Decode Disaggregation Roadmap (2026 Q2)*
- **Source URL:** <https://github.com/sgl-project/sglang/issues/21703>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new `KVConnectorBase_V1` implementation (e.g., `RingStagingConnector` under `vllm/distributed/kv_transfer/kv_connector/v1/`) and register it at `vllm/distributed/kv_transfer/kv_connector/factory.py:149-228` via an additional `KVConnectorFactory.register_connector(...)` call, alongside existing entries like `NixlConnector` and `P2pNcclConnector`. The connector maintains a fixed-size GPU staging buffer with a dynamic ring allocator on both prefill and decode sides. On the prefill/save path, per-layer per-head KV slices for a request (or a small batch of requests) are gathered into a contiguous region of the ring buffer before a single RDMA/NCCL send; on the decode/load path, the contiguous chunk is received into the ring buffer and scattered back into the destination KV cache layout. The connector keeps the existing scheduler-side contract (connector metadata, finished request sets, invalid block reporting) unchanged, so correctness can be validated against an existing connector under `tests/v1/kv_connector/unit/` and end-to-end remote-prefill output tests. Heterogeneous-TP support is the primary motivator: when prefill TP != decode TP, head slices on each side are small, and coalescing them through the staging ring amortizes per-transfer setup. Selection is via `KVTransferConfig.kv_connector` set to the new registered name; `kv_connector_module_path` continues to work for out-of-tree variants.

**Proposal rationale.**

The candidate is the exact seam for plugging in a new transport/admission strategy without touching scheduler or model execution code, and the finding contributes a concrete, transferable mechanism (ring-based GPU staging buffer to unite many small KV head slices into one large transfer) that is directly relevant to TTFT in multi-turn agentic workloads where remote-prefill transfer setup dominates. Existing connectors here (Nixl, Mooncake, P2pNccl) issue per-layer/per-head transfers that become small under heterogeneous TP; the finding's coalescing approach plausibly raises effective bandwidth utilization and lowers TTFT, while keeping the V1 contract fixed so the correctness oracle described in `evolve_rationale` still applies. The proposal lives entirely behind the registration seam, which is the minimum-risk location to evolve transport mechanics.

---

### 7. Pre-warm KV connector handshakes at factory creation time
- **Finding:** `find-0007` — *Prefill-decode disaggregation — Ray 2.55.1*
- **Source URL:** <https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the connector seam in vllm/distributed/kv_transfer/kv_connector/factory.py (lines 149-228 registration block plus the adjacent create_connector path) so that built-in connectors can declare an eager pre-warm step that runs immediately after instantiation, before the first real request. Concretely: (1) augment register_connector to accept an optional warmup callable or rely on a standard KVConnectorBase_V1 hook (e.g. prewarm()/establish_handshakes()) that subclasses may override; (2) in create_connector, after constructing the selected class from the registry (or from kv_connector_module_path), invoke that hook when enabled by config (e.g. a new KVTransferConfig.kv_connector_prewarm flag, default-on for connectors that benefit such as NixlConnector and the remote-prefill paths in p2p_nccl_connector and mooncake_connector); (3) implement the hook on NixlConnector to eagerly perform the prefill<->decode metadata/agent handshake that today happens lazily on the first request, and on OffloadingConnector to pre-touch the offload backend so admission paths are warm. Registration entries for connectors that do not need warmup (e.g. SharedStorageConnector) simply inherit the no-op base implementation. Existing unit tests in tests/v1/kv_connector/unit/ continue to validate the scheduler/worker contract; pre-warm only affects setup timing, not correctness oracles (metadata, finished sets, invalid blocks, token outputs).

**Proposal rationale.**

The candidate is the lifecycle entry point where a connector class transitions from registered-but-lazy to a live instance, which is precisely where a handshake pre-warm hook belongs. The Ray Serve prefill/decode disaggregation guidance explicitly calls out that NIXL-style connectors require a prefill<->decode handshake that today happens eagerly on the first request, paying that cost in TTFT. The caller cares about median TTFT and TPOT for a multi-turn agentic workload, where the first turn of each session (and any newly-scaled replica) would otherwise absorb that handshake latency. Wiring a standardized prewarm hook into the factory seam transfers the finding's recommendation into a single, low-risk surface that all built-in v1 connectors and external kv_connector_module_path plugins can opt into without changes to scheduler or model-execution code.

---

### 8. Add a non-blocking KV transfer connector with partial-readiness admission
- **Finding:** `find-0008` — *Disaggregated Serving | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new KVConnectorBase_V1 implementation registered at vllm/distributed/kv_transfer/kv_connector/factory.py (lines 149-228) via an additional KVConnectorFactory.register_connector(...) call (e.g., name 'NonBlockingDisaggConnector'). The connector wraps an existing transport (NIXL or P2P-NCCL) but reshapes its scheduler-side contract so KV transfer is treated as background work that does not gate decode. Concretely: (1) extend the connector's metadata/state returned to the scheduler so blocks expose finer-grained readiness (per-block or per-layer 'transfer_in_flight' vs 'ready') rather than only an all-or-nothing finished-request set; (2) admit a request to decode as soon as the minimum prefix needed for the next forward pass is ready, while remaining blocks continue arriving; (3) keep the worker-side save/load on a non-default CUDA stream so forward passes for other requests overlap transfer, mirroring the Dynamo 'KV transfer is non-blocking' pattern. The plugin seam itself does not change shape: only one new lazy-loaded class is added to _registry, and selection flows through KVTransferConfig.kv_connector. Correctness is checked against the same oracle described in evolve_rationale: identical token outputs and identical finished/invalid-block reporting versus a synchronous baseline connector under tests/v1/kv_connector/unit/ and remote-prefill/offload e2e tests, with only timing and admission order differing.

**Proposal rationale.**

The candidate is the registration site where new connector behaviors are introduced without touching scheduler or model-execution code, and KVConnectorBase_V1 already exposes the metadata + finished-request hooks needed to encode partial readiness. The finding contributes a concrete, transferable design idea from NVIDIA Dynamo: treat KV transfer as background work and let forward passes continue during transfer. For the stated multi-turn agentic workload, the dominant TTFT cost on cache-miss turns is waiting for a full prefix handoff, and the dominant TPOT cost is per-step save/load synchronization; non-blocking transfer with finer readiness directly attacks both by letting decode start on the earliest-ready prefix and overlapping in-flight transfer with other requests' forward passes. This is a new connector variant rather than a restatement of any currently registered one (Nixl/Offloading/Mooncake/P2pNccl), so it fits the plugin-seam evolution intent without inventing locations outside this seam.

---

### 9. Add a Mooncake-Store-backed KV connector with multi-replica, striped, multi-NIC RDMA transport
- **Finding:** `find-0009` — *Mooncake/docs/source/design/mooncake-store.md at main · kvcache-ai/Mooncake*
- **Source URL:** <https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/mooncake-store.md>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Register a new connector at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 (e.g., `KVConnectorFactory.register_connector("MooncakeStoreConnector", "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.connector", "MooncakeStoreConnector")`) that implements the existing `KVConnectorBase_V1` contract from vllm/distributed/kv_transfer/kv_connector/v1/base.py. The new connector wraps the Mooncake Store distributed object cache and exploits four features called out in the finding: (1) multi-replica object placement for hot shared prefixes so concurrent multi-turn requests do not converge on a single source worker; (2) zero-copy transfers between the connector buffer and the GPU KV layout to remove a host-side memcpy from the load/save path used by `start_load_kv`/`save_kv_layer`; (3) multi-NIC RDMA pooling to aggregate bandwidth across the NICs available on a remote-prefill node; and (4) striping of large KV objects (long-context prefix blocks) across storage nodes so a single block read is parallelized rather than serialized on one peer. Selection remains via `KVTransferConfig.kv_connector="MooncakeStoreConnector"`, mirroring how the existing `MooncakeConnector` (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py) is wired in. Correctness is validated against the same scheduler/worker oracle the rationale already cites: identical connector metadata, finished-request sets, invalid-block reports, and generated tokens versus `OffloadingConnector` and `NixlConnector` under tests/v1/kv_connector/unit/ and the existing remote-prefill/offload end-to-end output tests; only transport mechanics differ.

**Proposal rationale.**

The candidate is explicitly a plugin seam whose impact knob is connector choice, and the stated objective is reducing median TTFT and TPOT for a multi-turn agentic workload — a regime dominated by shared-prefix reuse across requests. The finding contributes concrete, transferable transport mechanisms (multi-replica placement, zero-copy, multi-NIC RDMA pooling, striping) that target exactly the failure modes that bound remote-prefix reuse today: hot-spotting on the worker that originally produced a popular prefix, single-NIC bandwidth caps on large block transfers, and host-copy overhead in the per-step save/load path. None of the currently registered connectors (NixlConnector, OffloadingConnector, MooncakeConnector, P2pNcclConnector) combine all four of these properties; the existing `MooncakeConnector` integrates with Mooncake transport but does not expose the Store's multi-replica/striping policy at the connector layer. Adding the connector is a localized change at this seam — one `register_connector` call plus a new module implementing `KVConnectorBase_V1` — and is therefore a plausible, bounded path to higher remote-prefix hit rate (TTFT) and lower per-step transfer overhead (TPOT) without touching scheduler or model-execution code.

---

### 10. Add an LMCache-style connector with batched KV movement and compute/I/O pipelining
- **Finding:** `find-0010` — *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*
- **Source URL:** <https://arxiv.org/abs/2510.09665>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new V1 connector (e.g. `LMCacheConnector` implementing `KVConnectorBase_V1` from `vllm/distributed/kv_transfer/kv_connector/v1/base.py`) and register it at the existing plugin seam in `vllm/distributed/kv_transfer/kv_connector/factory.py:149-228` via a single `KVConnectorFactory.register_connector("LMCacheConnector", "vllm.distributed.kv_transfer.kv_connector.v1.lmcache.lmcache_connector", "LMCacheConnector")` call, alongside the existing Nixl/Offloading/Mooncake/P2pNccl entries. The connector internalizes LMCache's two headline techniques: (1) batched KV data movement — coalesce per-block save/load operations across layers and requests into a small number of large transfers between GPU, pinned CPU, local storage, and the network tier, instead of issuing per-block ops; (2) compute/I/O pipelining — overlap KV save/load with model forward execution by issuing transfers on dedicated streams/threads scheduled around the connector's `start_load_kv` / `wait_for_layer_load` / `save_kv_layer` / `wait_for_save` worker hooks, and by emitting `KVConnectorMetadata` from the scheduler-side hooks (`get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`) that lets the worker prefetch the next layer's blocks while the current layer computes. The connector is selected at runtime via `KVTransferConfig.kv_connector="LMCacheConnector"`, and `KVTransferConfig.kv_connector_module_path` continues to allow out-of-tree variants without further registry changes. No edits to scheduler or model-execution code are required — the seam already isolates this.

**Proposal rationale.**

The candidate seam exists precisely to admit new transport/admission strategies that conform to `KVConnectorBase_V1` while leaving scheduler and model code untouched. The LMCache paper reports that its performance comes from batched KV data movement and compute/I/O pipelining across a multi-tier (GPU/CPU/storage/network) cache — exactly the two levers that the candidate's evolve_rationale identifies as moving median TTFT (higher remote-prefix hit rate, lower transfer setup cost) and median TPOT (reduced per-step save/load overhead). For the multi-turn agentic workload in the caller context, prior-turn prefixes are the dominant reuse opportunity, and batching plus overlap directly target the per-block setup cost and the stall-on-load tax that current connectors like `OffloadingConnector` and `NixlConnector` pay on each layer boundary. The change is also conservative with respect to the correctness oracle: only transport mechanics and admission policy differ, so existing `tests/v1/kv_connector/unit/` and end-to-end remote-prefill/offload output tests remain valid checks.

---

### 11. Add a direct-to-host RDMA KV connector with GPU hot-buffer + top-k swap
- **Finding:** `find-0011` — *HiSparse: Hierarchical Sparse Attention - SGLang Documentation*
- **Source URL:** <https://docs.sglang.io/docs/advanced_features/hisparse_guide>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Register a new KV transfer connector at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 (e.g. KVConnectorFactory.register_connector("DirectToHostConnector", ...)) backed by a new implementation under vllm/distributed/kv_transfer/kv_connector/v1/ that subclasses KVConnectorBase_V1 (vllm/distributed/kv_transfer/kv_connector/v1/base.py). The connector's worker side, on remote-prefill pull, would RDMA-write the prefill KV blocks directly into a host-resident pool (pinned/CPU memory) instead of staging them through GPU HBM, mirroring HiSparse's PD-disaggregation direct-to-host path. A small fixed-size GPU "hot" KV buffer is maintained per layer; the scheduler-side hooks (get_num_new_matched_tokens / build_connector_meta / request_finished) and worker-side hooks (start_load_kv / wait_for_layer_load / save_kv_layer) would be specialized to (a) treat host-pool blocks as locally hit (no recompute) and (b) admit only top-k blocks (by recent-use or attention-score signal supplied via connector metadata) into the GPU hot buffer on demand, evicting cold blocks back to host. Only the new file plus one register_connector(...) line in factory.py would be added; the V1 contract surface is unchanged so correctness can be checked against existing tests/v1/kv_connector/unit/ and the remote-prefill/offload e2e output tests by comparing generated tokens against NixlConnector / OffloadingConnector on the same request stream.

**Proposal rationale.**

The candidate is exactly the plugin seam for transport + admission policy, and the finding contributes a concrete, transferable transport idea: bypass GPU on the decode side during remote-prefill transfer and keep only a hot subset on GPU. This addresses two specific gaps in current built-in connectors registered at this seam: NixlConnector pulls remote KV onto GPU (transient HBM pressure that limits decode concurrency on long contexts, hurting TPOT), and OffloadingConnector targets local CPU/disk reuse rather than remote-prefill staging into host (so its TTFT win on remote-prefix-hit cases is bounded by GPU staging). A direct-to-host connector with top-k GPU residency can plausibly improve median TTFT on multi-turn agentic workloads by raising effective remote-prefix hit rate (host pool is larger and cheaper than HBM, so more turns hit) and improve median TPOT by freeing HBM for larger decode batch sizes during transfer. The change is scoped to a new file plus one registration line, matching the seam's intent.

---

### 12. Add a MemServe-style MemPool KV connector with prompt-tree locality
- **Finding:** `find-0012` — *MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool*
- **Source URL:** <https://arxiv.org/abs/2406.17565>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new `KVConnectorBase_V1` implementation (e.g. `MemPoolConnector` under `vllm/distributed/kv_transfer/kv_connector/v1/mempool/`) and register it at the plugin seam in `vllm/distributed/kv_transfer/kv_connector/factory.py` (lines 149-228) via an additional `KVConnectorFactory.register_connector("MemPoolConnector", ...)` call alongside the existing `NixlConnector`, `OffloadingConnector`, `MooncakeConnector`, and `P2pNcclConnector` entries. The connector wraps an elastic distributed memory pool (MemPool) shared across prefill/decode/offload instances and exposes it through the existing scheduler/worker contract: `get_num_new_matched_tokens` consults a global prompt-tree index keyed by token-block hashes to decide how many prefix tokens are remotely cached and where, `build_connector_meta` packs (request_id, block_ids, source_instance) lookup directives, and worker-side `start_load_kv` / `wait_for_save` execute placement-aware fetches from the pool instead of issuing one-off point-to-point copies. Placement and admission inside the pool follow MemServe's global prompt-tree locality policy (prefer the instance already holding the longest matching prefix, with elastic membership so instances can join/leave the pool), while the connector itself only needs the new module + the registration line; no scheduler or model-execution changes are required because the existing `KVConnectorBase_V1` interface already expresses match/load/save semantics.

**Proposal rationale.**

The candidate seam is exactly the place where a new transport+admission policy becomes selectable via `KVTransferConfig.kv_connector`, and the caller's workload is multi-turn agentic with TTFT/TPOT objectives — the regime MemServe targets. Existing connectors (Nixl, P2pNccl, Mooncake) treat each P/D transfer as a point-to-point copy and rely on the local prefix cache; OffloadingConnector adds a local tier but no cross-instance locality. MemServe's contribution — an elastic shared pool plus a global prompt-tree locality policy — addresses the concrete gap that, in multi-turn workloads, the instance with the best prefix match for a follow-up turn is often not the one routed to, causing redundant recomputation or full prefix transfers. Slotting this in as a new connector at the registration block preserves the correctness oracle in `evolve_rationale` (same scheduler/worker contract, same finished/invalid-block semantics; only transport and admission differ) while plausibly raising remote-prefix hit rate (lower median TTFT) and reducing per-step save/load overhead from coordinated, locality-aware placement (lower median TPOT).

---

### 13. Register a cache-aware connector that exposes KV overlap and transfer-cost signals for routing
- **Finding:** `find-0013` — *Router Guide | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.dynamo.nvidia.com/dynamo/user-guides/kv-cache-aware-routing>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new built-in connector at the KVConnectorFactory registration seam (vllm/distributed/kv_transfer/kv_connector/factory.py:149-228) whose KVConnectorBase_V1 implementation surfaces the signals needed for cache-aware routing of multi-turn requests. Concretely: (1) implement a new connector class under vllm/distributed/kv_transfer/kv_connector/v1/ (e.g., a 'CacheAwareConnector' that wraps or extends an existing transport such as NixlConnector or OffloadingConnector), and (2) add one register_connector(...) call in the same block alongside the existing 'NixlConnector', 'OffloadingConnector', 'MooncakeConnector', 'P2pNcclConnector' entries. The new connector's scheduler-side methods (get_num_new_matched_tokens / build_connector_meta and any added query API) expose, per request and per candidate endpoint: KV-block overlap with the endpoint's locally cached / offloaded blocks, count of in-flight transfers, and block-availability/admission state. These signals replicate the Dynamo router's cost-model inputs (active-decode load, new-prefill work, KV overlap) at the connector boundary so an upstream router can pick endpoints that avoid redundant transfer and recompute. The connector keeps the fixed v1 contract (connector metadata, finished-request sets, invalid-block reporting, token outputs identical to a baseline connector under tests/v1/kv_connector/unit/), changing only transport mechanics and admission policy.

**Proposal rationale.**

The candidate is the plugin seam where a new connector becomes selectable via KVTransferConfig.kv_connector. The finding contributes a concrete, transferable design: a cost model combining decode load, prefill work, and KV overlap to minimize redundant computation. The current built-in connectors transport KV but do not expose overlap / inflight-transfer / availability signals usable by a router, so multi-turn agentic requests can land on endpoints that re-fetch or recompute prefixes already cached elsewhere - directly inflating median TTFT and adding per-step save/load overhead that affects TPOT. Registering a connector that publishes these signals at this exact seam closes that gap without touching scheduler or model-execution code, and aligns with the candidate's own evolve_rationale that connector swaps should change only transport and admission policy.

---

### 14. Register a cache-aware CPD connector that classifies requests by warm/cold cache-hit class
- **Finding:** `find-0014` — *Cache-aware prefill–decode disaggregation (CPD) for up to 40% faster long-context LLM serving*
- **Source URL:** <https://www.together.ai/blog/cache-aware-disaggregated-inference>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new built-in KV transfer connector registration in `vllm/distributed/kv_transfer/kv_connector/factory.py` (lines 149-228) — e.g. `KVConnectorFactory.register_connector("CacheAwareCPDConnector", "vllm.distributed.kv_transfer.kv_connector.v1.cache_aware_cpd_connector", "CacheAwareCPDConnector")` — implementing `KVConnectorBase_V1` (see `vllm/distributed/kv_transfer/kv_connector/v1/base.py`). The connector wraps an underlying transport (composing or subclassing `NixlConnector` / `OffloadingConnector` from `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py` and `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`). On the scheduler side, in `get_num_new_matched_tokens` / `update_state_after_alloc` / `build_connector_meta`, it inspects each request against the local prefix-cache hit length and any remote-prefix lookup result to compute a cache-class label (`warm` if hit_ratio ≥ threshold or remote-prefix transfer is small/cheap, `cold` otherwise) plus an estimated transfer cost (bytes × per-link cost). It surfaces these as per-request fields in the connector metadata returned to the engine and as a sidecar API (e.g. `classify_request(request) -> {class, est_transfer_bytes, est_setup_cost}`) that an external router or the V1 scheduler's request-prioritization hook can read to keep cold prefills out of the warm path. Worker-side `start_load_kv` / `wait_for_save` behavior is unchanged for correctness — only ordering and admission are influenced — so existing unit tests in `tests/v1/kv_connector/unit/` and remote-prefill/offload output tests remain valid oracles. Configuration knobs (warm threshold, cost weights, underlying transport name) flow through `KVTransferConfig.kv_connector_extra_config`.

**Proposal rationale.**

The candidate seam is exactly where new connector behaviors are bound, and the registry pattern allows adding a CPD-aware variant without touching scheduler or model execution. The finding contributes a concrete, transferable idea — classify requests by cache-hit rate / transfer cost so warm reusable contexts are not queued behind cold prefills — that the current connectors (Nixl, Offloading, Mooncake, P2pNccl) do not expose: today they transport blocks but emit no warm/cold class or transfer-cost estimate the caller can route on. For the stated multi-turn agentic workload, prior-turn prefixes dominate the warm class, so exposing this classification at the connector boundary directly attacks median TTFT (warm requests skip the cold queue) and reduces tail TPOT (less interference between large cold prefills and decoding steps). The change is bounded to one new registration plus one new module implementing the fixed `KVConnectorBase_V1` contract, so the correctness oracle described in `evolve_rationale` is preserved.

---

### 15. Register a tier-aware KVCache-centric connector composing DRAM/SSD/NIC tiers with reuse-driven admission
- **Finding:** `find-0015` — *Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot*
- **Source URL:** <https://www.usenix.org/system/files/fast25-qin.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new registration entry in the KVConnectorFactory block at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 (alongside existing OffloadingConnector, SimpleCPUOffloadConnector, MooncakeConnector, HF3FSKVConnector, and MultiConnector entries) for a new connector class implementing KVConnectorBase_V1 (vllm/distributed/kv_transfer/kv_connector/v1/base.py) that exposes a Mooncake-style multi-tier KV cache fabric. The new connector should: (1) treat GPU HBM, host DRAM, local/remote SSD, and remote-prefill peers as ordered tiers behind one connector, leveraging the existing OffloadingConnector and HF3FSKVConnector implementations as building blocks rather than reimplementing transports; (2) implement get_num_new_matched_tokens / start_load_kv to query each tier in cost order and report the longest hit so the scheduler can skip prefill on multi-turn prefixes; (3) drive an admission policy in save_kv_layer / wait_for_save that promotes/demotes blocks across tiers based on observed reuse probability and tier capacity, biasing toward keeping blocks that recently produced hits in the fastest reachable tier; (4) expose tier capacities and weights via fields on KVTransferConfig so operators can trade storage for compute without code changes. The registration call is a single new register_connector(...) line in this block; the rest of the seam (KVTransferConfig.kv_connector dispatch, kv_connector_module_path override) is unchanged.

**Proposal rationale.**

The candidate is a plugin seam whose evolve_rationale explicitly invites new KVConnectorBase_V1 implementations registered with one line here, and the caller workload is multi-turn agentic with explicit TTFT/TPOT targets. The Mooncake paper's specific contribution beyond point-to-point disaggregation (which the existing MooncakeConnector covers) is the KVCache-centric tier fabric that pools cluster-wide CPU, DRAM, SSD, and NIC capacity and trades storage for prefill recompute. Multi-turn agentic prefixes are exactly the regime where this trade pays off in TTFT, because long shared prefixes that miss in HBM today either trigger recompute or fall back to a single-tier offload. The current registry already has single-tier offload connectors (SimpleCPUOffloadConnector, HF3FSKVConnector) and a transport-oriented MooncakeConnector but no connector that unifies them with reuse-aware admission, so the finding contributes a concrete and transferable architectural idea at the granularity of one new registered connector.

---

### 16. Add a CacheBlend-style chunk-reuse KV connector at the factory registration seam
- **Finding:** `find-0016` — *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*
- **Source URL:** <https://www.microsoft.com/en-us/research/wp-content/uploads/2024/09/eurosys25-final999.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new built-in connector (e.g., `CacheBlendConnector`) implementing `KVConnectorBase_V1` and register it at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 with a single `KVConnectorFactory.register_connector("CacheBlendConnector", "vllm.distributed.kv_transfer.kv_connector.v1.cacheblend.cacheblend_connector", "CacheBlendConnector")` call, mirroring the lazy-loaded pattern used by `NixlConnector`, `OffloadingConnector`, and `MooncakeConnector`. The connector's scheduler-side role (`get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`, `request_finished`) would (a) match request prompts against a chunk-addressed external KV store keyed by content hashes of arbitrary (non-prefix) chunks, (b) emit connector metadata that lists, per chunk, the source location to fetch and a small set of "recompute marker" token positions selected per CacheBlend's selective-recompute heuristic to repair cross-chunk attention. The worker-side role (`start_load_kv`, `wait_for_layer_load`, `save_kv_layer`, `wait_for_save`, `get_finished`) would pipeline chunk KV retrieval with the targeted recompute of marker tokens so transfer latency is hidden behind partial prefill, then write fused chunks back to the store on completion. Only the transport/admission policy and the recompute-marker selection live inside this connector; scheduler and model-execution code remain unchanged because the contract is the existing `KVConnectorBase_V1`. Configuration would flow through `KVTransferConfig.kv_connector="CacheBlendConnector"` (or `kv_connector_module_path` for an out-of-tree variant), and correctness can be validated against an existing connector under `tests/v1/kv_connector/unit/` plus end-to-end RAG-style multi-turn output tests.

**Proposal rationale.**

The candidate is explicitly the plugin seam where a new transport+admission policy can be added with one registration line, and the evolve_rationale calls out remote-prefix hit rate and per-step save/load overhead as the levers that move TTFT/TPOT. CacheBlend directly attacks both: it lifts reuse from prefix-only to arbitrary recurring chunks (raising hit rate for multi-turn agentic/RAG prompts assembled from recurring pieces, which matches the caller workload hint), and it pipelines KV retrieval with a small selective recompute, which keeps the per-step transfer cost bounded while preserving cross-chunk correctness. The contribution is concrete and transferable: chunk-addressed fetches plus recompute markers map cleanly onto the existing `build_connector_meta`/`start_load_kv`/`save_kv_layer` hooks, so the finding adds a new, plausibly-better point in the connector design space rather than restating the prefix-cache behavior already covered by `NixlConnector` or local-offload behavior covered by `OffloadingConnector`.

---

### 17. Register a KIVI-quantized KV transfer connector with asymmetric per-channel/per-token wire format
- **Finding:** `find-0017` — *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*
- **Source URL:** <https://proceedings.mlr.press/v235/liu24bz.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new built-in connector registration in vllm/distributed/kv_transfer/kv_connector/factory.py (lines 149-228) that binds a name (e.g. "KIVIQuantizedConnector") to a lazy-loaded class implementing KVConnectorBase_V1. The new connector wraps an existing transport (e.g. NixlConnector or P2pNcclConnector) but applies asymmetric low-bit quantization on the KV blocks before they are placed on the wire and dequantizes on the receiving worker: keys quantized per-channel, values quantized per-token, following KIVI's asymmetric scheme. Only the transport/serialization path changes; the scheduler-side contract (connector metadata, finished/invalid block reporting, request lifecycle) and the worker-side load/save semantics defined by KVConnectorBase_V1 in vllm/distributed/kv_transfer/kv_connector/v1/base.py remain identical, so the existing correctness oracle in tests/v1/kv_connector/unit/ and end-to-end remote-prefill/offload tests still apply (with a tolerance on token-level outputs to account for quantization noise). The connector exposes the bit-width and per-axis layout via KVTransferConfig fields so the same registration entry covers different precision points without new seams.

**Proposal rationale.**

The candidate is exactly the seam where new connector implementations are bound; transport mechanics and admission policy can vary while the V1 contract is preserved. The stated impact pathway is that a better connector can reduce TTFT by lowering transfer setup/bandwidth cost for multi-turn prefixes pulled from remote prefill or offload tiers. KIVI's asymmetric per-channel-key / per-token-value 2-bit quantization is a concrete, well-characterized format that directly attacks the bytes-on-the-wire term: at ~8x compression versus fp16, transfer-bandwidth-bound prefix loads shrink proportionally, which is the dominant cost in agentic multi-turn workloads where the prefix-to-decoded-token ratio is high. The finding contributes the specific, transferable design choice (asymmetry across the K/V axes) that makes 2-bit KV viable without per-model tuning, which is what justifies wiring it in at the connector seam rather than requiring model-side changes.

---

### 18. Register an InfiniGen-style importance-aware prefetching KV connector
- **Finding:** `find-0018` — *InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management*
- **Source URL:** <https://www.usenix.org/system/files/osdi24-lee.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new connector implementation under vllm/distributed/kv_transfer/kv_connector/v1/ (e.g., InfiniGenConnector) that conforms to KVConnectorBase_V1 and bind it via a new KVConnectorFactory.register_connector(...) call in vllm/distributed/kv_transfer/kv_connector/factory.py within the existing registration block (lines 149-228), alongside NixlConnector, OffloadingConnector, MooncakeConnector, and P2pNcclConnector. The connector wraps an underlying transport (offload-to-host or remote prefill peer) and adds an importance-prediction layer modeled on InfiniGen: at decode-step boundaries it runs a lightweight rehearsal over a small projection of the upcoming layer's query against cached key summaries to score blocks, then issues prefetch requests in importance order for the per-step KV needed by the worker. Blocks predicted as important are pulled into the GPU-resident cache eagerly to overlap with prior-layer compute, while remaining blocks are still fetched on demand to preserve correctness against the existing connector oracle (matching finished-request sets, invalid-block reports, and token outputs in tests/v1/kv_connector/unit/ and the remote-prefill/offload e2e suites). The rehearsal state and per-block importance scores live entirely inside the connector's scheduler/worker objects, so no scheduler or model-runner code changes are required; the only edit outside the new connector module is the single register_connector(...) line.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names transfer mechanics and admission policy as the in-scope axes for new connectors, and the caller objective is to reduce median TTFT and TPOT under multi-turn agentic workloads where per-step KV transfer from offload/remote tiers dominates. InfiniGen's central transferable idea -- predict the small subset of KV entries that the next layer will actually attend to and prefetch those first -- maps directly onto the connector's prefetch/admission decisions when KV must be moved from host memory or a remote prefill worker to GPU. By ordering transfers by predicted importance, the connector can deliver the KV needed for the critical path of attention earlier, shrinking per-step stall time (TPOT) without changing the contract surface, while a fall-through fetch for non-predicted blocks preserves exact correctness against the existing connector oracle. This is a concrete, plugin-local change that fits the seam at lines 149-228 without touching scheduler or model code.

---

### 19. Register a frequency-admission KV connector variant with CMS-gated local hot tier
- **Finding:** `find-0019` — *[STORE] feat: Frequency admission + LRU lock optimization for local hot cache*
- **Source URL:** <https://github.com/kvcache-ai/Mooncake/pull/1596>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new connector implementation behind the existing plugin seam in vllm/distributed/kv_transfer/kv_connector/factory.py (lines 149-228). The new class implements KVConnectorBase_V1 (vllm/distributed/kv_transfer/kv_connector/v1/base.py) and is wired via a single KVConnectorFactory.register_connector(...) call alongside the current built-ins (NixlConnector, OffloadingConnector, MooncakeConnector, P2pNcclConnector). Internally, the connector layers a local hot KV tier in front of its remote/offload backend (this can be done by composing with OffloadingConnector's local store path or by adding a small in-process cache in front of a Mooncake/Nixl-style transport). Admission to the hot tier is gated by a Count-Min Sketch counter keyed by block hash (or KV cache key), with promotion only when access count reaches a configurable threshold K (default K=2 per the Mooncake PR). The read path uses shared/reader locks for lookup and defers LRU recency updates (e.g. via a small bounded queue flushed under a writer lock) so concurrent hits do not serialize on a global mutex. Selection is opt-in via KVTransferConfig.kv_connector = "<new-name>", preserving the existing scheduler/worker contract; only admission and locking policy change.

**Proposal rationale.**

The candidate seam exists precisely to allow new connectors that differ only in transport and admission policy while preserving the KVConnectorBase_V1 contract — its evolve_rationale calls this out explicitly. The finding contributes two concrete, transferable mechanisms that target the gap most relevant to the stated caller objective (reduce median TTFT/TPOT under a multi-turn agentic workload): (1) CMS frequency admission directly counters hot-cache pollution by one-shot prefixes that are common in agent traces, raising hit rate for the recurring system/tool prefixes that dominate multi-turn TTFT; (2) shared-lock reads with deferred LRU touches reduce per-step lookup overhead, which shows up in TPOT when many concurrent decode steps probe the cache. Both are localized to a connector implementation and require only the standard one-line registration in factory.py, so the change fits the seam exactly without touching scheduler or model execution code.

---

### 20. Register a bandwidth-aware KV connector variant exposing measured transfer cost
- **Finding:** `find-0020` — *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*
- **Source URL:** <https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new connector registration in the KVConnectorFactory block at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 (e.g. `register_connector("BandwidthAwareNixlConnector", ...)`) that wraps an existing V1 transport (NixlConnector or P2pNcclConnector) with a thin policy layer implementing DistServe-style bandwidth-aware placement. The new connector would: (1) at startup, probe per-peer bandwidth and persist a cost matrix in the connector's worker-side state; (2) expose those measurements through the existing KVConnectorBase_V1 scheduler-side metadata path so that connector metadata returned to the scheduler includes per-endpoint transfer cost estimates; (3) when multiple remote-prefill sources are eligible for a given prefix, pick the lowest-cost source rather than first-match. The KVConnectorBase_V1 contract in vllm/distributed/kv_transfer/kv_connector/v1/base.py is unchanged — only the connector implementation differs — so the correctness oracle (matching finished sets, invalid blocks, and generated tokens vs. NixlConnector) still applies under tests in tests/v1/kv_connector/unit/. Selection is via KVTransferConfig.kv_connector, exactly the seam the candidate describes.

**Proposal rationale.**

The candidate is precisely the place where a new transport/policy variant is plugged in, and its evolve_rationale calls out admission policy and transfer mechanics as the dimensions a new connector may differ on. DistServe's contribution — placing prefill/decode to minimize communication caused by disaggregation — maps directly onto remote-prefill connector source selection: today the connectors do not consult measured bandwidth when choosing which peer to pull KV from, so under multi-turn agentic workloads with multiple candidate prefill workers, TTFT can regress on bandwidth-skewed clusters. Adding bandwidth measurement as connector metadata addresses that gap without changing scheduler/model code, and is registered through the single seam this candidate identifies.

---

### 21. Register a bandwidth-aware selective remote-prefill connector at the factory seam
- **Finding:** `find-0021` — *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*
- **Source URL:** <https://arxiv.org/html/2604.15039v1>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new KVConnectorBase_V1 implementation (e.g. `SelectivePrefillConnector`) and wire it into the registration block at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 via a new `KVConnectorFactory.register_connector("SelectivePrefillConnector", ...)` call, alongside existing entries like NixlConnector, OffloadingConnector, MooncakeConnector, and P2pNcclConnector. The connector wraps an underlying remote-prefill transport (Nixl or P2pNccl) and an optional local offload tier, but interposes an admission policy in its scheduler-side `get_num_new_matched_tokens` / `update_state_after_alloc` path that decides per-request whether to (a) pull KV from a remote prefill worker, (b) load from local offload, or (c) fall back to recomputation. The decision uses three inputs available at scheduler time: prompt length and matched-prefix length (from the request and block manager), an EWMA estimate of effective inter-cluster bandwidth and current outbound queue depth maintained by the connector's worker side, and a cache-aware placement hint (which remote worker most likely holds the prefix, derived from a small consistent-hash table over prompt-prefix hashes that the connector already has for prefix matching). Admission rule: choose remote prefill only when `predicted_transfer_time(matched_tokens, bw, queue_depth) + remote_compute_residual < local_recompute_time(prompt_len)`; otherwise fall back to local. A `kv_transfer_config.extra_config` block exposes the bandwidth EWMA half-life, queue-depth backpressure threshold, and a min-prefix-length gate so the policy degrades to today's behavior when set conservatively. No change to KVConnectorBase_V1 is required; the contract (connector metadata, finished request sets, invalid block reporting, generated tokens) remains identical to the wrapped connector, which is exactly the correctness oracle described in the candidate.

**Proposal rationale.**

The candidate is explicitly the seam where transport and admission policy can be swapped without touching scheduler or model execution, and its evolve_rationale singles out transfer mechanics and admission policy as the dimensions that can move TTFT/TPOT. The finding's PrfaaS contribution is precisely an admission/placement layer (bandwidth-aware scheduling + cache-aware placement + selective offload) that the existing connectors lack: NixlConnector and P2pNcclConnector decide transport mechanics but not whether transferring beats recomputing, and OffloadingConnector handles only local tiering. For the multi-turn agentic workload in the caller context, where prefix reuse is high but variable across turns, gating remote prefill on predicted transfer-vs-recompute cost addresses a concrete gap: today an unlucky bandwidth dip or a short matched prefix can make remote prefill slower than recompute, hurting median TTFT, while always-local recompute forfeits the TPOT/TTFT wins on long shared prefixes. The change is transferable (an isolated new connector class plus one register_connector line) and falsifiable against existing tests in tests/v1/kv_connector/unit/ plus end-to-end remote-prefill output tests, satisfying the candidate's stated correctness oracle.

---

### 22. Add a chunk-streaming KV connector that emits KV per prefill chunk
- **Finding:** `find-0022` — *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*
- **Source URL:** <https://www.microsoft.com/en-us/research/publication/sarathi-efficient-llm-inference-by-piggybacking-decodes-with-chunked-prefills/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new KVConnectorBase_V1 implementation (e.g., ChunkStreamingConnector) and register it in vllm/distributed/kv_transfer/kv_connector/factory.py at the registration block (lines 149-228) alongside NixlConnector, OffloadingConnector, MooncakeConnector, and P2pNcclConnector via KVConnectorFactory.register_connector(...). The connector adapts SARATHI's chunked-prefill idea to KV transfer: instead of waiting for an entire long prompt's KV to be materialized before save/load, the connector tracks per-chunk KV-block readiness and streams completed chunk-aligned block ranges to the receiver (offload tier or remote prefill peer) as soon as each chunk's prefill step finishes. Concretely, the worker-side hooks (save_kv_layer / wait_for_save / get_finished) report block ranges at chunk granularity rather than only at request completion, and the scheduler-side connector metadata advertises partial-prefix availability so a decode-side or downstream worker can begin load/compute on the already-transferred prefix while later chunks are still in flight. The factory entry is the minimal surface change required here; the bulk of the logic lives in a new module under vllm/distributed/kv_transfer/kv_connector/v1/. Selection is via KVTransferConfig.kv_connector="ChunkStreamingConnector" (or via kv_connector_module_path for out-of-tree variants), and behavior must match an existing connector on the unit tests in tests/v1/kv_connector/unit/ and end-to-end remote-prefill/offload output tests for finished-request sets, invalid-block reporting, and generated tokens; only transfer timing and admission policy differ.

**Proposal rationale.**

The candidate is explicitly a plugin seam whose only required change to add a new transport/admission policy is one register_connector call, and the evolve_rationale identifies transfer setup cost and remote-prefix availability as the main TTFT/TPOT levers. SARATHI's chunked-prefill decomposition is directly transferable to the KV-transfer layer: chunk boundaries provide a natural unit at which KV blocks become consistent and shippable, so overlapping transfer with prefill compute is well-defined rather than ad-hoc. For the stated multi-turn agentic workload, where long shared prefixes drive TTFT, exposing partially-ready prefixes to consumers earlier reduces head-of-line wait on full-prefill-then-transfer pipelines used by current connectors, addressing a concrete gap (whole-request granularity in save/load and finished-set reporting) rather than restating the existing approach.

---

## Agent proposals

### 1. Register an inter-turn idle-window KV pre-promotion connector
- **Agent:** claude

**Detailed description.**

Add a new built-in connector at vllm/distributed/kv_transfer/kv_connector/factory.py:149-228 (e.g., `KVConnectorFactory.register_connector("IdlePromotionConnector", "vllm.distributed.kv_transfer.kv_connector.v1.idle_promotion.connector", "IdlePromotionConnector")`) implementing `KVConnectorBase_V1`. The connector composes over an existing tiered transport (offload/host-pinned/GPU and optionally a remote prefill peer) and adds a low-priority background promotion engine driven by per-session liveness, not request liveness. On `request_finished`, it (1) tags the request's KV block hashes with a session id (passed in via `KVTransferConfig.kv_connector_extra_config` or sampling-params metadata), (2) records the blocks in a session-resident set with a TTL keyed off observed inter-turn arrival distribution, and (3) enqueues a background task that, while the session is alive but no request from it is queued, migrates those blocks up the tier hierarchy (disk/SSD -> host-pinned -> a small GPU-resident reserved pool) using a token-bucket throttled to spare transfer bandwidth measured by the connector's worker side. When the next turn for the session arrives, `get_num_new_matched_tokens` reports the longer hit length because blocks have already been pre-promoted, and `start_load_kv` resolves to the closest tier without paying first-turn cold-fetch latency. Eviction prefers blocks whose session TTLs have expired so the promotion engine cannot starve interactive traffic. Selection is via `KVTransferConfig.kv_connector="IdlePromotionConnector"`; the V1 contract (connector metadata, finished/invalid block reporting, generated tokens) is preserved, so the correctness oracle in `tests/v1/kv_connector/unit/` and the remote-prefill/offload e2e tests still apply -- only transfer timing changes.

**Novelty rationale.**

All listed deep_research_proposals are reactive: they trigger transport at request admission or per-step boundaries (find-0001/0002 delta-prefix at admit; find-0011/0015 tier composition on demand; find-0018 intra-request InfiniGen prefetch within a step; find-0022 chunk-streaming during prefill). The cache/locality-aware variants (find-0009 multi-replica, find-0012 MemServe locality, find-0013/0014 routing signals) optimize placement at write time or route requests to where blocks already live, but none actively migrate blocks up the tier hierarchy during the human-in-the-loop idle gap between turns. Pre-warming proposals exist only for handshakes (find-0007), not for KV data movement. The novelty is exploiting agentic-workload inter-turn idleness as a scheduling signal for opportunistic promotion using spare bandwidth -- a new admission policy at the connector seam that targets median TTFT on the second-and-subsequent turns where existing connectors still pay cold-tier latency on first access.

---

### 2. Register a single-flight KV connector that deduplicates concurrent block transfers
- **Agent:** codex

**Detailed description.**

Add a new built-in `KVConnectorBase_V1` implementation, for example `SingleFlightNixlConnector`, and register it in `vllm/distributed/kv_transfer/kv_connector/factory.py:149-228` with a `KVConnectorFactory.register_connector(...)` entry. The connector wraps an existing transport such as NIXL or Mooncake but maintains an in-flight table keyed by canonical KV block identity: block hash, layer, model/adaptor id, dtype, and KV layout. In `start_load_kv`, the first request that needs a missing remote block starts one transfer into a shared GPU or pinned-host staging slot; concurrent requests for the same block attach as waiters, then receive local device-to-device copies or scatters into their allocated KV slots after the transfer completes. In `save_kv_layer`, duplicate concurrent saves for the same block hash are similarly collapsed so one writer publishes the object while later writers reuse or pin the published entry. The connector exposes `kv_connector_extra_config` knobs for coalescing window, max waiters per block, and staging-buffer budget. The V1 contract remains unchanged: per-request metadata, finished sets, invalid-block handling, and generated tokens should match the wrapped baseline connector, while bytes-on-the-wire and duplicate transport setup are reduced for shared system/tool prefixes common in multi-turn agentic workloads.

**Novelty rationale.**

This is not the same as the listed delta-prefix, compression, quantization, batching, overlap, tiering, routing, bandwidth-aware, or chunk-streaming proposals. Those reduce transfer size, hide transfer latency, choose better sources, or batch descriptors, but they still generally treat simultaneous requests for the same KV content as separate logical loads or saves. `find-0010` batches KV movement across requests, but batching distinct transfers is different from content-addressed single-flight de-duplication with waiter fanout. `find-0009`, `find-0012`, and `find-0016` improve placement or chunk reuse, but do not collapse an in-flight thundering herd on the same block inside one connector instance. Agent A's idle-promotion proposal moves existing KV up tiers between turns; it does not address concurrent duplicate transfers at admission time.

---
