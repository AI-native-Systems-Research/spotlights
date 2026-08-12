# MoRIIOConnectorWorker._read_blocks

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py) (lines 2541–2640)
- **Symbol:** `MoRIIOConnectorWorker._read_blocks`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_distributed_kv_transfer-0016`

## Description
Posts MoRIIO READ operations for a request by iterating every registered layer, computing offsets, synchronously calling read_remote_data, and retrying SQ-full responses with exponential sleep backoff.

## Current approach
Layer transfers are submitted sequentially. Each iteration recomputes list(self.layer_name_to_local_kv_cache_metadata.keys()).index(layer_name), calls _compute_block_transfer_offsets, then blocks in a while True retry loop with time.sleep backoff up to 50 ms before recording status under moriio_wrapper.lock.

## Estimated impact explanation
MoRIIO READ mode is connector-specific, but when enabled this sequential per-layer posting is on remote-prefill TTFT and can delay layer barriers during decode.

## Evolve rationale
The file already notes a TODO for multi-session batch-read. Precomputing layer-to-session index once at registration, batching read descriptors when the backend supports it, or replacing sleep backoff with queue-depth-aware admission are in-repo scheduling changes preserving recorded transfer status. Oracle: tests/v1/kv_connector/unit/test_moriio_connector.py, test_moriio_routing_fairness.py, and test_moriio_kv_layout.py cover routing, offsets, transfer-id mapping, and completion semantics.

## Deep research proposals

### 1. Batch MoRIIO READ descriptors per request with topology-aware slicing
- **Finding:** `find-vllm_distributed_kv_transfer-0005` — *Transfer Engine*
- **Source URL:** <https://aionw.github.io/design/transfer-engine/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor MoRIIOConnectorWorker._read_blocks (vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:2541-2640) so that a single request no longer submits one synchronous read_remote_data per layer. Concretely: (1) precompute a stable layer_name -> sess_idx map at layer-registration time to eliminate the per-iteration list(...).index(layer_name) O(L^2) scan; (2) build a BatchTransfer-style scatter list by collecting the (local_addr, remote_addr, size) descriptors produced by _compute_block_transfer_offsets for every registered layer of this request, coalescing physically adjacent block ranges within a layer before appending; (3) submit the batch via a new moriio_wrapper batch-read entry point when the MoRIIO backend advertises it (guarded by a capability probe, keeping today's per-layer fallback for older backends), splitting the batch into topology-informed chunks sized to the QP/NIC send-queue depth rather than sleeping on SQ-full; (4) replace the time.sleep exponential backoff with a queue-depth-aware admission gate that waits on the CQ-drain thread's SQ-freed signal, still bounded by transfer_timeout, and record per-layer transfer_status entries in self._recving_transfers under moriio_wrapper.lock exactly as today so completion accounting via get_finished and the (remote_host, port, transfer_id) callback registration are preserved. The TODO already noted in the file (`apply multi-session batch-read when moriio support it`) is the extension point for step 3.

**Proposal rationale.**

The finding describes Mooncake Transfer Engine's BatchTransfer + topology-aware path selection + large-transfer slicing model, which directly targets the two gaps in this candidate: per-layer descriptor/initiation overhead and SQ-full contention handled by blind sleep backoff. Batching descriptors across layers of one request cuts L synchronous posts to O(1) submission, aligning with BatchTransfer's non-contiguous scatter semantics; topology-informed slicing turns SQ-full rejections into pre-sized admission rather than reactive sleeps; and precomputing sess_idx removes an incidental O(L^2) cost. All three are in-repo scheduling changes that preserve per-layer transfer status, callback addressing, and the file's existing multi-session batch-read TODO, and they plausibly reduce remote-prefill TTFT and per-layer decode barrier latency on the multi-turn agentic workload named in the caller context.

---

### 2. Pipeline MoRIIO layer READs with async submission and completion polling
- **Finding:** `find-vllm_distributed_kv_transfer-0006` — *NVMe Driver*
- **Source URL:** <https://spdk.io/doc/nvme.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Rework `MoRIIOConnectorWorker._read_blocks` (vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:2541-2640) to submit RDMA READs asynchronously across layers and poll completions incrementally, mirroring SPDK's NVMe submit/reap pattern. Concretely: (1) precompute a `layer_name -> sess_idx` map once at layer registration so the O(N) `list(...).index(layer_name)` inside the hot loop becomes an O(1) dict lookup; (2) fan out reads for all layers of a request without waiting between iterations - post each `read_remote_data` and stash the returned transfer handle/status in `_recving_transfers[request_id]` immediately; (3) replace the `while True: time.sleep(_backoff)` SQ-full retry with queue-depth-aware admission: track outstanding in-flight reads against `qp_per_transfer`, only post when a slot is free, and drain completions from the existing CQ-poll thread through a lightweight notification (condition variable or semaphore) rather than exponential wall-clock sleeps. Keep the transfer_timeout deadline as a bounded wait guard, but wake on completion rather than on a timer. Preserve the recorded-status semantics under `moriio_wrapper.lock` and the failure path that lets `get_finished` notify the prefill side non-fatally.

**Proposal rationale.**

SPDK's NVMe I/O model - submit-and-return plus nonblocking completion polling on the CQ - directly addresses two hotspots in this candidate. First, the sequential per-layer submission means TTFT on remote-prefill decode waits for every earlier layer's synchronous post to return before the next is issued; keeping descriptors in flight across layers overlaps submission with the CQ-poll thread's completion draining. Second, the `time.sleep(1ms..50ms)` SQ-full backoff is precisely the kind of blind wait SPDK avoids by polling completions to reclaim queue depth - a queue-depth-aware admission gate driven by the existing off-thread CQ drain replaces up to 50 ms of stalled sleep with an immediate wake as soon as SQ capacity frees. The finding also motivates removing the O(N^2) `list(...).index()` recomputation, which is a trivial precomputation once submission is reorganized. This aligns with the file's own TODO for multi-session batch-read and with the caller's TTFT/TPOT objective for the multi-turn agentic workload where remote-prefill layer barriers gate first-token emission.

---

## Agent proposals

### 1. Layer-order-prioritized progressive READ completion for remote-prefill TTFT
- **Agent:** claude

**Detailed description.**

Modify MoRIIOConnectorWorker._read_blocks (vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:2541-2640) and its completion path (get_finished at ~line 1837 and _recving_transfers accounting) so that a request's TTFT is gated by the earliest layers landing, not the last. Concretely: (1) Post the per-layer read_remote_data calls in strict forward-execution order (layer 0 first, layer L-1 last) — today the loop iterates dict-insertion order of layer_name_to_local_kv_cache_metadata, which is usually but not guaranteed to be forward order; assert/sort by the model's forward index once at registration and store an ordered layer list alongside the existing layer_name -> sess_idx map. (2) Reserve a small dedicated pool of SQ slots (e.g. qp_per_transfer // 4, min 1) on a separate high-priority QP set for the first K layers (K configurable, default ~2), so layer 0's READ is not head-of-line-blocked behind bulk reads from other requests when the shared SQ is contended; the current single _sq_deadline / _is_sq_full_status backoff loop becomes two-tier — high-priority slot first, fall back to the general pool for tail layers. (3) Expose progressive-ready state through a new API on the connector: get_finished continues to return only fully-complete request_ids for compatibility, but add get_layer_ready(request_id) -> int returning the highest contiguous layer index whose transfer_status in _recving_transfers[request_id] is Success under moriio_wrapper.lock, and record per-layer completion timestamps in _recving_transfers so the scheduler/attention runner can begin the first forward pass once layers [0..K_min) are ready while the tail is still in-flight. (4) Preserve today's failure semantics exactly — any layer failure still surfaces via get_finished's non-fatal drop path (host/port/transfer_id callback via _recving_transfers_callback_addr is unchanged), and the transfer_timeout deadline still bounds the request. Validation extends the existing test_moriio_connector.py / test_moriio_kv_layout.py suites with a scenario that injects staggered per-layer completions and asserts get_layer_ready advances monotonically before get_finished fires.

**Novelty rationale.**

Both existing deep_research_proposals (find-...-0005 BatchTransfer-style scatter, find-...-0006 SPDK-style submit-and-poll) optimize *how* the L layer reads are submitted and how SQ-full is handled, but both preserve the all-or-nothing 'request finished when every layer's transfer_status is Success' contract that get_finished enforces at line 1962-2022. Neither proposes (a) enforcing forward-execution layer order for submission priority so the earliest layers land first, (b) reserving a high-priority QP slot pool for the first K layers to bypass contention from tail-layer reads of other requests, or (c) exposing a *progressive* per-layer readiness signal to the scheduler so decode's first forward pass can start before the last layer lands. Those three items change *when* decode is unblocked, not just how quickly the submission burst completes, and they compose with either existing proposal's batching/async submission changes rather than duplicating them. The forward-order guarantee also removes an incidental correctness fragility (dict iteration order coincidence) that neither existing proposal names.

---

### 2. Prune MoRIIO READ ranges for chunked-local attention layers
- **Agent:** codex

**Detailed description.**

Modify `MoRIIOConnectorWorker._read_blocks` in `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:2541-2640` to use the existing `block_window_per_layer` metadata populated for Llama 4 local-attention layers. Iterate layers with their stable layer index; before calling `_compute_block_transfer_offsets`, if `block_window_per_layer[layer_idx]` is not `None`, slice the paired `(local_block_ids, remote_block_ids)` lists down to the most recent `block_window` blocks for that layer, leaving global-attention layers and non-Llama-4 models unchanged. This turns local-attention layers from transferring the full multi-turn prefix into transferring only the KV blocks that the layer can actually attend to. Add a focused unit test in the existing MoRIIO connector/layout tests that sets mixed global/local `block_window_per_layer` values, invokes `_read_blocks`, and asserts the mocked `read_remote_data` byte ranges for local layers cover only the trailing window while global layers still cover all requested blocks.

**Novelty rationale.**

The deep-research proposals optimize submission mechanics: batch descriptors, async posting, precomputed session indexes, coalescing adjacent ranges, and queue-depth-aware SQ admission. Agent A optimizes readiness ordering and prioritizes early layers. None of them reduce the amount of KV data requested per layer based on attention semantics. This proposal is a data-volume reduction specific to the existing MoRIIO `_read_blocks` path and its already-populated `block_window_per_layer` field, so it composes with batching, async submission, and progressive readiness without duplicating them.

---
