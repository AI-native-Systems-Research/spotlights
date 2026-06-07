# P2pNcclEngine.send / P2pNcclEngine.recv

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`](vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py) (lines 589–625)
- **Symbol:** `P2pNcclEngine.send / P2pNcclEngine.recv`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0006`

## Description
P2P NCCL send and receive wrappers that issue `ncclSend` or `ncclRecv` on a CUDA stream and immediately synchronize that stream before returning.

## Current approach
Each wrapper enters `torch.cuda.stream(stream)`, launches a single NCCL operation, then calls `stream.synchronize()`. Callers such as `send_sync`, `recv_tensor`, and the PUT/GET request handler therefore host-block on every tensor transfer rather than recording events and waiting only when data is actually consumed.

## Estimated impact explanation
Layer-wise KV transfer can incur many host-side synchronizations per request. Eliminating per-layer stalls can reduce remote-prefill TTFT and lower producer-side save latency, which also helps median TPOT when saves overlap with decode.

## Evolve rationale
The in-repo synchronization policy is the optimization unit. Replacing per-transfer `stream.synchronize()` with CUDA events, grouped waits, or a pipelined async path preserves the wait-before-read invariant while removing unnecessary host stalls. Correctness oracle: transmitted tensors must be byte-equal at the receiver, request stores must only become visible after transfer completion, and P2P connector tests can validate PUT/GET and cleanup behavior with deterministic fake peers.

## Deep research proposals

### 1. Batch per-layer NCCL transfers via ncclGroupStart/End with a single end-of-group sync
- **Finding:** `find-0004` — *[PD] optimize kv cache transfer directly using batch transfer*
- **Source URL:** <https://github.com/sgl-project/sglang/pull/9149>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the P2P NCCL engine at vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:589-625 with batched send/recv entry points (e.g. send_batch / recv_batch) that accept a list of tensors (typically one per layer) and submit them inside a single ncclGroupStart/ncclGroupEnd region on the engine's send_stream / recv_stream. Replace the current pattern of one ncclSend/ncclRecv followed by stream.synchronize() per tensor with: (a) iterate the list inside the group region issuing ncclSend/ncclRecv for each layer's data_ptr without any intermediate sync, (b) close the group, and (c) issue exactly one stream.synchronize() (or record a single CUDA event) after the group completes. Update the call sites that today invoke send/recv per layer (the PUT/GET request handler around lines 366/404/456/533 and send_sync / recv_tensor at lines 500 and 308) to gather all layer tensors for a request and call the new batched API once instead of looping over the per-tensor wrappers. Preserve the existing single-tensor send/recv as thin wrappers that call the batched API with a one-element list so callers that genuinely transfer a single tensor are unaffected. The wait-before-read invariant is maintained because consumers still observe a synchronize/event-wait before touching the receive buffers; only the number of host stalls drops from O(num_layers) to O(1) per request.

**Proposal rationale.**

The candidate's bottleneck is host-side stalls from stream.synchronize() executed once per tensor transfer; with KV caches transmitted layer-by-layer this is O(num_layers) host blocks per request. The finding's core idea — pack all layers' transfer parameters into one batch transfer call rather than issuing many separate executor operations — maps directly onto NCCL's group-call API and collapses both the launch overhead and the per-layer synchronization that the candidate's evolve_rationale calls out. This addresses the gap (per-transfer host blocking) without weakening the correctness oracle (transmitted tensors must be byte-equal, request stores visible only post-completion), since group-end + single sync still completes all layers before any consumer reads, and is plausibly impactful for remote-prefill TTFT and producer-side save latency in multi-turn agentic workloads where many layers transfer per request.

---

### 2. Replace per-transfer stream.synchronize() with event-based wait-before-read to overlap KV transmission with concurrent compute
- **Finding:** `find-0005` — *Disaggregated Serving — TensorRT LLM*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0rc4/features/disagg-serving.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py at lines 589-625, rework P2pNcclEngine.send and P2pNcclEngine.recv so they no longer call stream.synchronize() inline after issuing ncclSend/ncclRecv. Instead, after launching the NCCL op on the provided (or current) CUDA stream, record a torch.cuda.Event on that stream and return it (or stash it in the per-request transfer record kept by the PUT/GET handler and the send/recv stores). Update callers — including send_sync, recv_tensor, and the PUT/GET request handler around the send_request_id_to_tensor_ids / recv_request_id_to_tensor_ids stores — so that the wait moves to the moment the data is actually consumed: producers wait on the send-completion event before freeing the source buffer / pool slot or marking the request finished_sending; consumers wait on the recv-completion event before exposing the tensor to the model (or making the recv_store entry visible) and before reporting finished_recving. This preserves the wait-before-read invariant (transmitted tensors must be byte-equal at the receiver and request stores only become visible after completion) but lets unrelated forward-pass work and other in-flight requests' transfers progress on the same device while the NCCL kernel completes. Optionally batch consecutive layer transfers under a single ncclGroupStart/ncclGroupEnd on the dedicated NCCL stream and record one event per request/layer-group, so a single event-wait replaces N per-layer host stalls on the hot path.

**Proposal rationale.**

The candidate's gap is exactly the one TensorRT-LLM's disaggregated-serving design targets: each ncclSend/ncclRecv host-blocks via stream.synchronize(), so a single in-flight transfer prevents the worker from making progress on independent requests' compute. The finding's transferable idea — overlap KV transmission for one request with computation for other independent requests — maps directly to replacing the inline synchronize with a deferred event wait that is only joined at the consumption point. This is concrete and local to the candidate symbol, addresses the layer-wise N-synchronizations-per-request cost called out in the candidate's impact rationale, and fits the multi-turn agentic workload where remote-prefill TTFT and producer-side save latency overlap with decode are the dominant levers for median TTFT and TPOT.

---

### 3. Make P2P NCCL send/recv non-blocking with event-based readiness
- **Finding:** `find-0008` — *Disaggregated Serving | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor `P2pNcclEngine.send` and `P2pNcclEngine.recv` (vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:589-625) to remove the unconditional `stream.synchronize()` after each `ncclSend`/`ncclRecv`. Instead, after issuing the NCCL op on the dedicated transfer stream, record a `torch.cuda.Event` that captures completion of that op and return (or store) the event alongside the tensor handle. Callers in this file — `send_sync`, `recv_tensor`, and the PUT/GET request handler that drives layer-wise KV transfer — should be updated to (a) make the producer-side compute stream wait on the recorded event only when the underlying buffer is about to be reused/freed, and (b) only block the consumer host thread when the receiver is actually about to read the KV slab (e.g., immediately before decode admission for that layer/request). For PUT_ASYNC, allow the request store/bookkeeping entry to be marked "ready" via an event-completion callback (or a lightweight polling pass) rather than gating store visibility on a host-side synchronize. Preserve the wait-before-read invariant by ensuring every read path (decode kernel launch, response send, buffer reuse) issues `event.wait(stream)` or `event.synchronize()` exactly once before touching the data. Add a fallback synchronous path behind a flag so the change can be staged.

**Proposal rationale.**

The NVIDIA Dynamo disaggregated-serving design explicitly states that KV transfer should be non-blocking so GPU forward passes can keep serving other requests during the transfer. The candidate's current behavior is the exact opposite: every per-tensor (and in layer-wise mode, per-layer) send/recv host-blocks the caller via `stream.synchronize()`, which serializes KV movement against decode and other forward passes. Replacing per-call synchronizes with recorded events plus targeted waits is the canonical mechanism to realize Dynamo's non-blocking-transfer pattern on this code path, directly addressing the candidate's identified gap (host stalls in `send`/`recv`) while keeping the correctness oracle intact (events ensure data is ready before any reader touches it). For the stated multi-turn agentic workload optimizing media TTFT and median TPOT, eliminating these per-layer host stalls is the most direct way the finding's idea improves this specific candidate.

---

### 4. Replace per-transfer stream.synchronize() with batched, pipelined NCCL transfers using CUDA events
- **Finding:** `find-0010` — *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*
- **Source URL:** <https://arxiv.org/abs/2510.09665>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:589-625, refactor P2pNcclEngine.send and P2pNcclEngine.recv so that they no longer call stream.synchronize() inline after each ncclSend/ncclRecv. Instead: (1) record a CUDA event on the NCCL stream after the collective is enqueued and return that event (or a lightweight handle) to the caller; (2) provide a batched submission path that groups consecutive per-layer transfers for one request inside a single ncclGroupStart/ncclGroupEnd region and records one terminal event for the whole group; (3) update callers (send_sync, recv_tensor, and the PUT/GET request handler around lines 589-625) so host-side waits happen only at the wait-before-read consumption point — e.g., recv consumers stream-wait the recorded event before reading, and the request store/cleanup only becomes visible after the event has fired. This preserves the existing correctness oracle (byte-equal tensors at the receiver; request stores visible only after transfer completion) while letting NCCL transfer of layer k overlap with the model's compute on layer k+1 (or with another request's transfer), mirroring the LMCache pattern of batched KV data movement plus compute/I/O pipelining for layer-wise KV transfers.

**Proposal rationale.**

The candidate's evolve_rationale explicitly identifies the synchronization policy as the optimization unit and lists CUDA events, grouped waits, and pipelined async paths as desirable replacements for the current per-transfer stream.synchronize(). LMCache's reported gains attribute their performance specifically to 'batched data movement operations, compute and I/O pipelining' applied to KV cache movement — the same workload the P2P NCCL connector handles. This is a direct, transferable mechanism: batching layer-wise NCCL ops in a group region and replacing host syncs with event-based waits addresses the host-stall bottleneck that drives TTFT in remote-prefill and producer-side save latency in multi-turn agentic workloads, without changing the wait-before-read invariant the tests already validate.

---

## Agent proposals

### 1. Pre-register KV cache slabs with NCCL via ncclCommRegister for zero-copy P2P transfers
- **Agent:** claude

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py around lines 589-625 (and the engine init/teardown paths that allocate/free the per-rank send/recv buffers and the request stores), add a one-time NCCL user-buffer registration step for the long-lived KV slabs that flow through P2pNcclEngine.send/recv. Concretely: (1) at engine startup, after the dedicated send_stream / recv_stream and the NCCL communicator(s) are created, iterate the persistent KV cache tensors (or the pool of staging tensors used by send_sync / recv_tensor / the PUT/GET handler) and call ncclCommRegister(comm, ptr, size) for each, caching the returned registration handle on the engine keyed by data_ptr; (2) on every subsequent P2pNcclEngine.send / P2pNcclEngine.recv call, look up the handle for the tensor's data_ptr and pass the registered buffer to NCCL (or use ncclMemAlloc-allocated buffers for the staging pool so they are registered by construction); (3) on engine shutdown / pool eviction, call ncclCommDeregister to release handles and avoid leaking pinned IB MRs. Add a feature flag (e.g. VLLM_P2P_NCCL_REGISTER_BUFFERS) and a graceful fallback when the linked NCCL is older than 2.19 or when registration fails (e.g. non-IB transport). Do not change the call sequencing or the synchronize/event policy in this proposal — it is orthogonal to and composable with the batching/event-based proposals already on the candidate.

**Novelty rationale.**

All four existing deep_research_proposals (find-0004, find-0005, find-0008, find-0010) target the *synchronization policy*: batching ncclSend/Recv inside ncclGroupStart/End and/or replacing per-transfer stream.synchronize() with recorded CUDA events plus deferred wait-before-read. None of them touch how the buffers themselves are presented to NCCL. ncclCommRegister / ncclMemAlloc (NCCL >= 2.19) eliminate per-call memory-region setup, enable NVLink SHARP and IB GPUDirect zero-copy fast paths, and reduce per-transfer launch cost — a different bottleneck than host-side stalls. The change is concrete to this exact symbol/file, preserves the byte-equality and store-visibility correctness oracle, and is composable with (not redundant to) the event/group-based proposals already listed.

---

### 2. Receive contiguous KV block ranges directly into the destination KV cache
- **Agent:** codex

**Detailed description.**

Add a fast path around P2pNcclEngine.send/recv in vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:589-625 and its p2p_nccl_connector.py call sites that avoids the current producer-side gather tensor and consumer-side temporary receive tensor when request.block_ids form a contiguous run. In save_kv_layer, detect contiguous block_ids and pass a contiguous view to send: for MLA/FlashInfer layout use kv_layer.narrow(0, start, count); for FlashAttention layout send the K and V planes as two contiguous views, e.g. kv_layer[0].narrow(0, start, count) and kv_layer[1].narrow(0, start, count), with deterministic tensor IDs or a small plane-count metadata field. In start_load_kv, mirror that by receiving directly into the matching destination KV cache views instead of allocating torch.empty in recv_tensor and then copying via inject_kv_into_layer. Keep the existing advanced-indexing extract/temporary-recv/scatter path for non-contiguous block_ids, and add an explicit is_contiguous guard in send/recv so raw data_ptr NCCL transfers never operate on a strided view accidentally. This preserves byte equality while removing one device-to-device pack on the producer, one allocation on the consumer, and one device-to-device scatter before decode admission for the common sequential-block case.

**Novelty rationale.**

The listed deep_research_proposals all change transfer scheduling: NCCL grouping, deferred CUDA-event waits, or pipelining instead of per-call stream.synchronize. They do not change what memory NCCL reads from or writes into, and they still assume per-layer tensors are first materialized and later injected into the paged KV cache. Agent A's proposal changes NCCL buffer registration for the same buffers, but likewise does not remove the producer gather tensor or receiver staging tensor. This proposal targets a different cost center: avoid extra GPU memory traffic and allocations by using contiguous KV cache views as the actual ncclSend/ncclRecv buffers, with fallback for non-contiguous layouts.

---
