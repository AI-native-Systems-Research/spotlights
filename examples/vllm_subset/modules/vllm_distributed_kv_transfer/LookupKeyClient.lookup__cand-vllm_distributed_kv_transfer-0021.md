# LookupKeyClient.lookup

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py) (lines 1989–2028)
- **Symbol:** `LookupKeyClient.lookup`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_distributed_kv_transfer-0021`

## Description
MooncakeStore async prefix lookup client serializes lookup RPCs through one worker thread and one blocking ZMQ REQ socket.

## Current approach
LookupKeyClient creates ThreadPoolExecutor(max_workers=1). lookup() stores one Future per request; the worker calls _lookup(), which sends multipart frames and blocks on socket.recv(). With non_block=True, scheduler calls return None until the queued future completes.

## Estimated impact explanation
This is connector-specific, but for MooncakeStore deployments it gates get_num_new_matched_tokens; reducing lookup queueing lowers median TTFT under many concurrent agentic turns.

## Evolve rationale
Concurrent multi-turn requests can queue TTFT-critical prefix lookups behind one blocking client lane even if the server/store can handle more work. A bounded socket pool, DEALER-style multiplexing, or scheduler-step batching of lookup frames would preserve per-request hit_length results while exposing parallelism. Oracle: tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py, test_mooncake_store_connector.py, and test_mooncake_store_worker.py cover async lookup behavior, request metadata, and returned hit lengths.

## Deep research proposals

### 1. Replace REQ+single-thread executor with a DEALER/ROUTER multiplexed lookup client
- **Finding:** `find-vllm_distributed_kv_transfer-0004` — *Advanced Request-Reply Patterns*
- **Source URL:** <https://zguide.zeromq.org/docs/chapter3/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Rework LookupKeyClient in vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (lines 1971-2028) so lookups no longer serialize through one blocking REQ socket and one ThreadPoolExecutor worker. Concretely: (1) change self.socket from zmq.REQ to zmq.DEALER and update the matching LookupKeyServer socket to zmq.ROUTER; (2) prepend an identity frame (the req_id encoded as bytes) to every LOOKUP_MSG multipart frame sent from lookup(), so the ROUTER can echo it back on the reply; (3) replace the single-worker executor with a small background reader thread that loops on self.socket.recv_multipart(), parses (req_id, hit_len), and completes the corresponding Future stored in self.futures via future.set_result(int.from_bytes(...)); (4) have lookup() create the Future eagerly, send the tagged frames directly (protected by a zmq send lock), register the Future in self.futures[req_id], and honor non_block by returning None while future.done() is False; (5) keep _reset()/reset() correct by either routing it through the same dispatcher with its own tag or using a separate short-lived REQ socket, since reset is a rare synchronization point. discard() continues to pop the Future and cancel it; a late reply for a discarded req_id is dropped. This preserves the public lookup(req_id, num_tokens, block_hashes, non_block) contract, hit_length semantics, and the tests in tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py, test_mooncake_store_connector.py, and test_mooncake_store_worker.py, while allowing many prefix-lookup RPCs to be in flight over a single socket at once.

**Proposal rationale.**

The candidate's bottleneck is exactly the REQ/REP lockstep the ZMQ guide's DEALER/ROUTER section is written to remove: with REQ+max_workers=1, a second lookup cannot leave the client until the first recv() returns, so TTFT-critical prefix lookups for concurrent agentic turns queue behind one lane even when the LookupKeyServer/Mooncake store could service them in parallel. DEALER sockets are asynchronous by design (per the finding's quoted evidence) and support multiple outstanding sends before any reply arrives, so tagging each request with its req_id and demultiplexing replies in a reader thread directly unlocks the parallelism the candidate's evolve_rationale asks for, without changing the per-request hit_length return value that callers of get_num_new_matched_tokens depend on.

---

### 2. Replace REQ+single-thread serialization with DEALER-style async submit and completion polling in LookupKeyClient
- **Finding:** `find-vllm_distributed_kv_transfer-0006` — *NVMe Driver*
- **Source URL:** <https://spdk.io/doc/nvme.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py at LookupKeyClient (lines ~1979-2028), replace the ThreadPoolExecutor(max_workers=1) + blocking zmq.REQ socket with a submit/poll design modeled on SPDK's NVMe queue-pair pattern:

1. Submission path: keep lookup() synchronous from the scheduler's perspective, but on the first call for a given req_id, tag the multipart frames with a request-id frame and send them non-blocking on a zmq.DEALER socket (which permits multiple outstanding requests without REQ's strict send/recv lockstep). Record the req_id in an in-flight table with its submission metadata; return None when non_block=True.

2. Completion path: on each subsequent lookup() call (or on a lightweight poll() invoked at the scheduler-step boundary), drain any available replies from the DEALER socket using zmq.Poller with a zero timeout — the analogue of SPDK's nonblocking queue-pair completion poll. Match each reply's req_id tag to the in-flight table and store the hit_length. If the current req_id's result is ready, return it; otherwise return None so the scheduler retries next step.

3. Server side (LookupKeyServer) is updated symmetrically to echo the req_id tag in its response so replies can be demultiplexed. The reset admin command retains synchronous request/reply semantics by draining outstanding lookups first (or by carrying its own tag).

This preserves the existing per-request hit_length return contract used by the scheduler and covered by tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py, test_mooncake_store_connector.py, and test_mooncake_store_worker.py, while allowing many prefix lookups to be in flight concurrently against the LookupKeyServer.

**Proposal rationale.**

The candidate identifies serialization at two layers: one worker thread and one blocking REQ socket, both of which force TTFT-critical prefix lookups to queue head-of-line under concurrent multi-turn agentic workloads. SPDK's NVMe driver documents the general submit/poll pattern the candidate's evolve_rationale explicitly gestures at: 'The function returns immediately, prior to the completion of the command' and callers 'must poll for I/O completion.' That decoupling of submission from completion is directly transferable to the ZMQ RPC lane here — a DEALER socket permits multiple outstanding requests, and polling for replies at scheduler-step boundaries mirrors the completion-reaping loop. It addresses the exact constraint the candidate cites (single-lane blocking recv) with a concrete mechanism (tagged multi-in-flight requests + nonblocking completion poll) that is already idiomatic in ZMQ and preserves per-request hit_length semantics needed by get_num_new_matched_tokens.

---

## Agent proposals

### 1. Coalesce prefix lookups at each scheduler step into one batched RPC to LookupKeyServer
- **Agent:** claude

**Detailed description.**

Change LookupKeyClient.lookup (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py, lines 2007-2028) from one-RPC-per-request to a scheduler-step-batched round trip, orthogonal to any REQ→DEALER rewrite. Concretely: (1) In lookup(req_id, num_tokens, block_hashes, non_block), instead of eagerly submitting to the executor on the first call for a req_id, append (req_id, num_tokens, block_hashes) to a pending queue self._pending and, when non_block=True and the result is not yet cached in self._results, return None. (2) Add a new public method flush()/poll_step() that the scheduler invokes once per step (via a small hook in MooncakeStoreConnector.get_num_new_matched_tokens's caller, or from an existing per-step boundary such as scheduler.schedule()). flush() drains self._pending into a single LOOKUP_BATCH_MSG multipart frame — a new protocol tag alongside LOOKUP_MSG — whose payload is a length-prefixed array of (req_id_bytes, num_tokens, hash_len, block_hashes_blob) entries, sends it once, and receives one reply frame containing an array of (req_id, hit_length) pairs; populate self._results[req_id] = hit_length. (3) LookupKeyServer.process_request grows a matching LOOKUP_BATCH_MSG branch that loops over the entries, calls store_worker.lookup(num_tokens, block_hashes) for each, and returns the concatenated results in one send; the existing LOOKUP_MSG branch stays for backward compat / RESET_MSG. (4) Retain the executor-based single-shot path as a fallback for the blocking (non_block=False) case, or promote everything through the batched channel with an immediate flush() when non_block=False. discard() removes a req_id from both _pending (before flush) and _results (after). This preserves the per-request hit_length return value observed by tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py, test_mooncake_store_connector.py, and test_mooncake_store_worker.py — each individual lookup(req_id, ...) still returns the same integer — while collapsing N per-step lookups into one round trip and one server-side thread traversal.

**Novelty rationale.**

Both existing deep_research_proposals (find-0004 and find-0006) attack the client's serial socket lane by converting REQ→DEALER and multiplexing per-request replies via a reader thread or nonblocking completion poll — they still perform one RPC per lookup, only concurrently. This proposal is orthogonal on a different axis: it reduces the number of RPCs from N to 1 per scheduler step by coalescing all pending lookups into a single LOOKUP_BATCH_MSG frame, exploiting the fact that non_block=True calls already arrive in a burst at each schedule() invocation. Batching cuts wire round trips, ZMQ frame overhead, and server-side thread-wakeup churn regardless of whether the socket is REQ or DEALER, and it composes cleanly with either existing proposal (DEALER + batched frames would stack). Neither listed proposal introduces a new LOOKUP_BATCH_MSG protocol tag, a per-step flush hook, or a scheduler-driven drain point — those are the load-bearing pieces here.

---
