# ObjectStoreSecondaryTierManager._submit_transfer/_poll_active_transfers

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/tiering/obj/manager.py`](vllm/v1/kv_offload/tiering/obj/manager.py) (lines 217–345)
- **Symbol:** `ObjectStoreSecondaryTierManager._submit_transfer/_poll_active_transfers`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0017`

## Description
Object-store secondary tier builds NIXL OBJ descriptors, submits asynchronous transfers, polls in-flight handles, and releases transfer resources.

## Current approach
Each job converts block IDs, creates one OBJ descriptor per key with monotonically increasing dev IDs, registers object memory, prepares a transfer dlist, starts a NIXL transfer, and stores the handle in _transfers. Completion polling scans all active transfers, classifies states, releases handles/dlists/memory, and buffers one JobResult per job.

## Estimated impact explanation
Object-store hits avoid recomputation but can add substantial secondary promotion delay before TTFT. Better batching or polling can reduce that delay for multi-turn agentic reuse, although network/object-store latency limits the median improvement.

## Evolve rationale
Descriptor granularity, dev-id allocation, transfer submission batching, all-transfer polling, and cleanup ordering are owned scheduling choices around the NIXL object backend. Correctness oracle: tests/v1/kv_offload/tiering/test_obj_tier.py, exactly one JobResult per submitted job, failed-transfer lookup verdict invalidation, and byte-identical load/store round trips.

## Deep research proposals

### 1. Offload NIXL OBJ transfer submission and polling to a dedicated background I/O thread
- **Finding:** `find-vllm_v1_kv_offload-0001` — *Serving Agentic Workloads at Scale with vLLM x Mooncake*
- **Source URL:** <https://vllm.ai/blog/2026-05-06-mooncake-store>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Move the synchronous NIXL calls in ObjectStoreSecondaryTierManager._submit_transfer and ._poll_active_transfers (vllm/v1/kv_offload/tiering/obj/manager.py:217-345) off the scheduler thread onto a dedicated background I/O worker, mirroring Mooncake's design where 'all RDMA operations run on a dedicated background I/O thread.' Concretely: (1) introduce a single long-lived worker thread owned by the manager that consumes a submission queue populated by submit_store/submit_load; (2) have _submit_transfer's register_memory/prep_xfer_dlist/make_prepped_xfer/transfer sequence execute on that thread so descriptor construction, dev-id allocation, and NIXL submissions do not stall the scheduler; (3) run the _poll_active_transfers loop on the same thread (or a paired poller thread) so check_xfer_state / release_xfer_handle / release_dlist_handle / deregister_memory happen in the background, and push completions into a thread-safe _pending_results queue that get_finished_jobs drains from the scheduler thread without blocking. Preserve the existing invariants: exactly one JobResult per submitted job, xfer-handle release before publishing results (so primary-tier memory is not reused while a transfer is live), and unique monotonically increasing dev-ids per OBJ descriptor (which now requires the _next_obj_dev_id counter to be accessed only from the I/O thread or under a lock). Batching of multiple job submissions per NIXL prep_xfer_dlist/make_prepped_xfer call is a natural follow-on the background queue enables, but is not required for the core change.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'transfer submission batching' and 'all-transfer polling' as owned scheduling choices, and its impact narrative highlights promotion delay before TTFT for multi-turn agentic reuse — exactly the workload the Mooncake blog optimizes for. Today _submit_transfer performs five synchronous NIXL agent calls per job on the caller's thread, and _poll_active_transfers scans every in-flight transfer synchronously via get_finished_jobs; both run on the scheduler thread and add latency to every scheduling tick that touches the OBJ tier. The finding's transferable idea — a dedicated background I/O thread for all RDMA/object-store operations, decoupled from the scheduler — maps directly onto these two methods and addresses the median-TTFT gap for agentic workloads by removing object-store submission/polling latency from the scheduler critical path, without changing the correctness oracle (one JobResult per job, failed-transfer verdict invalidation, byte-identical round trips).

---

### 2. Split OBJ tier submissions into progressive sub-transfers to cut HOL blocking on long promotions
- **Finding:** `find-vllm_v1_kv_offload-0003` — *[RFC]: Progressive KV Cache CPU Onloading*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/33526>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/v1/kv_offload/tiering/obj/manager.py` (`ObjectStoreSecondaryTierManager._submit_transfer` and `_poll_active_transfers`, lines ~217-345), replace the current one-NIXL-transfer-per-job model with a progressive-batch model along the lines of the RFC. Instead of issuing a single `make_prepped_xfer`/`transfer` over the entire `block_ids_list`/`nixl_files`, chunk the (block_id, obj_key) pairs into an ordered list of sub-batches with a small first batch and geometrically (or arithmetically) growing later batches (e.g. 1, 2, 4, ... up to a cap). Register memory and prep the OBJ dlist once for the full descriptor set (preserving the current unique-dev-id invariant and single `deregister_memory`/`release_dlist_handle` at teardown), then submit one prepped xfer per sub-batch using the corresponding slice of block indices and OBJ dev-id indices. Change `_transfers[job_id]` to hold an ordered list of xfer handles (plus the shared `files_desc`/`obj_handle`); `_poll_active_transfers` polls each active sub-transfer, releases its xfer handle as soon as it reaches `NIXL_DONE`/error, and only appends a `JobResult(success=all_sub_ok)` and tears down the shared dlist/memory once the last sub-transfer resolves. On any sub-transfer failure, mark the job failed but still drain the remaining outstanding sub-transfers before cleanup so no data transfer outlives the released memory. Keep exactly one `JobResult` per submitted job and preserve the failed-promotion → verdict-invalidation path. Expose the batching schedule as a config knob defaulting to a conservative first-batch size (e.g. 1-2 blocks) so the head-of-line effect is bounded even under object-store tail latency.

**Proposal rationale.**

The candidate today issues a single NIXL OBJ transfer covering all blocks for a job, so `get_finished_jobs` only surfaces completion when the entire multi-block promotion lands. Under the stated multi-turn agentic workload, a long promotion job sharing a prefix with a shorter request can delay the shorter request's readiness on the same polling cycle, directly hurting median TTFT — the exact HOL-blocking scenario the RFC targets. The finding contributes a concrete, transferable scheduling idea (small-first, growing-later sub-batches within one logical transfer request) that maps cleanly onto NIXL's prepped-xfer API: the shared `register_memory`/`prep_xfer_dlist` per job stays intact, only `make_prepped_xfer`/`transfer` are split, and the correctness oracle (one JobResult per job, byte-identical round trips, failed-transfer verdict invalidation) is preserved. It addresses a real gap — coarse per-job transfer granularity — rather than restating the current approach, and its impact ceiling matches the candidate's medium impact estimate given object-store network latency.

---

### 3. Batch NIXL OBJ transfers into chunked object I/O with slack-aware polling
- **Finding:** `find-vllm_v1_kv_offload-0006` — *Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving*
- **Source URL:** <https://arxiv.org/abs/2605.03375>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/kv_offload/tiering/obj/manager.py (ObjectStoreSecondaryTierManager._submit_transfer at lines 217-290 and _poll_active_transfers at lines 291-345), replace the current per-key OBJ descriptor + one-transfer-per-job pattern with a coarser-grained, chunked object abstraction that groups contiguous block IDs of a job (and, where safe, cross-job co-scheduled keys targeting the same object-store shard) into a small number of larger NIXL OBJ descriptors before calling prep/transfer. Concretely: (1) In _submit_transfer, collapse the block_id->key list into run-length chunks so each NIXL descriptor spans multiple contiguous blocks with a single (base dev_id, length) instead of allocating monotonically increasing dev_ids per key; register object memory once per chunk and prepare a single dlist per job whose entries correspond to chunks, not individual blocks. (2) Introduce a slack-aware submission path: when the scheduler indicates the request is not yet on the critical TTFT/TPOT path (e.g. prompt tokens still being tokenized or decode still has runway), defer submission briefly to coalesce with sibling jobs to the same tier; when it is on the critical path, submit immediately with the largest chunk currently ready. (3) In _poll_active_transfers, instead of scanning every active transfer on every tick, prioritize polling handles whose owning jobs are closest to being consumed by the primary tier (slack-aware polling order), and release dlist/memory registrations in a single grouped cleanup per completed job so cleanup cost scales with jobs rather than keys. Preserve the existing correctness contract enforced by tests/v1/kv_offload/tiering/test_obj_tier.py: exactly one JobResult per submitted job, failed-transfer lookup verdict invalidation on any chunk failure, and byte-identical load/store round trips (chunk boundaries must be reconstructed on load).

**Proposal rationale.**

The candidate's current approach is exactly the fragmented pattern Tutti argues against: one descriptor per block/key with per-key dev_id allocation, per-key memory registration, and undifferentiated polling across all in-flight transfers. This produces many small random object-store operations and treats every in-flight handle as equally urgent, which directly inflates median TTFT for secondary-tier promotions in multi-turn agentic reuse (the caller's stated objective). The finding contributes two concrete, transferable ideas that map onto this specific code region: a coarser KV-cache object abstraction (chunked descriptors + bulk transfers) and slack-aware scheduling of both submission and completion handling. Both can be implemented within _submit_transfer/_poll_active_transfers without changing the JobResult contract, so the correctness oracle in test_obj_tier.py still applies. The gap this addresses is scheduling granularity around the NIXL object backend, which the candidate explicitly owns; the impact is bounded by network/object-store latency (matching the medium impact estimate) but should reduce the per-job fixed overhead and the tail of polling latency that currently contributes to secondary-tier promotion delay before TTFT.

---

### 4. Coalesce concurrent object-store jobs into a single batched NIXL submission
- **Finding:** `find-vllm_v1_kv_offload-0007` — *GPUDirect Storage Overview Guide*
- **Source URL:** <https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/kv_offload/tiering/obj/manager.py:217-345, change ObjectStoreSecondaryTierManager._submit_transfer to accumulate jobs submitted within a short scheduling window (or all jobs submitted between get_finished_jobs polls) and issue a single fused NIXL OBJ transfer per window instead of one per job. Concretely: (1) buffer incoming submit_store/submit_load calls into a pending queue keyed by op ('WRITE'/'READ'); (2) on the next flush point (end of on_schedule_end, or when the queue reaches a size threshold), build one combined nixl_files descriptor list spanning all queued jobs' obj_keys with monotonically increasing dev IDs, one combined block_ids list, do a single register_memory + prep_xfer_dlist + make_prepped_xfer + transfer, and record the mapping from (start_idx, end_idx) ranges back to job_ids in TransferEntry. (3) In _poll_active_transfers, when a fused handle transitions to DONE/failed, emit one JobResult per constituent job_id derived from that mapping (preserving the current 'exactly one JobResult per submitted job' invariant and per-job failure semantics — if the fused transfer fails, mark every constituent job failed). Keep the current per-job path as a fallback for singleton flushes. This amortizes the fixed register_memory/prep_xfer_dlist/transfer overhead across all jobs in the batch while preserving byte-identical load/store round trips.

**Proposal rationale.**

The candidate currently pays a full register_memory + prep_xfer_dlist + make_prepped_xfer + transfer setup for every submitted job, even when multiple jobs are submitted back-to-back within one scheduler step. The GPUDirect Storage guide explicitly identifies this pattern — 'batching reduces the overhead by amortizing that fixed overhead across the transactions in the batch' (Sections 1.2.2.4-1.2.2.5) — as the primary lever for reducing storage-backed I/O latency. Under the caller's multi-turn agentic workload, secondary-tier promotions frequently arrive in bursts (prefix reuse across concurrent turns), so batching per-window rather than per-job directly attacks the median TTFT contribution of object-store hits. The principle is backend-agnostic: NIXL OBJ descriptors have the same monotonic-devId requirement whether packed into one dlist or many, so a fused submission is a well-defined transformation of the existing scheduling choices this candidate owns.

---

### 5. Reuse prepared NIXL OBJ dlist handles and batch descriptors across jobs
- **Finding:** `find-vllm_v1_kv_offload-0008` — *Enhancing Distributed Inference Performance with the NVIDIA Inference Transfer Library*
- **Source URL:** <https://developer.nvidia.com/blog/?p=113426>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/kv_offload/tiering/obj/manager.py (lines 217-345), replace the per-job register_memory + prep_xfer_dlist + release cycle with reusable prepared descriptor lists sized to a target batch of blocks. Concretely: (1) pre-register a persistent pool of OBJ descriptor slots at manager init (parameterized to, e.g., a multiple of the primary tier's prepped block count), each slot owning a stable devId; (2) in _submit_transfer, instead of building a fresh nixl_files list and calling register_memory/prep_xfer_dlist per job, acquire a contiguous run of free slots from the pool, patch the (addr, len, devId, obj_key) tuples for the run in-place, and issue make_prepped_xfer against the already-prepared dlist handle using the run's indices; (3) at _poll_active_transfers cleanup, only release_xfer_handle and return the slots to the pool (do not release_dlist_handle or deregister_memory); (4) coalesce ready jobs of the same op (WRITE/READ) enqueued within a single get_finished_jobs / submit_* cycle into a single make_prepped_xfer call that spans multiple jobs' block runs, tracking per-job index ranges in TransferEntry so completion still emits exactly one JobResult per submitted job. Grow the pool on demand when the working set exceeds capacity, and keep _next_obj_dev_id logic only for pool growth (not per job).

**Proposal rationale.**

The finding describes NIXL as a descriptor-list abstraction with explicit memory registration and prepared transfer handles, whose stated design intent is to let higher-level schedulers batch descriptors and reuse prepared handles without changing semantics. The candidate today pays register_memory + prep_xfer_dlist + release_dlist_handle + deregister_memory on the critical path of every promotion job, which for the multi-turn agentic workload directly inflates secondary-tier promotion delay before TTFT. Amortizing those NIXL setup calls across many jobs via a persistent pool and coalescing eligible same-op jobs into one xfer is the transferable idea the blog surfaces, applied precisely to the candidate's owned scheduling choices (descriptor granularity, dev-id allocation, submission batching, cleanup ordering) while preserving the correctness oracle (per-job JobResult, byte-identical round trips, failed-transfer verdict invalidation).

---

### 6. Compress KV blocks with adaptive bitrate before NIXL OBJ transfer to shrink promotion bytes
- **Finding:** `find-vllm_v1_kv_offload-0009` — *CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving*
- **Source URL:** <https://arxiv.org/abs/2310.07240>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Insert a CacheGen-style KV tensor encoder into the object-store secondary tier submit/complete path in vllm/v1/kv_offload/tiering/obj/manager.py (ObjectStoreSecondaryTierManager._submit_transfer around lines 217-270 and the completion side of _poll_active_transfers around lines 302-345). On store (NIXL_WRITE), before calling register_memory/prep_xfer_dlist/make_prepped_xfer, encode each block's KV bytes with the CacheGen tensor encoder into a compact per-key payload, register the compressed staging buffers rather than the raw block regions, and record the encoded length/metadata alongside the object key so a subsequent load can locate and decode it. On load (NIXL_READ), issue the transfer into a compressed staging buffer of the recorded size, and after check_xfer_state returns NIXL_DONE decode the payload back into the primary-tier block region before publishing the JobResult (so failed transfers still invalidate lookup verdicts unchanged). Choose the compression level per submission based on an estimate of currently available object-store bandwidth (e.g., an EWMA of recent effective bytes/second observed across _transfers), matching the paper's adaptive-bitrate behavior — heavier compression when bandwidth is scarce, lighter compression when it is plentiful. Preserve exactly-one JobResult per submitted job, keep the existing cleanup ordering (release xfer handle first, then dlist, then deregister memory), and keep the correctness oracle (tests/v1/kv_offload/tiering/test_obj_tier.py, byte-identical KV round trips after decode) intact by treating the encoder as lossless enough that decoded blocks satisfy the tier's identity contract, or by exposing a config knob that gates lossy modes off by default.

**Proposal rationale.**

The candidate's estimated_impact_explanation explicitly notes that object-store promotion latency is often the bottleneck before TTFT, and that network/object-store bandwidth limits how much better polling or batching alone can do. The finding directly attacks that bandwidth constraint: CacheGen adapts KV compression level to available bandwidth, reducing bytes moved per promotion. Because the compression sits between the primary-tier block region and the NIXL OBJ descriptor, it is a local change to _submit_transfer and the completion side of _poll_active_transfers — the descriptor/dev-id/polling/cleanup structure the candidate owns is preserved. In multi-turn agentic workloads (the caller context), reused prefixes are large and repeatedly promoted from the object tier, so shrinking each transfer amortizes across many hits and plausibly improves median TTFT and, by making promotion faster overall, median TPOT for turns whose decode is stalled behind a promotion.

---

## Agent proposals

### 1. Issue parallel NIXL OBJ xfer stripes per job to saturate object-store concurrent-connection bandwidth
- **Agent:** claude

**Detailed description.**

In vllm/v1/kv_offload/tiering/obj/manager.py:217-345, change ObjectStoreSecondaryTierManager._submit_transfer so a single job whose block_ids_list length exceeds a threshold is issued as K equal-sized parallel NIXL prepped-xfer stripes over the SAME pre-registered files_desc / obj_handle, instead of one xfer over the full block set. Concretely: (1) after the existing register_memory(nixl_files, 'OBJ') + prep_xfer_dlist(...) succeed, compute stripe_count = min(K_max, ceil(len(block_ids_list)/min_blocks_per_stripe)) using a config knob (default K_max ~ 4-8, tuned to typical object-store per-connection caps of ~90 MB/s for S3), and partition both block_ids_list and the range(len(nixl_files)) index list into stripe_count contiguous slices; (2) call make_prepped_xfer once per stripe against the SHARED self._dram_prepped_handle and obj_handle with the corresponding block_ids slice and obj-dlist-index slice, then agent.transfer(...) each returned xfer_handle immediately so all K stripes are outstanding to the object store concurrently on independent NIXL connections; (3) change TransferEntry to hold a list of xfer_handles alongside the shared files_desc + obj_handle, and change self._transfers[job_id] to that struct. In _poll_active_transfers, check_xfer_state each active stripe, release_xfer_handle for stripes that reach NIXL_DONE/error as soon as they finish, and only append a single JobResult(success=all_stripes_ok) and call release_dlist_handle + deregister_memory once every stripe has resolved — matching the existing 'release xfer handles before publishing results, then dlist, then memory' cleanup ordering so primary-tier memory is not reused while any stripe is still live. On any stripe failure, mark the job failed but continue draining the remaining stripes' handles before shared-resource teardown so no data transfer outlives the released memory. Preserve exactly one JobResult per submitted job, the unique-monotonic _next_obj_dev_id invariant (dev ids are still allocated once for the whole job across all stripes), and byte-identical load/store round trips. Fall back to the current single-stripe path when the job is smaller than min_blocks_per_stripe.

**Novelty rationale.**

None of the six existing deep_research_proposals proposes intra-job parallel stripe transfers to exploit object-store concurrent-connection bandwidth scaling. Proposal #2 (progressive sub-transfers) splits a job into ordered small-first/growing sub-batches issued sequentially/pipelined with the explicit goal of reducing head-of-line blocking on early-block readiness — a serial schedule; this proposal instead issues K equal-sized stripes for one job SIMULTANEOUSLY to overcome the well-known per-connection throughput cap of S3/GCS-class object stores (aggregate bandwidth scales with parallel range GET/PUT connections, not with chunking within one connection). Proposals #4 and #7 fuse multiple JOBS into one submission (cross-job coalescing) — the opposite direction from splitting one job across stripes. Proposal #3 discusses chunked descriptors + slack-aware polling but explicitly collapses runs into fewer descriptors (coarser granularity, still one transfer), not multiple concurrent xfers. Proposals #5, #6, #8 concern descriptor-pool reuse, background threads, and payload compression — orthogonal mechanisms. The bandwidth-parallelism lever for a single large promotion job is uniquely addressed here.

---

### 2. Prioritize demand loads over cascade stores with OBJ transfer admission control
- **Agent:** codex

**Detailed description.**

In `vllm/v1/kv_offload/tiering/obj/manager.py:217-345`, add a small admission layer around `_submit_transfer` so object-store READ promotions are not forced to compete equally with background WRITE cascades. Concretely, split `_submit_transfer` into a lightweight enqueue path plus an internal `_start_transfer` containing the current descriptor/register/prep/transfer logic. Track pending jobs in two queues keyed by op, active counts for READ and WRITE, and config knobs such as `max_active_obj_transfers`, `max_active_obj_reads`, and `reserved_read_slots`. `submit_load` should enqueue READ jobs and immediately drain the READ queue while capacity exists; `submit_store` should enqueue WRITE jobs but only start them when total capacity remains after reserving read slots, or when no READs are pending/in flight. `_poll_active_transfers` should keep the existing completion and cleanup ordering, then call a `_drain_pending_transfers()` helper after each released handle so newly freed capacity starts READs first and WRITE cascades second. `drain_jobs()` should flush all pending jobs before blocking, and `shutdown()` should either fail queued-but-not-started jobs with exactly one `JobResult` or clear them only after the manager no longer expects results. This preserves the current per-job `JobResult` contract and byte-identical transfer behavior while preventing best-effort stores from inflating TTFT/TPOT for requests waiting on secondary-tier promotions.

**Novelty rationale.**

The existing proposals focus on moving work to a background thread, changing descriptor granularity, batching/coalescing submissions, reusing prepared descriptor lists, compressing payloads, slack-aware ordering, or splitting a single job into parallel/progressive sub-transfers. None introduces operation-class admission control that explicitly reserves object-store transfer capacity for demand READ promotions and throttles background WRITE cascades. Agent A's proposal increases intra-job parallelism for large jobs; this proposal may actually cap or delay lower-priority WRITE starts so READ promotions get bandwidth first, making it orthogonal to striping and batching.

---
