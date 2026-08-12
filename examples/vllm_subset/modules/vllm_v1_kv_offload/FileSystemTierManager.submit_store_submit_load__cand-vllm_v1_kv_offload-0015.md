# FileSystemTierManager.submit_store/submit_load

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/tiering/fs/manager.py`](vllm/v1/kv_offload/tiering/fs/manager.py) (lines 217–267)
- **Symbol:** `FileSystemTierManager.submit_store/submit_load`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0015`

## Description
FS secondary-tier load/store submission builds paths and offsets, wraps the whole job as one task, and enqueues it into the dual-queue pool.

## Current approach
submit_store and submit_load each create one functools.partial over all block paths and offsets, then call enqueue_store or enqueue_load with n_tasks=1. Large jobs therefore occupy a single worker task, while small jobs are not coalesced across submissions.

## Estimated impact explanation
For FS-backed multi-turn reuse, secondary-to-primary loads are on the TTFT path. Better task sizing can use configured read threads more effectively and reduce promotion latency, while store-side changes can reduce interference with read bursts.

## Evolve rationale
Task granularity, chunking, and coalescing are owned scheduling choices distinct from the thread-pool priority rule. Evolution can split large jobs across workers, coalesce small adjacent jobs, or choose chunk sizes by block count and filesystem behavior. Correctness oracle: FileSystemTierManager tests, byte-identical store/load round trips, one JobResult per submitted job, and preservation of partial-load failure reporting.

## Deep research proposals

### 1. Progressive-batch chunking of submit_load with small early sub-tasks
- **Finding:** `find-vllm_v1_kv_offload-0003` — *[RFC]: Progressive KV Cache CPU Onloading*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/33526>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Change FileSystemTierManager.submit_load in vllm/v1/kv_offload/tiering/fs/manager.py (lines 217-267) so that a load job whose block count exceeds a small threshold is decomposed into an ordered sequence of sub-tasks rather than a single functools.partial wrapping every path/offset. Split the (paths, offsets) lists into N contiguous chunks using a progressive size schedule: the first chunk is small (e.g. 1-2 blocks) and subsequent chunks grow (e.g. doubling up to a cap tied to n_read_threads and block size). Wrap each chunk in its own load_task closure that calls batch_load_block on that slice, and hand them to self._pool.enqueue_load(job_id, n_tasks=len(chunks), tasks=[...]) so the DualQueueThreadPool already-in-place aggregation reports a single JobResult per submitted job. Preserve the existing partial-load failure semantics by threading per-chunk offsets into self._load_progress: on OSError in a chunk, translate exc.num_succeeded within that slice into a global count so get_finished_jobs still yields successful_keys = load_keys[:num_succeeded] and failed = load_keys[num_succeeded:]. submit_store may keep its single-task form or apply the same schedule symmetrically; the load path is where TTFT wins are expected.

**Proposal rationale.**

The candidate's current_approach explicitly notes that submit_load wraps the entire job as one task with n_tasks=1, so a large secondary->primary promotion pins one read-thread worker and blocks any short-request load that shares the read queue. The finding's core idea - split a single large CPU->GPU (here FS->CPU) reload into progressive batches with small early sub-tasks - maps directly onto this gap: a small first chunk lets a short co-scheduled request's blocks interleave on other read threads sooner, while later, larger chunks amortize per-task overhead. This targets the caller's stated goal (reduce median TTFT on multi-turn agentic workloads where secondary->primary loads sit on the TTFT path) and stays within the evolve_rationale's owned scheduling choices (task granularity and chunking) without changing correctness: JobResult-per-job is preserved because DualQueueThreadPool already supports enqueue with n_tasks>1, and partial-failure reporting is preserved by adjusting the load_progress accounting.

---

### 2. Chunk FS tier jobs into multiple sub-tasks and add slack-aware load/store scheduling
- **Finding:** `find-vllm_v1_kv_offload-0006` — *Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving*
- **Source URL:** <https://arxiv.org/abs/2605.03375>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/kv_offload/tiering/fs/manager.py FileSystemTierManager.submit_store and submit_load (lines 217-267), replace the current single-task enqueue (n_tasks=1 wrapping the whole block list) with a chunked submission: split the (paths, offsets) list into K contiguous chunks sized by a configurable target (e.g. min(n_read_threads, ceil(len(paths)/chunk_size))) and enqueue K partials via self._pool.enqueue_load(job_id, K, tasks). Preserve the existing JobResult contract by having each sub-task record its own num_succeeded into a per-chunk slot, and reconstruct the earliest failing block position in get_finished_jobs so partial-load failure reporting (successful_keys/failed_keys via mark_miss) still points at the first bad block in submission order. For submit_store, apply the same chunking but additionally coalesce or defer store chunks when the load queue is non-empty (a slack-aware hint added to DualQueueThreadPool, e.g. store-priority workers skip a store chunk pull when self._load_q is non-trivially populated), so promotion loads on the TTFT path are not blocked by in-flight store bursts. Chunk size should default conservatively (single chunk for jobs <= some threshold, e.g. 8 blocks) so small jobs still travel as a single task and short-job overhead is unchanged.

**Proposal rationale.**

The candidate's current_approach explicitly wraps each job as one task, so a large promotion load occupies a single FS worker while other read threads sit idle. Tutti's contribution is (a) bulk KV-cache object transfers replacing tiny per-block random I/O and (b) slack/timing-aware scheduling to avoid GPU stalls on SSD tiers. The candidate already achieves (a) at the syscall layer via batch_load_block, but not at the task layer: it cannot use multiple configured read threads for a single job. Splitting into K sub-tasks addresses exactly this gap and directly targets median TTFT for the multi-turn agentic workload, where secondary-to-primary promotion is on the critical path. Slack-aware store deferral maps Tutti's timing-aware scheduling onto the DualQueueThreadPool priority rule and addresses the estimated_impact_explanation's concern about stores interfering with read bursts. Both changes are within the evolve_rationale's stated scope (task granularity, chunking, coalescing) and preserve the correctness oracle (one JobResult per job, byte-identical round trips, partial-failure reporting).

---

### 3. Chunk FS submit_store/submit_load into per-worker batched I/O tasks
- **Finding:** `find-vllm_v1_kv_offload-0007` — *GPUDirect Storage Overview Guide*
- **Source URL:** <https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify FileSystemTierManager.submit_store and submit_load (vllm/v1/kv_offload/tiering/fs/manager.py:217-267) so that instead of wrapping every block in the job into a single functools.partial with n_tasks=1, the manager splits the (paths, offsets) list into N chunks sized to the pool's read/write thread count and enqueues N batched tasks via _pool.enqueue_store/enqueue_load(job_id, N, tasks). Each chunk still calls batch_store_block/batch_load_block over its slice, preserving the batched syscall path (so per-chunk fixed overhead is amortized across the blocks in that chunk), but multiple chunks now run in parallel across worker threads. Chunk size is chosen from the block count and the configured thread count (e.g. ceil(len/threads) with a min-chunk floor to avoid degenerate 1-block chunks) so small jobs stay single-task and large jobs fan out. Preserve the existing partial-load failure reporting by aggregating num_succeeded across chunks in the load_task closure and only recording _load_progress[job_id] when a chunk fails, and by keeping the one-JobResult-per-job contract via the pool's n_tasks argument. No changes to the enqueue_store/enqueue_load or batch_*_block APIs are required.

**Proposal rationale.**

The candidate today builds one giant partial per job and enqueues it as n_tasks=1, so a large multi-turn KV promotion can only use a single FS worker thread while other threads sit idle — directly limiting the FS-to-primary load latency on the TTFT path called out in the candidate's estimated_impact. The GPUDirect Storage guide's core scheduling lesson is that fixed per-submission overhead should be amortized by *batching within a submission*, and that many batched submissions can then be issued in parallel to hide storage latency. That maps cleanly onto splitting one job into a small number of still-batched chunks: each chunk keeps the amortization benefit (batch_store_block/batch_load_block already issues one syscall path over its list), while chunking lets the dual-queue pool actually use its configured read/write threads for a single large job. This addresses the specific gap the candidate names — task granularity and chunk-size choice — without touching the pool's priority rule or the file mapper, and it stays within the correctness oracle (byte-identical round trips, one JobResult per job, and partial-failure key reporting are all preserved by aggregating num_succeeded across chunks).

---

## Agent proposals

### 1. Cross-submission store coalescing with a short debounce and a job-fanout JobState
- **Agent:** claude

**Detailed description.**

Change FileSystemTierManager.submit_store in vllm/v1/kv_offload/tiering/fs/manager.py (lines 217-267) so that store submissions are not enqueued immediately as one-task partials. Instead, buffer arriving store JobMetadata in a small manager-owned staging list guarded by a lightweight lock, and either (a) enqueue immediately once the buffered block count crosses a threshold (e.g. blocks_per_chunk or a small multiple of it), or (b) release the buffer via a short debounce (a single armed threading.Timer or a dedicated flusher thread using Condition.wait(timeout=few_ms)). When the buffer flushes, concatenate the buffered jobs' (paths, offsets) lists into one merged batch_store_block partial and enqueue it as a single task via self._pool.enqueue_store. To preserve the existing one-JobResult-per-submitted-job contract without changing the pool API, introduce a JobFanoutState that wraps one underlying JobState but distributes the single completion signal back to N separate finished_q entries (one per original job_id) by installing per-original-job entries into self._pool._finished_q on completion; alternatively, register N synthetic JobStates against the pool but with all of them sharing the same task callable via a JobGroup wrapper that calls task_done on each member state in one shot. The KV-event bookkeeping (self._store_job_keys) is unaffected because each original job_id keeps its own keys entry, and get_finished_jobs already loops over finished_q. On buffer flush, the merged task calls batch_store_block over the union of paths/offsets; a partial failure inside the C batch (num_succeeded < total) is fanned out per-original-job by counting cumulative-block boundaries so each merged job reports success/failure independently and store events are only emitted for the fully-successful original jobs. The debounce window and coalesce threshold are configurable, defaulting to a few milliseconds and a target chunk size aligned with n_write_threads, so isolated stores still travel immediately. submit_load is left untouched by this change and remains free to adopt orthogonal chunking work.

**Novelty rationale.**

All three existing deep_research_proposals (findings 0003, 0006, 0007) SPLIT a single large job into multiple sub-tasks for intra-job parallelism, and finding 0006 additionally proposes slack-aware store deferral (a pool-priority-rule tweak) — none of them MERGE multiple separately-submitted store jobs into one task, and none introduce a per-submission staging buffer with a fanout completion mapping. This proposal targets the opposite regime the existing ones ignore: bursts of many small submit_store calls (typical when eviction promotes several short KV chunks in quick succession), where per-job fixed overhead (partial construction, deque append, condvar notify, JobState allocation, task_done bookkeeping) and per-task open/close syscalls dominate over the actual write payload. Cross-submission coalescing is called out in the candidate's own evolve_rationale ("coalesce small adjacent jobs") but is not touched by any of the three existing proposals, which stay strictly within a single-job scope. The fanout-JobState mechanism is a novel contribution: it lets the manager preserve one JobResult per submitted job (the correctness oracle) while letting the pool see only one merged task, which is the opposite direction from the N-sub-tasks-per-job pattern the other proposals share and therefore does not overlap them.

---

### 2. Move path and offset materialization out of scheduler-thread submission
- **Agent:** codex

**Detailed description.**

Change FileSystemTierManager.submit_store and submit_load in vllm/v1/kv_offload/tiering/fs/manager.py:217-267 so the scheduler thread no longer eagerly builds full paths and byte-offset lists for every block before enqueueing. Instead, capture only immutable copies of job_id, keys, block_ids, block_size, use_o_direct, file_mapper, and the primary memoryview in the submitted task closure, and materialize the path/offset arrays inside the worker task immediately before calling batch_store_block or batch_load_block. For submit_load, keep _load_job_keys[job_id] populated on the scheduler thread so failure handling still knows the original key order; the worker-side load_task should still record _load_progress[job_id] from exc.num_succeeded exactly as today. For submit_store with KV events enabled, keep _store_job_keys[job_id] on the scheduler thread as today. Add or extend a FileSystemTierManager test with a large synthetic JobMetadata and a FileMapper spy to assert submit_load/submit_store return after enqueueing without invoking get_file_name for every key, while get_finished_jobs still produces one JobResult per job and existing byte-identical round-trip and partial-load failure tests continue to pass.

**Novelty rationale.**

The three deep_research_proposals all change worker-task granularity by splitting one submitted load/store job into multiple chunks, and one also changes load/store queue priority. Agent A proposes the opposite direction for stores: buffering multiple submitted store jobs and faning one coalesced task completion back out. This proposal does neither splitting nor cross-submission coalescing. It targets a separate latency source visible in the candidate region: submit_store/submit_load currently perform O(number of blocks) Python path and offset construction synchronously on the scheduler thread before the job even reaches the pool. Moving that materialization into the existing worker task preserves the current one-task, one-JobResult contract and partial-load semantics, while reducing scheduler-thread stalls that can affect median TTFT and TPOT even when the actual filesystem I/O is asynchronous.

---
