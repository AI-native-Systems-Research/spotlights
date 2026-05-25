# AsyncOperationManager._handle_save_task / _handle_load_task

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py) (lines 268–410)
- **Symbol:** `AsyncOperationManager._handle_save_task / _handle_load_task`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0011`

## Description
HF3FS async save/load task handlers that allocate device buffers, gather or scatter KV pages, batch filesystem reads/writes, and complete the task future.

## Current approach
Save allocates one buffer set for all blocks, gathers KV on `_save_stream`, records an event, submits `client.batch_write` calls in chunks of `DEFAULT_MAX_IO_ENTRIES`, then waits on all futures synchronously before confirming metadata. Load similarly submits `client.batch_read` chunks, waits on all futures, scatters buffers to KV cache on `_load_stream`, then calls `_load_stream.synchronize()` before freeing buffers and completing the future.

## Estimated impact explanation
HF3FS loads are on the TTFT path for cache hits, and saves consume background resources during generation. Reducing fixed batch overhead and the final load-stream synchronization can lower median TTFT for disk-backed multi-turn reuse and reduce TPOT interference from save work.

## Evolve rationale
The in-repo batching, stream synchronization, and buffer lifetime policy are concrete optimization units around the HF3FS client. Evolutions include adaptive I/O chunk sizing, earlier metadata confirmation batching, event-based buffer reclamation, or overlapping read completion with scatter. Correctness oracle: saved pages must load back byte-identically into the same KV block IDs, failed reads/writes must set failed task results, and HF3FS unit tests for batch read/write plus connector stats can be extended to assert identical future outcomes and stats.

## Deep research proposals

### 1. Compress KV pages with a CacheGen-style encoder before HF3FS batch writes and decode after reads
- **Finding:** `find-0003` — *CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving*
- **Source URL:** <https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py` lines 268-410 (`AsyncOperationManager._handle_save_task` / `_handle_load_task`), insert a KV-specific compression stage between the gather/scatter on `_save_stream`/`_load_stream` and the `client.batch_write` / `client.batch_read` chunks. On save: after gathering KV pages into the device buffer set, run a CacheGen-style tensor encoder (exploiting KV distributional regularities, e.g. per-channel quantization plus entropy coding) to emit a compact bitstream, write that bitstream via the existing chunked `client.batch_write` path, and persist a small per-block codec descriptor in the existing metadata confirmation step. On load: have the chunked `client.batch_read` futures populate compressed buffers, then decode on `_load_stream` immediately before the existing scatter into the KV cache, so decode overlaps with subsequent read chunks and the final `_load_stream.synchronize()` covers both decode and scatter. Optionally make the compression level adaptive per task based on observed HF3FS throughput (mirroring CacheGen's bandwidth-adaptive chunk levels), e.g. choose a higher-fidelity codec when the queue is light and a more aggressive one under contention. Preserve correctness by ensuring round-trip byte-identical KV reconstruction is gated by the codec descriptor, by surfacing decode errors into `task.future.set_exception` analogously to current failed read/write handling, and by extending the existing HF3FS batch read/write and connector-stats unit tests to cover the compressed path and assert identical future outcomes/stats.

**Proposal rationale.**

The candidate's bottleneck on the TTFT path is the volume of KV bytes traversing HF3FS batch reads plus the trailing `_load_stream.synchronize()` before buffers can be freed and the future completed. CacheGen's specific contribution is a KV-tailored encoder that meaningfully shrinks KV bitstreams with negligible decode overhead, which directly attacks the bytes-on-the-wire term in load/save latency rather than just reordering existing work. Because the candidate already isolates gather/scatter on dedicated CUDA streams and chunks I/O through a single `client.batch_*` boundary, a codec stage drops in cleanly at exactly those seams and can overlap decode with the in-flight read chunks, addressing the synchronous wait that the candidate's own evolve_rationale calls out. The bandwidth-adaptive aspect of the finding maps onto the existing `DEFAULT_MAX_IO_ENTRIES` chunking knob, giving a principled way to trade fidelity for latency under a multi-turn agentic workload where cache-hit TTFT dominates.

---

### 2. Ring-buffered staging for HF3FS save/load to overlap gather, I/O, and scatter
- **Finding:** `find-0006` — *[Roadmap] Prefill-Decode Disaggregation Roadmap (2026 Q2)*
- **Source URL:** <https://github.com/sgl-project/sglang/issues/21703>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py` (lines 268-410, `AsyncOperationManager._handle_save_task` / `_handle_load_task`), replace the current one-shot "allocate one buffer set for all blocks, gather on _save_stream, then submit batch_write chunks and wait on all futures" pattern with a ring-allocated staging buffer sized to a small multiple of `DEFAULT_MAX_IO_ENTRIES` worth of pages. Each ring slot is associated with a CUDA event recorded on `_save_stream` (or `_load_stream`).

For save: as soon as a slot's gather event is recorded, submit `client.batch_write` for that slot's pages without waiting for the remaining slots to finish gathering, and as each write future completes, mark the slot free and record per-chunk metadata confirmation (instead of confirming all metadata only after the final future). Reclaim slots based on event completion rather than waiting on every future synchronously at the end.

For load: submit `client.batch_read` into ring slots, and as each chunk's read future completes, queue the scatter to KV cache on `_load_stream` against that slot (waiting on a per-slot read-done event) and free the slot when the scatter event fires. This eliminates the global `_load_stream.synchronize()` barrier in favor of per-slot event waits, allowing the next save/load task to begin reusing slots while the tail scatter is still draining.

Keep the current correctness oracle: saved pages must round-trip byte-identically into the same KV block IDs, any failed read/write future must still mark the task failed (propagated through the ring slot's completion callback), and connector stats must reflect the same per-block accounting.

**Proposal rationale.**

The finding's transferable idea is a ring-based staging buffer that decouples gather, transfer, and scatter so that many small operations can be pipelined rather than executed in lockstep batches with a global synchronize at the end. The HF3FS handlers today exhibit exactly the inefficiency the finding targets: a fixed gather phase, a serialized `batch_write/read` submission loop chunked at `DEFAULT_MAX_IO_ENTRIES`, a synchronous `gather()` on all futures, and (for load) a `_load_stream.synchronize()` before buffers are freed and the future completes. This forces TTFT-on-cache-hit to wait for the slowest chunk of a load and prevents overlapping scatter with the tail of read I/O. Applying the ring-allocator pattern enables (a) earlier metadata confirmation batching, (b) event-based buffer reclamation, and (c) overlap of read completion with scatter and of gather with write submission — each explicitly listed in the candidate's `evolve_rationale`. The mapping is direct enough to be concrete (replace one-shot buffer + final synchronize with ring slots + per-slot events) without inventing unrelated mechanisms, and it targets the exact median-TTFT/TPOT-interference axes called out in the candidate's estimated impact for disk-backed multi-turn reuse.

---

### 3. Pipeline HF3FS load reads with scatter via per-chunk readiness events
- **Finding:** `find-0008` — *Disaggregated Serving | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py` lines 268-410, restructure `AsyncOperationManager._handle_load_task` (and symmetrically `_handle_save_task`) so that batched HF3FS I/O is treated as non-blocking background work whose completion is signaled at sub-batch granularity instead of via a single synchronous wait on all futures plus a final `_load_stream.synchronize()`. Concretely, on load: submit `client.batch_read` in `DEFAULT_MAX_IO_ENTRIES` chunks as today, but instead of `wait_all` on every future before any scatter runs, attach a small completion callback (or poll futures in submission order) that, as each chunk's read completes, immediately enqueues the corresponding scatter into `_load_stream` for just that subset of blocks and records a CUDA event for it. The task future is then completed once the last per-chunk event is observed (e.g., via `event.synchronize()` on only the trailing event, or by gating the consumer on these events instead of a whole-stream sync), allowing earlier-completing chunks' scatter work to overlap with later-arriving reads and freeing their host buffers as soon as their event fires. On save, mirror the pattern: as each `batch_write` chunk acks, mark those blocks ready for metadata confirmation and release their buffers, instead of waiting for the entire write set. Preserve correctness by still gating final `set_result(True)` on all per-chunk events/futures and propagating any chunk failure to `set_result(False)`; keep the existing one-shot buffer allocation but add per-chunk reference counting so reclamation is event-driven rather than tied to the whole-task synchronize.

**Proposal rationale.**

The Dynamo doc's principle that KV transfer should be non-blocking with finer-grained readiness directly targets two concrete bottlenecks called out in this candidate's evolve_rationale: the final `_load_stream.synchronize()` and the all-or-nothing wait on batched HF3FS futures. Today, the first chunk's data is on-device and ready to scatter long before the last `batch_read` future resolves, and scattered KV is ready for compute before the entire stream drains, so the synchronous joins serialize work that the candidate's two dedicated streams are already structured to overlap. Applying the finding's pattern, exposing per-chunk readiness via events instead of whole-task sync, lets scatter overlap with in-flight reads and lets the engine admit decode for cache hits as soon as the relevant blocks land, which is the TTFT-on-cache-hit and TPOT-interference path explicitly highlighted in `estimated_impact_explanation`. The change is local to the two handlers, preserves the byte-identical-load oracle and failure semantics noted in evolve_rationale, and reuses the existing stream/event machinery rather than introducing new transport.

---

### 4. Pipeline HF3FS batch I/O with gather/scatter to overlap compute and disk transfer
- **Finding:** `find-0010` — *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*
- **Source URL:** <https://arxiv.org/abs/2510.09665>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py` at lines 268-410, restructure `AsyncOperationManager._handle_save_task` and `_handle_load_task` so KV gather/scatter on `_save_stream`/`_load_stream` is pipelined with `client.batch_write`/`client.batch_read` chunks rather than serialized behind a full barrier. Concretely: (1) Split blocks into pipeline stages sized around `DEFAULT_MAX_IO_ENTRIES`; for save, issue gather + CUDA event record for stage N while stage N-1's `batch_write` future is still in flight, so disk write begins as soon as the per-stage event signals rather than after all gathers complete. (2) For load, as each `batch_read` chunk future completes, immediately launch the scatter for that chunk on `_load_stream` (waiting on the chunk's completion event) instead of joining all reads before any scatter; replace the terminal `_load_stream.synchronize()` with per-chunk events so buffers are released as soon as their scatter finishes, shortening the critical path to the first usable block. (3) Confirm metadata for save in batches as each write chunk's future resolves, rather than after the full barrier. Preserve the existing failure semantics: any failed chunk future still sets the task result to failure and cancels in-flight work before completion.

**Proposal rationale.**

The finding identifies batched KV data movement combined with compute/I/O pipelining as the key driver of LMCache's KV transfer performance. The HF3FS handler already does step one (batched I/O via `DEFAULT_MAX_IO_ENTRIES` chunks) but is missing step two: it serializes gather→write and read→scatter behind whole-task barriers (`asyncio.gather` over all futures, then `_load_stream.synchronize()`), so disk latency and PCIe gather/scatter never overlap. For the stated multi-turn agentic workload where HF3FS loads sit on the TTFT path for cache hits, overlapping scatter with later chunk reads directly reduces the time to the first usable block, and overlapping gather with earlier writes reduces the save tail that otherwise contends with decode for SM and copy bandwidth (lowering TPOT interference). The finding gives a concrete, transferable structural idea — staged pipelining keyed on per-chunk completion events — that maps onto the existing CUDA stream and chunked-future scaffolding in this candidate without changing the on-disk format or the byte-identical correctness oracle.

---

### 5. Quantize KV pages with KIVI-style asymmetric 2-bit format on the HF3FS wire/storage path
- **Finding:** `find-0017` — *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*
- **Source URL:** <https://proceedings.mlr.press/v235/liu24bz.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py` lines 268-410, modify `AsyncOperationManager._handle_save_task` and `_handle_load_task` so the bytes that traverse `client.batch_write` / `client.batch_read` are an asymmetric quantized representation rather than the native KV dtype. On save: after the existing per-block gather on `_save_stream`, run a fused quantization kernel that produces (a) per-channel scale/zero for the key tensor and (b) per-token scale/zero for the value tensor at 2 bits, packs them into the device buffer set, and only then issues the chunked `batch_write` calls; persist the quant metadata (dtype tag, group sizes, scale/zero layout, original shape) alongside each block in the HF3FS metadata that is already confirmed at the end of the save. On load: after the chunked `batch_read` futures resolve, dispatch a dequantization kernel on `_load_stream` that consumes the packed buffer plus the saved scale/zero tables and writes back into the KV cache, before the final `_load_stream.synchronize()` and buffer free. The chunking strategy, event-based buffer lifetime, and failure paths (`set failed task results`) remain unchanged; the only new logic is the quantize-before-write and dequantize-after-read steps and the small metadata extension needed to round-trip scale/zero. Make the format optional behind a connector-level config so non-quantized callers keep byte-identical behavior; when enabled, the correctness oracle becomes "dequantized load matches the quantization of the saved tensor at the same KV block IDs" rather than strict byte-identity, and the existing HF3FS batch read/write unit tests plus connector stats assertions are extended to cover the new path.

**Proposal rationale.**

The candidate's `estimated_impact_explanation` explicitly identifies HF3FS load latency as a TTFT driver for disk-backed multi-turn reuse and save bandwidth as a TPOT-interference source, which is exactly the regime where reducing bytes-moved dominates. The current handlers spend most of their wall time in `client.batch_write` / `client.batch_read` over `DEFAULT_MAX_IO_ENTRIES` chunks of full-precision KV pages; KIVI's asymmetric scheme (per-channel keys, per-token values) is specifically designed to preserve KV-cache quality at 2 bits, which is the missing ingredient that lets the connector trade compute on the existing `_save_stream` / `_load_stream` for ~8x less I/O without degrading generation. The finding also matches the caller context's multi-turn agentic workload, where the same KV blocks are repeatedly saved and loaded, so any per-byte savings amortize across reuses. This is a concrete, transferable change to the gather/scatter framing already present at lines 268-410 rather than a topical restatement of the existing batching policy.

---

### 6. Pipeline HF3FS load chunks: scatter completed read batches while later chunks remain in flight
- **Finding:** `find-0022` — *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*
- **Source URL:** <https://www.microsoft.com/en-us/research/publication/sarathi-efficient-llm-inference-by-piggybacking-decodes-with-chunked-prefills/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `AsyncOperationManager._handle_load_task` (vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:268-410), replace the current pattern of submitting all `client.batch_read` chunks of `DEFAULT_MAX_IO_ENTRIES` and then synchronously waiting on every future before doing a single scatter on `_load_stream`, with a chunk-pipelined schedule. Concretely: keep the existing chunking but, as each chunk's read future completes (e.g. via `as_completed` or an ordered queue), enqueue a per-chunk scatter on `_load_stream` for just the blocks/buffers belonging to that chunk while subsequent chunks' reads are still in flight on the HF3FS client. Use a CUDA event per chunk to gate the scatter behind the chunk's H2D copy, and only call `_load_stream.synchronize()` (or wait on the final event) once the last chunk's scatter is enqueued, then free buffers and complete the future. Apply the symmetric idea in `_handle_save_task`: confirm metadata for chunks of writes as they complete instead of after the full barrier, so a long save does not block the connector's metadata path. Preserve correctness by ensuring (a) per-chunk failure still routes to `set_exception` on the task future, (b) buffer slots used by an in-flight chunk are not reclaimed until that chunk's scatter event fires, and (c) the final completion still observes byte-identical KV in the same block IDs covered by the existing HF3FS batch read/write unit tests and connector-stats assertions.

**Proposal rationale.**

SARATHI's central idea is that monolithic prefill work can be split into uniform chunks so downstream consumers do not have to wait for the whole unit before making progress. The candidate has the same shape of inefficiency on the KV-transfer path: load requests already chunk batch reads by `DEFAULT_MAX_IO_ENTRIES`, but the handler then waits on all chunk futures before any scatter and finally synchronizes the load stream — exactly the kind of full-batch barrier SARATHI argues against. Translating chunked prefill into chunk-level KV streaming lets decode-side work observe the first chunks' KV before later chunks finish reading from HF3FS, directly attacking median TTFT for cache-hit multi-turn workloads where loads land on the TTFT path. It also matches the evolve_rationale's call-out for `overlapping read completion with scatter` and `event-based buffer reclamation`, and addresses a concrete gap (final `_load_stream.synchronize()` and full-batch wait) rather than restating current behavior.

---

## Agent proposals

### 1. Cross-task I/O coalescing with TTFT-priority lanes in AsyncOperationManager for HF3FS save/load submissions
- **Agent:** claude

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py` (lines 268-410), restructure `AsyncOperationManager._handle_save_task` and `_handle_load_task` so they no longer submit their own `client.batch_write`/`client.batch_read` chunks directly. Instead, introduce two shared submission queues inside `AsyncOperationManager` — a high-priority `load_queue` and a low-priority `save_queue` — each holding (block_handle, device_buffer_slice, per_block_future) entries from any in-flight task. A single submitter coroutine drains these queues into HF3FS client calls with two new behaviors: (1) Cross-task coalescing — pack entries from multiple tasks into one `batch_read`/`batch_write` of up to `DEFAULT_MAX_IO_ENTRIES`, fanning the single client future out to per-block (and thus per-task) completion via an indexed split. This amortizes the HF3FS per-call submission overhead across concurrent tasks (e.g., a 30-block save and a 50-block save become one 80-block call rather than two short calls). (2) Priority lanes — the submitter always drains `load_queue` before `save_queue` and additionally reserves a configurable fraction of `DEFAULT_MAX_IO_ENTRIES` for loads so a long stream of background saves cannot head-of-line block a TTFT-critical load that arrives mid-flight. Per-task gather/scatter on `_save_stream`/`_load_stream` and the per-task future-resolution/buffer-free logic stay where they are: each task still owns its device buffer set, records its gather event before enqueuing entries, and completes its `task.future` once all of its per-block futures resolve. Failure semantics are preserved by routing any per-block exception to the owning task's `task.future.set_exception`, and the byte-identical-round-trip oracle is unchanged because no per-block payload is altered. Existing HF3FS batch read/write unit tests and connector-stats assertions can be extended with a multi-task concurrency case asserting (a) identical per-task outcomes, (b) reduced number of `client.batch_*` invocations under concurrency, and (c) load-task completion order under contention with saves.

**Novelty rationale.**

All six existing deep_research_proposals optimize the lifecycle of a *single* save/load task: find-0006 (ring buffer per task), find-0008/0010/0022 (per-chunk readiness events within one task), find-0003/0017 (per-block compression/quantization). None introduce inter-task batching across the `AsyncOperationManager`'s concurrent tasks, none coalesce multiple tasks' chunks into one HF3FS client call to amortize submission overhead, and none address head-of-line contention between background saves and TTFT-critical loads via priority lanes. The mechanism — a shared, priority-aware submission queue with cross-task batch packing and fan-out futures — is structurally distinct from the intra-task pipelining ideas already proposed, and it composes with (rather than duplicates) any of them.

---

### 2. Fuse per-block KV gather/scatter into batched kernels
- **Agent:** codex

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py` lines 268-410, replace the `_handle_save_task` and `_handle_load_task` calls to `_connector._gather_or_scatter_kv_caches(block_ids, buffers, ...)` with a batched gather/scatter helper. The current helper loops in Python per block, builds a `token_indices` list/tensor per block, and launches one Triton kernel per KV page. Add a helper that takes the whole `block_ids` list plus the buffer pointer list, materializes one device tensor of block IDs and one device tensor of buffer data pointers, and launches a single Triton grid over `(num_blocks, num_layers, block_size)` for gather or scatter. On save, run this batched gather on `_save_stream` before recording the existing `save_stream_event`; on load, run the batched scatter on `_load_stream` after successful reads before the existing stream synchronization/free path. Keep the HF3FS `batch_read`/`batch_write` chunking and buffer lifetime semantics unchanged. Validation should compare the old and new helper paths for MHA and MLA layouts and assert byte-identical saved-then-loaded KV pages for the same block IDs, while optionally checking reduced launch count for multi-block tasks.

**Novelty rationale.**

The listed deep_research_proposals optimize bytes moved via compression/quantization or scheduling via ring buffers, per-chunk readiness, pipelined read/scatter, metadata confirmation timing, and event-based reclamation. This proposal targets a different cost inside the same handlers: per-block Python overhead and many small gather/scatter kernel launches. It does not alter I/O ordering, overlap policy, storage format, buffer reclamation, or metadata confirmation. Agent A's proposal is inter-task HF3FS submission coalescing with priority lanes; this proposal stays within a single task's CUDA copy kernel granularity and composes with cross-task batching rather than duplicating it.

---
