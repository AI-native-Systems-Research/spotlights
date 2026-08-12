# UBatchWrapper._capture_ubatches and _run_ubatches

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_ubatch_wrapper.py`](vllm/v1/worker/gpu_ubatch_wrapper.py) (lines 212–341)
- **Symbol:** `UBatchWrapper._capture_ubatches and _run_ubatches`
- **Kind:** region
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0019`

## Description
Threaded CUDA graph capture and runtime execution for legacy DBO microbatches.

## Current approach
Starts one Python thread per ubatch, synchronizes with a Barrier and threading.Event handoff, and serializes completion through thread joins and sorted result concatenation.

## Estimated impact explanation
DBO exists to lower TPOT for mixed prefill/decode batches; reducing scheduler overhead or adapting the handoff policy improves overlap and median TPOT for non-uniform agent batches.

## Evolve rationale
The thread/barrier/event schedule is owned code and paired with UBatchContext in vllm/v1/worker/ubatching.py. UBatch and attention-splitting tests validate output ordering and metadata slicing invariants.

## Deep research proposals

### 1. Reuse ubatch CUDA graphs across token buckets via cudaGraphExecUpdate
- **Finding:** `find-vllm_v1_worker-0008` — *Employing CUDA Graphs in a Dynamic Environment*
- **Source URL:** <https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/v1/worker/gpu_ubatch_wrapper.py` around lines 212-303, `_capture_ubatches` currently allocates a fresh `torch.cuda.CUDAGraph()` per distinct `num_tokens` and stores it in `self.cudagraphs[num_tokens]`, forcing a full re-capture (including the multi-thread barrier/event bring-up in `_capture_ubatch_thread`) whenever a new bucket is encountered. Adopt the NVIDIA blog's dynamic-parameter pattern: when a previously captured ubatch graph exists whose per-kernel topology (layer sequence, comm/compute stream layout, kernel identities) matches an incoming shape and only launch parameters/tensor addresses differ, clone the executable and call `cudaGraphExecUpdate` (via `torch.cuda.CUDAGraph.update()` where available, or a small `cudaGraphExecUpdate` binding) instead of recapturing. Concretely: (1) key `self.cudagraphs` on a `(topology_signature, num_tokens)` tuple, where `topology_signature` is derived from the ubatch context configuration (num ubatches, attention split mode, comm/compute stream ids) rather than num_tokens alone; (2) on a miss, try `update()` from the nearest existing entry with matching signature before falling back to the current thread+barrier capture path in lines 259-302; (3) keep the thread/barrier/event handoff unchanged for the true first-capture path, and skip it entirely on the update path since no re-launch of `_capture_ubatch_thread` is required. Fall back to full recapture whenever `cudaGraphExecUpdate` returns a topology-mismatch status, exactly as the blog recommends.

**Proposal rationale.**

The candidate's cost centers are (a) the thread/barrier/event bring-up in `_capture_ubatches` and (b) the per-bucket recapture that must repeat that bring-up. The finding targets exactly the recapture case: recurring shapes where topology is stable and only kernel parameters change, which matches DBO ubatch capture across `num_tokens` buckets seen in multi-turn agentic serving. Replacing recapture with `cudaGraphExecUpdate` removes both the CUDA graph capture cost and the associated thread-orchestration prelude for the second-and-later occurrence of each topologically equivalent shape, directly attacking scheduler/dispatch overhead on the DBO path and improving median TPOT for the non-uniform prefill/decode batches the candidate is optimized for.

---

## Agent proposals

### 1. Reuse persistent ubatch worker threads instead of spawning per forward pass
- **Agent:** claude

**Detailed description.**

In `vllm/v1/worker/gpu_ubatch_wrapper.py` at lines 305-341, `_run_ubatches` constructs two fresh `threading.Thread` objects on every eager (non-graph) forward pass, starts them, waits on `self.ready_barrier`, joins them, then discards them. Thread construction, `pthread_create`, GIL acquisition, and the initial `set_device_index`/`current_blas_handle()` warmup happen every step and land on the critical TPOT path when the runtime falls off the CUDA graph replay path (e.g. shapes outside captured buckets, first token of a new bucket, or capture-disabled configs). Replace this with a long-lived worker pool owned by `UBatchWrapper`: at wrapper init, spawn `num_ubatches` daemon threads, each pinned to `self.device`, each holding an `Event`-guarded work queue of one slot (`(metadata, results, model, done_event)`). `_run_ubatches` becomes: write per-worker work slots, set each worker's `start_event`, `self.ready_barrier.wait()`, set `cpu_wait_event`, then wait on each worker's `done_event`; the workers themselves run `with ubatch_metadata.context: model(...)` in a `while not shutdown: start_event.wait(); ...; done_event.set()` loop. Results are written into a preallocated 2-slot list indexed by `context.id`, removing the `sorted(results)` step in lines 295 and 339. Do the same in `_capture_ubatches` (lines 259-270) so the capture path also reuses the persistent workers — the existing pre-capture BLAS-handle warmup in `_capture_ubatch_thread` (lines 240-242) becomes a one-time per-worker init at pool startup rather than a per-capture cost. Add a `__del__`/`close()` on `UBatchWrapper` that sets a `shutdown` flag, sets each `start_event`, and joins the pool. Keep `UBatchContext`'s existing barrier/event handshake in `vllm/v1/worker/ubatching.py` unchanged so context-manager semantics and `_THREAD_ID_TO_CONTEXT` bookkeeping still work — the workers enter/exit the context each iteration, exactly as the throwaway threads do today, so no changes to the ubatching state machine are required.

**Novelty rationale.**

The one existing deep_research_proposal (find-vllm_v1_worker-0008) targets only `_capture_ubatches` and only the CUDA-graph recapture cost across `num_tokens` buckets via `cudaGraphExecUpdate`. It leaves the non-graph runtime path (`_run_ubatches`, lines 305-341) untouched and does not remove the per-invocation `threading.Thread` construction or the sorted-results merge — both of which run every forward pass regardless of whether a graph is captured or replayed-with-update. Persistent worker threads attack a different cost center (Python/pthread thread-lifecycle overhead and result ordering) and compose cleanly with the graph-update proposal rather than overlapping with it.

---

### 2. Reuse ubatch synchronization contexts across eager forwards
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu_ubatch_wrapper.py`, add a small `UBatchWrapper`-owned cache of `UBatchContext` objects and their synchronization primitives so `_run_ubatches` and the true first-capture path in `_capture_ubatches` do not allocate fresh `threading.Event` and `torch.Event` objects through `make_ubatch_contexts` on every ubatched forward. Concretely, move the per-ubatch CPU wait/signal events and GPU comm/compute done events out of `make_ubatch_contexts`' per-call construction and into a reusable context pool keyed by `(num_ubatches, compute_stream, comm_stream)` or refreshed when the current compute stream changes. Add a `reset(...)` method on `UBatchContext` that updates `forward_context`, clears `cpu_wait_event`/`cpu_signal_event`, clears `recv_hook`, and leaves the reusable CUDA events ready to be re-recorded by the next forward. `_make_ubatch_metadata` can then request reusable contexts from the wrapper instead of creating new event objects each time, preserving the existing context-manager enter/exit protocol used by `_capture_ubatches` and `_run_ubatches` while removing repeated Python and CUDA event allocation from the DBO hot path.

**Novelty rationale.**

The deep-research proposal removes full CUDA graph recapture for compatible buckets via `cudaGraphExecUpdate`; it does not address per-eager-forward construction of `UBatchContext` synchronization primitives. Agent A's proposal reuses worker threads and preallocates result slots, but explicitly keeps the existing `UBatchContext` handshake semantics and still passes fresh per-call metadata/contexts into those workers. This proposal targets a separate allocation cost in the same threaded handoff path: reusing the CPU/GPU event objects and mutable context shells themselves.

---
