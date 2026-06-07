# SingleDirectionOffloadingHandler.transfer_async build+enqueue region

[← v1.kv_offload](../v1.kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/gpu_worker.py`](vllm/v1/kv_offload/cpu/gpu_worker.py) (lines 221–331)
- **Symbol:** `SingleDirectionOffloadingHandler.transfer_async build+enqueue region`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0003`

## Description
Per-transfer preparation and enqueue path: counts copy ops, allocates pointer and size arrays, fills them across KV-cache groups and layer refs, wraps arrays as CPU tensors, selects a stream and timing events, serializes against prior transfers, and records swap_blocks_batch.

## Current approach
Each transfer freshly np.empty's all_src/all_dst/all_sizes, loops over groups and data refs in Python, calls torch.from_numpy three times, pops or creates a CUDA stream plus two enable_timing=True events, and chains every transfer after the previous transfer's end_event via stream.wait_event before launching ops.swap_blocks_batch.

## Estimated impact explanation
This is the main owned transfer-submission path. Lowering per-transfer host work and improving copy/compute overlap can reduce median TPOT under offload pressure, while faster CPU->GPU load submission also improves TTFT for requests restored from the offloaded prefix cache.

## Evolve rationale
This region controls both submission latency and overlap. Headroom includes per-handler scratch-buffer reuse for all_src/all_dst/all_sizes, coalescing multiple queued transfers into one swap_blocks_batch, avoiding enable_timing events when metrics are not needed, and replacing the unconditional single-direction stream chain with a dependency policy that still preserves hazards. Oracles are the existing round-trip GPU-worker tests, the num_transfer_bytes invariant reported by get_finished, and ordering tests for overlapping source and destination block sets.

## Deep research proposals

### 1. Coalesce queued transfers into one swap_blocks_batch with hazard-aware pipelining
- **Finding:** `find-0004` — *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*
- **Source URL:** <https://arxiv.org/html/2510.09665v2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In SingleDirectionOffloadingHandler.transfer_async (vllm/v1/kv_offload/cpu/gpu_worker.py:221-331), replace the per-call build/enqueue path with a small batched submission stage inspired by LMCache's batched data-movement + compute/I/O pipelining design. Concretely: (1) Instead of immediately materializing all_src/all_dst/all_sizes and launching swap_blocks_batch on every transfer_async invocation, append the per-group pointer/size segments to a pending submission buffer keyed by direction (gpu_to_cpu vs cpu_to_gpu). (2) When the scheduler drains pending work (or when a soft-bound on bytes/ops is reached), flatten all queued segments into a single all_src/all_dst/all_sizes triple, wrap once with torch.from_numpy, and issue one swap_blocks_batch. Each logical job_id still gets its own end_event recorded after its slice of the batched op so get_finished's per-job num_transfer_bytes invariant is preserved. (3) Replace the unconditional 'wait on previous transfer's end_event' chain at lines 311-315 with a hazard-aware dependency policy: track the GPU block-id sets touched by in-flight transfers and only insert stream.wait_event when the new transfer's source or destination blocks intersect a prior transfer's set; otherwise let it run on a separate stream from the pool, enabling true compute/I/O pipelining as described in the LMCache control-plane design. Keep the gpu_to_cpu wait_stream(current_stream) hazard. This change affects only the build+enqueue region; the existing kv_cache_groups_data_refs traversal and compute_sub_block_ptrs calls are reused unchanged.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out coalescing queued transfers into one swap_blocks_batch and relaxing the unconditional single-direction stream chain into a hazard-preserving dependency policy as the main headroom levers. LMCache's contribution (i) names exactly these two techniques — 'batched data movement operations' and 'compute and I/O pipelining' — as what produced its measured KV-movement speedups, giving a concrete, transferable design direction rather than a restatement of current behavior (today the code batches within a single transfer_async call, but does not batch across queued calls and serializes all transfers regardless of block overlap). For the stated workload (multi-turn agentic, TTFT/TPOT-sensitive), reducing per-transfer host work and unlocking overlap between independent CPU<->GPU loads/stores directly attacks the bottleneck this candidate is positioned at.

---

## Agent proposals

### 1. Eliminate per-call host allocations and timing-event overhead in transfer_async submission
- **Agent:** claude

**Detailed description.**

In SingleDirectionOffloadingHandler.transfer_async (vllm/v1/kv_offload/cpu/gpu_worker.py:221-331), cut the fixed per-transfer host overhead that exists even when only one transfer is in flight (the typical case for CPU->GPU prefix-cache loads on the TTFT path). Two concrete changes, both confined to this build+enqueue region:

(1) Replace the three np.empty + three torch.from_numpy calls (lines 227-229 and 292-294) with a single pinned, reusable scratch buffer owned by the handler. On __init__, allocate one int64 host-pinned tensor of shape (3, max_ops) using torch.empty(..., pin_memory=is_pin_memory_available()) where max_ops is bounded by the largest transfer the handler can issue (derivable from sum(group_sizes)*sum(len(refs)) bound, e.g., max GPU blocks per call * total data refs). Expose three numpy views (all_src, all_dst, all_sizes) over rows 0/1/2 of the underlying storage via tensor.numpy()[i, :num_copy_ops]; compute_sub_block_ptrs and the all_sizes[op_idx:end_idx] = ... assignment continue to work unchanged because they only need a writable np.ndarray of the right dtype and length. The numpy views are sliced to num_copy_ops per call, and the corresponding torch slices (self._scratch[i, :num_copy_ops]) are passed straight to ops.swap_blocks_batch — eliminating both the np.empty allocations and the from_numpy wrap on every call. Because the scratch is pinned and the swap_blocks_batch op only reads the pointer arrays at kernel-launch time on the issuing stream, reuse is safe as long as we don't overwrite scratch before the previous launch is enqueued; we already complete launching synchronously in this Python frame, so a single buffer suffices for the non-coalesced path. If a future change introduces async scratch reuse, fall back to a tiny (size N, e.g. 2-4) ring of these buffers indexed by an outstanding-transfer counter.

(2) Drop enable_timing=True from the event pool unless timing is actually consumed. Today both start_event and end_event are constructed with enable_timing=True (lines 297-306) so get_finished can compute transfer.start_event.elapsed_time(transfer.end_event) for logging. CUDA timing events are materially more expensive to record than vanilla events because the driver inserts a profiler timestamp; for production runs we typically only need end_event for completion polling. Add a constructor flag `record_transfer_timing: bool` (default False, or driven by an existing logger.isEnabledFor(DEBUG) / VLLM_LOG_KV_TRANSFER_TIMES env knob) and: when False, allocate end_event with enable_timing=False, skip start_event entirely (set to None in the Transfer dataclass), and in get_finished compute transfer_time = 0.0 (or wall-clock from a Python-side time.perf_counter captured at enqueue) instead of elapsed_time. Pool selection logic at lines 297-306 stays, but the pool stores the cheaper events. This removes one event-record per transfer and halves event-pool traffic.

Oracles: existing GPU-worker round-trip tests (the same ones used in the candidate's evolve_rationale) and the num_transfer_bytes invariant from get_finished are unchanged — neither depends on np.empty identity or on enable_timing semantics. Add one regression test asserting that the scratch buffer's underlying int64 storage pointer is identical across two consecutive transfer_async calls of the same handler.

**Novelty rationale.**

The listed deep_research_proposal (find-0004) attacks cross-call coalescing into one swap_blocks_batch and replaces the unconditional stream-chain with a hazard-aware dependency policy — both of which only pay off when there are multiple queued transfers and/or non-overlapping block sets. This proposal is orthogonal: it attacks the per-call submission cost that remains even for a single in-flight transfer (the common TTFT case for restoring a prefix-cache from CPU). Specifically, find-0004 still calls torch.from_numpy three times and still records two enable_timing=True events per logical job (it explicitly preserves per-job end_event recording). The candidate's evolve_rationale lists 'per-handler scratch-buffer reuse for all_src/all_dst/all_sizes' and 'avoiding enable_timing events when metrics are not needed' as distinct levers from coalescing and stream policy; this proposal targets exactly those two and is composable with find-0004 (the pinned scratch can later back the coalesced submission's flattened buffers, and the cheaper events apply identically to per-job end events in the batched design).

---

### 2. Prioritize CPU-to-GPU restore streams over background offloads
- **Agent:** codex

**Detailed description.**

In `SingleDirectionOffloadingHandler.transfer_async` around the stream selection path, make stream creation direction-aware: CPU->GPU handlers should allocate high-priority CUDA streams, while GPU->CPU handlers should allocate low/default-priority streams. Add a handler field initialized from `torch.cuda.Stream.priority_range()` in `__init__`, then replace `torch.cuda.Stream()` with `torch.cuda.Stream(priority=self._stream_priority)` when the pool is empty. Keep the existing per-direction stream pools and ordering semantics unchanged; this only affects how independent CPU->GPU loads and GPU->CPU stores compete on CUDA work queues/copy engines. This targets the TTFT-sensitive restore path: a demand CPU->GPU prefix-cache load should not sit behind best-effort GPU->CPU offload traffic launched by the other handler. Validation can reuse the existing GPU-worker round-trip tests and add a CUDA-only stress test that enqueues a large GPU->CPU transfer plus a small CPU->GPU transfer and verifies correctness while recording the small restore latency under the priority policy.

**Novelty rationale.**

The deep_research_proposal covers cross-call coalescing and hazard-aware stream dependency policies, but it does not change CUDA stream priority or distinguish latency-critical restore streams from background store streams. Agent A covers reusable host scratch buffers and disabling timing events, but leaves stream selection at default priority. This proposal is orthogonal: it keeps the same batching, arrays, events, and dependency chain, and only changes the priority of newly created streams based on transfer direction.

---
