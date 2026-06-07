# TensorMemoryPool allocate/free/store/load staging path

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/p2p/tensor_memory_pool.py`](vllm/distributed/kv_transfer/kv_connector/v1/p2p/tensor_memory_pool.py) (lines 106–263)
- **Symbol:** `TensorMemoryPool allocate/free/store/load staging path`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0007`

## Description
Pinned host-memory buddy allocator and staging copy path used by the P2P NCCL engine when received tensors exceed the live GPU buffer threshold and need overflow storage.

## Current approach
The allocator rounds requests to powers of two, scans free lists upward, eagerly splits the first larger free block down to the requested size, and greedily merges buddies on free with a hard-coded merge-depth cap. `store_tensor` and `load_tensor` create ctypes/torch views over pinned memory and issue one blocking-looking `copy_` per tensor without size-class-aware batching or stream-aware copy scheduling.

## Estimated impact explanation
This path is used during P2P overflow staging. Better fragmentation and lower allocator/copy overhead reduce tail TTFT spikes and buffer-overflow warnings when multi-turn workloads create bursts of large live KV tensors.

## Evolve rationale
Allocator and staging policy have fixed semantics: address ranges must not overlap and staged tensors must round-trip byte-identically. Lazy merging, slab classes for common KV tensor sizes, best-fit selection, reusable typed views, or stream-aware copy scheduling can reduce allocator latency and fragmentation while preserving load/store semantics. Correctness oracle: allocated address ranges must never overlap, freeing must restore reusable capacity, invalid/double frees must still fail, and `store_tensor` followed by `load_tensor` must return a tensor equal to the original CUDA tensor.

## Deep research proposals

### 1. Add batched store_tensors/load_tensors path that packs all KV layers into one staging transfer
- **Finding:** `find-0004` — *[PD] optimize kv cache transfer directly using batch transfer*
- **Source URL:** <https://github.com/sgl-project/sglang/pull/9149>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend TensorMemoryPool (vllm/distributed/kv_transfer/kv_connector/v1/p2p/tensor_memory_pool.py:106-263) with batched companions to store_tensor/load_tensor that accept a list of CUDA tensors (or a list of (addr, dtype, shape) descriptors) covering all KV layers being staged for a single P2P overflow event. Concretely: (1) add allocate_batch(sizes) that performs all buddy allocations under a single critical section and returns a list of addresses, reusing the existing power-of-two rounding, free-list scan, and _split_block logic so address-overlap and capacity invariants are unchanged; (2) add store_tensors(tensors) that, after batch allocation, builds one list of pinned-memory torch views (via the existing ctypes.c_byte/torch.frombuffer recipe) and dispatches the H2D-side copies as a single batched operation -- e.g. torch._foreach_copy_(cpu_views, tensors) -- on a dedicated copy stream so the per-layer Python/launch overhead collapses to one batch; (3) add load_tensors(descriptors, device) that mirrors this on the D2H side by allocating one contiguous output tensor (or a pre-grouped list) and issuing a single _foreach_copy_ from pinned views to device tensors, preserving the byte-identical round-trip oracle. Keep the existing per-tensor entry points as thin wrappers over the batch API so callers on the P2P NCCL engine path can switch to packing all layers into one call without changing semantics. Allocator merge/split, double-free detection, and the MAX_MERGE_DEPTH cap remain untouched.

**Proposal rationale.**

The candidate explicitly flags the absence of size-class-aware batching and stream-aware copy scheduling in store_tensor/load_tensor, where each layer's KV tensor incurs its own allocate + frombuffer + copy_ launch. The sglang PR (find-0004) demonstrates that packing all layers' transfer descriptors into a single batch transfer -- rather than issuing per-layer executor work -- materially reduces launch/control overhead in KV transfer and improves TTFT for large multi-layer models. That is precisely the gap on this overflow staging path: a multi-turn agentic workload that triggers overflow tends to stage many same-shape per-layer KV tensors in lockstep, so amortizing the Python-side bookkeeping and collapsing N copy_ launches into one batched copy directly attacks the tail TTFT spikes called out in the candidate's impact note, while preserving the allocator's overlap and round-trip correctness invariants.

---

### 2. Make TensorMemoryPool store/load staging copies non-blocking and stream-aware
- **Finding:** `find-0008` — *Disaggregated Serving | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor the staging copy path in vllm/distributed/kv_transfer/kv_connector/v1/p2p/tensor_memory_pool.py (lines 106-263), specifically `store_tensor` and `load_tensor`, so that the host<->device `copy_` operations are issued on a dedicated CUDA copy stream with `non_blocking=True` against the pinned host buffer rather than as synchronous-looking copies on the default stream. Concretely: (1) maintain a per-pool dedicated `torch.cuda.Stream` (or a small pool of streams partitioned by size class) used solely for staging copies; (2) in `store_tensor`, perform `dst_view.copy_(src, non_blocking=True)` on that stream and record a CUDA event tied to the returned address handle so consumers can wait on completion lazily; (3) in `load_tensor`, similarly issue the H2D copy on the staging stream and return both the tensor view and a readiness event/handle, so the P2P NCCL engine's decode-side scheduling can admit work as soon as the event is signaled instead of after a blocking copy returns; (4) keep the allocator's address-range invariants and round-trip equality oracle intact by ensuring the allocator does not free or reuse a buffer until its associated event has completed. This preserves the correctness oracle (no overlap, byte-identical round-trip, double/invalid free still fails) while overlapping staging copies with ongoing GPU forward passes.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out stream-aware copy scheduling as a way to reduce overhead while preserving load/store semantics, and today `store_tensor`/`load_tensor` issue `copy_` operations without an explicit copy stream or event-based readiness signal, so callers effectively serialize on staging. The Dynamo disaggregated-serving guidance — that KV transfer should be non-blocking so GPU forward passes can continue serving other requests during the transfer — directly motivates routing the pinned-host staging copies onto a dedicated stream with event-based readiness. Under the multi-turn agentic workload and the TTFT/TPOT objectives, overlapping overflow staging copies with decode forward passes is a plausible mechanism to reduce tail TTFT spikes when bursts of large KV tensors hit the overflow path, which is exactly the gap the candidate's `current_approach` description identifies (one blocking-looking `copy_` per tensor, no stream-aware scheduling).

---

### 3. Batch and pipeline staging copies in TensorMemoryPool with stream-aware scheduling
- **Finding:** `find-0010` — *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*
- **Source URL:** <https://arxiv.org/abs/2510.09665>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify the staging path in vllm/distributed/kv_transfer/kv_connector/v1/p2p/tensor_memory_pool.py (lines 106-263), specifically `store_tensor` and `load_tensor`, to support batched, pipelined transfers rather than one blocking `copy_` per tensor. Concretely: (1) add batch entry points (e.g., `store_tensors`/`load_tensors`) that take a list of tensors plus their pre-allocated pool addresses and issue copies on a dedicated non-default CUDA stream, allowing the caller to overlap H<->D staging with compute/NCCL work; (2) coalesce adjacent or same-size-class allocations so the underlying `copy_` calls operate on contiguous pinned regions where possible, amortizing per-call overhead; (3) record a single CUDA event per batch instead of implicitly synchronizing on each tensor, exposing it to callers so they can wait only at the consumption point. Allocation semantics (power-of-two rounding, buddy splits/merges, non-overlapping address ranges) and the byte-identical round-trip oracle for store/load remain unchanged; only the copy-issuing layer and its synchronization model change. Reusable typed ctypes/torch views over pinned regions can be cached per size class to avoid rebuilding them on every call.

**Proposal rationale.**

The candidate explicitly notes that `store_tensor`/`load_tensor` issue blocking-looking `copy_` calls per tensor with no size-class-aware batching or stream-aware copy scheduling, which is exactly the gap LMCache addresses with batched KV data movement and compute/I/O pipelining. The finding provides a concrete, transferable design pattern (batch + pipeline + dedicated movement path) directly applicable to the P2P overflow staging hot path, where bursts of large KV tensors during multi-turn agentic workloads cause TTFT spikes. Applying batched/pipelined copies here can overlap host staging with GPU compute and NCCL transfers, reducing median TPOT and tail TTFT without altering allocator correctness invariants.

---

## Agent proposals

### 1. Add a slab cache for the dominant KV size class on top of the buddy allocator with cached pinned views
- **Agent:** claude

**Detailed description.**

Layer a small slab/free-list cache in front of the existing buddy allocator in vllm/distributed/kv_transfer/kv_connector/v1/p2p/tensor_memory_pool.py:106-263, targeted specifically at the one or two power-of-two size classes that dominate KV overflow staging (per-layer KV block tensors are highly uniform in shape/dtype within a model, so almost every store_tensor call rounds to the same `required_size`). Concretely: (1) add a bounded `slab_cache: dict[int, deque[MemoryBlock]]` keyed by `required_size`, plus a per-class capacity (e.g. derived from observed peak concurrency or a config knob); (2) in `allocate(size)`, after computing `required_size`, first pop from `slab_cache[required_size]` if non-empty and return that block's address without touching `free_lists` or `_split_block`; (3) in `free(addr)`, if the freed block's size is a tracked slab class and the slab is below its capacity, push it onto `slab_cache[block.size]` and skip `_merge_buddies` entirely — only fall through to the existing buddy merge path when the slab is full or the size class is untracked. This makes the steady-state hot path O(1), eliminates the split-on-allocate / merge-on-free churn that today walks `MAX_MERGE_DEPTH` levels for every uniform-size store, and reduces fragmentation because uniform KV blocks no longer get repeatedly carved out of and folded back into larger free regions. (4) Alongside each cached slab block, memoize the pinned-memory `ctypes.c_byte` buffer and a prebuilt `torch.frombuffer(...)` view for the slab's canonical dtype/shape (when `store_tensor`/`load_tensor` see a cache hit on shape+dtype, reuse the cached view instead of rebuilding `frombuffer`+`reshape`); fall back to the current dynamic view path on a miss. Slab classes can be discovered adaptively (track a small frequency table of `required_size` values seen by `allocate` and promote the top-K above a threshold) so the optimization needs no config and degrades gracefully on heterogeneous workloads. All correctness oracles are preserved: address ranges still cannot overlap (a slab-cached block remains owned by exactly one of `allocated_blocks` or `slab_cache`), `store_tensor` followed by `load_tensor` is still byte-identical (the underlying pinned region is unchanged), invalid/double `free` still raises (the existing `allocated_blocks` membership check runs before the slab path), and any block in `slab_cache` is reclaimable by the buddy allocator on demand by draining the slab back into `free_lists[block.size]` when an allocation of a different size hits `Insufficient memory`.

**Novelty rationale.**

All three listed deep_research_proposals (find-0004, find-0008, find-0010) attack the copy-issuing layer — batched store_tensors/load_tensors, a dedicated CUDA copy stream, foreach_copy_, and event-based readiness — and explicitly state that allocator semantics (power-of-two rounding, buddy split/merge, MAX_MERGE_DEPTH) remain unchanged. None of them modify the allocator policy itself. This proposal is orthogonal: it changes the allocator's hot path by short-circuiting buddy split/merge with a slab cache for the dominant KV size class, and additionally memoizes the pinned ctypes/torch views per slab block — the candidate's evolve_rationale explicitly calls out 'slab classes for common KV tensor sizes' and 'reusable typed views' as targets, neither of which is addressed by the existing proposals. It composes cleanly with any of them (a slab hit still feeds into a batched, stream-aware copy) rather than overlapping.

---

### 2. Add a consume-on-load path that frees staged pinned blocks immediately after restore
- **Agent:** codex

**Detailed description.**

Extend `TensorMemoryPool` in `vllm/distributed/kv_transfer/kv_connector/v1/p2p/tensor_memory_pool.py:222-263` with a `consume_tensor(addr, dtype, shape, device)` helper, or an explicit `free_after_load=True` mode on `load_tensor`, that validates the descriptor, copies the pinned host contents back to the target device exactly as `load_tensor` does today, and then returns the block to the buddy allocator immediately after the copy succeeds. Keep the existing `load_tensor` behavior for any caller that needs a non-consuming read. The P2P overflow receive path can then use the consuming variant when a tuple descriptor is first materialized back into a CUDA tensor, and clear or replace the descriptor in `recv_store` so the later request-finished cleanup does not double-free it. This shortens the lifetime of overflow pinned-memory allocations from request completion to first successful load, which reduces pool pressure and fragmentation during multi-turn bursts without changing address allocation, byte-identical round-trip semantics, or invalid/double-free checks.

**Novelty rationale.**

The listed deep_research_proposals focus on batching copies, using dedicated CUDA streams, event readiness, and copy pipelining; they do not change how long staged host-memory blocks remain allocated after a tensor has been restored. Agent A's proposal adds hot size-class slab caching and cached views, but also leaves staged-block lifetime unchanged. This proposal is orthogonal: it attacks pinned-pool occupancy by making `load_tensor` optionally consume the allocation, so capacity is recycled earlier even if the allocator policy and copy implementation remain exactly the same.

---
