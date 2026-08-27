# FlashAttentionMetadataBuilder.build

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flash_attn.py`](vllm/v1/attention/backends/flash_attn.py) (lines 479–762)
- **Symbol:** `FlashAttentionMetadataBuilder.build`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_attention-0002`

## Description
Builds per-step FlashAttention metadata, including AOT scheduler metadata, DCP lengths, cascade tensors, multimodal prefix ranges, and R-SWA buffers.

## Current approach
Runs a sequential Python-side build each step. The nested schedule() closure can recompute scheduler_metadata for repeated shapes; cascade paths allocate cu_prefix_query_lens and prefix_kv_lens tensors; multimodal and R-SWA paths perform staging-buffer copies into device buffers.

## Estimated impact explanation
For multi-turn decode this runs every model step and directly contributes to median TPOT; for prefills it also contributes to TTFT. Reducing Python work, allocation, and H2D preparation compounds across generated tokens.

## Evolve rationale
This method is on the per-token metadata path. Concrete tunables include caching schedule() results by repeated shape, reusing cascade prefix tensors, batching H2D staging copies, and skipping inactive subpaths earlier. Correctness oracle is existing FlashAttention metadata tests and end-to-end token/output equality against the current implementation.

## Deep research proposals

### 1. Make cascade dispatch shape-aware and cache prefix scheduler metadata in FlashAttentionMetadataBuilder.build
- **Finding:** `find-vllm_v1_attention-0001` — *Cascade Inference: Memory Bandwidth Efficient Shared Prefix Batch Decoding*
- **Source URL:** <https://flashinfer.ai/2024/02/02/cascade-inference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flash_attn.py lines 564-662, refine the cascade branch of FlashAttentionMetadataBuilder.build so it (1) gates cascade dispatch on a shape-aware policy rather than the coarse `use_cascade = common_prefix_len > 0` predicate, and (2) reuses cached prefix-side artifacts across steps whose (num_actual_tokens, common_prefix_len, num_reqs, causal, sliding_window, dtype) tuple is unchanged. Concretely: (a) introduce a small policy helper that decides between the non-cascade path and the cascade two-pass path based on common_prefix_len, num_actual_tokens, num_reqs, and max_query_len — the intent being that cascade fires only when the shared-prefix multi-query kernel + per-request batch-decode kernel + state-merge is expected to beat the single-kernel path (short shared prefixes or tiny batches should stay on the single path). (b) Cache the prefix-side `cu_prefix_query_lens` and `prefix_kv_lens` int32 tensors (currently allocated fresh at lines 639-644 on every step) inside the builder as persistent buffers indexed by a `(num_actual_tokens, common_prefix_len)` key, and update the two device scalars in place with non_blocking H2D copies instead of allocating new tensors. (c) Memoize the result of the nested `schedule()` closure (lines 536-562) inside the cascade branch by hashing (batch_size, max_query_len, max_seq_len, causal, qkv_dtype, num_splits, aot_sliding_window, cache pointer identity of the seqlens tensor) so that `prefix_scheduler_metadata` at line 647 is reused across steps with repeated shared-prefix shapes — the shared prefix in multi-turn agentic workloads is exactly where this cache hits. (d) Leave a hook (a comment plus a placeholder branch) for extending the current single-level cascade to multi-level shared prefixes, where more than one prefix length is passed and the state-merge is chained; do not implement multi-level dispatch in this change, but structure the cached prefix tensors as a list to permit it. Preserve exact numerical behavior on the non-cascade path and on cascade batches whose shape does not repeat; guard with existing FlashAttention metadata tests and end-to-end token/output equality against the current implementation.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out `schedule()` recomputation for repeated shapes and cascade prefix-tensor allocation as concrete tunables, and its cascade branch (lines 638-662) is precisely the code path the finding describes: a two-pass split into shared-prefix multi-query attention and per-request suffix decode attention, followed by state merging. The finding's specific contribution — that cascade selection should be more shape-aware and could be extended toward multi-level shared prefixes — targets a gap in this build: cascade currently activates on any positive `common_prefix_len` regardless of whether the two-pass path is actually cheaper, and there is no reuse of the prefix-side scheduler metadata or prefix cu_seqlens/kv_lens tensors across steps even though multi-turn agentic workloads (the caller's stated workload) hit the identical shared-prefix shape on every decode step. Together these changes reduce Python-side work and per-step allocation on the metadata path each generated token traverses, which directly maps to the candidate's stated median TPOT / TTFT objective without touching the kernel-selection contract used downstream in forward().

---

### 2. Cache cascade planning artifacts across steps in FlashAttentionMetadataBuilder.build
- **Finding:** `find-vllm_v1_attention-0004` — *flashinfer.cascade*
- **Source URL:** <https://docs.flashinfer.ai/api/cascade.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flash_attn.py FlashAttentionMetadataBuilder.build (lines 479-762), introduce a small shape-keyed cache for cascade planning outputs so that when the (num_reqs, common_prefix_len, block_table shape, page size, cascade config) key is stable across steps, the cascade auxiliary tensors (cu_prefix_query_lens, prefix_kv_lens, suffix_kv_lens) and the associated scheduler_metadata returned by the nested schedule() closure are reused instead of being reallocated and recomputed. Store the cached entries against preallocated CUDA-graph-friendly buffers already owned by the builder so that per-step work becomes an in-place update (or a no-op copy) rather than fresh allocation plus planning. Invalidate the cache when any of the keying shapes change or when cascade is disabled for a step; keep the fallback to the current path when the cache misses. Mirror the caching only for the cascade branch and the schedule() closure inside build; do not alter multimodal or R-SWA staging paths in this change.

**Proposal rationale.**

The finding documents that FlashInfer's cascade wrappers explicitly reuse auxiliary planning data structures across multiple decode attention calls, which is directly analogous to the per-step planning work FlashAttentionMetadataBuilder.build performs on the cascade path and inside schedule(). The candidate's evolve_rationale already calls out caching schedule() results by repeated shape and reusing cascade prefix tensors as concrete tunables, and in a multi-turn agentic decode workload the cascade shape key is typically stable across steps, so a shape-keyed reuse strategy targets exactly the median TPOT hot path this method sits on without changing numerical behavior (correctness is guarded by existing FlashAttention metadata tests and end-to-end token equality).

---

### 3. Coalesce per-step H2D metadata copies and eliminate cascade tiny-tensor allocations in FlashAttentionMetadataBuilder.build
- **Finding:** `find-vllm_v1_attention-0006` — *CUDA C++ Best Practices Guide*
- **Source URL:** <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Apply the CUDA C++ Best Practices guidance on H2D transfers ("batching many small transfers into one larger transfer performs significantly better") to FlashAttentionMetadataBuilder.build in vllm/v1/attention/backends/flash_attn.py (lines 479-762). Concretely: (1) On the cascade path (lines 639-644), replace the two per-step torch.tensor([...], device=self.device) allocations for cu_prefix_query_lens and prefix_kv_lens with a single persistent pinned-CPU staging buffer + persistent device buffer owned by the builder; write [0, num_actual_tokens] and [common_prefix_len] into two slots of one pinned tensor, issue a single non_blocking copy_ into the device buffer, and slice views for cu_prefix_query_lens and prefix_kv_lens. This removes two small H2D transfers and two device allocations per step whenever cascade is active. (2) Consolidate the mm_prefix_query_ranges copy (lines 740-743) and the R-SWA prefix-lens copy (lines 755-758) into a single coalesced staged transfer per step: pack both regions into one pinned staging buffer (or perform them back-to-back into slices of one persistent device staging buffer) so they share transfer setup/latency, matching the guide's "batch small transfers" recommendation. (3) Ensure the persistent buffers backing self._dcp_context_kv_lens, self.mm_prefix_query_ranges_cpu, and self.persistent_rswa_prefix_lens are pinned memory so their non_blocking copies actually overlap with device work; audit their allocation sites and add pin_memory=True where missing. Preserve the existing FlashAttentionMetadata shape and semantics; validate with existing FlashAttention metadata tests and end-to-end token/output equality.

**Proposal rationale.**

The finding directly addresses two concrete inefficiencies in build's per-step path that the candidate's evolve_rationale explicitly flags: (a) cascade paths allocate cu_prefix_query_lens and prefix_kv_lens tensors via torch.tensor with device=self.device, which triggers an implicit H2D transfer plus a fresh device allocation on every step cascade is active; and (b) mm_prefix and R-SWA both perform independent H2D staging copies. The CUDA Best Practices Guide's "batch many small transfers" and "use pinned memory for asynchronous copies" recommendations map 1:1 onto these sites. For a multi-turn agentic workload where build runs every decode step, cutting even a handful of small H2D launches per step reduces per-token host overhead and improves median TPOT; on prefill steps (where cascade and mm_prefix are more likely active) it also reduces TTFT. The change is transferable, scoped to build, and correctness is checkable via the existing metadata tests and end-to-end equality oracle called out in the candidate.

---

### 4. Cache CUDA-graph-compatible plans keyed by stable batch descriptors in FlashAttentionMetadataBuilder.build
- **Finding:** `find-vllm_v1_attention-0009` — *CUDA Graphs*
- **Source URL:** <https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flash_attn.py (FlashAttentionMetadataBuilder.build, lines 479-762), tighten the fast-plan and per-step metadata construction around a batch descriptor that captures the CUDA-graph-relevant shape (uniform-decode vs mixed/prefill, batch size, max_seq_len bucket, page/block layout, cascade on/off, R-SWA on/off, multimodal-prefix presence). Use this descriptor as a cache key for the nested schedule() closure's scheduler_metadata result and for the cascade cu_prefix_query_lens / prefix_kv_lens tensors, so that CUDA-graph-compatible decode routes (uniform-decode batches replayed across steps) reuse a single planned metadata object instead of recomputing it every step. Mixed/prefill batches whose descriptor does not match a cached entry fall back to the current full build path, mirroring the doc's full-vs-piecewise dispatch pattern. Skip inactive subpaths (cascade, multimodal-prefix, R-SWA) early when the descriptor indicates they are unused, and batch remaining H2D staging copies for the paths that are active.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names caching schedule() by repeated shape and reusing cascade prefix tensors as concrete tunables, and its hot path is exactly the per-step decode metadata build that the CUDA-graph design targets. The finding contributes the specific structuring idea of a batch-descriptor-driven dispatcher with a graph-compatible fast path and a safe fallback, which maps directly onto separating uniform-decode replays (dominant in multi-turn agentic TPOT) from mixed/prefill builds (TTFT). This addresses the gap that the current build recomputes scheduler_metadata and reallocates cascade tensors even when successive decode steps share the same shape, which is the common case under CUDA graphs.

---

## Agent proposals

### 1. Pipeline AOT scheduler_metadata across steps via a helper-thread build so build() returns before get_scheduler_metadata finishes
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/flash_attn.py, restructure FlashAttentionMetadataBuilder.build (lines 479-762) so that the AOT `get_scheduler_metadata` call inside the nested `schedule()` closure (lines 544-561) is issued asynchronously on a helper thread bound to a dedicated non-default CUDA stream, and consumed by the *next* step's build rather than the current one. Concretely: (1) Add two instance fields on the builder, `self._pending_sched` (a Future-like handle produced by the helper thread) and `self._pending_sched_key` (the shape/dtype key that produced it), initialized to None. (2) On entry to build(), before the cascade/DCP/non-cascade branch selection at lines 564-671, compute the current step's `sched_key = (num_reqs, max_query_len, max_seq_len, causal, qkv_dtype, aot_sliding_window, max_num_splits, id(seq_lens))`. If `self._pending_sched` is set and `self._pending_sched_key == sched_key`, await the helper's result (a Tensor already resident on device, filled by `get_scheduler_metadata` on the helper stream) and use it in place of the synchronous `schedule()` call for the non-cascade branch at line 664. Insert a `torch.cuda.current_stream().wait_stream(self._sched_stream)` before consuming so the compute stream cannot dequeue the metadata tensor before the helper's H2D/planning writes have retired. (3) Immediately after the current step's metadata is assembled (right before `return attn_metadata` at line 762), *speculatively schedule* the next step's build for the uniform-decode case: for a uniform-decode batch of size num_reqs where every request advances by one token, the next step's inputs are derivable in-place — next_seq_lens = seq_lens + 1 elementwise on the persistent GPU seq_lens buffer, next_max_seq_len = max_seq_len + 1, cu_query_lens is unchanged. Enqueue a `self._sched_executor.submit(...)` job (a single-worker ThreadPoolExecutor owned by the builder) that (a) sets the helper CUDA stream as current via `torch.cuda.stream(self._sched_stream)`, (b) makes the helper stream wait on the compute stream (`self._sched_stream.wait_stream(torch.cuda.current_stream())`) so its inputs are safe to read, and (c) calls `get_scheduler_metadata(...)` writing into a preallocated persistent device buffer of shape self.scheduler_metadata (already owned by the builder for full-CUDA-graph mode; extend to always-allocated). Record the resulting handle in `self._pending_sched` under `self._pending_sched_key = next_sched_key`. (4) Invalidate `self._pending_sched` on any of: fast_build=True, VLLM_BATCH_INVARIANT set, cascade path taken, DCP path taken, or when the observed `sched_key` on the following step does not match `self._pending_sched_key` — in the mismatch case, drop the speculation, do not await it (fire-and-forget; the helper's writes into the persistent buffer are harmless), and fall through to the synchronous `schedule()` call. (5) At builder shutdown, join the ThreadPoolExecutor and destroy the helper stream. Correctness oracle: existing FlashAttention metadata tests and end-to-end token/output equality against the current implementation on both cascade and non-cascade decode batches; add a test that alternates uniform-decode and mixed-prefill batches to exercise the speculation-miss fallback. Preserve exact numerical behavior — the helper thread computes the *same* `get_scheduler_metadata` call the current build would have computed, so success paths are bit-identical.

**Novelty rationale.**

None of the four listed deep_research_proposals move `get_scheduler_metadata` off the critical path via cross-step pipelining. Findings 0001 and 0004 propose *shape-keyed caching* of the cascade branch's `prefix_scheduler_metadata` and cascade prefix tensors — they only hit when the same shape recurs, and they still execute synchronously inside the current step on the first occurrence and after any invalidation. Finding 0009 introduces a batch-descriptor cache for CUDA-graph-compatible replay but likewise operates synchronously within one step and targets the descriptor-keyed cache lookup, not asynchronous overlap with the model forward. Finding 0006 targets H2D transfer batching and pinned buffers, not compute latency of the scheduler planner itself. This proposal is orthogonal: it addresses the *latency of the planner call*, not its allocation cost or its cacheability, by exploiting that in a uniform-decode multi-turn agentic workload the next step's shape is deterministically derivable one step in advance, so the planner call can execute concurrently with the current step's model forward on a helper stream and be awaited (as a cheap stream wait) at the next build() entry. This composes cleanly with all four listed cache-based proposals — a cache hit short-circuits before the speculation is consumed — and directly attacks median TPOT beyond what cache reuse alone can achieve, since it removes wall-clock time from the per-token critical path even on cache misses within the class of uniform-decode transitions.

---
