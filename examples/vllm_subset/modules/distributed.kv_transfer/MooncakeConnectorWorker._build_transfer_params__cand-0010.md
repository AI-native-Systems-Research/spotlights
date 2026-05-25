# MooncakeConnectorWorker._build_transfer_params

[← distributed.kv_transfer](../distributed.kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py) (lines 1159–1324)
- **Symbol:** `MooncakeConnectorWorker._build_transfer_params`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0010`

## Description
Builds Mooncake transfer descriptors for ready decode requests by trimming HMA groups, grouping concurrent contiguous block IDs, deciding whether regions can be coalesced, and appending source pointers, destination pointers, and lengths for the transfer engine.

## Current approach
For each ready request, the method flattens per-group local and remote block IDs, checks group-count and block-count invariants, calls `group_concurrent_contiguous`, then loops over transfer regions and grouped block IDs. It coalesces only when `_can_coalesce_block_transfers(...)` says the per-block copy is identical; otherwise it emits one descriptor per block per region.

## Estimated impact explanation
Descriptor count and coalescing determine how much work the prefill worker submits for a remote decode pull. Better batching lowers transfer setup overhead and can reduce median TTFT for multi-turn remote-prefill hits, especially with HMA or heterogeneous TP.

## Evolve rationale
The descriptor packing and coalescing policy is owned in this method. It can be evolved with better grouping across requests, larger contiguous descriptor formation, cached transfer plans for repeated block patterns, or topology-aware batching while preserving the same byte ranges. Correctness oracle: for the same `MooncakeXferMetadata`, transfer regions, and ready requests, the generated `(src_ptrs, dst_ptrs, lengths, err_reqs, err_msg)` must cover exactly the same KV byte ranges; `tests/v1/kv_connector/unit/test_mooncake_connector.py` and `test_mooncake_connector_hma.py` directly exercise `_build_transfer_params` trimming, mismatch errors, and responses.

## Deep research proposals

### 1. Stage heterogeneous-TP KV slices in a ring buffer to coalesce small Mooncake transfers
- **Finding:** `find-0006` — *[Roadmap] Prefill-Decode Disaggregation Roadmap (2026 Q2)*
- **Source URL:** <https://github.com/sgl-project/sglang/issues/21703>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` at `MooncakeConnectorWorker._build_transfer_params` (lines 1159-1324), extend the descriptor-building path so that when `_can_coalesce_block_transfers(...)` returns False (typically the heterogeneous-TP / HMA-trimmed case where per-block source layouts differ across regions), the method optionally routes the affected blocks through a pre-registered GPU staging ring buffer instead of emitting one descriptor per block per region. Concretely: (1) reserve a fixed-size, RDMA-registered staging region per worker with a dynamic ring allocator that hands out N contiguous bytes at a time and reclaims them once the corresponding transfer completes; (2) before populating `src_ptrs`/`dst_ptrs`/`lengths`, gather the small per-head/per-region slices for a request's grouped block IDs into a contiguous staging slab using a fused CUDA copy (or `cudaMemcpyAsync` from each region into the slab); (3) emit a single (or a few) large descriptors pointing at the slab plus the corresponding remote destination range, falling back to the current per-block path only when the staging buffer is exhausted or the gather kernel is unavailable. The receiving side already has the equivalent destination layout, so the scatter on the decode worker is symmetric. The exact byte coverage required by the correctness oracle is preserved because the gather is over the same `(local_block_ids, remote_block_ids)` set produced by `group_concurrent_contiguous`; only the transport intermediate changes. Keep the existing coalesced fast path untouched so homogeneous-TP cases see no overhead.

**Proposal rationale.**

The candidate explicitly identifies the non-coalescable HMA / heterogeneous-TP path as the source of high descriptor counts: when `_can_coalesce_block_transfers` is False, the method emits one descriptor per block per region, which is exactly the 'many small RDMA operations dominate TTFT' regime the finding targets. The SGLang roadmap entry proposes the concrete mechanism (GPU staging buffer with ring allocator that unites small heterogeneous-TP KV head slices into one large piece) that fills this gap without changing the byte ranges transferred, matching the candidate's correctness oracle. For the stated multi-turn agentic workload with remote-prefill hits, reducing per-request descriptor count from O(num_regions * num_blocks) toward O(1) plausibly lowers transfer setup overhead and median TTFT, which is the candidate's stated impact axis.

---

### 2. Aggregate Mooncake transfer descriptors across ready requests for batched submission
- **Finding:** `find-0010` — *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*
- **Source URL:** <https://arxiv.org/abs/2510.09665>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor `MooncakeConnectorWorker._build_transfer_params` (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1159-1324) so that descriptor construction is no longer per-request-isolated. Instead of producing `(src_ptrs, dst_ptrs, lengths)` by independently looping each ready request and only coalescing within that request's contiguous block groups (via `group_concurrent_contiguous` + `_can_coalesce_block_transfers`), accumulate all eligible descriptors from every ready request first, then run a second-pass merge that joins adjacent contiguous regions across requests when their layer/region pointer arithmetic and per-block lengths match. Concretely: (1) keep the existing trimming, group-count, and HMA invariants per request to preserve the correctness oracle; (2) emit per-request descriptor segments tagged with their `(transfer_region, src_base, dst_base, stride, length)` signature; (3) after the loop, sort/merge segments whose source and destination ranges are byte-contiguous and whose per-block stride is identical, collapsing them into single longer `(src_ptr, dst_ptr, length)` triples; (4) preserve the original `err_reqs` / `err_msg` semantics so the unit tests in `tests/v1/kv_connector/unit/test_mooncake_connector.py` and `test_mooncake_connector_hma.py` still see the same trimming and mismatch behavior. The output set must continue to cover exactly the same KV byte ranges; only the descriptor count and length distribution change. Optionally cache the merged descriptor template keyed by `(num_layers, region_layout, block_size, stride)` so repeat patterns across consecutive scheduler steps in a multi-turn agentic workload skip the merge work.

**Proposal rationale.**

The candidate's current approach explicitly coalesces only within a single request's contiguous group, leaving cross-request fragmentation that inflates descriptor count submitted to the Mooncake transfer engine. LMCache's reported gain from 'batched data movement operations' is precisely the lever this method lacks: by widening the unit of batching from one request's groups to the full set of ready requests in a step, the prefill worker submits fewer, longer transfers, amortizing per-descriptor setup. This directly targets the candidate's stated impact path (descriptor count and coalescing dominate transfer setup overhead and median TTFT for remote-prefill hits in multi-turn agentic workloads), without changing what bytes are moved.

---

## Agent proposals

### 1. Pipeline descriptor construction with in-flight Mooncake submissions via streaming chunks
- **Agent:** claude

**Detailed description.**

Refactor `MooncakeConnectorWorker._build_transfer_params` (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1159-1324) and its single caller in the surrounding handler (around line 1104) so that descriptor production is overlapped with active RDMA submission instead of fully accumulating then submitting. Today the method walks every ready request, every region, and every contiguous block group to build one monolithic `(src_ptrs, dst_ptrs, lengths)` triple, and the caller only kicks `self._send_blocks` once all of it is built — meaning the NIC sits idle while the Python interpreter performs trimming, `group_concurrent_contiguous`, `_get_sender_transfer_plan`, and per-region pointer arithmetic. Concretely: (1) convert `_build_transfer_params` into an async generator (or a callback-driven streamer) that yields `(chunk_src, chunk_dst, chunk_len, chunk_owner_reqs)` whenever a tunable threshold is crossed (e.g. CHUNK_DESCRIPTORS=512, or after a request's descriptors are complete), keeping the existing per-request trimming and HMA-group invariants so the correctness oracle still holds; (2) in the calling coroutine, await the executor submission for chunk K while the generator is producing chunk K+1, using an `asyncio.Queue` of bounded depth so descriptor building cannot run away from the transfer engine; (3) preserve the existing `(err_reqs, err_msg)` semantics by attaching, with each chunk, the list of request IDs whose bytes are in that chunk — if `_send_blocks` returns non-zero for a chunk, mark exactly those request IDs as failed (matching the current all-or-nothing behaviour today applied per chunk) so `tests/v1/kv_connector/unit/test_mooncake_connector.py` and `test_mooncake_connector_hma.py` still observe the same trimming/mismatch/error responses; (4) keep coalescing inside a chunk (the existing `_can_coalesce_block_transfers` fast path is unchanged), so when the chunked variant degenerates to a single chunk (small batches) it is byte-for-byte identical to today's path. The byte ranges transferred do not change — only the time at which they start moving. For the agentic multi-turn workload, the prefill worker spends measurable wall time building descriptors for many small remote-prefill hits at scheduler-step granularity; overlapping that with NIC submission shifts the critical path off Python and onto the NIC.

**Novelty rationale.**

find-0006 routes non-coalescable heterogeneous-TP slices through a GPU staging ring buffer to reduce descriptor count; find-0010 adds a second-pass cross-request merge to collapse byte-contiguous segments. Both target the *descriptor count* axis and both keep the existing build-then-submit ordering: `_build_transfer_params` still produces one full `(src_ptrs, dst_ptrs, lengths)` triple before `_send_blocks` is called. This proposal targets a different axis — temporal overlap between descriptor construction (Python-bound) and Mooncake transfer engine submission (NIC-bound). Neither existing proposal reorders the build/submit pipeline or chunks the submission, and the staging-buffer / cross-request merging mechanisms remain compatible with this change as inner optimizations of each chunk.

---

### 2. Hoist Mooncake region transfer-plan computation out of the request loop
- **Agent:** codex

**Detailed description.**

Refactor `MooncakeConnectorWorker._build_transfer_params` in `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1159-1324` to precompute the active `(local_region, remote_region, src_region_offset, dst_region_offset, transfer_len, can_coalesce)` plans once per call before iterating `ready_reqs`. Today `_get_sender_transfer_plan(...)`, bounds assertions, and `_can_coalesce_block_transfers(...)` run for every request even though their inputs only depend on `local_regions`, `remote_regions`, and `agent_meta.remote_tp_{rank,size}`, not on `d_req_id` or `send_meta`. Build an `active_region_plans` list up front, skip non-transfer regions there, assert offsets once, and have the per-request loop only perform trimming, block grouping, and pointer emission against those precomputed plans. Preserve the current debug logging by logging from the cached plan for each request if needed, and add/adjust unit coverage to assert that homogeneous, replicated-skip, heterogeneous-TP, and HMA-trimmed cases produce identical `src_ptrs`, `dst_ptrs`, `lengths`, `err_reqs`, and `err_msg`.

**Novelty rationale.**

The deep-research proposals change descriptor shape: one uses a GPU staging ring buffer for non-coalescable slices, and the other merges descriptors across requests. Agent A changes submission timing by streaming chunks while descriptor construction continues. This proposal preserves the exact descriptor set, ordering, and build-then-submit behavior, and instead removes redundant per-request transfer-plan work inside `_build_transfer_params`, targeting Python CPU overhead without overlapping or duplicating those mechanisms.

---
