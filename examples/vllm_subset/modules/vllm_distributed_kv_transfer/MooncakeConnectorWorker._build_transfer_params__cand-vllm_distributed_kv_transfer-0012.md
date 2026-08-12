# MooncakeConnectorWorker._build_transfer_params

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py) (lines 1424–1614)
- **Symbol:** `MooncakeConnectorWorker._build_transfer_params`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_distributed_kv_transfer-0012`

## Description
Builds Mooncake RDMA transfer descriptors for producer-to-consumer KV movement, grouping contiguous block runs and emitting either one descriptor per run or one descriptor per block.

## Current approach
Coalescing is all-or-nothing through _can_coalesce_block_transfers and only applies when source/destination offsets are zero and transfer lengths match full region block lengths. Otherwise descriptor creation falls back to per-block entries inside nested request/group/region loops.

## Estimated impact explanation
For Mooncake P/D reuse, descriptor setup and RDMA efficiency are on the turn-2 KV fetch path, so reducing descriptors can lower TTFT and improve producer throughput under parallel sessions.

## Evolve rationale
Descriptor count drives RDMA initiation overhead and metadata pressure. Relaxed scatter-list coalescing for partial-region runs, region-aware ordering, or splitting only at hardware-relevant boundaries can reduce descriptors while preserving byte ranges. Oracle: tests/v1/kv_connector/unit/test_mooncake_connector.py, hybrid/mamba tests, and Mooncake integration tests assert the same source/destination regions and byte-correct KV transfer.

## Deep research proposals

### 1. Relax coalescing to fuse partial-region contiguous runs into scatter-list batch descriptors
- **Finding:** `find-vllm_distributed_kv_transfer-0005` — *Transfer Engine*
- **Source URL:** <https://aionw.github.io/design/transfer-engine/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `MooncakeConnectorWorker._build_transfer_params` (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1424-1614), replace the all-or-nothing `_can_coalesce_block_transfers` gate with a scatter-list-friendly batching strategy modeled on Mooncake Transfer Engine's `BatchTransfer` for non-contiguous ranges. Concretely: (1) always coalesce a contiguous block run into a single descriptor whose length is `transfer_len * len(group_local_block_id)` whenever the source and destination strides equal `local_region.block_len` and `remote_region.block_len` respectively, even when `src_region_offset`/`dst_region_offset` are non-zero or `transfer_len` is less than the full region block length — the byte ranges emitted per block are still contiguous across blocks because each block advances by exactly one block-length stride. Only fall back to per-block descriptors when strides differ (e.g. differing local/remote block_len ratios that break stride uniformity) or when the plan reports partial-window semantics that require gaps. (2) For the remaining truly non-contiguous cases (e.g. remote replica selection producing gaps within a run), assemble a single `BatchTransfer`-style scatter list per (request, region) rather than one descriptor per block, so RDMA initiation cost is amortized across the batch rather than per-block. Keep the existing group_index/region alignment, `_get_sender_transfer_plan` short-circuit, and the assertions on `src_region_offset + transfer_len <= local_region.kv_block_len` unchanged so byte-correctness is preserved. Verify with the existing `tests/v1/kv_connector/unit/test_mooncake_connector.py`, hybrid/mamba tests, and the Mooncake integration suite named in the candidate's oracle.

**Proposal rationale.**

The candidate's current coalescing predicate rejects any contiguous run where `src_region_offset != 0`, `dst_region_offset != 0`, or `transfer_len != region.block_len`, forcing one descriptor per block on the common partial-region paths that arise from TP-replica selection and non-full-page transfers. The Transfer Engine documentation explicitly models `BatchTransfer` over non-contiguous ranges and highlights amortized initiation via batched scatter lists, which directly addresses the descriptor-count and RDMA initiation overhead the candidate identifies as the TTFT/turn-2 KV-fetch bottleneck. Stride uniformity within a `group_concurrent_contiguous`-produced run is a sufficient (and locally checkable) invariant to safely widen coalescing, so the change transfers a concrete idea from the finding — batched scatter-list transfers over non-contiguous ranges — into a byte-preserving reduction in descriptors on precisely the multi-turn agentic P/D path called out in the caller context.

---

## Agent proposals

### 1. Fuse contiguously-registered regions (K/V halves, adjacent layers) into per-block-run cross-region descriptors
- **Agent:** claude

**Detailed description.**

In `MooncakeConnectorWorker._build_transfer_params` (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1424-1614), add a pre-pass that groups the parallel `local_regions`/`remote_regions` lists into *super-regions* whose member regions are contiguous in memory on BOTH sides and share identical `block_len`, `kv_block_len`, `group_index`, and the same `_get_sender_transfer_plan(...)` result. Concretely: sort a stable index list of region pairs by `(group_index, local_region.base_addr)`; walk it and open a new super-region whenever the next pair's `local_region.base_addr != prev.local_region.base_addr + prev.local_region.kv_block_len` OR the corresponding condition fails on the remote side OR any of the above metadata differ. This trivially fuses the two halves emitted by `_expand_transfer_regions` for `split_kv_region=True` layouts (base_addr and base_addr+kv_block_len at lines 174-194) into one super-region of length `2*kv_block_len`, and also fuses adjacent layers whenever the KV allocator lays them out back-to-back (the common case for a single `torch.empty` pool). Then run the existing region-inner loop on super-regions instead of raw regions, treating `super.block_len == member.block_len` (unchanged — this is the per-block stride) and `super.kv_block_len == sum(member.kv_block_len)` so that a full-region transfer (`src_region_offset == 0`, `transfer_len == super.kv_block_len`) coalesces into one descriptor that carries R members × N contiguous blocks in a single RDMA WQE. Fall back to the current per-region emission for any super-region that turns out to be size 1 (non-contiguous memory) or when `_get_sender_transfer_plan` returned differing offsets/lengths across members. Preserve group_index alignment and all existing assertions on `src_region_offset + transfer_len <= member.kv_block_len` by validating them against member metadata during the fusion pre-pass, not against the fused super-region. Verify byte-correctness with `tests/v1/kv_connector/unit/test_mooncake_connector.py`, hybrid/mamba tests, and the Mooncake integration suite; add a targeted unit test that registers two adjacent layers and asserts descriptor count drops by ~R× compared to the pre-fusion path.

**Novelty rationale.**

The existing deep_research_proposal (find-0005) coalesces *within* a single region — it widens `_can_coalesce_block_transfers` and, for genuinely non-contiguous runs inside one region, proposes a BatchTransfer scatter list per (request, region). It never touches the outer `for local_region, remote_region in zip(local_regions, remote_regions)` loop, so even after that proposal lands, R separate descriptor sets are still emitted per block-run for the R contiguously-registered regions (K/V halves and adjacent layers). This proposal operates on an orthogonal axis: it fuses across regions before the inner emission runs, exploiting the concrete memory layout produced by `_expand_transfer_regions` (base_addr and base_addr+kv_block_len are literally adjacent) and the typical single-pool allocation of successive layers. The two changes compose — find-0005 reduces descriptors by a factor of `blocks_per_run`, this reduces them by an additional factor of R (regions_per_super) — and neither subsumes the other.

---

### 2. Add a final adjacency merge over emitted transfer descriptors
- **Agent:** codex

**Detailed description.**

In `MooncakeConnectorWorker._build_transfer_params` (`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1424-1614`), add a small post-processing pass before returning `src_ptrs`, `dst_ptrs`, and `lengths` that merges already-emitted descriptors when their byte ranges are exactly adjacent on both sides: `prev_src + prev_len == src` and `prev_dst + prev_len == dst`. Build descriptor triples during the existing loops, then either preserve emission order and merge adjacent triples as they are appended, or perform a stable sort by `(src_ptr, dst_ptr)` within the single `remote_session` before merging if Mooncake batch write ordering is confirmed to be irrelevant. This catches boundaries the current local coalescing cannot see, especially request boundaries where two ready requests happen to use adjacent local and remote cache blocks, and boundaries introduced by conservative per-block fallback. Keep the merge byte-exact: do not merge overlapping ranges, do not merge if either side has a gap, and leave `err_reqs`/`err_msg` handling unchanged. Add a targeted unit test that constructs two ready requests with adjacent local and remote block ids and asserts the returned descriptor count drops from two descriptors to one while preserving the same total byte count and start/end addresses.

**Novelty rationale.**

The deep_research_proposal reduces descriptors within a single `(request, region)` contiguous block run and adds scatter-list batching for non-contiguous cases. Agent A fuses adjacent registered regions before the inner loop. This proposal operates after descriptor emission and across request or fallback boundaries, merging only byte-adjacent descriptor triples that remain separate even if both earlier proposals are implemented. It is therefore orthogonal: it does not relax `_can_coalesce_block_transfers`, does not introduce scatter-list descriptors, and does not fuse K/V or layer regions.

---
