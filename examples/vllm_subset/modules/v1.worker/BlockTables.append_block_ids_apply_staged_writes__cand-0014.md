# BlockTables.append_block_ids/apply_staged_writes

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu/block_table.py`](vllm/v1/worker/gpu/block_table.py) (lines 96–113)
- **Symbol:** `BlockTables.append_block_ids/apply_staged_writes`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0014`

## Description
Stages new KV block IDs for one request across all KV-cache groups, then applies staged writes to the GPU block-table storage and refreshes the per-group num_blocks UVA tensor.

## Current approach
append_block_ids loops over KV-cache groups and stages one row write per group. apply_staged_writes then loops over block_tables and calls apply_write once per group; the in-code TODO notes this launches one kernel per block table and should be handled by one kernel across all block tables.

## Estimated impact explanation
This is not always on the steady-state single-token path, but it is active under request churn and prefix/cache growth. Fewer kernels and staging copies reduce TTFT for new turns and TPOT jitter when many requests extend block tables.

## Evolve rationale
Block-table updates happen during new request admission and cache growth for running requests. Headroom is in fusing staged writes across KV-cache groups, packing row/start/length metadata once, and applying all group writes with one kernel or one coalesced staging operation. Correctness oracle: compare block_tables GPU contents, num_blocks, gathered block tables, and downstream slot mappings after add/update request flows in model-runner and block-table tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Fuse per-group staged block-table writes into a single multi-group Triton kernel with one packed H2D transfer
- **Agent:** claude

**Detailed description.**

Refactor `BlockTables.append_block_ids` / `apply_staged_writes` (vllm/v1/worker/gpu/block_table.py:96-113) so that staging happens once at the `BlockTables` level instead of independently per `StagedWriteTensor`. Concretely: (1) Move the four staged-write lists (indices, starts, cu_lens, contents) up from each per-group `StagedWriteTensor` into `BlockTables`, and add a parallel `_staged_group_ids` list. In `append_block_ids`, append a single (group_id, req_index, start, len) record for each non-empty `new_block_ids[i]` and extend a single contents buffer; this also lets `num_blocks.np[i, req_index]` be updated in the same loop as today. (2) In `apply_staged_writes`, perform one batched H2D using a single `UvaBufferPool` per metadata stream (group_ids, indices, starts, cu_lens) plus one `async_tensor_h2d` for the concatenated contents — replacing the current 4 H2D copies × G groups + G kernel launches with 5 H2D copies + 1 kernel launch. (3) Add a new `_apply_write_multi_group_kernel` parameterized by program index `pid` over the total number of staged writes. Each program loads `group_id`, then uses the already-cached `self.block_table_ptrs` / `self.block_table_strides` (populated by `init_block_table_layout_tensors`, lines 77-94) to indirect to the right block-table base — `_load_ptr(block_table_ptrs + group_id, tl.int32)` — and writes the slice exactly as `_apply_write_kernel` does today (lines 193-217). (4) Keep `StagedWriteTensor.apply_write` for callers that still need per-tensor staging (e.g., other consumers of `StagedWriteTensor`), but route block-table writes through the fused path. (5) Optionally co-stage the per-row `num_blocks` update on GPU inside the same kernel (writing `start + content_len` into `num_blocks.gpu[group_id, row_idx]`), removing the trailing `self.num_blocks.copy_to_uva()` from the hot path; this is a follow-on opt that can be guarded if any consumer reads `num_blocks.np` mid-step. Validation: existing block-table tests under `tests/v1/worker/` plus the model-runner add/update-request flows; correctness oracle is that `block_tables[i].gpu`, `num_blocks.np`/`.gpu`, and the downstream outputs of `gather_block_tables`/`compute_slot_mappings` match the pre-refactor implementation across multi-turn agentic workloads.

**Novelty rationale.**

The candidate has zero existing deep_research_proposals, and the in-code TODO at lines 109-110 only flags the problem ('one kernel per block table') without prescribing a solution. This proposal goes beyond restating the TODO by specifying the concrete data-structure refactor (move staging up to `BlockTables`, add a `group_ids` stream), the metadata-packing scheme that lets a single kernel dispatch across heterogeneous block-table strides via the existing `block_table_ptrs`/`block_table_strides` indirection tensors, the reduction in H2D copies from 4G to 5, and an opt-in co-staged GPU `num_blocks` update that eliminates the `copy_to_uva()` tail. None of these specifics is covered by an existing proposal.

---

### 2. Add a dirty fast path that skips block-table apply work when no block IDs changed
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu/block_table.py`, make `BlockTables` track whether `append_block_ids` actually changed block-table or `num_blocks` state since the last `apply_staged_writes`. Set a private dirty flag when `overwrite=True` or when any per-group `block_ids` list is non-empty; leave it unset for cached-request decode steps where `update_requests` has no new block IDs. Then have `apply_staged_writes` return immediately when the flag is false, avoiding the per-group Python loop over `StagedWriteTensor.apply_write()` and, more importantly, the unconditional full `self.num_blocks.copy_to_uva()` refresh on every `execute_model` call. Preserve correctness by still marking dirty for overwrite-with-empty-blocks so reused request rows publish `num_blocks=0`, and clear the flag only after the staged writes and `num_blocks` refresh complete. Validation should include a no-op decode-step unit test that monkeypatches `num_blocks.copy_to_uva` to assert it is not called without prior `append_block_ids`, plus an overwrite-empty case proving `num_blocks.gpu` is refreshed when a row is reset.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes fusing non-empty staged writes across KV-cache groups into one packed transfer and one Triton kernel, with an optional GPU-side `num_blocks` update. This proposal targets a different hot case: the current unconditional `apply_staged_writes` call from `execute_model` on steps where no block IDs were staged at all. It does not require the multi-group kernel, metadata repacking, or GPU-side `num_blocks` co-staging from Agent A; it is an independent guard that removes avoidable CPU/UVA work on unchanged decode steps.

---
