# BlockTable and MultiGroupBlockTable row mutation methods

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/block_table.py`](vllm/v1/worker/block_table.py) (lines 138–355)
- **Symbol:** `BlockTable and MultiGroupBlockTable row mutation methods`
- **Kind:** region
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0011`

## Description
CPU-side block-table row append/add/clear/move/swap operations used while applying scheduler block allocations to one or more KV-cache groups.

## Current approach
BlockTable.append_row performs per-row NumPy writes and hybrid block expansion; MultiGroupBlockTable fans row operations out through Python loops across KV-cache groups.

## Estimated impact explanation
Request churn makes this O(active changes x KV groups) host work visible in TTFT and TPOT; batching removes repeated Python calls for hybrid and multi-group models.

## Evolve rationale
The append_row/add_row fanout and per-group row mutation are concrete targets. Batched row updates across requests and groups can preserve num_blocks_per_row and block table invariants covered by block-table unit tests and compute_slot_mapping behavior.

## Deep research proposals

### 1. Batch block-table row mutations across requests and KV groups (MRV2-style persistent state)
- **Finding:** `find-vllm_v1_worker-0003` — *Model Runner V2 Design Document*
- **Source URL:** <https://docs.vllm.ai/en/v0.17.0/design/model_runner_v2/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor BlockTable and MultiGroupBlockTable row mutations in vllm/v1/worker/block_table.py (lines 138-355) to follow the MRV2 persistent-state pattern: instead of applying per-request Python calls to append_row/add_row/clear_row/move_row/swap_row and fanning each call out through a Python for-loop across KV cache groups, collect all per-step row deltas (row_idx, group_idx, block_ids, op-kind) from the scheduler into a compact CPU staging buffer once per step, then apply them in a single vectorized pass. Concretely: (1) Replace the current append_row implementation that does `num_blocks_per_row[row_idx] += num_blocks` and a scalar NumPy slice-write per call with a batched routine that concatenates block_ids from all active row-appends this step, computes per-row start offsets from a vectorized cumulative sum over num_blocks_per_row deltas, and performs one np.add.at / advanced-indexed assignment into self.block_table.np. (2) Fold the hybrid-block map_to_kernel_blocks expansion into that batched path so it runs once over the concatenated block_ids rather than per row. (3) In MultiGroupBlockTable, replace the per-group Python loops in append_row/add_row/clear_row/move_row/swap_row with a single dispatch that partitions the pre-collected delta list by group and calls each BlockTable's batched applier once. (4) Analogously batch clear_row, move_row and swap_row via advanced indexing on src/tgt arrays and a single fill for the vacated regions. (5) Preserve the invariants covered by existing block-table unit tests and compute_slot_mapping, and preserve the mamba-slot-zeroing semantics of move_row. The persistent num_blocks_per_row and block_table.np arrays remain the stable per-row state (matching the MRV2 idea of stable request rows); only the update path changes from per-row-per-group Python to per-step batched.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names append_row/add_row fanout and per-group row mutation as concrete targets, and its estimated_impact_explanation attributes visible TTFT/TPOT overhead to O(active changes x KV groups) host work per step. The MRV2 finding contributes exactly the transferable idea needed to attack that overhead: keep persistent per-row state stable across steps and drive per-step updates from batched inputs rather than per-request Python calls. In multi-turn agentic workloads (the caller's workload hint), request churn amplifies the per-row-per-group loop cost on every scheduling step, so collapsing it into one vectorized NumPy update per step and one partitioned dispatch across groups directly addresses the median TTFT and TPOT objective. The change is confined to the candidate's file/symbol, is bounded by existing block-table invariants and unit tests, and does not require altering the compute_slot_mapping Triton kernel or the GPU-side layout.

---

## Agent proposals

### 1. GPU-resident block-table applier: fused Triton kernel replaces per-step H2D row copies
- **Agent:** claude

**Detailed description.**

Refactor BlockTable and MultiGroupBlockTable in vllm/v1/worker/block_table.py (lines 138-355) so row mutations become a GPU-resident apply pass driven by a compact ops descriptor, instead of CPU NumPy writes followed by a per-step full-row H2D copy in commit_block_table. Concretely: (1) Promote block_table.gpu and a new num_blocks_per_row_gpu (int32 CpuGpuBuffer) to authoritative state; keep block_table.np only as a diagnostic mirror updated lazily. (2) Replace the current per-call append_row/add_row/clear_row/move_row/swap_row bodies with a lightweight recorder that pushes an entry into a pinned per-step ops buffer with fields (op_kind: uint8 in {APPEND, ADD, CLEAR, MOVE, SWAP}, row_idx: int32, group_idx: int32, ids_offset: int32, num_kv_blocks: int32, src: int32, tgt: int32), plus a parallel pinned int32 block_ids buffer indexed by ids_offset. (3) Add a commit path apply_pending_ops() that performs one small H2D of the descriptor + ids buffer and launches a new Triton kernel _apply_block_table_ops_kernel(block_tables_ptrs, strides, num_blocks_per_row_ptrs, ops, ids, num_ops, blocks_per_kv_block_per_group). The kernel dispatches per op: APPEND expands hybrid ids inline using group-specific blocks_per_kv_block (folding map_to_kernel_blocks into the kernel so expanded ids never touch CPU or DRAM twice), writes into block_table.gpu[row_idx, start:start+n], and updates num_blocks_per_row_gpu[row_idx]; ADD resets the counter first; CLEAR zero-fills; MOVE copies gpu row src->tgt then zero-fills src (preserving the mamba slot-zeroing invariant called out in move_row's existing comment); SWAP exchanges rows and counts. (4) MultiGroupBlockTable becomes a thin ops recorder over group_idx; append_row/add_row/clear_row/move_row/swap_row no longer iterate BlockTables in Python — they emit one op per group into the shared descriptor and issue one kernel launch per step covering all groups. (5) Replace commit_block_table's copy_to_gpu(num_reqs) full-row transfer with apply_pending_ops(); the CPU no longer needs to hold the up-to-date block table each step. (6) num_blocks_per_row (host) is refreshed from num_blocks_per_row_gpu only when scheduler-side code reads it; keep a small cached copy for hot host reads and invalidate on op record. (7) Preserve existing compute_slot_mapping (GPU-side, unchanged) and existing block-table unit tests by keeping public method signatures and end-of-step observable state identical; add tests that interleave APPEND/MOVE/SWAP/CLEAR across groups in one step and assert byte-exact equivalence with the current implementation. Expected wins on multi-turn agentic workloads: eliminates the per-step O(num_reqs * max_num_blocks_per_req) int32 H2D transfer, removes hybrid-expansion CPU work and memory traffic, and collapses per-group Python fanout into one kernel launch per step.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_worker-0003) explicitly keeps state CPU-authoritative: it batches per-step deltas into the existing self.block_table.np via cumulative-sum offsets and advanced NumPy indexing, then relies on the unchanged commit_block_table -> copy_to_gpu(num_reqs) path to transfer the full row region every step, and it states 'does not require altering the compute_slot_mapping Triton kernel or the GPU-side layout.' This proposal targets a distinct cost axis: it moves authority to the GPU, eliminates the per-step full-row H2D copy (transferring only a compact ops descriptor), and folds the hybrid map_to_kernel_blocks expansion into a new Triton applier kernel so expanded block ids never materialize on the host. The MultiGroupBlockTable per-group Python loop is also collapsed into a single kernel launch parameterized by group_idx in the descriptor, whereas the DR proposal still does a per-group partitioned dispatch on CPU. The two ideas are orthogonal and composable (the CPU-batched staging in the DR proposal could feed this GPU applier), but this proposal is not covered by the DR proposal's scope, mechanism, or claimed non-goals.

---

### 2. Copy only dirty block-table rows during commit
- **Agent:** codex

**Detailed description.**

Add dirty-row tracking to BlockTable in vllm/v1/worker/block_table.py so CPU-authoritative row mutations do not force commit_block_table(num_reqs) to copy every active row to the GPU. Concretely: initialize a boolean or compact int32 dirty-row buffer alongside num_blocks_per_row; mark row_idx dirty in append_row/add_row/clear_row, mark both src and tgt dirty in move_row/swap_row, and clear the dirty set after commit. Change commit_block_table to copy only dirty rows that are < num_reqs, using either coalesced contiguous slice copies for clustered rows or torch index_copy_ for sparse rows from the pinned CPU tensor into block_table.gpu. Keep the existing full block_table.copy_to_gpu(num_reqs) path as a fallback when the dirty count crosses a density threshold, after clear(), or for any path that cannot prove the row set. In MultiGroupBlockTable, commit each group with its own dirty rows, preserving the current public row mutation APIs and CPU state invariants. Add tests that mutate a sparse subset of rows with append/add/clear/move/swap, call commit_block_table, and assert the GPU tensor matches the CPU tensor for dirty rows while untouched rows are not recopied under a mocked/spied copy path.

**Novelty rationale.**

The deep_research_proposal batches CPU-side mutation work but explicitly keeps the existing commit_block_table full-row transfer unchanged, so it does not address avoiding H2D copies for clean rows. Agent A addresses the H2D problem by making GPU state authoritative and applying compact op descriptors with a new Triton kernel. This proposal is a lower-risk, CPU-authoritative alternative: no GPU applier kernel, no descriptor protocol, and no lazy CPU mirror. It targets the same candidate methods by using their existing row-level mutation knowledge to maintain an exact dirty set and shrink commit_block_table traffic for multi-turn workloads where only a small fraction of active request rows change per scheduler step.

---
