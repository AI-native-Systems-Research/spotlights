# _compute_slot_mapping_kernel

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/block_table.py`](vllm/v1/worker/block_table.py) (lines 318–373)
- **Symbol:** `_compute_slot_mapping_kernel`
- **Kind:** kernel
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0017`

## Description
Computes old-runner KV slot_mapping entries from query_start_loc, token positions, block-table rows, block size, and context-parallel rank, while padding the CUDA-graph tail with PAD_SLOT_ID.

## Current approach
The Triton kernel uses one program per request plus one tail-padding program. Each request program loops over its token range with a fixed BLOCK_SIZE supplied by BlockTable.compute_slot_mapping, performs integer division/modulo for virtual block offsets, indirectly loads block_numbers, applies context-parallel locality masking, and stores slot IDs.

## Estimated impact explanation
The kernel runs every step and scales with scheduled tokens and KV-cache groups. Optimizing it mainly reduces per-step overhead and TPOT jitter, with the largest effect under full CUDA graph padding, context parallelism, or high token-count prefills.

## Evolve rationale
Slot mapping is rebuilt every forward and is on the critical path between input prep and attention. Headroom is in tuning the BLOCK_SIZE launch parameter, specializing the CP-disabled common case to remove locality arithmetic, reducing padding work when num_tokens equals max_num_tokens, and using vector-width/alignment assumptions for block-table loads. Correctness oracle: exact slot_mapping equality, including PAD_SLOT_ID placement and context-parallel interleave behavior, in block-table tests and downstream attention logits parity in GPU model-runner tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Reshape grid to token-chunk-parallel with per-chunk request lookup, removing long-request serialization and the dedicated padding program
- **Agent:** claude

**Detailed description.**

Change `_compute_slot_mapping_kernel` (vllm/v1/worker/block_table.py:318-373) and its launcher in `BlockTable.compute_slot_mapping` (lines 141-164) so the Triton grid is `(cdiv(max_num_tokens, BLOCK_SIZE),)` — one program per fixed BLOCK_SIZE chunk of the *output* slot_mapping — instead of the current `(num_reqs + 1,)` per-request grid. Each program: (1) computes its global token offset range `[base, base + BLOCK_SIZE)`, (2) for tokens beyond `num_tokens` writes `PAD_ID` and returns (this absorbs the special tail-padding program — line 336-345 — into the same uniform loop body), (3) for valid tokens, locates the owning request via a small in-kernel binary search over `query_start_loc_ptr[:num_reqs+1]` (vectorized with `tl.arange`+`tl.load` over a static `BS_BSEARCH = next_pow2(max_num_reqs)` upper bound, masked) to derive `req_idx` and `row_offset = req_idx * block_table_stride` per lane, (4) executes the existing arithmetic at lines 350-372 (positions→virtual block index, indirect block-table load, CP locality mask, slot id computation) using the per-lane `row_offset`. To bound binary-search cost, also pass an auxiliary `chunk_to_req` int32 tensor of length `cdiv(max_num_tokens, BLOCK_SIZE)` precomputed each step on the GPU side from `query_start_loc` via a tiny Triton helper (or torch.searchsorted, since `query_start_loc` is already on the device): each program loads `req_idx_lo = chunk_to_req[pid]` and `req_idx_hi = chunk_to_req[pid+1]`, narrowing the per-lane search range to ≤2 candidate requests in the typical case so the per-lane lookup becomes O(1) with a tiny mask compare instead of a real log search. Update the launcher to compute the chunk count and the auxiliary tensor, and keep the existing `BLOCK_SIZE=1024` constexpr. Result: long prefill requests are spread across many programs instead of one, the padding region becomes regular work (no separate program walking it serially), and per-program work is uniform — improving occupancy on mixed batches typical of agentic multi-turn traffic where one long context turn would otherwise hog a single program on the input-prep critical path. Correctness oracle (already cited in evolve_rationale) is unchanged: exact slot_mapping equality including PAD_SLOT_ID placement and CP interleave behavior.

**Novelty rationale.**

The candidate has no listed deep_research_proposals, so the only baseline to differentiate from is the evolve_rationale's optimization axes: tuning the BLOCK_SIZE launch parameter, specializing the CP-disabled common case to remove locality arithmetic, reducing padding work when num_tokens==max_num_tokens, and using vector-width/alignment assumptions for block-table loads. All four keep the existing per-request 1D launch geometry intact and merely tune the body of each request's program or the constexpr launch tile. This proposal instead restructures the launch *grid itself* — moving from per-request to per-output-chunk parallelism with an auxiliary chunk→request index — which is an orthogonal change that targets a different bottleneck (serialization within a single long request and load imbalance across SMs when request lengths are skewed) rather than per-token arithmetic or padding constant-work. It also subsumes the dedicated tail-padding program into the uniform grid as a side effect, which is a structurally different mechanism than 'reduce padding work when num_tokens==max_num_tokens'.

---

### 2. Fuse KV-cache groups into one grouped slot-mapping launch
- **Agent:** codex

**Detailed description.**

Change `MultiGroupBlockTable.compute_slot_mapping` in `vllm/v1/worker/block_table.py` to launch a group-aware variant of `_compute_slot_mapping_kernel` once with a 2D grid `(num_kv_cache_groups, num_reqs + 1)` instead of looping over `self.block_tables` and launching one kernel per group. Store small device metadata arrays on `MultiGroupBlockTable` for each group's block-table pointer, slot-mapping pointer, row stride, and effective kernel block size, refresh them when the tables are constructed or reallocated, and have the kernel select `group_id = tl.program_id(0)` before running the existing per-request body. This preserves the current per-request token loop and tail-padding semantics, but amortizes launch overhead and makes the grouped work visible to the scheduler as one launch. Validate by comparing every group's `slot_mapping.gpu` against the current implementation for mixed block sizes, hybrid block splitting, padding tails, and context-parallel ranks.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes changing the launch geometry within a single group from per-request to output-token chunks, plus absorbing tail padding. This proposal keeps the per-request mapping intact and instead fuses the outer KV-cache-group loop in `MultiGroupBlockTable`, targeting repeated kernel launches and duplicated tail-padding launches across groups. It is also distinct from the candidate's listed tuning ideas around `BLOCK_SIZE`, CP-disabled arithmetic specialization, padding elision when fully occupied, and vector/alignment assumptions.

---
