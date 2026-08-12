# postprocess_mamba_fused_kernel

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/mamba_utils.py`](vllm/v1/worker/mamba_utils.py) (lines 154–277)
- **Symbol:** `postprocess_mamba_fused_kernel`
- **Kind:** kernel
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0013`

## Description
Fused Triton kernel that updates accepted-token counts and copies Mamba/SSM/conv state blocks during spec-decode postprocess alignment.

## Current approach
Uses a grid over requests and flattened state types with COPY_BLOCK_SIZE fixed at 1024 at call sites, and the dim-first conv path loops over dim rows inside each program.

## Estimated impact explanation
This kernel is on hybrid Mamba spec-decode TPOT; better tiling and vectorization can reduce memory-copy time for recurrent states in long-context agent workloads.

## Evolve rationale
COPY_BLOCK_SIZE, the request/state grid, and conv row loop are owned kernel parameters. tests/v1/worker/test_mamba_utils.py and mamba hybrid/spec-decode tests validate alignment, block-table stride, and accepted-token behavior.

## Deep research proposals

### 1. Replace copy-based accepted-token alignment with ring-buffered input cache and matmul replay
- **Finding:** `find-vllm_v1_worker-0010` — *flashinfer.mamba.checkpointing_ssu*
- **Source URL:** <https://docs.flashinfer.ai/generated/flashinfer.mamba.checkpointing_ssu.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Investigate replacing the block-copy postprocess path in `vllm/v1/worker/mamba_utils.py::postprocess_mamba_fused_kernel` (lines 154-277) with a FlashInfer `checkpointing_ssu`-style design. Concretely: (1) maintain a small GPU-resident ring buffer of recent Mamba/SSM inputs `x_cache` per request (sized to the max draft length used in spec decode), populated during the forward pass; (2) at spec-decode postprocess, instead of computing `aligned_new_computed`, allocating a new dest block, and issuing per-state block-sized memcpys via `_copy_mamba_state_block` for every layer/state-type, replay only the accepted-token subrange from `x_cache` against the last committed state using a matmul-based parallel token processor (as `flashinfer.mamba.checkpointing_ssu` exposes); (3) keep host-owned flush bookkeeping (block table updates, `num_accepted_tokens_out`) but drive it once per request rather than in a `(num_reqs, num_layers * num_state_types)` grid of copiers. The `_copy_mamba_state_block` path would remain only as a fallback for full-block flushes when `aligned_new_computed` crosses a block boundary; the common accepted-token-bias case would be handled by the replay.

**Proposal rationale.**

The candidate's current approach launches a 2D grid of copy programs and, for each accepted-token event, moves entire recurrent/conv state blocks whose size grows with hidden dim and conv width. On hybrid Mamba spec-decode TPOT — the stated impact target — this bandwidth-bound copy is the dominant cost. The finding describes a concrete alternative already realized in FlashInfer: cache the small inputs (x) instead of the large states, and reconstruct accepted-token state via matmul replay. Matmul replay is compute-bound and scales with `num_accepted_tokens * hidden_dim`, whereas the current copy scales with `num_layers * num_state_types * block_bytes` per request, so for the long-context multi-turn agentic workload in the caller context (many layers, modest accepted counts) the replay is plausibly faster and avoids the fixed COPY_BLOCK_SIZE=1024 tuning pressure the candidate calls out. This addresses the specific gap the candidate identifies (memory-copy time for recurrent states) rather than merely re-tuning tiling.

---

## Agent proposals

### 1. Split into compacted decision prepass + 3D copy grid with per-state-type autotuned COPY_BLOCK_SIZE
- **Agent:** claude

**Detailed description.**

Restructure `postprocess_mamba_fused_kernel` in vllm/v1/worker/mamba_utils.py (lines 154-277) into a two-stage launch that stops paying for the full `(num_reqs, num_layers * num_state_types)` grid on the common no-copy / self-copy path and unlocks better tiling for the copy body itself.

Stage 1 — decision compaction (tiny 1D kernel over `num_reqs`): compute per-request `num_accepted`, `num_tokens_running_state`, `aligned_new_computed`, `needs_copy`, `dest_block_idx`, `accept_token_bias`, and the self-copy flag exactly as lines 224-256 do today, but write results into a compact work-list. Emit (a) a scalar `num_copy_reqs` via `atomic_add`, and (b) per-slot `(req_idx, bt_row_idx, src_block_idx, dest_block_idx, token_bias)` tuples into a preallocated GPU buffer for only those requests that pass `needs_copy` and are not pure self-copies. Also write the `num_accepted_tokens_out_ptr[req_idx] = 1` result for the src==dst && state_idx==0 case here (one program per request instead of `num_layers * num_state_types`).

Stage 2 — copy kernel with a 3D grid: `(num_copy_reqs, num_states, DIM_ROW_TILES)`. `program_id(0)` indexes the compacted work-list (so no wasted programs on skipped requests, no re-derivation of decision logic, and no divergence from HAS_IDX_MAPPING / PRECOMPUTED_NEW_COMPUTED branches inside the copy body). `program_id(1)` picks the (layer, state_type) as today. `program_id(2)` parallelizes the DS-conv dim-row loop currently at line 92 (`for d in range(0, dim_rows)`), replacing that sequential per-program loop with `dim_rows` cooperating programs each handling one row (falling back to `program_id(2)==0` for temporal/non-DS states via `tl.static_assert`-guarded specialization). This directly attacks the candidate's `evolve_rationale` bullet about the conv row loop and turns the current serialization into extra grid parallelism, which matters most for DS conv layers with large `dim_rows`.

Autotune the copy body per state-type family instead of the single hard-coded `COPY_BLOCK_SIZE=1024` at the three call sites (lines 796/839/892). Introduce three specialized copy entry points via `triton.autotune` on `(copy_size_u64,)` (for temporal), `(per_row_bytes,)` (for DS conv), and `(copy_size,)` (for SD conv), with configs sweeping COPY_BLOCK_SIZE ∈ {256, 512, 1024, 2048, 4096} and num_warps ∈ {2, 4, 8}. Temporal states already vectorize via uint64 (lines 129-141) but small hidden-dim / GQA-shaped SSM states are oversized at 1024; DS conv per-row copies are usually much smaller (`(conv_width - token_bias) * elem_size`) and want a much smaller tile. Cache autotune keys by state-family plus `(state_inner_size, state_elem_size, conv_width, dim_rows)` bucket so autotuning cost is paid once per model config, not per launch.

Wiring: the three existing Python call sites (postprocess and V2 align) each become `decision_prepass_kernel[(num_reqs,)](...)` followed by `copy_kernel[(num_copy_reqs, num_states, max_dim_row_tiles)](...)`. `num_copy_reqs` stays GPU-resident to preserve the no-CPU-sync property; use `triton.cdiv(max_reqs, 1)` as the grid bound with an early-return on `program_id(0) >= num_copy_reqs` loaded from the counter (same idiom as today's `if batch_idx >= num_reqs` at line 214).

Validation: extend tests/v1/worker/test_mamba_utils.py with (a) a mixed-batch case where only a subset of requests trigger `needs_copy`, asserting the copy kernel's grid usage via the work-list length, (b) a DS-conv case with `dim_rows > 1` to exercise the new `program_id(2)` parallelism, and (c) parametric assertions that the autotuned COPY_BLOCK_SIZE never breaks the byte-tail path for sub-8B slices (lines 143-150).

**Novelty rationale.**

find-vllm_v1_worker-0010 proposes eliminating the block-copy mechanism entirely by caching inputs (x) in a ring buffer and replaying accepted-token state via a FlashInfer matmul-based `checkpointing_ssu`. That is an architectural swap: the copy path becomes a fallback, and the wins come from changing what is stored (inputs, not states) and what is executed (compute-bound matmul, not memcpy). My proposal is orthogonal: it keeps the copy mechanism and directly attacks the three parameters the candidate itself flags as owned — COPY_BLOCK_SIZE (via per-state-family autotune instead of a single 1024), the `(num_reqs, num_layers*num_state_types)` grid (via a compacted work-list so only actually-copying requests launch copy programs, and per-request decision logic runs once instead of `num_layers*num_state_types` times), and the sequential DS-conv `for d in range(0, dim_rows)` row loop (via a new `program_id(2)` dim-row axis). It requires no new host-side caching, no matmul kernel, no `x_cache` allocation, and no FlashInfer dependency, and it delivers wins on the exact code path the deep_research proposal treats only as the fallback — including the DS-conv row-loop parallelism which the replay design does not address at all.

---

### 2. Write accepted-token outputs inside the fused kernel and drop the V1 pre-copy
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/mamba_utils.py::run_fused_postprocess`, remove the eager `self.num_accepted_tokens_out[:num_reqs].copy_(num_accepted_tokens_gpu[:num_reqs])` initialization and make `postprocess_mamba_fused_kernel` write `num_accepted_tokens_out_ptr[req_idx]` exactly once per request from the `state_idx == 0` program. The write should occur after the decision values are computed: store `1` when `needs_copy && src_block_idx == dest_block_idx`, otherwise store the original `num_accepted`. This preserves the existing public output contract while eliminating a separate device-to-device copy/torch op from the V1 postprocess path. Keep the V2 align snapshot path unchanged unless it is later split into a true prepass, because V2 currently reads and writes the same accepted-token buffer and cannot safely remove the snapshot without inter-program ordering.

**Novelty rationale.**

The deep_research proposal replaces state copying with an input ring buffer plus matmul replay, which is an architectural alternative and does not address the existing V1 output initialization copy. Agent A proposes a compacted decision prepass, a 3D copy grid, and COPY_BLOCK_SIZE autotuning; while that prepass could eventually own per-request bookkeeping, it does not call out removing the current `num_accepted_tokens_out.copy_` launch in `run_fused_postprocess` or the narrower low-risk kernel change that writes unchanged accepted counts from `state_idx == 0`. This proposal is therefore a distinct incremental optimization on the current fused kernel path.

---
