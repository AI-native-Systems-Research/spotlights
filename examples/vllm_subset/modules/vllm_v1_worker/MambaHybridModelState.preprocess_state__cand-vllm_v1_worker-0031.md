# MambaHybridModelState.preprocess_state

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/model_states/mamba_hybrid.py`](vllm/v1/worker/gpu/model_states/mamba_hybrid.py) (lines 160–207)
- **Symbol:** `MambaHybridModelState.preprocess_state`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0031`

## Description
New GPU runner Mamba align-mode pre-forward state migration across block boundaries.

## Current approach
Runs a per-step preprocess_mamba_align_fused_kernel launch with fixed block=256 for all requests and then invokes ctx.run_fused_precopy, relying on GPU fast-exit when no state copy is needed.

## Estimated impact explanation
The impact is specific to hybrid Mamba/SSM models, but those workloads pay this on every decode step; avoiding no-op launches or improving tiling reduces median TPOT in long agent loops.

## Evolve rationale
The unconditional per-step launch, block=256 constant, and ctx.run_fused_precopy handoff are concrete stream/kernel scheduling targets. tests/v1/worker/test_mamba_hybrid_model_state.py, tests/v1/worker/test_mamba_utils.py, and tests/kernels/mamba/test_precopy_mamba_align.py validate align-mode state migration semantics.

## Deep research proposals

### 1. Amortize preprocess_state launches across multi-step decode windows
- **Finding:** `find-vllm_v1_worker-0001` — *[RFC]: Multi-Step Scheduling*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/6854>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend MambaHybridModelState.preprocess_state (vllm/v1/worker/gpu/model_states/mamba_hybrid.py:160-207) so the align-mode pre-forward migration participates in a worker-local multi-step decode loop. Instead of unconditionally launching preprocess_mamba_align_fused_kernel with block=256 and calling ctx.run_fused_precopy on every decode iteration, gate the launch on whether the current step is a boundary step within a lookahead window: on step 0 of the window, precompute block/boundary metadata for all n lookahead steps in a single fused kernel (or a small persistent kernel that advances an on-GPU step counter), and on subsequent steps skip the Python-side preprocess_state entrypoint entirely, letting ctx.run_fused_precopy consume the pre-staged metadata. Keep the tensors resident on GPU across steps so no host sync is required, mirroring the RFC's guidance that next-step input updates should be driven by CUDA kernels rather than Torch/Python. Preserve current semantics validated by tests/v1/worker/test_mamba_hybrid_model_state.py, tests/v1/worker/test_mamba_utils.py, and tests/kernels/mamba/test_precopy_mamba_align.py by falling back to the per-step path whenever the scheduler indicates the lookahead window is exhausted or a request's block layout changes mid-window.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out the unconditional per-step kernel launch and Python-side handoff as scheduling targets, and the caller context prioritizes median TPOT on multi-turn agentic traffic. The multi-step scheduling RFC's core idea, amortizing per-step Python input prep and metadata updates across n decode steps using GPU-resident kernels, maps directly onto preprocess_state: today it pays a Python entry, a fixed-block kernel launch, and a fused precopy handoff every token, most of which are no-ops on non-boundary steps. Batching the boundary computation over a lookahead window and skipping the Python path on interior steps is a concrete, transferable application of the finding to this specific hot path without inventing new abstractions.

---

### 2. Fold align-mode precopy into SSM update via separate src/dst state indices
- **Finding:** `find-vllm_v1_worker-0011` — *flashinfer.mamba.selective_state_update*
- **Source URL:** <https://docs.flashinfer.ai/generated/flashinfer.mamba.selective_state_update.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In MambaHybridModelState.preprocess_state (vllm/v1/worker/gpu/model_states/mamba_hybrid.py:160-207), the align-mode boundary migration currently runs two per-step kernel launches: preprocess_mamba_align_fused_kernel to compute per-request src_col/token_bias, followed by ctx.run_fused_precopy which physically copies the previous window block into the new destination block before the forward. Adapt the flashinfer selective_state_update pattern of accepting separate `state_batch_indices` and `dst_state_batch_indices` so that the downstream Mamba SSM update reads from the pre-boundary source slot and writes to the post-boundary destination slot in one pass, eliminating the standalone precopy kernel. Concretely: (1) keep the lightweight index-preparation step (preprocess_mamba_align_fused_kernel) but have it emit both a src-slot and dst-slot index array per request instead of driving a copy; (2) drop the ctx.run_fused_precopy call at line 201, so on non-crossing steps preprocess_state performs no data-movement kernel at all; (3) thread the src/dst index tensors through the align-mode Mamba forward path (state_batch_indices + dst_state_batch_indices) so the SSM/conv update writes the migrated state in place. This removes both the fixed-block=256 precopy launch and its per-request GPU fast-exit branch, replacing them with additional pointer indirection inside the already-scheduled forward kernel. tests/v1/worker/test_mamba_hybrid_model_state.py, tests/v1/worker/test_mamba_utils.py, and tests/kernels/mamba/test_precopy_mamba_align.py already validate align-mode migration semantics and can be extended to cover the separate-index path.

**Proposal rationale.**

The candidate's stated evolve targets are exactly the ctx.run_fused_precopy handoff and the unconditional per-step launch with fixed block=256; the finding provides a specific, upstream-proven mechanism (dst_state_batch_indices on selective_state_update) for merging that precopy into the SSM update itself. Because a multi-turn agentic decode workload pays this migration cost on every step but only a small fraction of steps actually cross a Mamba block boundary, removing the second kernel launch — rather than relying on its GPU fast-exit — directly reduces per-decode scheduling overhead and median TPOT on hybrid Mamba/SSM models, which is the caller's stated objective.

---

## Agent proposals

### 1. Fuse preprocess and precopy Mamba-align kernels into a single launch
- **Agent:** claude

**Detailed description.**

Replace the two-kernel sequence in MambaHybridModelState.preprocess_state (vllm/v1/worker/gpu/model_states/mamba_hybrid.py:189-207) with a single fused Triton kernel that computes per-request metadata and performs the boundary-crossing state copy in one launch. Today, preprocess_mamba_align_fused_kernel (vllm/v1/worker/mamba_utils.py:281-323) writes src_col, src_off, and the advanced state_idx into _mamba_src_col_gpu / _mamba_src_off_gpu / _mamba_state_idx_gpu, and then ctx.run_fused_precopy launches precopy_mamba_align_fused_kernel to read those same buffers back and dispatch the state-copy loop. Because the metadata producer and consumer are separate kernels, they pay two launch overheads and a global-memory round-trip for src_col/src_off per request. Fuse them: emit a single Triton kernel with a 2-D grid over (num_reqs_block, state_dim_tiles) — or a cooperative kernel where each request tile first computes its (src_col, src_off, new_state_idx) in registers, writes the advanced state_idx and reset num_accepted flag (to preserve today's global-memory side effects), broadcasts src_col/src_off into shared memory, and then, in the same program, evaluates the existing fast-exit predicate (src_col < 0 or src_col == dst_col) and loops over its assigned state-dim tiles to perform the copy. Keep the _mamba_src_col_gpu / _mamba_src_off_gpu buffers only if any consumer outside preprocess_state reads them; otherwise drop them from the hot path. The Python wrapper collapses to one kernel launch replacing lines 189-207. Extend tests/v1/worker/test_mamba_hybrid_model_state.py, tests/v1/worker/test_mamba_utils.py, and tests/kernels/mamba/test_precopy_mamba_align.py with an equivalence test that pins the fused kernel's outputs against the current two-kernel path across the covered boundary/no-boundary/accepted-token scenarios.

**Novelty rationale.**

find-vllm_v1_worker-0001 amortizes preprocess_state across a multi-step decode window by skipping the Python entry and kernel launch on non-boundary interior steps — a temporal-scheduling change that still uses today's two-kernel structure on boundary steps. find-vllm_v1_worker-0011 eliminates the precopy launch by threading separate state_batch_indices / dst_state_batch_indices into the downstream Mamba SSM update, moving data movement into the forward. This proposal is orthogonal to both: it preserves per-step launches and preserves a dedicated pre-forward data-movement path, but collapses the two per-step Triton launches (preprocess_mamba_align_fused_kernel + precopy_mamba_align_fused_kernel via ctx.run_fused_precopy) into a single fused kernel, removing one launch and the src_col/src_off global-memory round-trip. Neither existing proposal proposes intra-launch fusion of the metadata-producer and copy-consumer kernels.

---

### 2. Specialize the align preprocess launch for active request count buckets
- **Agent:** codex

**Detailed description.**

In `MambaHybridModelState.preprocess_state` (`vllm/v1/worker/gpu/model_states/mamba_hybrid.py:189-207`), replace the unconditional `block = 256` Triton launch shape with a small set of cached, request-count-aware launch variants for `preprocess_mamba_align_fused_kernel`. For small decode batches common in multi-turn agentic serving, use narrower blocks such as 32 or 64 so the metadata kernel does not schedule mostly masked lanes for a handful of active requests; keep 128/256 for larger batches where occupancy and launch count dominate. The wrapper can choose the block from `num_reqs` using fixed buckets, e.g. `<=32`, `<=64`, `<=128`, else `256`, so Triton compiles only a bounded number of specializations and the rest of the align-mode path remains unchanged. Add a targeted benchmark or unit-level launch-shape test around the existing Mamba hybrid state tests to ensure all buckets produce identical `_mamba_state_idx_gpu`, `_mamba_src_col_gpu`, `_mamba_src_off_gpu`, and `num_accepted_tokens_gpu` results across boundary and non-boundary steps.

**Novelty rationale.**

The deep research proposals focus on temporal amortization across multi-step decode windows and eliminating the precopy by threading separate source/destination indices into the forward path. Agent A proposes fusing the metadata and precopy kernels into one launch. This proposal deliberately keeps the current per-step structure and the `ctx.run_fused_precopy` handoff intact; it targets only the candidate's fixed `block=256` launch constant, reducing wasted masked work for small active batches without changing scheduling across steps or fusing kernels.

---
