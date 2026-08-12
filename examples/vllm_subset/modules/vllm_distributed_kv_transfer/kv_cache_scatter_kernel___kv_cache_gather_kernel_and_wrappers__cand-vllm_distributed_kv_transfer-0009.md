# kv_cache_scatter_kernel / kv_cache_gather_kernel and wrappers

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py`](vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py) (lines 9–247)
- **Symbol:** `kv_cache_scatter_kernel / kv_cache_gather_kernel and wrappers`
- **Kind:** kernel
- **Estimated impact:** medium
- **Id:** `cand-vllm_distributed_kv_transfer-0009`

## Description
Triton gather/scatter kernels move KV data between contiguous HF3FS buffers and paged KV cache storage, while wrappers construct token_indices tensors and launch one grid over layer and token.

## Current approach
Grid is (num_layers, num_tokens_in_block), BLOCK_SIZE is hard-coded to 128, num_warps/num_stages use defaults, and each wrapper builds token_indices as a CPU tensor then transfers it to device on every launch.

## Estimated impact explanation
For HF3FS-backed reuse, launch and hidden-size scan latency is paid per loaded or saved block; many small multi-turn chunks can make this visible in TTFT.

## Evolve rationale
Kernel launch parameters, vector width, grid layout, and CPU-to-GPU token-index setup are local tuning surfaces. Autotuning BLOCK_SIZE/warps, fusing K/V layout work where possible, reusing pinned or device scratch indices, and grouping small blocks into one launch can preserve the gather/scatter contract. Oracle: HF3FS unit tests and gather/scatter round-trip checks assert equal outputs for MLA and MHA layouts.

## Deep research proposals

### 1. Autotune BLOCK_SIZE, num_warps, and num_stages for HF3FS gather/scatter kernels
- **Finding:** `find-vllm_distributed_kv_transfer-0011` — *triton.autotune*
- **Source URL:** <https://triton-lang.org/main/python-api/generated/triton.autotune.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Apply `@triton.autotune` above `kv_cache_scatter_kernel` and `kv_cache_gather_kernel` in vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py (lines 9-122), keyed on `hidden_size` and `is_mla`. Provide a small set of `triton.Config` entries that sweep BLOCK_SIZE across the vector widths that plausibly match MLA and MHA head-dim shapes (e.g. 64, 128, 256, 512), combined with num_warps in {2, 4, 8} and num_stages in {2, 3, 4}. Because BLOCK_SIZE is currently a `tl.constexpr` used both to size `tl.arange` and to stride the `for i in range(0, hidden_size, BLOCK_SIZE)` loop, the autotune key must include `hidden_size` so Triton caches the best config per shape and never mixes configs across MLA/MHA. Drop the hardcoded `BLOCK_SIZE = 128` assignment in `scatter_kv_caches` (line 165) and `gather_kv_caches` (line 235) and stop passing `BLOCK_SIZE=` explicitly at the kernel launch sites (lines 176, 246); autotune will inject the chosen value. Semantics are preserved because BLOCK_SIZE only affects vector width of masked loads/stores over `hidden_size`, which the existing MLA/MHA round-trip tests in tests/ for the HF3FS connector already exercise.

**Proposal rationale.**

The candidate identifies BLOCK_SIZE=128 and default num_warps/num_stages as local tuning surfaces whose optimal value depends on `hidden_size` (which differs between MLA latent dim and MHA per-head dim) and on the target GPU. `triton.autotune` is designed for exactly this: it evaluates configurations keyed by tensor-shape arguments and caches the winner, without changing kernel semantics. For multi-turn agentic workloads, gather/scatter launches are paid per loaded/saved KV block on the TTFT critical path, so cutting the hidden-size scan latency by picking a better vector width and warp count directly reduces the transfer-side GPU time the candidate flags as visible in TTFT. This addresses the exact gap the candidate calls out and requires no change to the gather/scatter contract.

---

### 2. Fuse per-block gather/scatter launches into a single batched kernel across multiple HF3FS blocks
- **Finding:** `find-vllm_distributed_kv_transfer-0012` — *Kernel Fusion in NVIDIA CUDA: Optimizing Memory Traffic and Launch Overhead*
- **Source URL:** <https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend `kv_cache_scatter_kernel` and `kv_cache_gather_kernel` in vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py (lines 9-122) plus their `scatter_kv_caches` / `gather_kv_caches` wrappers (lines 125-247) to accept a batch of blocks per launch instead of one block per launch. Concretely: (1) change the wrappers so callers that currently invoke gather/scatter once per HF3FS block can pass a list of `token_indices` groups plus per-group source/destination offsets, packed into device tensors (e.g., a flat `token_indices` tensor plus a `block_offsets` / `block_ids` tensor and a `src_base_ptrs` tensor when source tensors are not already contiguous across blocks); (2) grow the Triton grid from `(num_layers, num_tokens_in_block)` to `(num_layers, num_tokens_in_block, num_blocks)` (or a fused axis with `tl.program_id(2)` selecting the block), and derive `source_offset` / `target_offset` inside the kernel from the block id, keeping the MLA and MHA layouts and their K/V fusion untouched so the gather/scatter contract and HF3FS unit tests / round-trip checks continue to pass; (3) at the wrapper level, replace the per-block `torch.tensor(token_indices, device='cpu').to(device, non_blocking=True)` pattern with a single batched host-to-device transfer (ideally out of a pinned staging buffer) that covers all blocks in one submission, eliminating the per-block CPU-tensor construction and copy. Keep the existing single-block signatures as thin shims that call the batched form with `num_blocks=1` to preserve backward compatibility.

**Proposal rationale.**

The finding's core pattern - combining adjacent GPU operations into one kernel to cut launch count and separate host-to-device transfers - maps directly onto this candidate's current shape. Today each HF3FS block triggers one Triton launch plus one CPU->GPU `token_indices` copy; in the target multi-turn agentic workload many small external-cache hits mean these fixed costs are paid repeatedly and are visible in TTFT. The candidate already fuses K/V within a launch, but does not fuse across blocks, so a batched launch is a strict superset of what it does. The change is transferable because Triton natively supports 3D grids and per-program-id offset math, the gather/scatter round-trip oracles (MLA and MHA equality checks) still constrain correctness, and shims preserve current call sites. This addresses the candidate's explicit `evolve_rationale` gap of 'grouping small blocks into one launch' and 'reusing pinned or device scratch indices'.

---

## Agent proposals

### 1. Persist token_indices in a device-resident ring buffer keyed by block layout to eliminate per-launch H2D copies
- **Agent:** claude

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py (lines 125-247), replace the per-call `torch.tensor(token_indices, device='cpu').to(device, non_blocking=True)` pattern in `scatter_kv_caches` and `gather_kv_caches` with a module-level, device-resident cache of precomputed `token_indices` tensors keyed by `(block_size, start_token_idx, num_tokens_in_block, device)`. Because HF3FS blocks are aligned to `block_size` and `token_indices` is deterministically `torch.arange(start_token_idx, start_token_idx + num_tokens_in_block)` shifted by a base offset, the vast majority of gather/scatter calls in a multi-turn agentic workload reuse an extremely small set of index patterns (typically one or two: a full-block index and a partial-tail-block index per sequence length modulo `block_size`). Implement an LRU dict (small, e.g. 16 entries) at module scope guarded by a lock, populated lazily via `torch.arange(..., device=device)` directly on the GPU — bypassing the CPU tensor construction and the host-to-device copy entirely. For the rarer case where the caller passes non-contiguous `token_indices`, fall back to the current path. Keep the kernel signatures and launch shape untouched so all HF3FS round-trip tests for MLA and MHA layouts pass unchanged.

**Novelty rationale.**

Proposal #1 (autotune) only tunes kernel meta-parameters and does not touch the `token_indices` construction path. Proposal #2 (batched multi-block kernel) restructures the kernel grid and batches multiple blocks into a single launch, using a per-call batched H2D transfer out of a pinned staging buffer — it still allocates and copies indices on every call, just amortized across blocks. This proposal is orthogonal: it eliminates the H2D copy altogether by exploiting the fact that `token_indices` is a deterministic `arange` derived from `(block_size, start_token_idx)` and thus can be constructed once on-device and memoized across calls. It composes with either existing proposal (a batched kernel would still benefit from device-resident per-block-slot indices; an autotuned single-block kernel would still pay the H2D cost per launch without this change) and specifically targets the candidate's `evolve_rationale` clause 'reusing pinned or device scratch indices' from a caching angle rather than a batching angle.

---

### 2. Add contiguous-index fast paths that compute cache token offsets in-kernel
- **Agent:** codex

**Detailed description.**

In `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py`, add specialized gather/scatter fast paths for the common case where `token_indices` is exactly a contiguous range for one HF3FS block. Instead of materializing any `token_indices` tensor, have the wrappers detect the contiguous case and launch variants of `kv_cache_scatter_kernel` / `kv_cache_gather_kernel` that accept `start_token_idx` and compute `cache_block_idx = (start_token_idx + token_offset) // block_size` and `cache_block_offset = (start_token_idx + token_offset) % block_size` directly inside the Triton program. Keep the current tensor-index kernels as the fallback for non-contiguous or future irregular layouts. This removes both wrapper-side index setup and the per-token global load from `token_indices` in the kernel while preserving the existing one-grid-over-layer-and-token contract for MLA and MHA layouts.

**Novelty rationale.**

This is not covered by the autotuning proposal, which only changes Triton meta-parameters. It is not covered by the batched-kernel proposal, which still uses packed index tensors and focuses on amortizing launches across blocks. It is also distinct from Claude's device-resident ring buffer: that proposal avoids repeated H2D copies but still creates, caches, passes, and loads `token_indices` tensors. This proposal removes the index tensor entirely on the dominant contiguous-block path and leaves the existing indexed path only for irregular cases.

---
