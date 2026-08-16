# make_local_attention_virtual_batches

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/utils.py`](vllm/v1/attention/backends/utils.py) (lines 348–491)
- **Symbol:** `make_local_attention_virtual_batches`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0012`

## Description
Splits local-attention requests into virtual batches and rebuilds local query lengths, KV lengths, computed-token counts, and block tables.

## Current approach
Performs multiple numpy repeat/cumsum/fancy-index steps, materializes batch_indices and block_indices, uploads both index arrays H2D, and then gathers block_table_local with PyTorch indexing.

## Estimated impact explanation
Local-attention Gemma-family models are relevant to agentic serving. This per-step metadata transformation feeds scheduling and attention launch, so reducing CPU/H2D work improves median TPOT.

## Evolve rationale
Optimizable constructs are the numpy repeat/cumsum pipeline and the two async_tensor_h2d index uploads. Correctness oracle is exact equality of returned CommonAttentionMetadata fields and block_table_local for randomized local-attention batches, plus existing local-attention tests.

## Deep research proposals

### 1. Coalesce four H2D uploads in make_local_attention_virtual_batches into one staged transfer
- **Finding:** `find-vllm_v1_attention-0006` — *CUDA C++ Best Practices Guide*
- **Source URL:** <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/utils.py:348-491 (make_local_attention_virtual_batches), replace the four independent async_tensor_h2d calls (batch_indices, block_indices, cu_seqlens_q_local, seqlens_k_local at lines 464-465, 479-480) with a single coalesced host-to-device staging transfer. Concretely: allocate one pinned-memory numpy/torch buffer on the CPU side sized to hold the concatenation of batch_indices (int32), block_indices (int32), cu_seqlens_q_local (int32), and seqlens_k_local (int32); copy each array into its slice of that pinned buffer; issue one async_tensor_h2d for the combined buffer; then create views/slices on-device to recover the four tensors (batch_indices_torch, block_indices_torch, query_start_loc, seq_lens). This preserves the exact returned CommonAttentionMetadata fields and block_table_local semantics because the values are unchanged and fancy indexing still uses the same device tensors, just aliased into a single buffer. Optionally, hoist the tiny scalar max_query_len/max_seq_len reductions to CPU (already CPU) so no extra D2H sync is introduced. Correctness oracle: exact equality of returned CommonAttentionMetadata fields and block_table_local versus the current implementation over randomized local-attention batches, plus the existing local-attention tests.

**Proposal rationale.**

The candidate's current_approach explicitly identifies the two async_tensor_h2d index uploads (plus the two additional H2D uploads for query_start_loc and seq_lens inside the returned CommonAttentionMetadata) as an optimization target. The CUDA C++ Best Practices Guide (section 10.1) states that 'batching many small transfers into one larger transfer performs significantly better' and recommends pinned memory for asynchronous copies. Coalescing these four separate H2D copies — issued every scheduler step for local-attention (Gemma-family) models — into one staged pinned-memory transfer directly reduces per-step CPU overhead and PCIe transaction count on the critical path that feeds attention launch, which improves median TPOT under the multi-turn agentic workload described in the caller context. This is a concrete, transferable application of the finding to the exact hotspot the candidate calls out, not a topical restatement.

---

## Agent proposals

### 1. Compute batch_indices/block_indices on-device from small per-request tensors instead of materializing large numpy arrays and uploading them
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/utils.py:348-491 (make_local_attention_virtual_batches), eliminate the CPU-side materialization and H2D upload of the two largest index arrays — `batch_indices` (line 453-456) and `block_indices` (line 449-452), each of size `virtual_batches * pages_per_local_batch` — by deriving them on-device from small per-request tensors. Concretely: (a) keep only the small per-request numpy arrays on CPU: `local_blocks` (length `actual_batch_size`), `block_starts_per_virtual_batch` = `k_seqstarts_absolute // block_size` (length `virtual_batches`, still much smaller than the final index arrays when `pages_per_local_batch > 1`), and the cumulative sums `cu_local_blocks` and `cu_pages = np.cumsum(local_blocks * pages_per_local_batch)`. (b) Upload only these small tensors to device (single coalesced transfer, sized O(actual_batch_size) + O(virtual_batches), not O(virtual_batches * pages_per_local_batch)). (c) On-device, reconstruct `batch_indices_torch` via `torch.repeat_interleave(torch.arange(actual_batch_size, device=device), (local_blocks * pages_per_local_batch)_gpu)` and `block_indices_torch` via `block_starts_gpu.unsqueeze(1) + torch.arange(pages_per_local_batch, device=device).unsqueeze(0)` followed by `.reshape(-1).clamp_(max=block_table.shape[1]-1)`. (d) Then perform the existing fancy indexing `block_table[batch_indices_torch, block_indices_torch].view(virtual_batches, -1)` unchanged. Correctness oracle: exact equality of returned CommonAttentionMetadata fields and block_table_local versus the current implementation across randomized local-attention batches, plus existing local-attention tests in tests/v1/attention/. This is beneficial specifically when `pages_per_local_batch > 1` (e.g., attn_chunk_size=4096 with block_size=16 gives pages_per_local_batch=256), where the current code inflates a per-virtual-batch scalar into 256x more data on CPU before shipping it across PCIe.

**Novelty rationale.**

The existing deep_research_proposal (find-vllm_v1_attention-0006) coalesces the four existing H2D transfers into one staged pinned-memory transfer but still transfers the fully-materialized `batch_indices` and `block_indices` arrays of size `virtual_batches * pages_per_local_batch`. This proposal is orthogonal and complementary: it changes *what* is transferred, not *how* it is transferred. By shifting the `np.repeat` and broadcast-arange expansions from CPU-numpy to on-device torch ops, it reduces both the CPU work (two `np.repeat`/`np.arange` allocations on the largest arrays in the function are removed) and the PCIe payload by a factor of `pages_per_local_batch` for the two dominant index arrays — a reduction the coalescing proposal cannot achieve because it preserves the payload size. It also removes the numpy `.reshape(-1).clip(...)` on the large intermediate. The two optimizations can be stacked (compute-on-device first, then coalesce the residual small uploads), so this is not a restatement of the existing proposal.

---

### 2. Add a single-local-block fast path for decode-style batches
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/backends/utils.py:348-491` (`make_local_attention_virtual_batches`), add an early branch after `q_tokens_in_first_block`, `tokens_in_last_block`, and `local_blocks` are computed for the common TPOT case where every request maps to exactly one local virtual batch, i.e. `np.all(local_blocks == 1)` (notably decode with `q_seqlens == 1`). In this branch, avoid constructing `cu_num_blocks`, `block_offsets`, `arange`, `rarange`, repeated `seqlens_q_local`, repeated `tokens_in_last_block`, and `k_seqstarts_absolute`. Instead set `virtual_batches = actual_batch_size`, `seqlens_q_local = q_seqlens.astype(np.int32, copy=False)`, `cu_seqlens_q_local` from the existing query lengths, `seqlens_k_local = tokens_in_last_block.astype(np.int32, copy=False)`, and `num_computed_tokens_local = seqlens_k_local - seqlens_q_local`. Compute `block_starts` directly as `(seq_lens_np - tokens_in_last_block) // block_size`, then build the same `block_indices`/`batch_indices` and return the same `CommonAttentionMetadata` fields as today. Correctness oracle: compare exact equality with the current implementation for randomized batches where all `local_blocks == 1`, including decode at different positions inside the local attention window and boundary cases where `seq_lens_np % attn_chunk_size == 0`, plus existing local-attention tests.

**Novelty rationale.**

The deep_research proposal optimizes the number of H2D transfers while preserving the existing CPU-side arrays. Agent A's proposal changes where the large `batch_indices` and `block_indices` expansions are computed. This proposal targets a different source of overhead: it bypasses the virtual-batch splitting pipeline entirely when no split is needed, reducing several numpy repeat/cumsum/intermediate-array operations for the common per-token decode path. It can be combined with either transfer coalescing or on-device index generation because it changes the control flow before those index-transfer decisions.

---
