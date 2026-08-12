# GPUModelRunner._prepare_input_ids

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 1825–1954)
- **Symbol:** `GPUModelRunner._prepare_input_ids`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0003`

## Description
Legacy async-scheduling path that scatters previously sampled and drafted tokens into input_ids.gpu for the next step.

## Current approach
Walks num_reqs in Python, builds four Python index lists, calls cu_num_tokens[cur_index].item() on a NumPy scalar, materializes multiple pinned torch.tensor objects, and launches separate scatter operations for sampled and draft tokens.

## Estimated impact explanation
Async scheduling exists to lower TPOT; four micro-copies plus Python list construction per step are fixed overhead that is most visible in single-token multi-turn agent decode.

## Evolve rationale
The Python list construction, pinned tensor materialization, and separate scatter calls are on the async decode fast path. A packed index buffer or fused scatter preparation can preserve behavior covered by tests/v1/worker/test_gpu_model_runner.py::test_update_states_pp_async_multi_request_keeps_rank_state_consistent and async spec-decode coverage in tests/v1/spec_decode/test_eagle.py.

## Deep research proposals

### 1. Replace async-path index-list construction and dual scatter with a fused CUDA kernel for next-step input_ids updates
- **Finding:** `find-vllm_v1_worker-0001` — *[RFC]: Multi-Step Scheduling*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/6854>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `GPUModelRunner._prepare_input_ids` (vllm/v1/worker/gpu_model_runner.py:1825-1954), replace the Python-side loop that builds `sample_flattened_indices`, `spec_flattened_indices`, `prev_draft_token_indices`, and `prev_indices` — followed by four `torch.tensor(list, pin_memory=PIN_MEMORY).to(device, non_blocking=True)` uploads and two separate `input_ids.gpu.scatter_` calls — with a persistent, pre-allocated pinned index buffer that is filled in-place from `self.prev_positions.np`, `cu_num_tokens`, and the `scheduled_spec_decode_tokens` counts, and then handed to a single fused CUDA kernel that writes both the sampled token and the (optional) draft tokens into `input_ids.gpu` in one launch. Concretely: (1) precompute per-request `draft_len` into an existing CPU-side numpy staging buffer alongside `cu_num_tokens` so no per-request Python `.item()`/`extend`/`append` is needed; (2) materialize a single packed `(prev_index, flattened_index, draft_len, draft_offset)` int32 tensor into a reused pinned staging area and copy it once with `non_blocking=True`; (3) launch one CUDA kernel that, given that packed descriptor plus `prev_sampled_token_ids` and (when present) `_draft_token_ids`, scatters the sampled token at `flattened_index - draft_len` and the draft tokens at `flattened_index - draft_len + 1 .. flattened_index` for every common request. Keep the existing fast-path shortcut when `common_indices_match and max_flattened_index == num_common_tokens - 1` (single contiguous copy) and the early exit when no draft tokens exist. This preserves the semantics exercised by `tests/v1/worker/test_gpu_model_runner.py::test_update_states_pp_async_multi_request_keeps_rank_state_consistent` and the async spec-decode coverage in `tests/v1/spec_decode/test_eagle.py`, while removing the four per-step host-tensor materializations, the Python list construction, and one of the two scatter launches.

**Proposal rationale.**

The candidate is exactly the async fast path whose overhead the RFC targets: it is called every decode step, it builds inputs for the next step in Python, and it uploads several small pinned tensors per step. The RFC's transferable insight — 'We use Cuda kernels for faster updates because Torch is too slow' — applies directly: the four `torch.tensor([...], pin_memory=True).to(device, non_blocking=True)` calls plus two `scatter_` launches per step are the Torch-based next-step input update that a fused CUDA kernel replaces. Even without adopting the full multi-step loop, folding this single-step update into one kernel launch driven by a persistent packed index buffer captures the same overhead-amortization idea at the granularity that fits this method, addressing the fixed per-step Python/CPU-to-GPU cost that most visibly hurts median TPOT on multi-turn agentic decode.

---

## Agent proposals

### 1. Drop per-step int32 cast and intermediate gather-materializations by using index_copy_ with pre-cast draft buffer
- **Agent:** claude

**Detailed description.**

In `GPUModelRunner._prepare_input_ids` (vllm/v1/worker/gpu_model_runner.py:1919-1954), eliminate two per-step GPU-side overheads that are independent of index-list construction and are not addressed by the existing packed-buffer/fused-kernel proposal: (1) the full-tensor dtype conversion `draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)` — this allocates and rewrites the entire `[max_num_reqs, prev_num_spec_tokens]` draft buffer on every async step even though only `total_num_spec_tokens` entries are ever read; (2) the intermediate advanced-indexing materializations `self.input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0]` and `draft_token_ids.flatten()[prev_draft_token_indices_tensor]`, each of which allocates a fresh contiguous GPU tensor solely to serve as `src` for the subsequent `scatter_`. Concretely: (a) Change `_draft_token_ids` to be produced as `torch.int32` at its origin (the drafter/spec-decode path that assigns `self._draft_token_ids`) so no per-step cast is needed here — add a one-line assertion in this method rather than a conversion. If cross-drafter dtypes vary, cache the int32 view in a persistent buffer `self._draft_token_ids_int32` and lazily refresh only when the underlying tensor identity changes (`self._draft_token_ids.data_ptr()` check). (b) Replace the two `input_ids.gpu.scatter_(dim=0, index=idx, src=<gather-expression>)` calls with `input_ids.gpu.index_copy_(0, idx, <gathered-src>)` where the `<gathered-src>` is written directly into a pre-allocated persistent output-shaped staging buffer using `torch.index_select(..., out=self._sampled_src_staging[:num_common_tokens])` and `torch.index_select(..., out=self._draft_src_staging[:total_num_spec_tokens])`. `index_select` with `out=` avoids the anonymous intermediate allocation that advanced indexing forces, and `index_copy_` is a cheaper primitive than `scatter_` for the 1-D case here (no atomics, contiguous write). (c) Additionally reuse `sampled_tokens_index_tensor`, `prev_common_req_indices_tensor`, `draft_tokens_index_tensor`, and `prev_draft_token_indices_tensor` as slices of persistent int64 GPU tensors (already allocated at `max_num_reqs` and `max_num_reqs * prev_num_spec_tokens`) filled via one `copy_(non_blocking=True)` from a persistent pinned staging tensor, so index tensor allocation is amortized to zero on the steady-state async decode path. These changes preserve the semantics exercised by `tests/v1/worker/test_gpu_model_runner.py::test_update_states_pp_async_multi_request_keeps_rank_state_consistent` and `tests/v1/spec_decode/test_eagle.py`, and target GPU-side allocator/kernel overhead that persists even if the Python list construction is later replaced by a packed buffer or fused kernel.

**Novelty rationale.**

The existing deep_research_proposal focuses on Python-side index-list construction, packed pinned staging of `(prev_index, flattened_index, draft_len, draft_offset)` descriptors, and a fused CUDA kernel replacing the two `scatter_` launches. It does not address three orthogonal GPU-side costs that this method still pays: the whole-buffer `_draft_token_ids.to(dtype=torch.int32)` re-cast every step, the anonymous intermediate GPU tensors materialized by the two advanced-indexing gather expressions used as `src`, and the choice of `scatter_` over `index_copy_` for what is effectively a 1-D non-overlapping index write. This proposal is compatible with the fused-kernel direction (they operate at different layers: this one shrinks per-step GPU allocations and dtype work even without a custom kernel, and the persistent int32 draft view plus `index_select` out= staging remain useful if a fused kernel later replaces `index_copy_`), and it stands on its own by removing overhead that survives after Python-side packing lands.

---

### 2. Copy only CPU-origin input_ids ranges before async overwrite
- **Agent:** codex

**Detailed description.**

In `GPUModelRunner._prepare_input_ids` (`vllm/v1/worker/gpu_model_runner.py:1898-1905`), replace the mixed-batch fallback `self.input_ids.copy_to_gpu(total_num_scheduled_tokens)` with a range-copy path that uploads only slots not supplied from `prev_sampled_token_ids` or `_draft_token_ids`. While walking requests, track CPU-origin flattened spans for requests with `prev_index < 0`; for common async requests, skip both the sampled slot and its scheduled draft-token slots because those are overwritten later on GPU. If the CPU-origin spans collapse to the full `[0, total_num_scheduled_tokens)` range, keep the existing single prefix copy; otherwise issue slice `copy_` calls for the usually small number of new/prefill spans. Apply the same span-limited copy to `inputs_embeds` when `enable_prompt_embeds` is true, while preserving the existing full `is_token_ids.copy_to_gpu(...)` refresh because that buffer is read independently by the multimodal path. Add coverage to `tests/v1/worker/test_gpu_model_runner.py` for an async mixed batch with one retained decode request and one newly scheduled request, asserting that retained sampled/draft GPU slots are populated from previous GPU state and new-request slots still come from CPU input IDs.

**Novelty rationale.**

The deep-research proposal attacks Python index-list construction, pinned index uploads, and the sampled/draft scatter launches via a packed descriptor and fused CUDA kernel. Agent A targets GPU-side dtype conversion, advanced-indexing source materialization, `scatter_` versus `index_copy_`, and persistent index tensors. Neither proposal addresses the separate host-to-device prefix copy performed before those scatters when `num_common_tokens < total_without_spec`; that copy can redundantly upload async decode slots that are immediately overwritten. This proposal is a narrower data-movement reduction on the CPU-origin fallback path and remains useful even if the later sampled/draft update is fused or converted to `index_copy_`.

---
