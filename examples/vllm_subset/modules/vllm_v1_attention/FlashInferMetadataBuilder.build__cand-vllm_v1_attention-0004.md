# FlashInferMetadataBuilder.build

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flashinfer.py`](vllm/v1/attention/backends/flashinfer.py) (lines 1119–1520)
- **Symbol:** `FlashInferMetadataBuilder.build`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_attention-0004`

## Description
Builds FlashInfer per-step metadata, splits decode and prefill regions, computes required paged-KV metadata, and plans native FlashInfer, TRTLLM prefill, XQA, trtllm-gen decode, or cascade paths.

## Current approach
A monolithic dispatch recomputes path decisions every step and invokes wrapper plan calls for native FlashInfer paths even when batch structure repeats. It conditionally materializes CPU sequence lengths and prepares H2D metadata copies for paged-KV indices when native paths need them.

## Estimated impact explanation
This method runs once per step for the FlashInfer backend. Reducing plan overhead and transfer preparation moves median TPOT in stable decode batches and TTFT for first-step prefill-heavy agentic turns.

## Evolve rationale
Owned constructs include needs_seq_lens_cpu, prefill_use_trtllm, decode_with_flashinfer_trtllm_api, needs_paged_kv_indices, and native wrapper plan calls. A cache or skip layer keyed by stable batch/page shapes can be validated by existing FlashInfer backend tests and output equality because the underlying attention kernels remain unchanged.

## Deep research proposals

### 1. Shape-aware, multi-level cascade dispatch in FlashInferMetadataBuilder.build
- **Finding:** `find-vllm_v1_attention-0001` — *Cascade Inference: Memory Bandwidth Efficient Shared Prefix Batch Decoding*
- **Source URL:** <https://flashinfer.ai/2024/02/02/cascade-inference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flashinfer.py inside FlashInferMetadataBuilder.build (lines 1119-1520), replace the current single-level cascade gate (`use_cascade = common_prefix_len > 0` at line 1155) with a shape-aware selector that plans cascade attention as recursive attention-state merging over one or more shared-prefix levels. Concretely: (1) accept/derive multiple prefix-length levels from CommonAttentionMetadata (e.g., a list of common_prefix_lens across nested groups of requests sharing system/tool/document prefixes), (2) at build time evaluate a cost model based on num_prefills, num_decodes, num_qo_heads/num_kv_heads, page_size, per-level prefix block counts, and per-request suffix lengths to decide (a) whether cascade is worth planning at all and (b) which kernel to bind at each level - multi-query prefill/append for the shared prefix levels and batch decode for the per-request suffix, matching the Cascade Inference algorithm steps 1-2, (3) extend the current cascade early-out (lines 1296-1344) to plan N shared levels: build shared_qo_indptr/shared_kv_page_indptr/shared_kv_page_indices arrays per level and pass a list of length N+1 into cascade_wrapper.plan(qo_indptr_arr=..., paged_kv_indptr_arr=..., paged_kv_indices_arr=..., paged_kv_last_page_len=...), reusing the existing block_table_tensor slicing pattern (line 1314) recursively so each level strips its prefix blocks before the next level's indices are computed, and (4) fall back to the current non-cascade prefill/decode dispatch when the cost model rejects cascade (e.g., tiny shared prefix, few sharing requests, or when trtllm-gen decode + trtllm prefill are already selected for all requests). Keep the existing needs_seq_lens_cpu / needs_paged_kv_indices guards (lines 1235, 1279-1281) correct by treating multi-level cascade like the current cascade branch. The output tensor shape and semantics remain unchanged; only the plan-time dispatch and per-level index construction change.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out `use_cascade` selection as a coarse gate and lists cascade planning as an owned construct that can be optimized without changing kernel outputs. The finding describes cascade inference as recursive attention-state merging that dispatches multi-query prefill kernels to the shared prefix and batch decode kernels to per-request suffixes - exactly the two kernel families the builder already binds separately for prefill and decode. For the caller's multi-turn agentic workload, repeated system/tool/document prefixes across requests are the canonical case cascade inference targets, so a shape-aware, multi-level selector directly reduces redundant KV reads for the shared prefix in decode-heavy batches (TPOT) and for prefill-heavy first-turn planning (TTFT). The change is localized to the plan-time dispatch in build(), leaves the underlying FlashInfer kernels untouched, and can be validated by existing FlashInfer backend tests plus output-equality checks against the non-cascade path, matching the candidate's stated validation approach.

---

### 2. Cache FlashInfer plan artifacts across steps keyed by stable batch/page shape
- **Finding:** `find-vllm_v1_attention-0004` — *flashinfer.cascade*
- **Source URL:** <https://docs.flashinfer.ai/api/cascade.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In FlashInferMetadataBuilder.build (vllm/v1/attention/backends/flashinfer.py:1119-1520), introduce a small planning cache that memoizes wrapper plan outputs and auxiliary paged-KV metadata across steps when the batch structure is stable. Key the cache on a tuple of stable shape descriptors (num_decodes, num_prefills, page_size, num_kv_heads, num_qo_heads, head_dim, cascade level configuration, and the presence of TRTLLM/XQA/trtllm-gen dispatch flags such as prefill_use_trtllm, decode_with_flashinfer_trtllm_api, needs_paged_kv_indices, and needs_seq_lens_cpu). When the key matches the previous step, skip re-invoking native FlashInfer wrapper .plan() calls, reuse the previously prepared paged-KV indices/indptr device buffers, and avoid recomputing path-decision predicates. Preallocate the CUDA-graph-friendly buffers used by the native path once per unique key and refill them in place. When the key changes (new sequence structure, new page allocation, cascade level change), fall back to the current full build path and repopulate the cache entry. This mirrors FlashInfer's own guidance that auxiliary planning structures can be reused across multiple batch-decode attention calls at the same shape.

**Proposal rationale.**

The candidate explicitly identifies wrapper plan calls and paged-KV metadata preparation as per-step overhead that repeats when batch structure is stable, and calls out a cache/skip layer keyed by stable batch/page shapes as the intended lever. The FlashInfer cascade docs directly state that auxiliary data structures can be reused across multiple batch decode attention calls, which is the same reuse pattern the candidate wants to apply at metadata-build time. In multi-turn agentic decode, batches remain shape-stable across many steps, so skipping redundant plan() work and buffer preparation targets median TPOT without altering the underlying attention kernels, keeping existing FlashInfer backend tests and output-equality checks as the validation surface.

---

### 3. Replace fixed split-KV constants with an occupancy-aware policy in FlashInferMetadataBuilder.build
- **Finding:** `find-vllm_v1_attention-0005` — *Multi-Head, Multi-Query, and Group-Query Attention*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flashinfer.py FlashInferMetadataBuilder.build (lines 1119-1520), the plan calls into the native FlashInfer prefill and decode wrappers pass `fixed_split_size=self.prefill_fixed_split_size` / `self.decode_fixed_split_size` and `disable_split_kv=self.disable_split_kv` as static configuration (see the prefill_wrapper.plan calls at ~lines 1405/1434 and fast_plan_decode at ~line 1497). Introduce a small occupancy-aware policy computed inside build() that decides, per step, whether to enable split-KV / multi-block decode and what split size to use — driven by the actual (num_decodes, num_decode_tokens, num_prefills, max_seq_len, num_qo_heads, num_kv_heads) already available in the method plus a cached device property (SM count). The policy would estimate whether a one-block-per-KV-head launch under-fills the SMs (low occupancy) and, only then, enable a multi-block/split-KV plan for the native decode and native prefill paths; when occupancy is already sufficient it keeps split-KV disabled to avoid the extra reduction. The same occupancy signal can also nudge the existing `decode_with_flashinfer_trtllm_api` / native decode / XQA-eligible selection in Step 1 so that batches that already saturate SMs skip multi-block modes. Keep the current `self.prefill_fixed_split_size` / `self.decode_fixed_split_size` / `self.disable_split_kv` as optional overrides so behavior is unchanged when a user pins them.

**Proposal rationale.**

The candidate explicitly notes that its dispatch 'recomputes path decisions every step' and calls out fixed split constants and native wrapper plans as owned constructs a policy layer could tune. The finding describes exactly this: TRTLLM's generation-phase multi-block mode is engaged only when low occupancy would otherwise leave SMs idle, using batch size, head count, and SM count as inputs. Applying that heuristic to select `disable_split_kv` / `fixed_split_size` (and, secondarily, to bias trtllm-gen vs XQA vs native decode selection) targets the same TPOT-in-stable-decode and TTFT-in-prefill-heavy-first-step regimes the candidate is trying to improve for the multi-turn agentic workload, without changing any underlying attention kernels — so existing FlashInfer backend tests and output-equality checks remain a valid validation surface.

---

### 4. Batch small paged-KV metadata H2D copies into one pinned staging transfer in FlashInferMetadataBuilder.build
- **Finding:** `find-vllm_v1_attention-0006` — *CUDA C++ Best Practices Guide*
- **Source URL:** <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flashinfer.py FlashInferMetadataBuilder.build (lines 1119-1520) and its helper _compute_flashinfer_kv_metadata (lines 1062-1117), replace the multiple small non_blocking H2D copies used to publish per-step paged-KV metadata with a single packed transfer from a pinned staging buffer. Concretely: (1) allocate one pin_memory=True staging tensor sized to hold [paged_kv_indptr[:num_reqs+1] || paged_kv_last_page_len[:num_reqs]] (and, in the cascade path at lines 1296-1344, also shared_qo_indptr, shared_kv_page_indptr, shared_kv_last_page_len) as int32 contiguous regions. (2) Write the CPU-side cumsum and last_page_len np arrays into slices of that staging buffer instead of into separate self.paged_kv_indptr.cpu / self.paged_kv_last_page_len.cpu tensors. (3) Issue exactly one tensor.copy_(staging, non_blocking=True) into a matching GPU buffer, then create views (paged_kv_indptr.gpu, paged_kv_last_page_len.gpu, cascade shared tensors) as slices of the destination. (4) For the cascade branch, stop constructing shared_qo_indptr_cpu / shared_kv_page_indptr_cpu / shared_kv_last_page_len_cpu as fresh unpinned torch.tensor(...) each step (they trigger separate implicit H2D copies inside cascade_wrapper.plan) and instead reuse persistent pinned slots inside the packed staging buffer. (5) Keep the block-table-driven paged_kv_indices Triton kernel (lines 1099-1105) unchanged since it already runs on GPU. The change preserves all path decisions and kernel invocations; only the transfer schedule and pinning behavior change.

**Proposal rationale.**

The CUDA C++ Best Practices Guide section 10.1 explicitly recommends coalescing many small transfers into one contiguous transfer and using pinned memory for asynchronous copies. FlashInferMetadataBuilder.build runs every decode step and currently issues at least two separate non_blocking H2D copies (paged_kv_indptr and paged_kv_last_page_len in _compute_flashinfer_kv_metadata), plus several additional small unpinned CPU tensor creations in the cascade branch that cause implicit synchronous copies inside cascade_wrapper.plan. In stable multi-turn agentic decode batches where FlashInfer native or cascade paths are selected, these per-step transfers sit on the TPOT critical path. Packing them into a single pinned staging buffer with one copy_ call reduces launch and DMA overhead per step without altering any attention kernel behavior, directly addressing the candidate's evolve_rationale about metadata H2D preparation overhead.

---

### 5. Key FlashInfer plan cache by CUDA-graph batch descriptor to skip redundant planning on stable decode batches
- **Finding:** `find-vllm_v1_attention-0009` — *CUDA Graphs*
- **Source URL:** <https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In FlashInferMetadataBuilder.build (vllm/v1/attention/backends/flashinfer.py:1119-1520), introduce a batch-descriptor-keyed fast path that mirrors vLLM's CUDA graph dispatcher's split between uniform-decode and mixed/prefill batches. Compute a stable descriptor from the inputs that already drive the current dispatch (num_decodes, num_prefills, per-request paged-KV page counts/shape, block_table stride, page_size, dtype, and the boolean toggles this method already owns: needs_seq_lens_cpu, prefill_use_trtllm, decode_with_flashinfer_trtllm_api, needs_paged_kv_indices). Cache the last successful native FlashInfer wrapper plan output and the prepared paged-KV index buffers keyed by that descriptor; on a hit for a pure-decode batch that matches the prior descriptor, reuse the plan and skip both the wrapper plan call and the CPU seq-len materialization / H2D paged-KV index copy. On any descriptor change, on a mixed prefill+decode batch, or when a non-native path (TRTLLM prefill, XQA, trtllm-gen decode, cascade) is selected, fall back to the current full recomputation path unchanged. Keep the cache single-slot and per-builder so it is invalidated implicitly whenever batch shape shifts, matching the doc's fallback discipline.

**Proposal rationale.**

The candidate explicitly calls out that path decisions and wrapper plan calls are recomputed every step even when batch structure repeats, and identifies needs_seq_lens_cpu / prefill_use_trtllm / decode_with_flashinfer_trtllm_api / needs_paged_kv_indices / native wrapper plan as the owned constructs a cache/skip layer could gate. The finding's CUDA-graph dispatcher design contributes a concrete, transferable idea: use a batch-descriptor key to split uniform-decode (cacheable) from mixed/prefill (must re-plan) with an explicit capability-based fallback. Applied here, that pattern directly reduces per-step plan overhead on stable multi-turn agentic decode batches (improving median TPOT) while leaving first-step and mixed batches on the safe recomputation path (preserving TTFT correctness), which matches the caller's stated TTFT/TPOT objective. Correctness is checkable via existing FlashInfer backend tests since kernels are unchanged.

---

## Agent proposals

### 1. Fuse per-step FlashInfer metadata prep into a single Triton kernel to eliminate CPU cumsum and multiple H2D copies
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/flashinfer.py, FlashInferMetadataBuilder.build (lines 1119-1520) and its helper _compute_flashinfer_kv_metadata (lines 1062-1117) currently execute a CPU-side pipeline every step: numpy cumsum to produce paged_kv_indptr, numpy arithmetic to derive paged_kv_last_page_len, torch.from_numpy conversions into pre-allocated pinned CPU tensors, and one or more non_blocking H2D copies. Only paged_kv_indices is already produced on GPU via the existing block-table Triton kernel (lines 1099-1105). Replace this hybrid CPU/GPU pipeline with a single fused Triton (or lightweight CUDA) kernel that reads the existing on-device inputs — block_table_tensor, seq_lens_gpu (or num_computed_tokens_gpu + query_lens_gpu), and page_size — and writes, in one launch, all three device-side outputs required by native FlashInfer wrapper plan calls: (a) paged_kv_indptr (exclusive cumsum of per-request num_blocks), (b) paged_kv_indices (gathered block-table entries, replacing the current dedicated kernel), and (c) paged_kv_last_page_len ((seq_len - 1) % page_size + 1 per request, with the empty-request guard preserved). Structure the kernel as: one program per request writing its per-request num_blocks and last_page_len into scratch, followed by an in-kernel prefix-sum stage (or a two-pass launch with a tiny reduce kernel) that materializes indptr and simultaneously gathers indices at the resolved offsets. In build(), delete the numpy cumsum path, the pinned-CPU staging tensors for indptr/last_page_len, and their non_blocking copy_ calls; replace them with slices of the kernel's output buffer. Keep the existing prefill/decode split, cascade branch, and needs_paged_kv_indices / needs_seq_lens_cpu gates intact — needs_seq_lens_cpu still triggers a separate D2H copy of seq_lens for wrappers that require CPU seq lens, but the paged-KV indptr/indices/last_page_len metadata never round-trips through CPU. For the cascade branch (lines 1296-1344), extend the same kernel with an optional 'strip_prefix_blocks' argument so shared_qo_indptr/shared_kv_page_indptr/shared_kv_last_page_len are produced on-device by the same launch instead of being built as fresh unpinned torch.tensor(...) each step. Wrapper plan output tensors and semantics are unchanged.

**Novelty rationale.**

The listed deep_research_proposals fall into three families: (1) shape-aware cascade dispatch (finding 0001), (2) plan-artifact caching keyed by stable batch/page shape (findings 0004 and 0009), (3) occupancy-aware split-KV policy (finding 0005), and (4) batching multiple small H2D copies into one pinned staging transfer (finding 0006). Proposal 0006 is the closest neighbor — it also targets the metadata transfer schedule — but it still constructs indptr and last_page_len on CPU via numpy cumsum and simply coalesces the resulting host tensors into one pinned staging buffer with a single non_blocking copy. This proposal is materially different: it eliminates the CPU-side cumsum and arithmetic entirely by producing indptr, indices, and last_page_len directly on device in one Triton launch, so there is no host cumsum, no pinned staging buffer, and no H2D copy on the paged-KV metadata path at all — orthogonal to caching (findings 0004/0009), which still pays the full CPU+H2D cost on any descriptor miss, and complementary to cascade/split-KV dispatch changes (findings 0001/0005), which operate on plan-time policy rather than on how paged-KV metadata is materialized.

---

### 2. Materialize paged-KV metadata only for native FlashInfer request ranges
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/backends/flashinfer.py`, narrow the `needs_paged_kv_indices` path in `FlashInferMetadataBuilder.build` so `_compute_flashinfer_kv_metadata` is invoked only for the request ranges that will actually be consumed by native FlashInfer wrappers. Today, once any native/cascade path needs paged-KV metadata, the helper computes `paged_kv_indptr`, gathers `paged_kv_indices`, and copies `paged_kv_last_page_len` for all `num_reqs`. For mixed batches where decode is native but prefill uses TRTLLM, compute only the leading decode range (`[:num_decodes]`) because `fast_plan_decode` consumes only `self.paged_kv_indptr.cpu[:num_decode_tokens + 1]`, `paged_kv_indices`, and `last_page_len_cpu[:num_decode_tokens]`. For batches where prefill is native but decode uses TRTLLM/XQA/trtllm-gen, compute only the trailing prefill range (`[num_decodes:num_reqs]`) and normalize its indptr to start at zero before passing it to the prefill wrapper. Keep the full-range behavior for cascade and for batches where both decode and prefill use native FlashInfer. This avoids CPU cumsum work, last-page-length writes, H2D copies, and block-table gathers for requests handled by non-native kernels, while leaving all kernel choices and wrapper semantics unchanged.

**Novelty rationale.**

This is not the same as the listed cache proposals, which reuse full plan artifacts when a stable descriptor hits, nor the split-KV or cascade dispatch proposals, which change policy decisions. It is also distinct from the H2D batching proposal and Agent A's fused device-kernel proposal: those optimize how paged-KV metadata is produced, while this proposal reduces the amount of metadata produced by excluding request subranges that native FlashInfer will never read in mixed native/TRTLLM batches. The change is therefore orthogonal and can combine with either packed transfers or a future fused GPU metadata kernel.

---
