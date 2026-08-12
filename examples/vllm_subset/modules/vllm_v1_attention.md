# vllm/v1/attention

[← All modules](../index.md)

## Module
- **Path:** `vllm/v1/attention`
- **Description:** Attention backend selection and metadata for v1's paged/chunked scheduling.
- **Depends on:** _(none)_
- **Main files:**
  - `vllm/v1/attention/backends/flash_attn.py` — FlashAttention backend
  - `vllm/v1/attention/backends/flashinfer.py` — FlashInfer backend
- **Run status:** DEGRADED
- **Findings:** 9
- **Issues:** 2

## Candidates

| Candidate | Impact | Deep research proposals |
|---|---|---:|
| [`FlashInferMetadataBuilder.build`](vllm_v1_attention/FlashInferMetadataBuilder.build__cand-vllm_v1_attention-0004.md) | high | 5 |
| [`FlashAttentionMetadataBuilder.build`](vllm_v1_attention/FlashAttentionMetadataBuilder.build__cand-vllm_v1_attention-0002.md) | high | 4 |
| [`FlashAttentionImpl._forward_with_dcp`](vllm_v1_attention/FlashAttentionImpl._forward_with_dcp__cand-vllm_v1_attention-0003.md) | medium | 4 |
| [`fast_plan_decode`](vllm_v1_attention/fast_plan_decode__cand-vllm_v1_attention-0006.md) | medium | 3 |
| [`use_cascade_attention`](vllm_v1_attention/use_cascade_attention__cand-vllm_v1_attention-0001.md) | high | 2 |
| [`unified_attention launch-parameter selection`](vllm_v1_attention/unified_attention_launch-parameter_selection__cand-vllm_v1_attention-0008.md) | high | 2 |
| [`TritonAttentionMetadataBuilder 2D/3D tuning constants`](vllm_v1_attention/TritonAttentionMetadataBuilder_2D_3D_tuning_constants__cand-vllm_v1_attention-0009.md) | medium | 2 |
| [`FlexAttentionMetadata._build_block_mask_direct`](vllm_v1_attention/FlexAttentionMetadata._build_block_mask_direct__cand-vllm_v1_attention-0023.md) | medium | 2 |
| [`Triton MLA decode split policy`](vllm_v1_attention/Triton_MLA_decode_split_policy__cand-vllm_v1_attention-0025.md) | medium | 2 |
| [`FlashInferMetadataBuilder._compute_flashinfer_kv_metadata`](vllm_v1_attention/FlashInferMetadataBuilder._compute_flashinfer_kv_metadata__cand-vllm_v1_attention-0005.md) | medium | 1 |
| [`make_local_attention_virtual_batches`](vllm_v1_attention/make_local_attention_virtual_batches__cand-vllm_v1_attention-0012.md) | medium | 1 |
| [`get_dcp_local_seq_lens`](vllm_v1_attention/get_dcp_local_seq_lens__cand-vllm_v1_attention-0014.md) | low | 1 |
| [`_make_mm_prefix_mask_mod`](vllm_v1_attention/_make_mm_prefix_mask_mod__cand-vllm_v1_attention-0015.md) | medium | 1 |
| [`FlashInferMetadataBuilder fixed split policy`](vllm_v1_attention/FlashInferMetadataBuilder_fixed_split_policy__cand-vllm_v1_attention-0019.md) | medium | 1 |
| [`compute_mm_prefix_range_tensor`](vllm_v1_attention/compute_mm_prefix_range_tensor__cand-vllm_v1_attention-0020.md) | medium | 1 |
| [`resolve_seq_and_query_len / find_seq_idx`](vllm_v1_attention/resolve_seq_and_query_len___find_seq_idx__cand-vllm_v1_attention-0022.md) | medium | 1 |
| [`reorder_batch_to_split_decodes_and_prefills`](vllm_v1_attention/reorder_batch_to_split_decodes_and_prefills__cand-vllm_v1_attention-0010.md) | medium | 0 |
| [`split_decodes_and_prefills`](vllm_v1_attention/split_decodes_and_prefills__cand-vllm_v1_attention-0011.md) | medium | 0 |
| [`fill_mm_prefix_query_ranges`](vllm_v1_attention/fill_mm_prefix_query_ranges__cand-vllm_v1_attention-0013.md) | medium | 0 |
| [`FlashInferMetadataBuilder._get_workspace_buffer`](vllm_v1_attention/FlashInferMetadataBuilder._get_workspace_buffer__cand-vllm_v1_attention-0016.md) | low | 0 |
| [`AttentionMetadataBuilder._init_reorder_batch_threshold`](vllm_v1_attention/AttentionMetadataBuilder._init_reorder_batch_threshold__cand-vllm_v1_attention-0021.md) | medium | 0 |
| [`get_kernel_options`](vllm_v1_attention/get_kernel_options__cand-vllm_v1_attention-0024.md) | medium | 0 |

## Findings (full list)

1. **Cascade Inference: Memory Bandwidth Efficient Shared Prefix Batch Decoding**
   - Source type: blog
   - URL: <https://flashinfer.ai/2024/02/02/cascade-inference.html>
   - Technique: Use recursive attention state merging to split shared-prefix attention from per-request suffix attention, then dispatch each part to the kernel best suited for its reuse pattern. For multi-turn agentic workloads with repeated system/tool/document prefixes, the module could make cascade selection more shape-aware and extend it toward multi-level shared prefixes rather than relying on coarse gates.
   - Evidence: Quote: "Use multi-query (prefill/append) attention kernel" and "Use batch decode attention kernel". Pointer: Cascade Inference: The Algorithm, steps 1-2.
2. **Flash-Decoding for Long-Context Inference**
   - Source type: blog
   - URL: <https://princeton-nlp.github.io/flash-decoding/>
   - Technique: Parallelize decode attention over the KV sequence length and merge partial log-sum-exp attention states in a reduction pass. This is directly transferable to adaptive split-KV or 3D decode launch policies that improve TPOT when batch size is small and contexts are long.
   - Evidence: Quote: "adds a new parallelization dimension: the keys/values sequence length." Pointer: A faster attention for decoding section.
3. **Attention States and Recursive Attention**
   - Source type: docs
   - URL: <https://docs.flashinfer.ai/tutorials/recursive_attention.html>
   - Technique: Represent partial attention over any KV subset as an output plus log-sum-exp state, then merge states associatively. The module could use this as a unifying abstraction for DCP context/new-token combination, cascade attention, and split-KV decode so that communication and compute partitioning can be tuned independently while preserving exact attention semantics.
   - Evidence: Quote: "the merge operator can be generalized to any number of attention state inputs". Pointer: Attention States and Recursive Attention, lines 358-360.
4. **flashinfer.cascade**
   - Source type: docs
   - URL: <https://docs.flashinfer.ai/api/cascade.html>
   - Technique: Cache cascade planning auxiliary structures and reuse them across multiple layer calls for the same decode step. The module could mirror this at metadata-build time by keying stable shape/page/cascade plans and reusing preallocated CUDA-graph buffers to lower per-step planning overhead and median TPOT.
   - Evidence: Quote: "auxiliary data structures can be reused across multiple batch decode attention calls". Pointer: MultiLevelCascadeAttentionWrapper example note.
5. **Multi-Head, Multi-Query, and Group-Query Attention**
   - Source type: docs
   - URL: <https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html>
   - Technique: Use a generation-phase multi-block attention mode only when low occupancy makes one-block-per-head inefficient, guided by batch size, head count, SM count, and internal heuristics. The module could replace fixed split constants with an occupancy-aware policy for split-KV/XQA/TRTLLM/native decode selection.
   - Evidence: Quote: "multi-block version of the GPU kernel" and "internal heuristic". Pointer: Generation Phase and XQA Optimization sections.
6. **CUDA C++ Best Practices Guide**
   - Source type: docs
   - URL: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
   - Technique: Minimize host-device transfers, batch many small transfers into one contiguous transfer, and use pinned memory for asynchronous copies. The module could apply this to attention metadata construction by packing indptr, last-page lengths, multimodal ranges, and split metadata into fewer staged H2D copies or moving cheap prefix computations onto device.
   - Evidence: Quote: "batching many small transfers into one larger transfer performs significantly better". Pointer: Data Transfer Between Host and Device, section 10.1.
7. **FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention**
   - Source type: blog
   - URL: <https://pytorch.org/blog/flexattention/>
   - Technique: Build block-sparse masks with custom constructors and separate full blocks from partial blocks so kernels skip unnecessary per-element masking. The module could adapt this for multimodal prefix masks and direct BlockMask construction by producing compact full/partial block metadata instead of evaluating or filling token-level masks repeatedly.
   - Evidence: Quote: "Write a custom constructor for BlockMask." Pointer: FAQ, How can we compute BlockMask quicker.
8. **DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**
   - Source type: blog
   - URL: <https://raw.githubusercontent.com/deepspeedai/DeepSpeed/master/blogs/deepspeed-ulysses/README.md>
   - Technique: Convert sequence-sharded tensors to head-sharded tensors with all-to-all before attention, compute attention on full sequence for fewer heads, then all-to-all back. For DCP paths, this suggests an alternative to gather-heavy context attention when heads divide cleanly across ranks, potentially reducing communication volume and improving long-context TPOT.
   - Evidence: Quote: "attention computation is head parallelism with full attention per head". Pointer: Additional Highlights of DeepSpeed-Ulysses, lines 22-23.
9. **CUDA Graphs**
   - Source type: docs
   - URL: <https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/>
   - Technique: Use a batch-descriptor-driven CUDA graph dispatcher that separately handles uniform decode and mixed/prefill batches, with backend capability fallbacks. The attention module could tighten its fast-plan and metadata decisions around stable batch descriptors so CUDA-graph-compatible decode routes avoid redundant planning while mixed batches safely fall back.
   - Evidence: Quote: "dispatch between full and piecewise cudagraph at runtime". Pointer: Motivation, lines 2736-2742.

## Issues

### module_deep_research
- **[warning, recoverable]** codex: Local ripgrep was unavailable in the sandbox, so module inspection used sed and fallback shell reads instead.

### proposal_from_finding_creator
- **[error, recoverable]** agent failure (candidate_id=cand-vllm_v1_attention-0001, finding_id=find-vllm_v1_attention-0001): claude exit=1: stderr=''
