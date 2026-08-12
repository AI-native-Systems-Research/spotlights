# vllm/v1/worker

[← All modules](../index.md)

## Module
- **Path:** `vllm/v1/worker`
- **Description:** Per-device workers running the model forward pass with input batching and cudagraph capture.
- **Depends on:** _(none)_
- **Main files:**
  - `vllm/v1/worker/gpu_worker.py` — GPU worker process wrapper
  - `vllm/v1/worker/gpu_model_runner.py` — Input batching + forward + sampling on GPU
  - `vllm/v1/worker/gpu_input_batch.py` — Batched request state
  - `vllm/v1/worker/cpu_model_runner.py` — CPU counterpart of the GPU model runner
- **Run status:** DEGRADED
- **Findings:** 11
- **Issues:** 1

## Candidates

| Candidate | Impact | Deep research proposals |
|---|---|---:|
| [`GPUModelRunner._prepare_inputs`](vllm_v1_worker/GPUModelRunner._prepare_inputs__cand-vllm_v1_worker-0001.md) | high | 4 |
| [`GPUModelRunner._bookkeeping_sync`](vllm_v1_worker/GPUModelRunner._bookkeeping_sync__cand-vllm_v1_worker-0006.md) | high | 3 |
| [`GPUModelRunner.execute_model`](vllm_v1_worker/GPUModelRunner.execute_model__cand-vllm_v1_worker-0007.md) | medium | 3 |
| [`GPUModelRunner._calc_spec_decode_metadata`](vllm_v1_worker/GPUModelRunner._calc_spec_decode_metadata__cand-vllm_v1_worker-0004.md) | high | 2 |
| [`GPUModelRunner._build_attention_metadata`](vllm_v1_worker/GPUModelRunner._build_attention_metadata__cand-vllm_v1_worker-0005.md) | medium | 2 |
| [`preprocess_mamba`](vllm_v1_worker/preprocess_mamba__cand-vllm_v1_worker-0012.md) | high | 2 |
| [`GPUModelRunner.prepare_inputs`](vllm_v1_worker/GPUModelRunner.prepare_inputs__cand-vllm_v1_worker-0023.md) | high | 2 |
| [`MambaHybridModelState.preprocess_state`](vllm_v1_worker/MambaHybridModelState.preprocess_state__cand-vllm_v1_worker-0031.md) | medium | 2 |
| [`GPUModelRunner._update_states`](vllm_v1_worker/GPUModelRunner._update_states__cand-vllm_v1_worker-0002.md) | high | 1 |
| [`GPUModelRunner._prepare_input_ids`](vllm_v1_worker/GPUModelRunner._prepare_input_ids__cand-vllm_v1_worker-0003.md) | high | 1 |
| [`BlockTable and MultiGroupBlockTable row mutation methods`](vllm_v1_worker/BlockTable_and_MultiGroupBlockTable_row_mutation_methods__cand-vllm_v1_worker-0011.md) | high | 1 |
| [`postprocess_mamba_fused_kernel`](vllm_v1_worker/postprocess_mamba_fused_kernel__cand-vllm_v1_worker-0013.md) | high | 1 |
| [`DP synchronization post-processing helpers`](vllm_v1_worker/DP_synchronization_post-processing_helpers__cand-vllm_v1_worker-0015.md) | high | 1 |
| [`UBatchWrapper._capture_ubatches and _run_ubatches`](vllm_v1_worker/UBatchWrapper._capture_ubatches_and__run_ubatches__cand-vllm_v1_worker-0019.md) | high | 1 |
| [`sync_cudagraph_and_dp_padding`](vllm_v1_worker/sync_cudagraph_and_dp_padding__cand-vllm_v1_worker-0026.md) | medium | 1 |
| [`AsyncOutput.get_output`](vllm_v1_worker/AsyncOutput.get_output__cand-vllm_v1_worker-0027.md) | high | 1 |
| [`DefaultModelState.prepare_attn`](vllm_v1_worker/DefaultModelState.prepare_attn__cand-vllm_v1_worker-0030.md) | medium | 1 |
| [`GPUModelRunner._calc_mrope_positions`](vllm_v1_worker/GPUModelRunner._calc_mrope_positions__cand-vllm_v1_worker-0008.md) | medium | 0 |
| [`GPUModelRunner._determine_batch_execution_and_padding`](vllm_v1_worker/GPUModelRunner._determine_batch_execution_and_padding__cand-vllm_v1_worker-0009.md) | medium | 0 |
| [`_compute_slot_mapping_kernel`](vllm_v1_worker/_compute_slot_mapping_kernel__cand-vllm_v1_worker-0010.md) | high | 0 |
| [`stage_postprocess_inputs_to_gpu`](vllm_v1_worker/stage_postprocess_inputs_to_gpu__cand-vllm_v1_worker-0014.md) | medium | 0 |
| [`KVBlockZeroer`](vllm_v1_worker/KVBlockZeroer__cand-vllm_v1_worker-0016.md) | medium | 0 |
| [`EncoderCudaGraphManager budget generation and lookup`](vllm_v1_worker/EncoderCudaGraphManager_budget_generation_and_lookup__cand-vllm_v1_worker-0017.md) | high | 0 |
| [`EncoderCudaGraphManager._execute_local`](vllm_v1_worker/EncoderCudaGraphManager._execute_local__cand-vllm_v1_worker-0018.md) | medium | 0 |
| [`maybe_create_ubatch_slices`](vllm_v1_worker/maybe_create_ubatch_slices__cand-vllm_v1_worker-0020.md) | medium | 0 |
| [`WorkspaceManager._ensure_workspace_size`](vllm_v1_worker/WorkspaceManager._ensure_workspace_size__cand-vllm_v1_worker-0021.md) | medium | 0 |
| [`copy_kv_cache_blocks_inplace`](vllm_v1_worker/copy_kv_cache_blocks_inplace__cand-vllm_v1_worker-0022.md) | medium | 0 |
| [`BlockTables.gather_block_tables and compute_slot_mappings`](vllm_v1_worker/BlockTables.gather_block_tables_and_compute_slot_mappings__cand-vllm_v1_worker-0024.md) | high | 0 |
| [`StagedWriteTensor and FusedStagedWriter apply path`](vllm_v1_worker/StagedWriteTensor_and_FusedStagedWriter_apply_path__cand-vllm_v1_worker-0025.md) | medium | 0 |
| [`GPUModelRunner.add_requests and update_requests`](vllm_v1_worker/GPUModelRunner.add_requests_and_update_requests__cand-vllm_v1_worker-0028.md) | high | 0 |
| [`rejection_sample`](vllm_v1_worker/rejection_sample__cand-vllm_v1_worker-0029.md) | high | 0 |
| [`RopeState.prepare_positions and _prepare_rope_positions_kernel`](vllm_v1_worker/RopeState.prepare_positions_and__prepare_rope_positions_kernel__cand-vllm_v1_worker-0032.md) | medium | 0 |

## Findings (full list)

1. **[RFC]: Multi-Step Scheduling**
   - Source type: issue
   - URL: <https://github.com/vllm-project/vllm/issues/6854>
   - Technique: Adopt a worker-local multi-step decode loop that keeps sampled tokens on GPU, advances next-step input metadata with CUDA kernels, and delays scheduler/output synchronization until lookahead slots are exhausted. This directly targets median TPOT by amortizing per-token Python input prep, output materialization, and scheduler overhead across multiple decode iterations in agentic multi-turn traffic.
   - Evidence: Quote: "Multi-step decoding will be able to amortize all these overheads over n-steps at a time." Quote: "Update inputs for the next step. We use Cuda kernels for faster updates because Torch is too slow." Pointer: Motivation and High level Algorithm, lines 166-174 and 213-220.
2. **[Performance]: Fully Async Spec-Decoding | Make `seq_lens_cpu` in CommonAttentionMetadata optional**
   - Source type: issue
   - URL: <https://github.com/vllm-project/vllm/issues/29134>
   - Technique: Make CPU sequence-length metadata optional and push spec-decode metadata consumers toward device-resident sequence lengths and upper bounds. This would remove host/device syncs that block overlapping next-step input preparation with current forward execution, especially when verifying multiple drafted tokens.
   - Evidence: Quote: "Ultimately in-order to realize fully async spec decoding we need to build attention metadata without knowing `seq_lens_cpu` (using device `seq_lens` on device is fine...)." Pointer: Proposal to improve performance, lines 166-180.
3. **Model Runner V2 Design Document**
   - Source type: docs
   - URL: <https://docs.vllm.ai/en/v0.17.0/design/model_runner_v2/>
   - Technique: Decouple persistent request state from per-step input tensors, assign active requests stable rows, and gather per-step inputs from mostly GPU-resident state. This reduces request-churn bookkeeping and avoids tensor-wide reordering on the critical TTFT/TPOT path.
   - Evidence: Quote: "MRV2 decouples persistent state tensors from per-step input tensors." Quote: "Large state tensors are mostly stored on GPU memory, so gather runs in parallel on the GPU with low overhead." Pointer: Persistent Batch / MRV2's Solution, lines 2360-2367.
4. **vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction**
   - Source type: blog
   - URL: <https://vllm-project.github.io/2024/09/05/perf-update.html>
   - Technique: Overlap output processing with the next model execution step instead of synchronously converting GPU tensors to Python lists and checking stop criteria after every token. The worker output path could further delay or batch materialization to reduce TPOT bubbles from token IDs, logprobs, NaN counts, and fault flags.
   - Evidence: Quote: "we introduced asynchronous output processing, which overlaps the output processing with model execution." Quote: "vLLM now delays it, performing the processing of the `n`-th step output while executing the `n+1`-th step." Pointer: Asynchronous output processing section.
5. **FlashInfer Attention Kernels**
   - Source type: docs
   - URL: <https://docs.flashinfer.ai/api/attention.html>
   - Technique: Cache auxiliary attention planning structures and reuse them across layer calls and repeated execution, with user-provided stable buffers under CUDA graph mode. The worker attention-metadata path could apply a similar lifecycle for per-group/per-layer metadata to reduce repeated CPU recomputation and graph pointer churn.
   - Evidence: Quote: "FlashInfer’s batch decode attention creates some auxiliary data structures, these data structures can be reused across multiple batch decode attention calls." Pointer: BatchDecodeWithPagedKVCacheWrapper note, lines 486-510.
6. **gpt-attention.md**
   - Source type: docs
   - URL: <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/gpt-attention.md>
   - Technique: Pack input tokens without padding and order context-phase sequences before generation-phase sequences for inflight batches. The worker’s per-step input assembly and padding decisions could adapt this as a stricter packed-layout invariant to cut wasted token slots and simplify slot/position metadata.
   - Evidence: Quote: "For efficiency reasons... the support for inflight batching requires the input tensors to be packed (no padding)." Quote: "the sequences that are going through the context phase must be before the sequences in the generation phase in the input tensor." Pointer: In-flight Batching, lines 246-250.
7. **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**
   - Source type: paper
   - URL: <https://www.usenix.org/conference/osdi24/presentation/agrawal>
   - Technique: Use chunked-prefill, stall-free scheduling, and more uniform batches so prefill work fills decode slack without pausing ongoing decode iterations. Worker microbatch slicing and attention metadata generation could adopt uniform-token chunk policies to lower median TTFT while avoiding TPOT stalls in mixed agent workloads.
   - Evidence: Quote: "Sarathi-Serve introduces chunked-prefills which splits a prefill request into near equal sized chunks and creates stall-free schedules that adds new requests in a batch without pausing ongoing decodes." Pointer: USENIX abstract.
8. **Employing CUDA Graphs in a Dynamic Environment**
   - Source type: blog
   - URL: <https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/>
   - Technique: Represent dynamic execution modes with either recognized graph instances or `cudaGraphExecUpdate` when topology is stable and only parameters change. This could improve per-step cudagraph budget/dispatch decisions and reduce recapture or fallback overhead for recurring batch shapes.
   - Evidence: Quote: "To handle changing kernel launch parameters, two methods can be used: saving and recognizing CUDA graphs or updating existing graphs using cudaGraphExecUpdate." Pointer: Summary and Save/update sections, lines 26-28 and 126-150.
9. **SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification**
   - Source type: paper
   - URL: <https://www.alphaxiv.org/abs/2305.09781v4>
   - Technique: Verify a token tree of diverse draft candidates in parallel instead of a single speculative sequence, then accept the longest valid path with greedy or stochastic verification. This provides a design direction for increasing tokens accepted per target forward and improving the worker rejection-sampling pipeline’s payoff per metadata/setup cost.
   - Evidence: Quote: "These candidates are organized as a token tree... The correctness of all candidate token sequences is verified against the LLM in parallel." Pointer: SpecInfer overview and Algorithm 2 discussion.
10. **flashinfer.mamba.checkpointing_ssu**
   - Source type: docs
   - URL: <https://docs.flashinfer.ai/generated/flashinfer.mamba.checkpointing_ssu.html>
   - Technique: Use a ring of cached Mamba inputs and matmul-based parallel token replay for MTP/speculative accepted-token handling. This can reduce repeated per-request state-copy bookkeeping by turning accepted-token replay into a GPU-resident ring-buffer operation with host-owned flush bookkeeping.
   - Evidence: Quote: "Checkpointing SSU with MTP replay using matmul-based parallel token processing." Quote: "x_cache – Ring of cached x..." Pointer: API description and parameters, lines 347-360.
11. **flashinfer.mamba.selective_state_update**
   - Source type: docs
   - URL: <https://docs.flashinfer.ai/generated/flashinfer.mamba.selective_state_update.html>
   - Technique: Support separate source and destination state indices and optional intermediate-state buffers inside the Mamba update kernel. The worker’s Mamba preprocessing/postprocessing could adapt this to avoid separate copy-spec staging and make align-mode state migration more directly GPU-driven.
   - Evidence: Quote: "When provided, state is read from state_batch_indices and written to dst_state_batch_indices (enables separate read/write state slots)." Pointer: `dst_state_batch_indices` parameter, lines 371-376.

## Issues

### module_deep_research
- **[warning, recoverable]** codex: The workspace did not have `rg` available during module inspection; grep and direct file reads were used instead.
