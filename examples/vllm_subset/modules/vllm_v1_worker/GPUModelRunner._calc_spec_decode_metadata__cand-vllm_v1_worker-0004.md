# GPUModelRunner._calc_spec_decode_metadata

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 2891–2964)
- **Symbol:** `GPUModelRunner._calc_spec_decode_metadata`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0004`

## Description
Constructs cumulative draft/sample counts and logits index tensors for legacy speculative decoding.

## Current approach
Computes all index arrays on CPU with NumPy, then issues five independent async_tensor_h2d transfers for cu_num_draft_tokens, cu_num_sampled_tokens, logits_indices, target_logits_indices, and bonus_logits_indices; also returns num_draft_tokens as a Python list.

## Estimated impact explanation
Spec decode is a primary TPOT lever for agent serving; reducing per-step launch and copy overhead in its metadata path directly improves median TPOT for speculative workloads.

## Evolve rationale
The five H2D calls in this metadata builder are a concrete optimization unit. A packed struct-of-arrays transfer or GPU-side index kernel can keep the fixed contract while tests/v1/worker/test_gpu_model_runner.py::test_invalid_draft_suffixes_remain_rejected_in_metadata and rejection-sampler tests validate downstream semantics.

## Deep research proposals

### 1. Replace CPU-side spec-decode index computation and 5 H2D copies with a fused GPU kernel
- **Finding:** `find-vllm_v1_worker-0001` — *[RFC]: Multi-Step Scheduling*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/6854>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py:2891-2964 (GPUModelRunner._calc_spec_decode_metadata), replace the NumPy-on-CPU construction of cu_num_sampled_tokens, cu_num_draft_tokens, logits_indices, target_logits_indices, and bonus_logits_indices — followed by five independent async_tensor_h2d transfers — with a single GPU-resident computation. Concretely: (a) H2D-copy the small num_draft_tokens and cu_num_scheduled_tokens arrays once into device tensors (or reuse an existing device-side copy already produced upstream in input prep); (b) launch a small fused CUDA kernel (or torch.compile'd primitive over prefix-sum + repeat_interleave + arange) that produces all five index tensors directly on device in one launch, mirroring the exact arithmetic already documented in the comments (cumsum of num_sampled_tokens, repeat of segment starts, add arange, etc.); (c) return those device tensors as-is, and lazily materialize num_draft_tokens.tolist() only where SpecDecodeMetadata actually needs a Python list (or replace that field with the device tensor and update consumers). Keep the SpecDecodeMetadata dataclass contract stable so rejection-sampler tests and tests/v1/worker/test_gpu_model_runner.py continue to pass.

**Proposal rationale.**

The multi-step scheduling RFC's core empirical claim — 'We use Cuda kernels for faster updates because Torch is too slow' — targets exactly this class of per-step Python/NumPy input-metadata construction that feeds decode. This candidate exhibits the same anti-pattern the RFC calls out: index arithmetic is done on CPU with NumPy and then bounced to GPU via five separate async_tensor_h2d calls per speculative step. On agentic multi-turn workloads with speculative decoding enabled, this Python + copy overhead sits directly on the TPOT critical path once per decode step. Folding the five transfers into a single kernel (or a single packed transfer of the two small inputs plus device-side compute) removes four launch/copy round-trips per step and eliminates the CPU NumPy work, which is a transferable application of the finding's technique to this specific metadata builder without requiring the full multi-step scheduler restructuring.

---

### 2. Build spec-decode logits index tensors on-device to remove five H2D transfers and the num_draft_tokens tolist() sync
- **Finding:** `find-vllm_v1_worker-0002` — *[Performance]: Fully Async Spec-Decoding | Make `seq_lens_cpu` in CommonAttentionMetadata optional*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/29134>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Rework GPUModelRunner._calc_spec_decode_metadata (vllm/v1/worker/gpu_model_runner.py:2891-2964) so the five index arrays it currently constructs on the host with NumPy and copies via async_tensor_h2d — cu_num_draft_tokens, cu_num_sampled_tokens, logits_indices, target_logits_indices, and bonus_logits_indices — are produced on the device instead. Concretely: (1) upload num_draft_tokens and cu_num_scheduled_tokens once as a packed struct-of-arrays H2D (or reuse an already-device-resident copy if one exists on the input-batch path) rather than issuing five independent transfers; (2) implement a small fused CUDA/Triton kernel (or a torch.cumsum + torch.repeat_interleave + arange composition on the GPU) that derives all five outputs from num_draft_tokens and cu_num_scheduled_tokens on-device, keeping the exact tensor shapes and semantics that SpecDecodeMetadata's downstream consumers (rejection sampler, draft-token gather at input_ids.gpu[logits_indices]) already expect; (3) stop returning num_draft_tokens as a Python list in SpecDecodeMetadata.num_draft_tokens — keep it as a device tensor (or a lazily-materialized host view) so this builder no longer forces a .tolist() sync. Preserve the fixed contract validated by tests/v1/worker/test_gpu_model_runner.py::test_invalid_draft_suffixes_remain_rejected_in_metadata and rejection-sampler tests; update any consumer that iterates num_draft_tokens as a Python list to accept a tensor or defer the host materialization until after forward.

**Proposal rationale.**

The finding argues that realizing fully async spec decoding requires building spec-decode-adjacent metadata without host-side sequence-length views, so that next-step input preparation overlaps with the current forward pass instead of being serialized by host/device syncs. This candidate is exactly that kind of sync point on the spec-decode metadata path: five async_tensor_h2d launches plus a tolist() on num_draft_tokens together create a host-to-device barrier every step where drafts are verified. The finding's transferable idea — keep the metadata builder device-resident and drop the host mirrors — applies directly here: the inputs (num_draft_tokens, cu_num_scheduled_tokens) are cheap to keep on device, and all five outputs are pure functions of them expressible with cumsum/repeat_interleave/arange. Doing so shrinks per-step launch and copy overhead on the spec-decode critical path, which the candidate's estimated_impact and the caller's median-TPOT objective for multi-turn agentic workloads specifically target, without changing the SpecDecodeMetadata contract that downstream tests validate.

---

## Agent proposals

### 1. Amortize spec-decode metadata construction with a persistent pinned staging buffer plus one packed H2D and a torch.compile'd device kernel keyed by (batch_size, max_spec_len)
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py:2891-2964 (GPUModelRunner._calc_spec_decode_metadata), keep the builder's public SpecDecodeMetadata contract but replace both the five per-step async_tensor_h2d transfers and the per-step NumPy index arithmetic with an amortized two-part scheme:

1. Persistent pinned staging + single packed H2D. Allocate once (lazily, resized on growth like the existing input-batch buffers) a single pinned CPU tensor of shape [2, max_num_reqs] holding num_draft_tokens and cu_num_scheduled_tokens contiguously. In _calc_spec_decode_metadata, write the two int32 arrays into fixed row slices of this pinned buffer via a NumPy view (no per-call allocation), then issue exactly one async_tensor_h2d copy of the [2, num_reqs] slab into a preallocated device buffer. This removes four of the five H2D launches even before any kernel work, and it eliminates the per-step allocation of five small pageable host arrays that the current code round-trips through async_tensor_h2d.

2. Shape-specialized compiled derivation. Wrap the on-device derivation of cu_num_draft_tokens, cu_num_sampled_tokens, logits_indices, target_logits_indices, and bonus_logits_indices in a @torch.compile(dynamic=False, mode="reduce-overhead") function keyed by (bucketed_num_reqs, bucketed_max_num_draft_tokens_per_req) using the same bucketing scheme already used by CUDA graph capture in this file. This produces a captured graph per bucket that fuses cumsum + repeat_interleave + arange + add into a single launch, matches vLLM's existing compile/CUDA-graph strategy for the model forward, and avoids introducing a hand-written CUDA/Triton kernel. Fall back to eager for out-of-bucket shapes so correctness (and the tests/v1/worker/test_gpu_model_runner.py::test_invalid_draft_suffixes_remain_rejected_in_metadata assertions) is preserved regardless of shape.

Also: keep SpecDecodeMetadata.num_draft_tokens as a device tensor and add a cached .num_draft_tokens_cpu property that lazily .tolist()s on first host access, so callers that only need it on device (rejection sampler indexing) never trigger the sync, while callers that still need a Python list are unchanged.

**Novelty rationale.**

The two existing deep_research_proposals both propose moving the index math onto the device via a fused CUDA/Triton kernel (or cumsum+repeat_interleave+arange composition) and dropping the num_draft_tokens tolist() sync. Neither addresses two orthogonal costs that dominate at small batch sizes on this path: (a) per-step allocation and per-copy launch overhead of the H2D transfers themselves — this proposal collapses them into one packed transfer through a persistent pinned staging buffer sized like the existing input-batch buffers, which is a distinct optimization from fusing the compute, and (b) the launch/dispatch overhead of a device-side reduction+scatter pipeline at small num_reqs — this proposal specifically leverages vLLM's existing shape-bucketed torch.compile / CUDA-graph infrastructure to capture the derivation per bucket, rather than writing a new hand-rolled kernel that would still incur eager-mode dispatch. The lazy num_draft_tokens_cpu cached property is also a strictly weaker, backward-compatible variant of proposal 2's contract change, letting the optimization ship without touching every consumer in one PR.

---
