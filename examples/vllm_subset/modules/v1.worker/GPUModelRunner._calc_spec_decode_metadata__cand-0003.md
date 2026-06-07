# GPUModelRunner._calc_spec_decode_metadata

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 2596–2674)
- **Symbol:** `GPUModelRunner._calc_spec_decode_metadata`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0003`

## Description
Builds SpecDecodeMetadata for speculative decoding, including sampled-token, draft-token, target-logit, bonus-logit, and logit index tensors.

## Current approach
Uses numpy cumsum/repeat/subtract helpers to build several small CPU arrays, then performs five separate torch.from_numpy(...).to(device, non_blocking=True) transfers for cu_num_draft_tokens, cu_num_sampled_tokens, logits_indices, target_logits_indices, and bonus_logits_indices. The local TODO explicitly calls out optimizing the CPU to GPU copy.

## Estimated impact explanation
The cost is fixed per spec-decode step rather than proportional to model size. Coalescing these transfers should reduce host overhead and median TPOT for speculative decoding, but only affects deployments with spec decode enabled.

## Evolve rationale
Spec-decode steps repeatedly compute related index arrays from the same inputs, then pay fixed launch/copy overhead for each small tensor. Headroom is in packing the arrays into a single staging buffer, reusing pinned scratch storage, or computing derived indices on GPU from num_draft_tokens and cu_num_scheduled_tokens. Correctness oracle: field-by-field equality of SpecDecodeMetadata and rejection-sampler outputs in tests/v1/spec_decode and tests/v1/e2e/spec_decode.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Coalesce SpecDecodeMetadata transfers via a single packed pinned-CPU staging buffer; derive bonus_logits_indices on GPU
- **Agent:** claude

**Detailed description.**

Replace the five independent torch.from_numpy(...).to(device, non_blocking=True) calls at vllm/v1/worker/gpu_model_runner.py:2645-2659 with a single coalesced host->device transfer using one persistent pinned staging buffer.

Concrete change inside GPUModelRunner._calc_spec_decode_metadata (lines 2596-2674):

1. Allocation (one-time, near self._arange_scratch at line 761): add a pinned byte-tensor scratch sized for the worst case, e.g.
   spec_meta_bytes = (3 * self.max_num_reqs) * 4  # three int32 arrays of length num_reqs
              + (2 * self.max_num_tokens) * 8  # two int64 arrays bounded by total scheduled tokens
   self._spec_meta_pinned = torch.empty(spec_meta_bytes, dtype=torch.uint8, pin_memory=self.pin_memory)
   self._spec_meta_gpu = torch.empty_like(self._spec_meta_pinned, device=self.device).

2. In _calc_spec_decode_metadata, after computing the numpy arrays (cu_num_sampled_tokens int32, cu_num_draft_tokens int32, logits_indices int64, target_logits_indices int64), write them into typed views over self._spec_meta_pinned at fixed offsets — use np.frombuffer(self._spec_meta_pinned.numpy(), dtype=...) views and assign in place rather than allocating fresh numpy arrays via np.repeat/np.cumsum where possible (pass out= to cumsum; allocate the repeat target once, reuse).

3. Issue ONE non_blocking copy of the used prefix:
   used = int32_region_bytes + int64_region_bytes
   self._spec_meta_gpu[:used].copy_(self._spec_meta_pinned[:used], non_blocking=True)
   then construct typed GPU views with .view(torch.int32) / .view(torch.int64) and slice into cu_num_sampled_tokens, cu_num_draft_tokens, logits_indices, target_logits_indices. These are zero-copy aliases into self._spec_meta_gpu.

4. Drop the bonus_logits_indices transfer entirely: compute it on-device as bonus_logits_indices = cu_num_sampled_tokens_gpu - 1 (single elementwise op on a num_reqs-sized int32 tensor). This removes one of the five copies outright and eliminates the bonus_logits_indices CPU array.

5. Keep the existing draft_token_ids gather (lines 2663-2664) unchanged — it operates on the GPU views.

Net result on the hot path: 5 small host->device launches collapse into 1 coalesced launch + 1 trivial GPU subtraction. The numpy intermediates stay (still needed for the np.repeat structure), but their results land directly in the pinned buffer rather than in throwaway allocations. The fixed per-step launch overhead — which dominates these tiny (num_reqs- and num_drafts-sized) transfers — drops from O(5) to O(1), directly improving median TPOT for spec-decode-enabled deployments without changing any sampler-visible field of SpecDecodeMetadata.

Validation: run tests/v1/spec_decode and tests/v1/e2e/spec_decode for field-by-field equality (cu_num_draft_tokens, cu_num_sampled_tokens, logits_indices, target_logits_indices, bonus_logits_indices, draft_token_ids), and verify rejection-sampler outputs are bit-identical. Benchmark TPOT with VLLM speculative_config enabled on a representative agentic multi-turn workload to confirm reduction in per-step host overhead.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, so any concrete plan is novel. The evolve_rationale gestures at 'packing into a single staging buffer' and 'GPU derivation' as directions, but does not specify the concrete layout (byte-level packing of mixed int32/int64 regions in one pinned allocation), the single-copy + view-slicing protocol, or the specific opportunity to elide bonus_logits_indices entirely by computing it on-device from cu_num_sampled_tokens. This proposal commits to that exact recipe in code terms, naming the buffer, sizes, slicing, and the dropped transfer.

---

### 2. Narrow spec-decode index metadata to int32 end-to-end
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu_model_runner.py::_calc_spec_decode_metadata`, keep `logits_indices` and `target_logits_indices` as `np.int32` / `torch.int32` instead of allowing the shared int64 `_arange_scratch` path to promote them. Add a spec-decode-only int32 arange/scratch buffer near the existing `self.arange_np` / `self._arange_scratch` initialization, assert the per-step `cu_num_scheduled_tokens[-1]` is within int32 bounds, and build the two repeated index arrays with int32 bases plus int32 offsets. `cu_num_draft_tokens`, `cu_num_sampled_tokens`, and `bonus_logits_indices` are already int32, and `SpecDecodeMetadata.make_dummy` already uses int32 index tensors, so this aligns the production path with existing sampler expectations. This halves the host-to-device bytes for the two largest metadata tensors and reduces GPU index bandwidth in the downstream `input_ids`, logits gather, and scatter paths, while preserving field values exactly. Validation should compare old/new metadata values on representative mixed draft-count batches and run `tests/v1/spec_decode`, `tests/v1/e2e/spec_decode`, and `tests/v1/sample/test_rejection_sampler.py` to confirm CUDA int32 indexing behavior remains covered.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes coalescing transfers through one packed pinned staging buffer and deriving `bonus_logits_indices` on GPU, but it explicitly treats `logits_indices` and `target_logits_indices` as int64 regions and does not change metadata dtypes or reduce downstream index bandwidth. This proposal is a separate, compatible optimization focused on narrowing the index representation itself.

---
