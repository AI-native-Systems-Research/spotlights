# stage_postprocess_inputs_to_gpu

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/mamba_utils.py`](vllm/v1/worker/mamba_utils.py) (lines 1246–1292)
- **Symbol:** `stage_postprocess_inputs_to_gpu`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0014`

## Description
Stages per-request scalar inputs for the fused legacy Mamba postprocess kernel after forward.

## Current approach
Loops over num_reqs, performs several dict lookups per request, writes four pinned NumPy buffers, and then issues four copy_to_gpu calls.

## Estimated impact explanation
The cost is specific to hybrid spec-decode runs; collapsing four per-step copies into one packed buffer reduces launch overhead and improves TPOT for that workload slice.

## Evolve rationale
The four separate staging buffers and H2D copies are the concrete target. Existing Mamba spec-decode parity tests and tests/v1/worker/test_mamba_utils.py validate the aggregate postprocess inputs and kernel outcomes.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Reuse runner's already-staged num_scheduled/num_computed GPU tensors and pack the remaining two per-request values into a single H2D copy
- **Agent:** claude

**Detailed description.**

Rework `stage_postprocess_inputs_to_gpu` (vllm/v1/worker/mamba_utils.py:1246-1292) and the `run_fused_postprocess` call site (lines 1228-1235) to stop restaging values the runner has already staged to GPU for this same batch, rather than merely packing all four buffers into one.

Concretely:

1. In `MambaSpecDecodeGPUContext.create` (line 593-596), drop `num_scheduled_tokens_buf` and `num_computed_tokens_buf`. Keep only a single packed pinned buffer, e.g. `postprocess_extras_buf: CpuGpuBuffer` of shape `[max_num_reqs, 2]` int32 whose two columns hold `mamba_state_idx` and `num_draft_tokens`. Update the four `assert ... is not None` tripwires accordingly at lines 1212-1215 and 1265-1268.

2. Rewrite the staging loop (lines 1277-1292). The runner (`vllm/v1/worker/gpu_model_runner.py:818,827`) already exposes `self.num_computed_tokens` and `self.num_scheduled_tokens` as GPU tensors populated by `_prepare_inputs` in exactly the same `req_ids[:num_reqs]` order that the fused postprocess kernel indexes. Plumb those two GPU tensors into the postprocess call path (a new arg pair on `run_fused_postprocess`, or store references on `ctx` during `initialize_from_forward_context` at line 1218). Then the CPU staging loop only needs the two remaining values:

   extras_np = ctx.postprocess_extras_buf.np  # shape [max_num_reqs, 2]
   scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
   for i in range(num_reqs):
       req_id = req_ids[i]
       state_idx = mamba_state_idx.get(req_id)
       assert state_idx is not None, (...existing message...)
       extras_np[i, 0] = state_idx
       extras_np[i, 1] = len(scheduled_spec_tokens.get(req_id, []))
   ctx.postprocess_extras_buf.copy_to_gpu(num_reqs)

3. Update `run_fused_postprocess` (called at lines 1228-1235) so it receives `mamba_state_idx_gpu = extras_gpu[:, 0]`, `num_draft_tokens_gpu = extras_gpu[:, 1]`, and takes `num_scheduled_tokens_gpu` / `num_computed_tokens_gpu` directly from the runner-owned tensors (sliced to `[:num_reqs]`). The kernel signature does not change; only the pointers passed in do.

Net effect on this specific hot path in hybrid spec-decode: 4 H2D copies collapse to 1, 4 pinned buffers collapse to 1, per-request Python work drops from 4 dict lookups + 4 numpy writes to 2 dict lookups + 2 writes, and the redundant staging of `num_computed_tokens` / `num_scheduled_tokens` disappears entirely. Existing coverage in `tests/v1/worker/test_mamba_utils.py` and the mamba spec-decode parity tests validates the postprocess inputs and kernel outcome unchanged.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, so the only overlap concern is with the candidate's own `evolve_rationale`. That rationale points at the direction 'collapse four per-step copies into one packed buffer,' which is a mechanical staging-buffer merge. This proposal is strictly stronger and structurally different: it observes that two of the four values (`num_scheduled_tokens`, `num_computed_tokens`) are *already* staged to GPU by `gpu_model_runner._prepare_inputs` in the same batch order (`vllm/v1/worker/gpu_model_runner.py:818,827`) and reuses those existing tensors rather than restaging them. That eliminates two of the four buffers outright, halves the per-request Python work, and reduces H2D copies to one — a cross-module data-flow change rather than a local buffer repack. Nothing in the candidate metadata or the (empty) prior proposals mentions reusing the runner's already-staged tensors.

---

### 2. Overlap postprocess input staging copies with the model forward pass
- **Agent:** codex

**Detailed description.**

Change `stage_postprocess_inputs_to_gpu` in `vllm/v1/worker/mamba_utils.py:1246-1292` so its H2D transfers are issued on a dedicated staging CUDA stream owned by `MambaSpecDecodeGPUContext`, then record a CUDA event after the copies. In `postprocess_mamba_align_gpu`, before `ctx.run_fused_postprocess(...)`, make the current stream wait on that event. The scalar inputs are already known in `_prepare_inputs` before model execution, but the fused postprocess kernel only consumes them after `_update_states_after_model_execute`; moving the `copy_to_gpu(num_reqs)` calls under a non-default stream lets those small pinned-buffer transfers overlap the forward/sampling work instead of sitting directly in the pre-forward critical path. Add the stream/event fields when creating `MambaSpecDecodeGPUContext`, guard the path for `num_reqs == 0`, and update `tests/v1/worker/test_mamba_utils.py` to assert the staged values are still visible after synchronizing the recorded event/current stream.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes reducing redundant staging by reusing runner-owned GPU tensors and packing the remaining staged values into one copy. This proposal is different: it does not depend on changing which tensors are staged or the kernel argument set, and it is still useful even if the four buffers remain separate. It targets the timing of the existing staging work by overlapping the H2D copies with forward execution, which neither the candidate rationale nor Agent A's proposal covers.

---
