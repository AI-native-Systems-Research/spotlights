# GPUModelRunner.prepare_inputs

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu/model_runner.py`](vllm/v1/worker/gpu/model_runner.py) (lines 704–850)
- **Symbol:** `GPUModelRunner.prepare_inputs`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0013`

## Description
Builds the modular GPU InputBatch for a scheduler step: orders requests, maps batch rows to persistent request-state rows, prepares query starts, prefill input IDs, positions, sequence lengths, speculative logits indices, and CPU upper bounds.

## Current approach
Every step sorts request IDs by scheduled-token count, builds numpy arrays with fromiter/cumsum, performs several small async_copy_to_gpu transfers, allocates torch.arange/torch.zeros tensors in the no-draft path, and launches separate Triton kernels for prefill inputs, positions/sequence lengths, DCP sequence lengths, and sampled/draft token combination.

## Estimated impact explanation
The method runs once per forward pass before GPU execution. Reducing sorting, allocation, H2D, and launch overhead directly targets median TPOT and TTFT in multi-turn workloads.

## Evolve rationale
This is the central per-step input-prep method in the modular GPU runner. Headroom is in reusing scratch buffers, avoiding full sort when the scheduler already provides a stable decode/prefill partition, coalescing small H2D copies, caching common no-draft tensors by shape, and fusing adjacent input-prep kernels where contracts allow. Correctness oracle: field-by-field equality of returned InputBatch, prepared input_ids/positions/seq_lens/logits_indices, and end-to-end logits/sampling parity in v1 GPU model-runner v2 and streaming-input tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Pipeline prepare_inputs of step N+1 with the forward pass of step N on a side stream
- **Agent:** claude

**Detailed description.**

Today GPUModelRunner.prepare_inputs (vllm/v1/worker/gpu/model_runner.py:704-850) runs serially on the CPU/H2D path between the prior step's forward and the current step's forward, so its sort, several np.fromiter/cumsum builds, multiple async_copy_to_gpu transfers (idx_mapping, cu_num_logits, query_start_loc), torch.arange/torch.zeros allocations in the no-draft branch, and the four small Triton launches (prepare_prefill_inputs, prepare_pos_seq_lens, prepare_dcp_local_seq_lens, combine_sampled_and_draft_tokens) all sit on the critical path of TPOT. Restructure execution so that prepare_inputs for step N+1 begins as soon as the scheduler hands the runner its SchedulerOutput for that step, on a dedicated 'input-prep' CUDA stream, while the model forward of step N is still running on the main compute stream. Concretely: (1) keep all writes targeted at the existing self.input_buffers.* preallocated tensors and idx_mapping/cu_num_logits/logits_indices/expanded_* outputs, but issue every async_copy_to_gpu and every Triton launch from prepare_inputs on a stream owned by the runner (e.g. self._input_prep_stream); (2) at the boundary between steps, record a CUDA event after the previous step's forward writes to last_sampled_tokens / draft_tokens / num_computed_tokens.gpu (the only req_states fields that combine_sampled_and_draft_tokens / prepare_pos_seq_lens read) and have the input-prep stream wait on that event before running combine_sampled_and_draft_tokens and prepare_pos_seq_lens, so the dependency on freshly produced sampled/draft tokens is honored; (3) record a second event at the end of prepare_inputs and have the main compute stream wait on it before launching the next forward; (4) double-buffer only the small CPU-side scratch arrays that prepare_inputs writes (query_start_loc_np, cu_num_logits_np, seq_lens_cpu_upper_bound_np, idx_mapping_np) so that step N+1's CPU-side construction does not race with InputBatch consumers from step N; the large GPU buffers in self.input_buffers do not need duplication because their consumers are gated by the events above. This converts prepare_inputs latency from additive on the critical path into time hidden behind the previous forward, which directly attacks median TPOT in the multi-turn agentic workload where prefill is rare and the per-step forward is short enough that input-prep overhead is a meaningful fraction.

**Novelty rationale.**

The candidate has no listed deep_research_proposals. The evolve_rationale only suggests intra-step micro-optimizations (skip sort, coalesce H2D, cache shape-based scratch tensors, fuse adjacent kernels). This proposal is orthogonal: it does not change what prepare_inputs computes or which kernels it launches, but rather overlaps the entire method with the prior step's forward pass via a side CUDA stream and event-based synchronization on the specific req_states fields produced by sampling. None of the listed angles touch stream pipelining or event-driven dependency tracking between steps.

---

### 2. Bound padding work to the actual batch descriptor
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu/model_runner.py:758-803`, `prepare_inputs` already computes `num_reqs_padded = batch_desc.num_reqs or num_reqs`, but several operations still run to `self.max_num_reqs` every step. Change the query-start path to fill and copy only `query_start_loc_np[:num_reqs_padded + 1]`, padding `[num_reqs + 1:num_reqs_padded + 1]` only when FULL CUDA graph request padding is actually present. Then update `prepare_pos_seq_lens` to accept a `num_reqs_padded`/`pad_to_num_reqs` argument: for PIECEWISE/eager, launch only the `num_reqs` real request programs and skip the extra padding program; for FULL graphs, zero only `seq_lens[num_reqs:num_reqs_padded]` instead of `seq_lens[num_reqs:self.max_num_reqs]`. Apply the same bound to the DCP local seq-lens helper so it pads only the returned slice. This preserves the visible `InputBatch` fields while removing max-capacity CPU tail fills, H2D bytes, and Triton pad-loop work from the common case where the selected graph shape is smaller than `max_num_seqs`. Validate with field-by-field equality for PIECEWISE and FULL descriptors, with and without DCP, especially when `max_num_reqs > num_reqs_padded > num_reqs`.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A proposed overlapping step N+1 input preparation with step N forward on a side CUDA stream; this proposal does not change scheduling or stream synchronization. It is also distinct from generic H2D coalescing, scratch reuse, no-draft tensor caching, and kernel fusion: the specific change is to honor the existing `BatchExecutionDescriptor` bounds so prepare_inputs stops doing max-capacity padding work that no downstream consumer can observe.

---
