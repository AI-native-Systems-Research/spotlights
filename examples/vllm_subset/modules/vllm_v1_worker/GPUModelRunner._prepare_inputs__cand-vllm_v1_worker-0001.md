# GPUModelRunner._prepare_inputs

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 2001–2323)
- **Symbol:** `GPUModelRunner._prepare_inputs`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0001`

## Description
Main legacy runner per-step CPU-to-GPU input preparation for request indices, positions, token indices, query_start_loc, optimistic sequence lengths, discard masks, slot mapping, and spec-decode counts.

## Current approach
Uses several NumPy/Torch CPU passes, torch.index_select, repeated CpuGpuBuffer.copy_to_gpu calls, a per-request prompt-embeds loop, a req_ids list walk for num_tokens, and an event synchronize before staging num_accepted_tokens.

## Estimated impact explanation
It is one of the largest steady per-step host costs in the runner; packing small H2D copies and removing per-request Python work directly targets median TPOT, especially in low-batch multi-turn agent decode.

## Evolve rationale
The method runs every non-empty legacy runner step on the TTFT/TPOT path. Coalescing H2D staging, removing Python request walks, and replacing the accepted-token sync contract can preserve positions, query_start_loc, slot_mapping, and request-state semantics covered by tests/v1/worker/test_gpu_model_runner.py::test_update_states_* and tests/v1/attention/test_attention_backends.py.

## Deep research proposals

### 1. Amortize _prepare_inputs via worker-local multi-step decode with on-GPU input advancement
- **Finding:** `find-vllm_v1_worker-0001` — *[RFC]: Multi-Step Scheduling*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/6854>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend GPUModelRunner._prepare_inputs (vllm/v1/worker/gpu_model_runner.py:2001-2323) to support a worker-local multi-step decode mode. Instead of executing the full CPU-side preparation pass every decode iteration, prepare inputs once for a batch and then, for N-1 lookahead steps, advance positions, query_start_loc, slot_mapping, and optimistic seq_lens directly on the GPU using small CUDA kernels (or fused torch ops) that consume the previous step's on-device sampled tokens. Keep sampled tokens on GPU across steps, defer the req_ids Python walk and CpuGpuBuffer.copy_to_gpu staging until the lookahead window is exhausted, and remove the per-step event synchronize before staging num_accepted_tokens by promoting num_accepted_tokens to a GPU-resident tensor updated in place by the same advancement kernel. Discard masks and spec-decode counts would be recomputed on-device from the GPU token buffer; the per-request prompt-embeds loop remains gated to prefill-only steps so pure-decode multi-step iterations skip it entirely. The scheduler-visible outputs are materialized only at the end of the multi-step window, matching the RFC's 'delay scheduler/output synchronization until lookahead slots are exhausted' contract, while preserving the invariants exercised by tests/v1/worker/test_gpu_model_runner.py::test_update_states_* and tests/v1/attention/test_attention_backends.py.

**Proposal rationale.**

The candidate's dominant per-step costs are exactly the ones the RFC identifies as amortizable: repeated small H2D copies via CpuGpuBuffer.copy_to_gpu, NumPy/Torch CPU passes to rebuild positions/query_start_loc/slot_mapping, a Python req_ids walk for num_tokens, and an event synchronize before staging num_accepted_tokens. The finding's core technique (keep tokens on GPU, advance next-step metadata with CUDA kernels, defer sync) maps directly onto these hotspots and targets median TPOT on the multi-turn agentic workload named in the caller context, where decode-heavy steps dominate. It does not merely restate the current approach: it replaces per-step CPU staging with on-device advancement and changes the accepted-token sync contract, which the candidate's evolve_rationale explicitly flags as in-scope.

---

### 2. Defer seq_lens host materialization in _prepare_inputs to enable async spec-decode overlap
- **Finding:** `find-vllm_v1_worker-0002` — *[Performance]: Fully Async Spec-Decoding | Make `seq_lens_cpu` in CommonAttentionMetadata optional*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/29134>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In GPUModelRunner._prepare_inputs (vllm/v1/worker/gpu_model_runner.py:2001-2323), stop treating the CPU seq_lens buffer as a required output of per-step input prep. Concretely: (1) build the CommonAttentionMetadata for this step with seq_lens_cpu marked optional/None whenever downstream attention backends and spec-decode metadata consumers can accept device-resident seq_lens plus an upper bound; (2) remove the event.synchronize() gate that currently precedes staging num_accepted_tokens, replacing it with a device-side path that derives accepted-token counts from device tensors so the host does not need to wait on the prior step's sampler event; (3) coalesce the small CpuGpuBuffer.copy_to_gpu calls (positions, query_start_loc, slot_mapping, seq_lens when still needed) into a single staged H2D so the remaining host work does not re-serialize with the forward. Keep a CPU fallback path for backends that still require seq_lens_cpu, selected once at init rather than per-step, so tests/v1/worker/test_gpu_model_runner.py::test_update_states_* and tests/v1/attention/test_attention_backends.py continue to see identical positions, query_start_loc, slot_mapping, and request-state semantics.

**Proposal rationale.**

The candidate explicitly calls out an event.synchronize() before staging num_accepted_tokens and repeated small H2D copies as the dominant per-step host costs on the TPOT path; the finding argues that removing the requirement to know seq_lens_cpu when building attention metadata is precisely what unblocks overlapping next-step input prep with the current forward, which is the same sync this candidate wants to remove. Making seq_lens_cpu optional in the metadata contract is the enabling change that lets _prepare_inputs skip the host-visible sequence-length materialization and the accepted-token sync, directly targeting the median TPOT in low-batch multi-turn agentic decode where the sync-per-step tax dominates. It is transferable rather than topically adjacent because the finding names the exact metadata field and consumer pattern (spec-decode verifying multiple drafted tokens) that this method assembles.

---

### 3. Adopt MRV2 persistent-state pattern: gather per-step inputs on GPU from GPU-resident state
- **Finding:** `find-vllm_v1_worker-0003` — *Model Runner V2 Design Document*
- **Source URL:** <https://docs.vllm.ai/en/v0.17.0/design/model_runner_v2/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor GPUModelRunner._prepare_inputs (vllm/v1/worker/gpu_model_runner.py, lines 2001-2323) to follow the Model Runner V2 design: decouple persistent request state (num_computed_tokens, token_ids, num_accepted_tokens, req_id_to_index) from per-step input tensors and keep the persistent state predominantly GPU-resident with stable row assignments for active requests. Concretely: (1) Compute positions and token_indices on GPU by gathering from a GPU-resident num_computed_tokens_gpu and token_ids_gpu using an already-uploaded req_indices tensor, instead of the current CPU pipeline (positions_np = num_computed_tokens_cpu[req_indices] + query_pos.np, followed by torch.index_select on CPU into input_ids.cpu at lines 2033-2065). (2) Replace the per-request Python walk that builds num_tokens (line 2138: `num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]`) with a GPU-resident num_tokens tensor maintained by state-update paths, so discard_request_mask can be computed on GPU without a CPU list comprehension. (3) Coalesce the many small CpuGpuBuffer.copy_to_gpu calls (query_start_loc, discard_request_mask, req_indices, query_pos, num_scheduled_tokens, num_accepted_tokens, num_decode_draft_tokens, prev_positions, prev_num_draft_tokens) into a single packed H2D transfer per step, since MRV2's stable rows shrink per-step CPU-authored data to only what changed. (4) Remove the num_accepted_tokens_event.synchronize() at line 2157 by producing num_accepted_tokens directly on GPU in the previous step's D2H-free path, using MRV2's stable row identity to avoid the async-mode reordering branch (lines 2159-2170). The refactor should preserve the observable contracts covered by tests/v1/worker/test_gpu_model_runner.py::test_update_states_* and tests/v1/attention/test_attention_backends.py (positions, query_start_loc, slot_mapping, request-state semantics).

**Proposal rationale.**

The finding is vLLM's own MRV2 design document, which explicitly names the exact bottlenecks called out in the candidate's evolve_rationale: per-step host bookkeeping and tensor-wide reordering on the TTFT/TPOT path. The quoted claim that 'Large state tensors are mostly stored on GPU memory, so gather runs in parallel on the GPU with low overhead' directly targets the candidate's CPU-side positions/token_indices/index_select pipeline and the many small H2D copies. Stable row assignments (persistent batch) eliminate the async-mode prev_positions reordering, which is the specific reason the num_accepted_tokens D2H sync exists today. Because MRV2 is a vLLM design doc rather than an external technique, the mapping to this file is concrete and low-risk: state tensors already have a CpuGpuBuffer abstraction that can be repurposed to hold GPU-primary state. This especially helps the caller's stated multi-turn agentic workload, where low-batch decode makes the fixed per-step Python/H2D overhead dominant in median TPOT.

---

### 4. Remove the per-step accepted-tokens sync from _prepare_inputs by deferring num_accepted_tokens materialization one step
- **Finding:** `find-vllm_v1_worker-0004` — *vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction*
- **Source URL:** <https://vllm-project.github.io/2024/09/05/perf-update.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py::GPUModelRunner._prepare_inputs (lines 2001-2323), replace the current event-synchronize-then-stage pattern for num_accepted_tokens with an asynchronous contract analogous to vLLM v0.6.0's async output processing. Concretely: keep the previous step's accepted-token counts in a GPU-resident (or double-buffered pinned) tensor produced by the sampler/spec-decode kernel, and consume that tensor on-device for step n+1's input staging (query_start_loc adjustment, slot_mapping fill, spec-decode counts) instead of forcing a host readback via CUDA event sync inside _prepare_inputs. Where a host value is genuinely required (e.g., early exit or CPU-side bookkeeping), route it through the same 'overlap with next model execution' pattern the blog describes: perform the synchronization/materialization one step later, while the next forward pass is already in flight, so _prepare_inputs itself never blocks on the previous step. Preserve request-state semantics covered by tests/v1/worker/test_gpu_model_runner.py::test_update_states_* and the attention-backend contracts in tests/v1/attention/test_attention_backends.py by keeping the observable positions, query_start_loc, and slot_mapping values byte-identical; only their production timing changes. Do not touch the token-id/logprobs materialization path, which lives outside _prepare_inputs.

**Proposal rationale.**

The candidate explicitly lists 'an event synchronize before staging num_accepted_tokens' as a per-step host cost on the TTFT/TPOT path, and the finding's central technique is exactly this: eliminate a host-side synchronous checkpoint by overlapping it with the next step's model execution ('performing the processing of the n-th step output while executing the (n+1)-th step'). The gap addressed is the CPU bubble the sync introduces on every non-empty legacy step, which is disproportionately painful in the caller's stated low-batch multi-turn agentic decode workload where per-step CPU overhead dominates TPOT. The rest of the finding (delaying token-id/logprobs conversion) is not applicable to _prepare_inputs and is deliberately excluded from the proposal.

---

## Agent proposals

### 1. Add a pure-decode fast path in _prepare_inputs that skips np.repeat, cumsum, and index_select when every request has num_scheduled_tokens==1
- **Agent:** claude

**Detailed description.**

In GPUModelRunner._prepare_inputs (vllm/v1/worker/gpu_model_runner.py:2001-2323), branch on a cheap steady-state predicate at the top of the method: `is_pure_decode = (total_num_scheduled_tokens == num_reqs) and not use_spec_decode and not self.enable_prompt_embeds and not self.input_batch.req_prompt_embeds and not self.uses_mrope and self.uses_xdrope_dim == 0 and not self.lora_config`. Under this predicate, replace the generic CPU pipeline (lines 2022-2072, 2113-2120, 2141-2146, 2273-2308) with a specialized fast path that produces byte-identical outputs: (1) `req_indices = self.arange_np[:num_reqs]` — no np.repeat allocation; (2) skip `_get_cumsum_and_arange` entirely — `cu_num_tokens = self.arange_np[1:num_reqs+1]` (precomputable once), `query_pos` slice is all-zeros so the additive term in positions is skipped; (3) `positions_np = self.input_batch.num_computed_tokens_cpu[:num_reqs]` — a direct view, no allocation; (4) collapse `torch.index_select` into a single 2D gather: `self.input_ids.cpu[:num_reqs].copy_(self.input_batch.token_ids_cpu_tensor[torch.arange(num_reqs), self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]])`, which is a single kernel over num_reqs rather than a flatten+index_select over num_reqs elements of a (num_reqs, max_len) tensor; (5) `self.query_start_loc.np[:num_reqs+1] = self.arange_np[:num_reqs+1]` via a single precomputed slice write, then a single `copy_to_gpu`; (6) `discard_request_mask` collapses to `(num_computed_tokens_cpu + 1) < num_tokens_cached_np`, where `num_tokens_cached_np` is a small companion buffer maintained alongside `num_computed_tokens_cpu` in InputBatch state-update paths (add if not present), removing the Python `[self.requests[r].num_tokens for r in self.input_batch.req_ids]` list comprehension at line 2138; (7) `logits_indices = query_start_loc[1:] - 1` is already the pure-decode result — computed directly. Any predicate miss falls back to the existing slow path unchanged, so the diff is additive. This targets the exact steady state of low-batch multi-turn agentic decode where TPOT is dominated by Python/NumPy CPU overhead per step, not by H2D bandwidth or GPU compute.

**Novelty rationale.**

The four listed deep_research_proposals all target orthogonal mechanisms: on-GPU input advancement across a multi-step window (#1), making seq_lens_cpu optional in the attention-metadata contract (#2), MRV2 GPU-resident persistent state with stable rows (#3), and deferring the num_accepted_tokens D2H sync one step (#4). None of them eliminate the per-step CPU work of `np.repeat`, `_get_cumsum_and_arange`, the flatten+index_select gather, the `query_start_loc` NumPy write, or the `num_tokens` Python list comprehension — they all leave that code executing on the critical path and instead attack surrounding sync/staging. The fast path proposed here is a localized branch that pays off immediately without changing state ownership, attention-metadata contracts, or step boundaries, and it composes with all four: it makes the boundary step of proposal #1 cheaper, it removes work that proposals #2/#4 leave intact, and it is a much smaller intervention than the state refactor in #3.

---
