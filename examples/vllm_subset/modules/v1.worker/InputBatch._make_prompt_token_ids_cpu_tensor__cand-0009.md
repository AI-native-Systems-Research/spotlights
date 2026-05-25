# InputBatch._make_prompt_token_ids_cpu_tensor

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_input_batch.py`](vllm/v1/worker/gpu_input_batch.py) (lines 960–975)
- **Symbol:** `InputBatch._make_prompt_token_ids_cpu_tensor`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0009`

## Description
Materializes a padded CPU prompt-token tensor for sampling penalties, logits processors that need token IDs, and step-pooling metadata.

## Current approach
Allocates a fresh pinned CPU tensor sized to the current maximum prompt length, copies token_ids_cpu, then loops over requests in Python to fill each padded suffix with vocab_size.

## Estimated impact explanation
This does not affect the all-greedy/no-processor path, but when penalties or token-aware processors are active it removes an allocation and Python loop from per-step metadata refresh, improving median TPOT.

## Evolve rationale
The method is called whenever prompt token IDs are needed for penalties or token-aware processing. Headroom is in reusing a scratch buffer, vectorizing padding with an arange/broadcast mask, and avoiding work when prompt lengths are uniform or unchanged. Correctness oracle: exact prompt_token_ids tensors in tests/v1/worker/test_gpu_input_batch.py and sampling parity for penalty, bad-word, and logits-processor cases.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Maintain a persistent pre-padded prompt-token-ids CPU buffer updated at slot mutation time
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu_input_batch.py around lines 960-975, replace the per-step allocation+Python pad loop in `InputBatch._make_prompt_token_ids_cpu_tensor` by maintaining a persistent CPU tensor `self.prompt_token_ids_padded_cpu` of shape (max_num_reqs, max_model_len), pinned when `self.pin_memory` is set, pre-filled once with `self.vocab_size`. Push the work to the (rare) slot-mutation sites that already touch `self.token_ids_cpu`: in `add_request` (around line 362) write `prompt_token_ids_padded_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids` and reset the trailing region of just that single row to `vocab_size` (it may contain a prior occupant's prompt); in `condense` (around line 741) mirror the row-copy from `last_req_index` to `empty_index` and reset that row's suffix to `vocab_size`; in `swap_states` (around line 607) swap the corresponding rows. Then `_make_prompt_token_ids_cpu_tensor` becomes a non-allocating slice: `return self.prompt_token_ids_padded_cpu[:num_reqs, :int(self.num_prompt_tokens[:num_reqs].max())]` (optionally `.clone()` only if downstream callers require ownership; current callers in `get_pooling_metadata` and the sampler's penalty/bad-words path treat it as read-only metadata). Crucially, prompt tokens never change for a given request slot once added (output tokens are written to indices >= num_prompt_tokens in `token_ids_cpu`, but the *padded prompt buffer* is a separate array, so output tokens never contaminate the padded suffix). This removes one pinned allocation and one Python `for i in range(num_reqs)` loop from every step that uses penalties / bad-words / token-aware logits processors / step-pooling — exactly the hot loop in agentic multi-turn workloads where penalties and stop-token processors are common. Verification: tests/v1/worker/test_gpu_input_batch.py already asserts exact equality of the padded tensor; sampling parity for penalties, bad-words, and pooling token-id paths is the secondary oracle.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's own evolve_rationale only suggests three runtime-side optimizations (scratch-buffer reuse, vectorized arange/broadcast pad, skip-when-uniform). My proposal is structurally different: it eliminates the per-step work entirely by pushing the padded representation to the request-slot lifecycle (add/condense/swap), exploiting the invariant — not stated in the rationale — that prompt_token_ids for a given slot never mutate after `add_request`. A scratch buffer still pays per-step copy+pad costs; this design pays them once per add/condense and turns the per-step path into a pure slice.

---

### 2. Skip prompt-token materialization for frequency/presence-only penalties
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu_input_batch.py` around `_make_sampling_metadata`, make `needs_prompt_token_ids` depend on repetition penalties, not on `not self.no_penalties`: frequency and presence penalties in the current penalty implementation only use generated-output token counts/masks, while prompt token IDs are only needed for repetition penalties and token-ID-aware pooling/logits consumers. Add a small metadata flag such as `has_repetition_penalties` or `no_repetition_penalties`, set from `self.repetition_penalties_reqs`, and allow `SamplingMetadata.prompt_token_ids` to remain `None` when the batch has only frequency and/or presence penalties. Then update `vllm/v1/sample/sampler.py`, `vllm/v1/sample/rejection_sampler.py`, and `vllm/v1/sample/ops/penalties.py` so the frequency/presence-only path computes `output_tokens_t`, `output_bin_counts`, and `output_mask` as today but skips the prompt-mask/repetition branch entirely; keep the existing assert and full `apply_all_penalties` path when any request has a non-default repetition penalty. This removes `_make_prompt_token_ids_cpu_tensor()` allocation, padding, and CPU-to-device copy for common frequency/presence-only sampling while preserving exact behavior for repetition penalties and pooling token-id consumers. Add tests covering frequency-only and presence-only batches where `sampling_metadata.prompt_token_ids is None`, plus parity checks against the existing penalty outputs.

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate. Agent A proposes maintaining a persistent padded prompt-token buffer and updating it on slot lifecycle mutations; this proposal does not add or reuse a padded prompt buffer. It instead avoids calling `_make_prompt_token_ids_cpu_tensor` at all for a semantic subset of penalty workloads where prompt IDs are unnecessary under the current implementation, so it reduces work by tightening the consumer contract rather than by changing how the padded tensor is stored.

---
