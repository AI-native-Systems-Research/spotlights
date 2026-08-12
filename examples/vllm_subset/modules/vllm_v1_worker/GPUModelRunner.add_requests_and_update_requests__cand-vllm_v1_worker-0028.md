# GPUModelRunner.add_requests and update_requests

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/model_runner.py`](vllm/v1/worker/gpu/model_runner.py) (lines 879–954)
- **Symbol:** `GPUModelRunner.add_requests and update_requests`
- **Kind:** region
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0028`

## Description
New GPU runner per-step request admission and cached-request state update path before input preparation.

## Current approach
Loops over scheduled_new_reqs and scheduled_cached_reqs, repeatedly removes/re-adds request state, performs dict lookups, stages request/model/sampler writes separately, appends block IDs per request, and updates CPU mirrors with per-request assignments plus a full np.minimum pass.

## Estimated impact explanation
Admission and cached-request churn are common in multi-turn agent workloads; batching state deltas reduces TTFT for new turns and TPOT during high-turnover decode.

## Evolve rationale
The for-loops in add_requests/update_requests and the staged write calls are the concrete optimization unit. tests/v1/worker/test_gpu_input_batch.py, tests/v1/worker/test_gpu_block_table.py, and streaming-input v2 tests validate request-index, staged-write, and block-table mutation semantics.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Vectorize update_requests via a single scatter over pre-packed CPU arrays
- **Agent:** claude

**Detailed description.**

Rewrite GPUModelRunner.update_requests (vllm/v1/worker/gpu/model_runner.py:935-954) so the per-request Python for-loop becomes one vectorized scatter. Concretely: (1) In the scheduler-output producer for scheduled_cached_reqs, materialize req_ids as a Python list already keyed to indices, and pass num_computed_tokens as a contiguous np.int32 array plus a parallel np.int32 array of req_indices resolved via a single dict.get lookup over req_ids (or better, maintain req_id_to_index as a small perfect-hash / np.searchsorted lookup so all indices can be gathered in one C-level pass). (2) Replace the per-iteration num_computed_tokens_np[req_index] = num_computed_tokens assignment with a single np.put/np.ndarray.__setitem__ scatter: num_computed_tokens_np[idx_arr] = ntok_arr. (3) Batch block-id appends: instead of calling self.block_tables.append_block_ids(req_index, req_new_block_ids, overwrite=False) once per request (each of which typically stages/writes a small slice and updates CPU length mirrors), add a BlockTable.append_block_ids_batched(idx_arr, flat_block_ids, offsets) that takes a jagged CSR-style (offsets, values) representation and performs one np.add.at on num_blocks_per_row plus one contiguous CPU->pinned staging copy; the existing per-request path stays as a thin wrapper. (4) Fold the trailing np.minimum(num_computed_tokens_np, prefill_len.np, out=num_computed_prefill_tokens) so it only operates on the touched slice (np.minimum.at on idx_arr, or a masked view) to avoid a full-batch pass every step even when only a few cached reqs updated. Do the analogous batching in add_requests for the fields that are cheap to pre-pack (req_index assignment, block_ids append, lora_state.add_request calls that are pure bookkeeping) while leaving the sampler/model_state staged writes as-is (they already batch via apply_staged_writes). Add microbenchmark coverage in benchmarks/kernels/ measuring wall time of update_requests at batch sizes {64,256,1024} with 1-3 new blocks each, and extend tests/v1/worker/test_gpu_input_batch.py / test_gpu_block_table.py to assert equivalence between the scalar and batched paths on random schedules.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete direction is novel by construction. The specific mechanism proposed — treating scheduled_cached_reqs as columnar arrays and doing a single np.put scatter for num_computed_tokens plus a CSR-style batched block_ids append, and reducing the full-batch np.minimum to a touched-slice update — targets the exact hot path (per-step admission/cached-req churn) called out in evolve_rationale and is not implied elsewhere in the candidate description.

---

### 2. Preserve request slots for streaming-input re-admission
- **Agent:** codex

**Detailed description.**

Change `GPUModelRunner.add_requests` in `vllm/v1/worker/gpu/model_runner.py:879-934` so streaming-input updates do not go through the full `_remove_request(req_id)` followed by `RequestState.add_request(...)` path. Add a `RequestState.replace_request(req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens) -> int | None` helper that overwrites the existing slot in place when the request is already present, stages the same tensor writes as `add_request`, zeros draft tokens, and returns the preserved `req_index`. In `add_requests`, call this helper first; only fall back to `_remove_request`/`add_request` for true fresh admission or for states that explicitly require teardown. Then refresh the dependent per-request state at that preserved index (`model_state.add_request`, `block_tables.append_block_ids(..., overwrite=True)`, `lora_state.add_request`, sampler/prompt-logprob state when present) without removing and re-inserting dictionary entries or pushing the index through `free_indices`. This keeps request-index identity stable for streaming chunks, avoids two dict mutations plus a free-list pop/push per chunk, and reduces churn in multi-turn/streaming agent workloads before input preparation. Cover it by extending the streaming-input v2 tests and `tests/v1/worker/test_gpu_input_batch.py` to assert that a re-admitted request keeps the same `req_index`, refreshes prompt/prefill/block-table state, and leaves unrelated active request indices unchanged.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal focuses on vectorizing `update_requests` with scatter writes, batched block appends, and touched-slice `np.minimum`, with only a brief analogous batching note for `add_requests`. This proposal targets a different source of overhead in `add_requests`: eliminating the remove/re-add lifecycle for streaming-input re-admission by preserving the existing request slot and refreshing state in place.

---
