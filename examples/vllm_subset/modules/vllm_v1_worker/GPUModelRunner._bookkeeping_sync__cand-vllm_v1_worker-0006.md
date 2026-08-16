# GPUModelRunner._bookkeeping_sync

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 3763–3910)
- **Symbol:** `GPUModelRunner._bookkeeping_sync`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0006`

## Description
Synchronous legacy post-sampling bookkeeping that consumes sampled token IDs, discards invalid rows, updates token_ids_cpu/is_token_ids/num_tokens_no_spec, and extends request output tokens.

## Current approach
Converts sampled_token_ids to Python lists, then loops over sampled requests to perform CPU tensor slices, dict lookups, and list extensions.

## Estimated impact explanation
This is the largest post-sample host roundtrip in synchronous legacy scheduling; deferring or coalescing CPU conversion reduces median TPOT and improves overlap after decode.

## Evolve rationale
The self._to_list(sampled_token_ids) call and per-sampled-request update loop are concrete sync and Python-work targets. Async output staging or vectorized CPU scatter can preserve observable output semantics covered by tests/v1/worker/test_gpu_model_runner.py::test_get_nans_in_logits, test_sample_tokens_*, and tests/v1/streaming_input/test_gpu_model_runner_streaming.py.

## Deep research proposals

### 1. Batch multiple decode steps before running _bookkeeping_sync to amortize its host work
- **Finding:** `find-vllm_v1_worker-0001` — *[RFC]: Multi-Step Scheduling*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/6854>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Wrap the sync bookkeeping in a worker-local multi-step decode loop as described in the RFC. Instead of calling _bookkeeping_sync (vllm/v1/worker/gpu_model_runner.py:3763-3910) after every sampled token, run N consecutive decode iterations in which: (1) sampled_token_ids stay on GPU and are appended to a per-step GPU-side ring buffer, (2) the per-step input metadata (positions, slot mappings, num_tokens_no_spec) is advanced by dedicated CUDA kernels rather than the current Python loop over range(num_sampled_tokens) that does token_ids_cpu / is_token_ids / num_tokens_no_spec scatter and req_state.output_token_ids.extend, and (3) the self._to_list(sampled_token_ids) D2H sync at line 3829 plus the invalid-row masking are deferred to the end of the N-step window. When the lookahead window exhausts (or a request finishes / a stop condition trips on GPU), a single _bookkeeping_sync-equivalent finalizer performs one coalesced D2H copy of the accumulated GPU-side token buffer, one vectorized CPU scatter into token_ids_cpu/is_token_ids/num_tokens_no_spec, and one batched extend of req_state.output_token_ids, and only then hands control back to the scheduler. Prompt-logprobs computation stays in the finalizer path; discard_sampled_tokens masking is applied to the accumulated buffer before the scatter.

**Proposal rationale.**

The candidate's hottest costs are exactly the two the RFC calls out: a synchronous D2H list conversion (_to_list at 3829) and per-decode-step Python bookkeeping that must run once per generated token. The existing async-scheduling branch already keeps sampled_token_ids on GPU for a single step, which shows the mechanism is viable here; the finding extends that idea by amortizing the same host work across N steps, directly cutting median TPOT for the multi-turn agentic workload named in the caller context, where decode dominates and per-token host overhead is repeatedly paid. The RFC's specific recommendation to use CUDA kernels for next-step input updates maps cleanly onto replacing the per-request scatter loop at 3868-3893.

---

### 2. Adopt MRV2 persistent-state/per-step decoupling to eliminate _bookkeeping_sync host roundtrip
- **Finding:** `find-vllm_v1_worker-0003` — *Model Runner V2 Design Document*
- **Source URL:** <https://docs.vllm.ai/en/v0.17.0/design/model_runner_v2/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor GPUModelRunner._bookkeeping_sync (vllm/v1/worker/gpu_model_runner.py:3763-3910) so that the post-sampling bookkeeping no longer needs to materialize sampled_token_ids on the host via self._to_list(...) and then loop per-sampled-request over token_ids_cpu/is_token_ids/num_tokens_no_spec/output_token_ids. Following the MRV2 design, treat token_ids_cpu, is_token_ids, and num_tokens_no_spec as persistent GPU-resident state tensors keyed by stable per-request row indices, and perform the append of newly sampled tokens as a vectorized GPU scatter into those persistent tensors using the already-known (row, position) indices computed from num_tokens_no_spec. The Python-side output_token_ids extension for request objects is then reduced to a lazy, asynchronous host copy of only the sampled column (or deferred until a scheduler/detokenizer boundary actually needs it), rather than being on the critical path of the decode step. Invalid-row discard is expressed as a mask applied to the same scatter (or a follow-up masked write) instead of a Python for-loop over valid_sampled_token_ids. Observable output semantics covered by tests/v1/worker/test_gpu_model_runner.py::test_get_nans_in_logits, test_sample_tokens_*, and tests/v1/streaming_input/test_gpu_model_runner_streaming.py must be preserved: the same token IDs must eventually appear in the same request output slots, only the timing/location of the host-visible copy changes.

**Proposal rationale.**

The candidate explicitly calls out self._to_list(sampled_token_ids) and the per-sampled-request update loop as the concrete sync/Python-work targets that hurt median TPOT under multi-turn agentic workloads. The MRV2 finding contributes exactly the transferable idea needed to attack that: keep large per-request state tensors GPU-resident and gather/scatter per-step inputs and outputs on the GPU rather than reordering or copying them back to host every step. That directly addresses the gap in _bookkeeping_sync where a full device->host sync plus a Python loop is used to maintain what is fundamentally row-indexed persistent state, and it plausibly reduces the post-sample host stall that dominates TPOT after decode in the legacy synchronous path.

---

### 3. Pipeline sync bookkeeping behind the next step to eliminate the _to_list D2H stall
- **Finding:** `find-vllm_v1_worker-0004` — *vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction*
- **Source URL:** <https://vllm-project.github.io/2024/09/05/perf-update.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor `GPUModelRunner._bookkeeping_sync` (vllm/v1/worker/gpu_model_runner.py:3763-3910) so the sync-scheduling branch no longer blocks the current step on `self._to_list(sampled_token_ids)` and the subsequent per-request CPU updates. Specifically, when `not self.use_async_scheduling`: (1) enqueue the D2H copy of `sampled_token_ids` (and logprobs tensors, `num_nans` counts, routed-experts buffers already staged at lines 3816-3823) into a pinned CPU buffer plus a CUDA event, and return a lightweight `DeferredBookkeeping` handle instead of the materialized `valid_sampled_token_ids` / `logprobs_lists`; (2) move the loop at lines 3868-3893 that writes `token_ids_cpu`, `is_token_ids`, `num_tokens_no_spec`, and `req_state.output_token_ids` into a follow-up `finalize_bookkeeping()` method that the model runner invokes after launching the next step's forward, mirroring the pattern the v0.6.0 blog attributes to `_process_model_outputs` in v0. Keep the existing async-scheduling branch unchanged; the new deferred path becomes the sync-mode analogue. Update the single-token fast-path (`max_gen_len == 1`) and the spec-decode `RejectionSampler.parse_output` path to both produce deferred handles, and preserve the discard-mask semantics by capturing `discard_sampled_tokens_req_indices` at issue time so `finalize_bookkeeping` can still clear invalid rows. Semantics observable by `test_get_nans_in_logits`, `test_sample_tokens_*`, and `test_gpu_model_runner_streaming` must remain identical because finalization runs before outputs are returned to the engine.

**Proposal rationale.**

The finding describes the exact optimization vLLM v0.6.0 applied at the engine level — overlap output processing (GPU→CPU token materialization, stop-criteria checks) with the next step's model execution — and this candidate is the surviving synchronous instance of that pattern inside the v1 worker. Today `_to_list` forces an `event.synchronize()` on the async copy stream (per the comment at lines 3812-3815) inside the current step's critical path, and the CPU-side loop at 3868-3893 runs before the runner can launch the next forward. Both are pure host work that could be scheduled after the next `execute_model` launches, directly attacking median TPOT on the multi-turn agentic workload described in the caller context. The async-scheduling branch already demonstrates that the per-step CPU tensor updates are not required for the *next* step's input preparation (it caches `prev_sampled_token_ids` on GPU instead), which is precisely the invariant the async-output-processing technique relies on — strong evidence the transfer of the idea is feasible for the sync path rather than merely topical.

---

## Agent proposals

### 1. Issue sampled_token_ids D2H immediately after sampling and vectorize the per-request CPU scatter with NumPy fancy indexing
- **Agent:** claude

**Detailed description.**

Two orthogonal changes to _bookkeeping_sync (vllm/v1/worker/gpu_model_runner.py:3763-3910) plus a small hoist in its caller (execute_model at ~4740-4817):

(A) Pull the D2H issue earlier in the step. Immediately after sampler_output is produced upstream — before the spec-decode drafting block that runs prepare_next_token_ids_padded / update_token_ids_ngram / _copy_valid_sampled_token_count / _copy_draft_token_ids_to_cpu at lines ~4740-4799 — enqueue the existing pinned-buffer non-blocking copy currently hidden inside _to_list: pinned = self.sampled_token_ids_pinned_cpu[:N]; pinned.copy_(sampled_token_ids, non_blocking=True); self.transfer_event.record(). Then have _to_list (vllm/v1/worker/gpu_model_runner.py:7878-7891) skip the copy/record when the caller has already issued them for this step and just do transfer_event.synchronize() + tolist(). The routed-experts D2H already enqueued at 3816-3823 stays as-is; both copies now overlap the drafter block instead of stalling inside _bookkeeping_sync. By the time transfer_event.synchronize() is reached, the copy is typically already complete, converting the current fixed sync stall into near-zero on the critical path.

(B) Vectorize the per-request scatter loop at lines 3868-3893 (fast path max_gen_len == 1) with NumPy fancy indexing on the existing pinned CPU tensors — token_ids_cpu, is_token_ids, and num_tokens_no_spec are already NumPy views over pinned torch tensors per gpu_input_batch.py:140/147/158. Replace the Python for-loop with: (1) build a boolean row mask from valid_sampled_token_ids that filters out discarded and empty rows (discarded rows already cleared to [] by the existing loop at 3831-3832); (2) rows = np.nonzero(mask)[0]; (3) tokens = np.fromiter((valid_sampled_token_ids[r][0] for r in rows), dtype=token_ids_cpu.dtype, count=rows.size) — a single tight Python pass rather than a per-row scatter; (4) starts = self.input_batch.num_tokens_no_spec[rows]; (5) self.input_batch.token_ids_cpu[rows, starts] = tokens; self.input_batch.is_token_ids[rows, starts] = True; self.input_batch.num_tokens_no_spec[rows] = starts + 1; (6) a small trailing Python loop only over rows to call self.requests[req_ids[r]].output_token_ids.append(int(tokens[k])) — unavoidable because output_token_ids is a plain Python list on the request object, but this pass touches only surviving rows and does no NumPy slice arithmetic per row. Keep the assertion end_idx <= self.max_model_len as a single vectorized (starts + 1).max() <= self.max_model_len check. The spec-decode path (max_gen_len > 1) continues to use RejectionSampler.parse_output at 3838-3843 and its variable-length-per-row scatter, but still benefits from (A). Async-scheduling branch (3844-3860) is untouched — its Python loop at 3868-3893 only sets sampled_ids=[-1] and can stay per-row or be similarly vectorized in a follow-up.

Observable semantics: token_ids_cpu, is_token_ids, num_tokens_no_spec, output_token_ids, discard-mask behavior, and generator offset rewinds at 3794-3797 are unchanged. Tests named in evolve_rationale (tests/v1/worker/test_gpu_model_runner.py::test_get_nans_in_logits, test_sample_tokens_*, tests/v1/streaming_input/test_gpu_model_runner_streaming.py) observe identical results. Only the wall-clock schedule of the D2H and the Python-vs-NumPy nature of the scatter change.

**Novelty rationale.**

None of the three listed deep_research_proposals contain either half of this proposal. (1) The RFC multi-step batching proposal (find-vllm_v1_worker-0001) explicitly amortizes bookkeeping across N steps and replaces per-step Python bookkeeping with CUDA kernels — a scheduler/RFC-level rewrite; this proposal preserves single-step semantics and does not touch the scheduler. (2) The MRV2 proposal (find-vllm_v1_worker-0003) migrates token_ids_cpu / is_token_ids / num_tokens_no_spec off host to persistent GPU tensors and does the scatter as a GPU kernel; this proposal keeps them as pinned CPU NumPy views and instead uses NumPy fancy indexing, which is far less invasive and does not require reworking every other consumer of those CPU views (gpu_input_batch.py:376-390, 526-527, 631-637, 765) that assume they are host-side arrays. (3) The v0.6.0-style async-output-processing proposal (find-vllm_v1_worker-0004) pushes the D2H wait and the per-request loop to AFTER the next execute_model launches, requiring a DeferredBookkeeping handle API and coordination with KV-connector finalize / EPLB / draft-after-bookkeeping consumers that today expect finalized state before execute_model returns. This proposal is orthogonal: it pulls the D2H ISSUE earlier within the current step (overlapping only with the drafter block that already runs before bookkeeping) rather than pushing the WAIT later across the step boundary, so downstream consumers still see finalized bookkeeping when execute_model returns. Additionally, the NumPy vectorization of the 3868-3893 loop attacks the pure Python per-request overhead that persists under proposals #1, #3, and #4 (and under #2 only if the full GPU-state migration ships), giving a meaningful TPOT win with a much smaller blast radius than any of the three existing proposals.

---

### 2. Fast-path bookkeeping for empty or fully discarded sampled rows
- **Agent:** codex

**Detailed description.**

Add an early-return/skip path in `GPUModelRunner._bookkeeping_sync` for steps where `num_sampled_tokens == 0` or where `discard_sampled_tokens_req_indices` covers every sampled request row before any sampled IDs are materialized into Python lists. In the current method, the discard mask is only applied after `sampled_token_ids = self._to_list(sampled_token_ids)`, so fully discarded rows still pay the synchronous D2H conversion, optional logprobs parsing, and the per-row bookkeeping setup even though they produce no request-visible token extensions. Concretely, compute a compact sampled-row count and, when all sampled rows are known invalid, return the same shape of empty `valid_sampled_token_ids`/logprob outputs expected downstream while still preserving the existing generator-offset rewinds at lines 3794-3797 and any NaN/error metadata that must be reported. For partial discard cases, build the valid-row indices first and copy/gather only those rows to host for the legacy path rather than converting the full sampled batch and then clearing invalid entries to `[]`. This keeps the observable behavior of discarded rows unchanged but avoids unnecessary host synchronization and Python list construction during preemptions, aborts, or requests whose sampled output is intentionally dropped.

**Novelty rationale.**

The listed deep-research proposals target broad scheduling or state-placement changes: multi-step batching, persistent GPU-resident state, or deferring bookkeeping behind the next step. Agent A targets earlier D2H issue and NumPy vectorization of the normal single-token scatter. None of them propose changing the discard handling order so invalid rows are filtered before `_to_list`, nor an early exit for fully discarded sampled batches. This is a narrower guard/compaction optimization around the existing discard semantics rather than an async pipeline, GPU-state migration, multi-step decode rewrite, or CPU scatter vectorization.

---
