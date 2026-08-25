# AsyncOutput.get_output

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/async_utils.py`](vllm/v1/worker/gpu/async_utils.py) (lines 32–102)
- **Symbol:** `AsyncOutput.get_output`
- **Kind:** region
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0027`

## Description
New GPU runner asynchronous output staging and final materialization of sampled tokens, logprobs, NaN counts, routed experts, and EP fault status.

## Current approach
Records a copy-stream event, synchronizes in get_output, converts sampled_token_ids and num_sampled_tokens NumPy arrays to Python lists, trims each row in a Python loop, and may call .item() plus mask.cpu().tolist() for fault reporting.

## Estimated impact explanation
This is the new runner counterpart to the legacy post-sample host roundtrip; reducing blocking conversion and Python row work lowers median TPOT in async multi-turn decode.

## Evolve rationale
The copy_event.synchronize, sampled_token_ids.tolist, num_sampled_tokens_np.tolist, per-row trim loop, and _has_fault.item are concrete sync/materialization targets. tests/v1/streaming_input/test_gpu_model_runner_v2_streaming.py, tests/v1/sample/test_logprobs.py, and tests/v1/worker/test_gpu_model_runner.py validate visible output, logprob, and error semantics.

## Deep research proposals

### 1. Batch AsyncOutput materialization across a multi-step decode window
- **Finding:** `find-vllm_v1_worker-0001` — *[RFC]: Multi-Step Scheduling*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/6854>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend AsyncOutput in vllm/v1/worker/gpu/async_utils.py (lines 32-102) so that when the runner is executing a lookahead window of N decode steps for the same request batch, per-step sampled_token_ids and num_sampled_tokens remain resident on GPU and are appended to a per-request GPU buffer instead of triggering a copy-stream D2H + copy_event.synchronize + tolist + per-row trim on every step. Concretely: (1) add a 'lookahead' mode to AsyncOutput where __init__ still records a copy event on the copy stream but only enqueues D2H copies for tensors that are actually needed to advance the next step's input metadata (which under multi-step is handled by GPU kernels, so nothing) and defers sampled_token_ids/num_sampled_tokens copies; (2) accumulate GPU-side per-step sampled tokens into a preallocated [N, max_reqs, max_spec+1] tensor; (3) on the final step of the window, issue a single D2H copy of the accumulated tensor plus num_sampled_tokens, and let get_output perform one tolist and one vectorized trim (numpy slicing keyed off num_sampled_tokens_np) that produces the list-of-lists ModelRunnerOutput.sampled_token_ids the scheduler expects; (4) for intermediate steps, return a lightweight AsyncModelRunnerOutput that produces an empty/no-op ModelRunnerOutput so the scheduler does not observe them. Logprobs, num_nans, routed_experts, and _has_fault paths retain their current per-step behavior when they are actually requested, since those are typically not used in pure decode lookahead; when they are requested the window collapses to N=1 to preserve semantics. The change keeps the existing single-step call path (N=1) byte-for-byte identical, so tests/v1/streaming_input/test_gpu_model_runner_v2_streaming.py, tests/v1/sample/test_logprobs.py, and tests/v1/worker/test_gpu_model_runner.py continue to exercise the current code path.

**Proposal rationale.**

The candidate's evolve_rationale explicitly identifies copy_event.synchronize, sampled_token_ids.tolist, num_sampled_tokens_np.tolist, the per-row trim loop, and _has_fault.item as concrete sync/materialization targets that dominate host overhead in the async decode path. The finding's core insight — amortize per-step output materialization and scheduler roundtrips across an N-step decode window while keeping sampled tokens on GPU — maps directly onto these targets: at N=8, the per-step tolist+trim+synchronize cost is paid once instead of eight times, which is exactly the median-TPOT lever the caller context (multi-turn agentic, TPOT-sensitive) is asking for. The finding also validates the specific mechanism (GPU-resident sampled tokens, delayed scheduler sync) that the candidate would need to adopt, and this candidate is the natural host-side integration point because it is already the single place where sampled tokens transition from GPU to the ModelRunnerOutput contract. The change is transferable and scoped to this candidate's file even though the broader multi-step architecture requires scheduler cooperation, because the AsyncOutput class is the exact boundary at which per-step host materialization can be turned into per-window host materialization.

---

## Agent proposals

### 1. Land sampler outputs directly in pinned/UVA host buffers to eliminate per-step D2H copy and synchronize
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu/async_utils.py:22-66, replace the pattern where sampled_token_ids, num_sampled_tokens, num_nans, and _has_fault are produced on GPU and then D2H-copied non-blocking on copy_stream with UVA-backed output buffers taken from a per-runner UvaBufferPool (already defined in vllm/v1/worker/gpu/buffer_utils.py:44-101, providing pinned CPU tensors with an accelerator UVA view). Concretely: (1) the sampler in vllm/v1/worker/gpu/sample/ writes sampled_token_ids and num_sampled_tokens into a UvaBuffer.uva view sized [max_reqs, max_spec+1] and [max_reqs], and similarly for num_nans and (as a single-element int8) the EP fault flag; (2) AsyncOutput.__init__ stops enqueuing async_copy_to_np for these tensors on copy_stream and instead retains references to the UvaBuffer.np NumPy views — no copy event needed for the integer-shaped outputs; (3) get_output waits on a lightweight main-stream event that fires when the sampling kernel finishes (writes through UVA to pinned host memory are visible to the host on kernel completion), and only synchronizes copy_event when logprobs_tensors / prompt_logprobs_dict / routed_experts_cpu are present (those paths keep their current copy-stream D2H); (4) replace the sampled_token_ids.tolist() + per-row Python trim loop at lines 75-78 with a vectorized construction that slices each row of the pinned NumPy view to num_sampled_tokens_np[i] before calling tolist(), avoiding the transient full-width max_reqs*(max_spec+1) Python-int allocation that the current code deletes immediately after creating; (5) for the EP-fault path (lines 94-100), the pinned single-element fault flag makes _has_fault a numpy scalar read rather than a .item() device sync, so the common no-fault path no longer forces a host wait even when check_ep_fault=True. Correctness with tests/v1/streaming_input/test_gpu_model_runner_v2_streaming.py, tests/v1/sample/test_logprobs.py, and tests/v1/worker/test_gpu_model_runner.py is preserved because the observable ModelRunnerOutput fields are byte-identical (same integer contents, same trimmed list-of-lists shape); the change is fenced behind is_uva_available() from vllm.utils.platform_utils with a fallback to the current D2H path.

**Novelty rationale.**

The existing deep_research_proposal (find-vllm_v1_worker-0001) amortizes per-step materialization across an N-step decode window but still relies on non-blocking D2H copies plus event synchronization at window boundaries, and explicitly collapses to N=1 when logprobs/nans/routed_experts are requested. This proposal is orthogonal: it eliminates the D2H copy path itself for the integer-shaped outputs by reusing the codebase's already-existing UvaBufferPool infrastructure in vllm/v1/worker/gpu/buffer_utils.py so kernel writes land directly in host-visible pinned memory. It (a) improves the N=1 single-step path that the batching proposal falls back to and does not accelerate — directly benefiting the multi-turn agentic TTFT/TPOT objective when the workload is not a clean lookahead window; (b) targets the _has_fault.item() device sync (lines 94) which the batching proposal does not touch; (c) replaces the transient full-width Python list-of-lists allocation on lines 75-78 with per-row numpy slicing, addressing Python-object churn that batching leaves in place. The two proposals compose cleanly: with UVA-backed outputs plus multi-step batching, an N-step window would have zero per-step D2H copies and a single vectorized host-side trim at window close.

---

### 2. Introduce a lazy ModelRunnerOutput token view to defer Python list materialization
- **Agent:** codex

**Detailed description.**

Change `AsyncOutput.get_output` in `vllm/v1/worker/gpu/async_utils.py` so the common sampled-token path returns a small sequence-like wrapper over the host `sampled_token_ids` and `num_sampled_tokens_np` arrays instead of immediately executing `sampled_token_ids.tolist()`, `num_sampled_tokens_np.tolist()`, and the per-row trim loop. The wrapper would implement the list-of-lists contract expected by downstream scheduler code (`__len__`, `__iter__`, `__getitem__`, equality against ordinary lists if tests require it), slicing `sampled_token_ids[i, :num_sampled_tokens_np[i]]` and converting only the row that is actually consumed. `ModelRunnerOutput.sampled_token_ids` can accept this wrapper via a narrow type alias such as `Sequence[Sequence[int]]`, while preserving an explicit `.to_list()` or compatibility conversion at serialization/test boundaries that truly require a concrete Python `list`. Keep logprobs, `num_nans`, routed experts, and EP fault reporting unchanged; this is strictly a host-side materialization change inside the candidate's finalization path. Add focused tests near the existing worker/sample suites that assert observable indexing/iteration/equality behavior matches the current trimmed list-of-lists for single-token and speculative multi-token rows.

**Novelty rationale.**

The deep_research_proposal batches GPU-resident sampled-token materialization across a multi-step decode window, but still performs concrete host list materialization at the window boundary. Agent A's proposal changes where GPU writes land by using pinned/UVA host buffers and also still constructs trimmed Python lists, just with less transient full-width allocation. This proposal is orthogonal: it does not change D2H transfer mechanics or decode-window scheduling at all. It targets the Python object churn after host arrays already exist by making `ModelRunnerOutput.sampled_token_ids` lazy and row-oriented, so workloads that only inspect or append a subset of rows avoid allocating every token id as a Python int on every output finalization.

---
