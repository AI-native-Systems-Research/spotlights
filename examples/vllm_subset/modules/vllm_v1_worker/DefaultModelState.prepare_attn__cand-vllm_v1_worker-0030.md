# DefaultModelState.prepare_attn

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/model_states/default.py`](vllm/v1/worker/gpu/model_states/default.py) (lines 125–185)
- **Symbol:** `DefaultModelState.prepare_attn`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0030`

## Description
New GPU runner default attention metadata construction for decoder-only models across cudagraph modes.

## Current approach
Builds CPU query_start_loc views every step, reads max_query_len and max_seq_len through tensor .max().item(), optionally computes multimodal prefix ranges in Python, and then calls build_attn_metadata with per-step host values.

## Estimated impact explanation
This runs every new-runner forward step, so removing scalar syncs and repeated CPU metadata work can reduce steady median TPOT, with extra TTFT benefit for multimodal prefix-LM turns.

## Evolve rationale
The max().item calls, torch.from_numpy CPU view construction, and compute_mm_prefix_ranges request walk are concrete per-step metadata targets. tests/v1/attention/test_attention_backends.py, tests/v1/attention/test_mla_backends.py, and tests/v1/worker/test_gpu_model_runner_mm_gather.py validate attention metadata and multimodal-prefix behavior.

## Deep research proposals

### 1. Eliminate host syncs in prepare_attn by making CPU seq/query metadata optional
- **Finding:** `find-vllm_v1_worker-0002` — *[Performance]: Fully Async Spec-Decoding | Make `seq_lens_cpu` in CommonAttentionMetadata optional*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/29134>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/worker/gpu/model_states/default.py (DefaultModelState.prepare_attn, lines 125-185), remove the per-step host/device synchronizations that block overlap with the current forward step. Specifically: (1) replace `input_batch.num_scheduled_tokens.max().item()` (line 147) and `seq_lens_cpu_upper_bound[:num_reqs].max().item()` (line 153) with device-resident upper bounds tracked on the host from the scheduler batch (already available as `seq_lens_cpu_upper_bound` and the pre-computed numpy `num_scheduled_tokens` array on `input_batch`), avoiding the tensor `.max().item()` sync path; (2) plumb `query_start_loc_cpu` and `is_prefilling` as optional through `build_attn_metadata`, computing them lazily inside backends that truly need CPU-side values, rather than materializing `torch.from_numpy(...)` slices every step at lines 143-145 and 180; (3) allow attention metadata builders to consume device `seq_lens` (already passed via `input_batch.seq_lens` at line 172) as the source of truth, and treat `seq_lens_cpu_upper_bound` as an optional upper bound for shape/kernel dispatch only. Keep the `for_capture` branch's `max_seq_len = self.max_model_len` fallback so cudagraph capture is unaffected. Validate with tests/v1/attention/test_attention_backends.py, tests/v1/attention/test_mla_backends.py, and tests/v1/worker/test_gpu_model_runner_mm_gather.py.

**Proposal rationale.**

The finding proposes making `seq_lens_cpu` in CommonAttentionMetadata optional so metadata construction does not force host/device syncs, enabling next-step input preparation to overlap with the current forward step. `prepare_attn` here is the exact per-step metadata construction site named by the candidate's evolve_rationale: it performs two `.max().item()` GPU→CPU syncs and eager `torch.from_numpy` CPU view materialization on every decode step. For the caller's multi-turn agentic workload aiming at lower median TPOT, removing these blocking syncs on the steady-state hot path is directly aligned with the finding's async-metadata direction, and paves the way toward the fully async spec-decode overlap the issue targets.

---

## Agent proposals

### 1. Cache multimodal prefix ranges per request to skip redundant Python walks on decode steps
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu/model_states/default.py (DefaultModelState.prepare_attn, lines 125-185), avoid recomputing `compute_mm_prefix_ranges` for every scheduled request on every forward step. The Python-side walk over each request's multimodal placeholder ranges is invariant across decode steps for a given request (the prefix positions do not change after prefill completes), yet it currently runs each step for any batch containing multimodal-prefix-LM requests. Introduce a per-request cache on `input_batch` (or on the scheduler-side request state) keyed by request id, storing the precomputed prefix-range tensor/array produced on the first step the request appears. On subsequent steps, `prepare_attn` should look up the cached ranges and only invoke `compute_mm_prefix_ranges` on cache miss (new requests entering the batch) or when the request signals its multimodal state changed. Invalidate the cache when a request is evicted/completed. Feed the cached ranges directly into `build_attn_metadata` at line 180's callsite instead of the fresh Python walk. Keep the `for_capture` path unchanged (cudagraph capture uses fixed shapes and does not hit this cache). Validate with tests/v1/worker/test_gpu_model_runner_mm_gather.py to confirm parity of prefix-range outputs across cached and uncached paths, and tests/v1/attention/test_attention_backends.py to ensure attention metadata remains identical.

**Novelty rationale.**

The existing deep_research_proposal (find-vllm_v1_worker-0002) focuses exclusively on eliminating host/device syncs from `.max().item()` calls and making `query_start_loc_cpu`/`seq_lens_cpu` optional in `build_attn_metadata`. It does not address the `compute_mm_prefix_ranges` Python walk, which is the third distinct per-step cost named in the candidate's current_approach. Caching prefix ranges across steps is an orthogonal optimization: it targets Python overhead for multimodal-prefix-LM workloads (a class of agentic multi-turn traffic explicitly relevant to the caller's TPOT goal), not GPU sync latency. The two changes compose additively and cover disjoint bottlenecks in the same code block.

---

### 2. Skip multimodal prefix metadata on pure decode steps
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu/model_states/default.py` (`DefaultModelState.prepare_attn`, lines 125-185), only compute and pass `mm_req_doc_ranges` when the current batch contains prefill/extend tokens. The multimodal PrefixLM ranges are used to make image/video prefix spans bidirectional while those prefix tokens are in the query side; on pure decode steps the new query positions are after the prefix ranges, so passing the ranges forces `compute_mm_prefix_ranges` plus backend-side range tensor/mask setup without changing the attention mask. Gate the existing `compute_mm_prefix_ranges(...)` block with the already-available `input_batch.is_prefilling_np` (for example `np.any(input_batch.is_prefilling_np)`) and pass `None` for `mm_req_doc_ranges` when all requests are decode-only. Add/extend tests around multimodal PrefixLM attention metadata to assert that prefill batches still populate ranges and decode-only batches leave them unset, alongside `tests/v1/worker/test_gpu_model_runner_mm_gather.py` for multimodal request coverage.

**Novelty rationale.**

The deep_research_proposal targets scalar host syncs and optional CPU metadata plumbing (`query_start_loc_cpu`, `is_prefilling`, `seq_lens_cpu_upper_bound`); it does not change when multimodal PrefixLM ranges are semantically needed. Claude's proposal caches `compute_mm_prefix_ranges` per request but still feeds cached ranges into every decode step, so attention backends may continue constructing mm-prefix range tensors or mask modifiers. This proposal is different: it eliminates the entire multimodal-prefix metadata path for pure decode batches where the ranges do not affect query-side masking.

---
