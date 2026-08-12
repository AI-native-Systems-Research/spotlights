# GPUModelRunner.execute_model

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 4259–4620)
- **Symbol:** `GPUModelRunner.execute_model`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0007`

## Description
Legacy top-level per-step orchestration for state updates, input prep, cascade attention, cudagraph dispatch, Mamba preprocessing, slot mappings, attention metadata, forward, and logits computation.

## Current approach
Builds num_scheduled_tokens_np from scheduler dict lookups each step, recomputes has_separate_kv_update through nested generators, and copies scheduler dictionaries for ngram GPU mode.

## Estimated impact explanation
These costs are smaller than input prep or sampling sync, but they run every step and scale with active requests, so they can move median TPOT in agentic decode.

## Evolve rationale
The concrete headroom is reducing top-of-step Python work and caching static attention-backend properties. End-to-end tests in tests/v1/e2e/ plus per-hook tests in tests/v1/worker/test_gpu_model_runner.py validate orchestration behavior.

## Deep research proposals

### 1. Skip CPU seq-lens materialization in execute_model spec-decode path
- **Finding:** `find-vllm_v1_worker-0002` — *[Performance]: Fully Async Spec-Decoding | Make `seq_lens_cpu` in CommonAttentionMetadata optional*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/29134>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In GPUModelRunner.execute_model (vllm/v1/worker/gpu_model_runner.py:4259-4620), the top-of-step preprocess block builds num_scheduled_tokens_np and then invokes self._build_attention_metadata(...) with use_spec_decode and num_scheduled_tokens=scheduler_output.num_scheduled_tokens (lines 4471-4485). Following the issue's proposal to make seq_lens_cpu optional in CommonAttentionMetadata, thread a `needs_seq_lens_cpu` signal (derived once from the attention backends selected for the active kv_cache_groups, cached alongside the has_separate_kv_update determination) into this call site so that when spec-decode is active and no consumer requires host seq_lens, execute_model stops assembling/passing CPU sequence-length arrays and lets _build_attention_metadata construct metadata from device seq_lens and device-side upper bounds. Concretely: (1) cache the per-backend `requires_seq_lens_cpu` flag on the runner at init (same place has_separate_kv_update is recomputed at lines 4403-4410, which is also flagged as top-of-step Python work), (2) gate the CPU seq-lens/upper-bound population inside _build_attention_metadata on that flag, and (3) when use_spec_decode is True and the flag is False, avoid the CPU-side seq_lens copy that currently forces a host/device sync before the forward launch at line 4535. Preserve current behavior for backends that still need seq_lens_cpu by keeping the existing path under the flag.

**Proposal rationale.**

The candidate's evolve_rationale explicitly targets reducing top-of-step Python work and caching static attention-backend properties that run every step in this orchestration method; spec-decode-heavy multi-turn agentic decode (the caller's stated workload) is exactly the regime where per-step host/device syncs from seq_lens_cpu block overlap between input prep and the current forward. The finding contributes a concrete transferable idea — treat seq_lens_cpu as optional and drive attention metadata from device seq_lens — that maps directly onto the _build_attention_metadata call and the has_separate_kv_update-style backend-property caching inside execute_model, addressing the same TPOT gap the candidate identifies. This is a narrow, plausibly-beneficial change at the exact call site named in the candidate, not a rewrite of unrelated code.

---

### 2. Adopt MRV2-style stable-row batch state to cut per-step Python overhead in execute_model
- **Finding:** `find-vllm_v1_worker-0003` — *Model Runner V2 Design Document*
- **Source URL:** <https://docs.vllm.ai/en/v0.17.0/design/model_runner_v2/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor the top-of-step orchestration in GPUModelRunner.execute_model (vllm/v1/worker/gpu_model_runner.py:4259-4620) to apply the Model Runner V2 pattern of decoupling persistent request state from per-step input tensors. Concretely: (1) Maintain persistent GPU-resident tensors indexed by stable row IDs for per-request metadata (num_scheduled_tokens, num_computed_tokens, num_accepted_tokens) so that _update_states writes only the deltas caused by newly scheduled/finished requests instead of forcing execute_model to rebuild `num_scheduled_tokens_np` from a Python dict comprehension every step (lines 4335-4336). Per-step input construction then becomes a small GPU gather over the active rows rather than a CPU-side list build + numpy conversion. (2) Cache `has_separate_kv_update` and the `pad_attn`-relevant attention-backend properties as instance attributes computed once at KV-cache-group initialization time; the nested generator over `self.attn_groups`/`kv_cache_groups` at lines 4403-4410 recomputes an invariant every step. (3) Because rows are stable across steps, avoid the ngram-GPU shallow-copy of `scheduler_output.num_scheduled_tokens` and `scheduled_spec_decode_tokens` (lines 4277-4285) by mutating a runner-owned persistent view rather than the scheduler payload, eliminating the per-step dict copies. The change is local to execute_model, _update_states, and the persistent-batch dataclass; validate via tests/v1/worker/test_gpu_model_runner.py and tests/v1/e2e/ orchestration tests plus microbenchmarks of median TPOT under multi-turn agentic workloads.

**Proposal rationale.**

The finding's MRV2 design directly targets the two hot spots called out in the candidate's evolve_rationale: 'top-of-step Python work' and 'caching static attention-backend properties.' MRV2's stable-row persistent state removes the need to rebuild per-step arrays from scheduler dicts (addresses the num_scheduled_tokens_np dict-comprehension and the ngram-mode dict copies), and its 'gather runs in parallel on the GPU with low overhead' quote justifies moving these small per-request metadata builds off the CPU critical path. Under the caller's multi-turn agentic workload — where per-step orchestration overhead dominates TPOT relative to compute — reducing top-of-step CPU work and cutting redundant Python-level recomputation is exactly the transferable idea this finding contributes. The proposal is scoped to the candidate's file and symbol and reuses the existing persistent-batch abstraction rather than inventing new locations.

---

### 3. Cache static attention-backend properties to eliminate per-step recomputation in execute_model
- **Finding:** `find-vllm_v1_worker-0005` — *FlashInfer Attention Kernels*
- **Source URL:** <https://docs.flashinfer.ai/api/attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `GPUModelRunner.execute_model` (vllm/v1/worker/gpu_model_runner.py:4259-4620), replace the per-step nested-generator recomputation of `has_separate_kv_update` (lines 4403-4410) with a value cached once during worker initialization (e.g., alongside `self.attn_groups`/`self.kv_cache_config` setup). Because `backend.forward_includes_kv_cache_update` and the `EncoderOnlyAttentionSpec` membership of each group are fixed for the life of the runner, the reduction over `self.attn_groups` and `self.kv_cache_config.kv_cache_groups` produces the same boolean every step. Store it as `self._has_separate_kv_update` after attention groups are constructed and read it here. Similarly, precompute and cache any other per-step-invariant attention-group booleans currently derived by iterating `self.attn_groups` at the top of `execute_model` (e.g., extend the same lifecycle pattern to static backend flags consulted before `_build_attention_metadata`). This mirrors FlashInfer's guidance to build auxiliary attention planning data once and reuse it across repeated attention calls with stable state, adapted to the worker's per-step orchestration path rather than kernel wrappers.

**Proposal rationale.**

The candidate's `evolve_rationale` explicitly calls out `has_separate_kv_update` recomputation via nested generators and caching static attention-backend properties as headroom items that run every step and scale with active requests / attention groups. The FlashInfer finding provides a directly transferable principle: attention-backend auxiliary state that is invariant across calls should be built once and reused, with reuse across repeated execution. Applied here, that reduces per-step CPU Python work in the hot `execute_model` preprocess block, which in multi-turn agentic decode contributes to median TPOT. The change is narrow, local to the identified lines, and does not alter semantics because the cached quantity is static after worker init.

---

## Agent proposals

### 1. Early-exit cascade prefix computation when no KV group has a shared prefix
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py at the top-of-step orchestration block (lines 4347-4353), the call to `self._compute_cascade_attn_prefix_lens(...)` runs every step whenever `self.cascade_attn_enabled` is true and ubatching is off. The callee at lines 2660-2696 unconditionally builds a `list[list[int]]` of length `num_kv_cache_groups`, iterates every `attn_group` in `self.attn_groups[kv_cache_gid]`, performs an `isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec)` check per group, and calls `_compute_cascade_attn_prefix_len` for each — even though that inner function returns 0 immediately at line 2724-2727 when `num_common_prefix_blocks * block_size == 0` (the code itself annotates this as the 'Common case.'). Under the caller's multi-turn agentic workload, running requests are typically distinct sessions with no cross-request shared prefix, so `scheduler_output.num_common_prefix_blocks` is a list of zeros on essentially every decode step, meaning the nested iteration and 2D-list allocation are pure per-step Python overhead. Concrete change: (1) at the execute_model call site around line 4347, gate the cascade computation on `any(n > 0 for n in scheduler_output.num_common_prefix_blocks)` (a cheap iterate over a short list, one entry per kv_cache_group) before calling `_compute_cascade_attn_prefix_lens`, and set `cascade_attn_prefix_lens = None` directly when the check fails; (2) as a defense-in-depth check, add the same short-circuit as the first statement of `_compute_cascade_attn_prefix_lens` itself so any other caller benefits. Semantics are preserved because when every `num_common_prefix_blocks[gid]` is 0, every `_compute_cascade_attn_prefix_len` call would return 0, `use_cascade_attn` would stay False, and the function would return `None` at line 2696 anyway — the early exit reaches the same end state without allocating the 2D list or touching `self.attn_groups`. Downstream, `use_cascade_attn=cascade_attn_prefix_lens is not None` at line 4366 continues to see False in this common case, keeping `_determine_batch_execution_and_padding` and `_build_attention_metadata` behavior identical.

**Novelty rationale.**

None of the three existing deep_research_proposals touch the cascade attention prefix computation. Finding-0002 targets `seq_lens_cpu` gating in `_build_attention_metadata` under spec-decode; finding-0003 is an MRV2-style refactor of persistent per-request batch state (num_scheduled_tokens / num_computed_tokens / num_accepted_tokens), which does not include cascade attention data flow; finding-0005 caches per-runner-invariant attention-backend booleans (`has_separate_kv_update` and similar static flags) at init time. Cascade prefix lengths are per-step, batch-dependent quantities driven by the scheduler's current `num_common_prefix_blocks`, so they cannot be lifted out at init time (finding-0005 does not apply), and the optimization is not a persistent-batch data-model change (finding-0003 does not apply). The proposal contributes a distinct axis of top-of-step Python savings — a runtime early-exit on the common `num_common_prefix_blocks == 0` case — that directly addresses the candidate's `evolve_rationale` about reducing per-step Python work that scales with active requests and attention groups, without overlapping any listed finding.

---

### 2. Defer microbatching veto computation until DP coordination actually needs it
- **Agent:** codex

**Detailed description.**

In `GPUModelRunner.execute_model` around the `_determine_batch_execution_and_padding(...)` call, stop eagerly evaluating `self._allow_microbatching(num_reqs, num_scheduled_tokens_np)` before entering `_determine_batch_execution_and_padding`. That value is only consumed by `coordinate_batch_across_dp(...)` inside `_determine_batch_execution_and_padding`, and that branch is guarded by `self.vllm_config.parallel_config.data_parallel_size > 1`. Move the `_allow_microbatching` call into `_determine_batch_execution_and_padding` immediately before `coordinate_batch_across_dp`, or pass a lazy/optional value and compute it only when `data_parallel_size > 1` and `parallel_config.use_ubatching` can make the veto relevant. This avoids the per-step NumPy work in `_allow_microbatching` for single-DP-rank runs and for configurations where DBO coordination will never execute, including the block-table scan path that can iterate every KV cache group and active request in mixed prefill/decode batches. Preserve behavior for multi-rank DP by computing the same boolean at the existing coordination point and passing it unchanged to `coordinate_batch_across_dp`.

**Novelty rationale.**

The deep research proposals cover spec-decode seq-lens CPU avoidance, persistent stable-row state for scheduler dictionaries, and caching static attention-backend booleans such as `has_separate_kv_update`; none address the eager `_allow_microbatching` call or the fact that its result is unused unless DP coordination runs. Agent A's proposal optimizes cascade-attention prefix computation based on `num_common_prefix_blocks`, which is a separate per-step path before cudagraph dispatch. This proposal targets a distinct avoidable top-of-step cost: a dynamic microbatching-safety computation that should be evaluated only on the branch that consumes it.

---
