# GPUModelRunner._build_attention_metadata

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 2325–2658)
- **Symbol:** `GPUModelRunner._build_attention_metadata`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0005`

## Description
Builds CommonAttentionMetadata and per-layer attention metadata across KV-cache groups and attention groups in the legacy runner.

## Current approach
Computes max_seq_len from a CPU tensor every step, walks multimodal-prefix ranges in Python, shallow-copies CommonAttentionMetadata per KV group, calls _get_encoder_seq_lens per group, and builds or updates metadata per attention group.

## Estimated impact explanation
Attention metadata is built every step, so memoizing group-invariant pieces and precomputing host values reduces steady TPOT overhead, though the main cost is CPU bookkeeping rather than a mandatory GPU stall.

## Evolve rationale
The headroom is CPU metadata recomputation and repeated per-group work. Tests in tests/v1/attention/test_attention_backends.py, test_mla_backends.py, test_attention_splitting.py, and test_mamba_update_block_table.py validate the per-group metadata contracts.

## Deep research proposals

### 1. Make seq_lens_cpu optional in _build_attention_metadata to unblock async spec-decode overlap
- **Finding:** `find-vllm_v1_worker-0002` — *[Performance]: Fully Async Spec-Decoding | Make `seq_lens_cpu` in CommonAttentionMetadata optional*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/29134>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Change GPUModelRunner._build_attention_metadata (vllm/v1/worker/gpu_model_runner.py:2325-2658) so that the CommonAttentionMetadata it constructs no longer requires a populated seq_lens_cpu / max_seq_len computed from a CPU tensor every step. Specifically: (1) treat seq_lens_cpu as Optional and only materialize it when a downstream attention backend actually consumes it; (2) replace the per-step max_seq_len computation currently derived from the CPU seq_lens tensor with either a device-resident upper bound (e.g., a scalar produced on-device or a precomputed budget derived from scheduler outputs) or lazy computation gated on the backend's declared need; (3) route spec-decode paths through the device seq_lens branch so building attention metadata for drafted/verified tokens does not force a host/device sync; (4) keep the shallow-copy-per-KV-group and per-attention-group construction intact but drop the seq_lens_cpu / max_seq_len bookkeeping from the group-invariant portion so it is not recomputed for each group. Downstream callers of CommonAttentionMetadata that still require CPU values should request them explicitly, matching the direction proposed in issue #29134.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out CPU metadata recomputation (max_seq_len from a CPU tensor every step) and repeated per-group work as the headroom, and the finding targets exactly that: it proposes making seq_lens_cpu optional in CommonAttentionMetadata so metadata construction does not depend on host-side sequence lengths. Removing the mandatory CPU dependency in _build_attention_metadata directly addresses the caller's TPOT objective for multi-turn agentic workloads by enabling overlap of next-step input preparation with current forward execution (fully async spec decoding), which is the exact blocker described in the linked issue.

---

### 2. Persist FlashInfer-style auxiliary attention planning across steps in _build_attention_metadata
- **Finding:** `find-vllm_v1_worker-0005` — *FlashInfer Attention Kernels*
- **Source URL:** <https://docs.flashinfer.ai/api/attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the intra-call `cached_attn_metadata` dict in GPUModelRunner._build_attention_metadata (vllm/v1/worker/gpu_model_runner.py:2517-2586) into a runner-owned, cross-step cache of per-(KVCacheSpec, AttentionMetadataBuilder) auxiliary planning structures, mirroring FlashInfer's BatchDecodeWithPagedKVCacheWrapper lifecycle where 'auxiliary data structures ... can be reused across multiple batch decode attention calls.'

Concretely: (1) Promote `cached_attn_metadata` to `self._attn_metadata_plan_cache` initialized in __init__, keyed by (kv_cache_spec, type(builder)) and, when relevant, by ubid. (2) On each step, for builders that expose `supports_update_block_table` (already the fast path at lines 2570-2586), prefer calling `builder.update_block_table(cached, block_table_tensor, slot_mapping)` on the persisted metadata object instead of rebuilding. (3) For builders that support it (notably FlashInfer decode/prefill wrappers), thread pre-allocated stable workspace / indptr / paged_kv buffers into the builder so the plan() call reuses buffers across steps — matching FlashInfer's guidance that under CUDA graph mode users must supply stable buffers. (4) Hoist per-KV-group invariants out of the `for kv_cache_gid` loop at lines 2601-2646: memoize `_get_encoder_seq_lens` results by (kv_cache_spec identity, num_scheduled_tokens fingerprint) instead of recomputing per group, and cache the `copy(cm_base)` template shape so only block_table_tensor / slot_mapping / encoder_seq_lens fields are patched per group. (5) Compute `max_seq_len` at line 2362 from an already-materialized host scalar when available (e.g., a running max maintained by _prepare_inputs) to remove the per-step `.numpy()[:num_reqs].max().item()` sync-and-scan. Invalidate the persistent cache on kv_cache_config changes, batch-size padding boundary changes, or CUDA graph capture transitions.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names CPU metadata recomputation and repeated per-group work as the headroom, and its current_approach already gestures at this idea with a per-call `cached_attn_metadata` dict plus `supports_update_block_table` fast path — but the cache is thrown away every step. FlashInfer's documented pattern of reusing auxiliary planning structures across calls (and requiring stable user-provided buffers under CUDA graph) is a direct fit: it addresses both the per-step Python bookkeeping (shallow copies, encoder_seq_lens recompute, max_seq_len host scan) and 'graph pointer churn' the finding calls out. For the caller's multi-turn agentic TPOT target, decode steps hit this path every token with mostly-invariant group topology, so persisting plan state across steps compounds savings without changing kernel semantics.

---

## Agent proposals

### 1. Precompute per-step group-invariant slices in _build_attention_metadata via a compiled numpy path
- **Agent:** claude

**Detailed description.**

In GPUModelRunner._build_attention_metadata (vllm/v1/worker/gpu_model_runner.py:2325-2658), replace the Python-level per-KV-group loop that repeatedly derives the same intermediate arrays (query_start_loc slicing, seq_lens indexing, num_computed_tokens gathers, logits_indices mask, multimodal-prefix range walks) with a single vectorized numpy precompute stage that runs once per step before the `for kv_cache_gid` loop. Concretely: (1) Hoist the multimodal-prefix range enumeration currently done in Python (walking mm_positions per request) into a single pass that produces a flat int32 numpy array of (req_idx, start, end) triples plus a per-request offset index; store this in a reusable pinned host buffer sized to max_num_reqs so the walk allocates nothing on hot path. (2) Precompute encoder_seq_lens once for the whole batch and slice it per KV group by an int32 gather instead of calling _get_encoder_seq_lens per group (the current implementation at lines ~2601-2646 pays the dict lookup and per-request Python loop repeatedly). (3) Replace the `copy(cm_base)` shallow-copy-per-group pattern with a dataclasses.replace call that only rebinds the three fields that actually vary per group (block_table_tensor, slot_mapping, encoder_seq_lens) — this avoids constructing a fresh CommonAttentionMetadata dict and its __post_init__ side effects per group. (4) For the max_seq_len computation, track a running host-side max in self._max_seq_len_host maintained incrementally by _prepare_inputs when it writes seq_lens, so _build_attention_metadata reads a plain Python int rather than calling `.numpy()[:num_reqs].max().item()` (an O(num_reqs) host scan plus an implicit sync if the tensor was device-side). Preserve the existing shape/dtype contracts asserted by tests/v1/attention/test_attention_backends.py, test_mla_backends.py, test_attention_splitting.py, and test_mamba_update_block_table.py; only the Python bookkeeping path changes.

**Novelty rationale.**

The two existing deep_research_proposals target (a) making seq_lens_cpu/max_seq_len optional at the CommonAttentionMetadata contract level to unblock async spec-decode overlap, and (b) persisting FlashInfer-style plan/auxiliary buffers across steps via a runner-owned cross-step cache. This proposal is orthogonal to both: it keeps the CommonAttentionMetadata contract unchanged (seq_lens_cpu remains populated) and does not persist any per-step plan objects across steps. Instead, it attacks the intra-step Python overhead — the multimodal-prefix walk, the per-group _get_encoder_seq_lens recomputation, the shallow-copy-per-group dict churn, and the max_seq_len host scan — by vectorizing them into a single pre-loop pass and maintaining an incremental host-side max. It composes cleanly with either existing proposal (the vectorized precompute stage still runs whether or not seq_lens_cpu is optional, and the reduced per-group work reduces the surface that a persistent plan cache would need to invalidate).

---

### 2. Make padded block-table tails lazily maintained instead of filling them every step
- **Agent:** codex

**Detailed description.**

In `GPUModelRunner._build_attention_metadata` (`vllm/v1/worker/gpu_model_runner.py:2340-2360`), change `_get_block_table` so it does not unconditionally execute `blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)` for every KV-cache group on every metadata build. Add a small per-runner/per-`kv_cache_gid` padding-tail state keyed by `(num_reqs, num_reqs_padded, block_table_storage_version)` or move the invariant into the block-table object: active rows are written by input preparation, and rows in `[num_reqs, num_reqs_padded)` are guaranteed to already contain `NULL_BLOCK_ID` unless the active-request boundary moved downward. Then only issue the device fill when `num_reqs_padded > num_reqs` and the newly padded tail may contain stale non-null rows. For `EncoderOnlyAttentionSpec`, also replace the per-call `torch.zeros((num_reqs_padded, 1), ...)` allocation with a reusable device buffer per padded request bucket, since it is purely synthetic metadata. Cover this with an attention metadata test that exercises changing `num_reqs` under CUDA-graph padding and verifies padded block-table rows remain `NULL_BLOCK_ID` across multiple KV groups.

**Novelty rationale.**

The existing deep-research proposals focus on making CPU sequence-length metadata optional and on persisting attention planner/build artifacts across steps, including block-table update paths. Agent A focuses on vectorizing intra-step host-side precomputations, encoder sequence lengths, shallow-copy churn, and max sequence length. This proposal targets a different hot-path cost inside `_get_block_table`: repeated device-side mutation/allocation of padded block-table tails. It does not change the `CommonAttentionMetadata` CPU contract, does not persist backend planning objects, and is not a numpy/vectorized precompute of host metadata; it specifically removes avoidable GPU-side fill/allocation work caused by CUDA-graph padding.

---
