# GPUModelRunner._build_attention_metadata

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 2098–2365)
- **Symbol:** `GPUModelRunner._build_attention_metadata`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0016`

## Description
Builds CommonAttentionMetadata and per-layer attention metadata for each KV-cache group, attention group, ubatch, encoder/cross-attention variant, and speculative-decoding path.

## Current approach
Every call computes max_seq_len from CPU tensors, conditionally allocates encoder-only block tables, fills padded block-table rows, builds a CommonAttentionMetadata object, copies DCP local sequence lengths when enabled, creates a fresh cached_attn_metadata dict, walks KV-cache groups and attention groups, repeatedly copies/splits metadata for ubatches, and assigns metadata to every layer name.

## Estimated impact explanation
The function is per-step but mostly host-side metadata work, so gains are bounded by model-forward cost. It can still reduce median TPOT for large-layer-count models, hybrid KV-cache layouts, DBO, and multimodal/cross-attention workloads where metadata assembly is repeated many times.

## Evolve rationale
The in-repo dispatch and metadata assembly run once per forward and scale with KV-cache groups, attention groups, layer count, and ubatching. Headroom is in caching static group-to-builder/layer mappings, preserving reusable per-group CommonAttentionMetadata shells, avoiding pad fills when no padded tail exists, hoisting encoder seq-len work for non-cross-attention groups, and reusing split metadata for stable ubatch shapes. Correctness oracle: compare PerLayerAttnMetadata keys and object fields, CommonAttentionMetadata fields, encoder seq-lens, DCP seq-lens, cascade prefix propagation, and logits parity across v1 attention, DBO, cross-attention, and spec-decode tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Hoist max_seq_len computation into _prepare_inputs and consume a cached scalar
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py at lines 2128–2134, `_build_attention_metadata` unconditionally runs `self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()` on every step (the only exception being CUDA-graph capture). Move this reduction to `_prepare_inputs` (around lines 1913–1932), where `optimistic_seq_lens_cpu` is freshly written via `torch.add(num_computed_tokens_cpu_tensor, num_scheduled_tokens, out=optimistic_seq_lens_cpu[:num_reqs])` and is already materialized as a numpy view two lines later for `discard_request_mask`. Concretely: (1) Add a `self.optimistic_max_seq_len: int` field next to the existing `optimistic_seq_lens_cpu` allocation (~line 690). (2) In `_prepare_inputs`, after the `torch.add(...)` that fills `optimistic_seq_lens_cpu[:num_reqs]`, reuse the same numpy view already needed for `discard_request_mask` to compute `self.optimistic_max_seq_len = int(opt_np.max())` in a single fused pass — better, derive it as `int((num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens).max())` to avoid a second array scan. (3) In `_build_attention_metadata`, replace the line-2134 reduction with `max_seq_len = self.optimistic_max_seq_len`. Keep the `for_cudagraph_capture` branch unchanged (still uses `self.max_model_len`). The same cached scalar can also feed line 3178 (`seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs]`) callers if they need a max. This eliminates a per-step O(num_reqs) numpy reduction plus `.item()` Python overhead from the metadata-build hot path; in long agentic batches with many concurrent requests, the reduction happens for every microstep, and folding it into the same loop where seq_lens are written keeps it cache-hot and removes one independent pass over the array.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any non-trivial proposal is novel by definition. More importantly, the `evolve_rationale`'s headroom list explicitly enumerates: (a) caching group-to-builder/layer mappings, (b) reusable per-group CommonAttentionMetadata shells, (c) avoiding pad fills, (d) hoisting encoder seq-len work for non-cross-attention groups, (e) reusing split metadata for stable ubatch shapes — none of which target the `max_seq_len` reduction at line 2134. The current_approach mentions `max_seq_len` is computed from CPU tensors but treats it as a given, not as headroom. Hoisting it to `_prepare_inputs` and reusing the already-materialized array consumed by `discard_request_mask` is a distinct optimization site.

---

### 2. Reuse a persistent encoder-only block table
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu_model_runner.py`, replace the per-call `torch.zeros((num_reqs_padded, 1), dtype=torch.int32, device=self.device)` allocation in `_build_attention_metadata`'s `_get_block_table` branch for `EncoderOnlyAttentionSpec` with a runner-owned buffer. Allocate it once after KV-cache groups are finalized, e.g. in `initialize_kv_cache` after `may_add_encoder_only_layers_to_kv_cache_config()`/attention backend initialization, only when any group has an `EncoderOnlyAttentionSpec`: `self.encoder_only_block_table = torch.zeros((self.max_num_reqs, 1), dtype=torch.int32, device=self.device)`. Then `_get_block_table` can return `self.encoder_only_block_table[:num_reqs_padded]` for encoder-only groups. Since `NULL_BLOCK_ID` is currently `0`, the pre-zeroed buffer is identical to the current actual-row and padded-row contents; if the code wants to preserve future-proofing, keep a guarded tail fill only when `num_reqs < num_reqs_padded and NULL_BLOCK_ID != 0`. Validate with encoder-only and multimodal prefix-LM paths, plus a CUDA-graph padded batch to confirm per-layer metadata fields and logits are unchanged.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal only hoists the `max_seq_len` reduction from `_build_attention_metadata` into `_prepare_inputs`; it does not address the encoder-only block-table allocation path. This proposal targets a different hot-path cost at lines 2140-2147: eliminating repeated device tensor allocation/zeroing for encoder-only attention groups while preserving the same metadata shape and values.

---
