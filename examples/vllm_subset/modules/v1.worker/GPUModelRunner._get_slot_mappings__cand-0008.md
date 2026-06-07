# GPUModelRunner._get_slot_mappings

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 3741–3813)
- **Symbol:** `GPUModelRunner._get_slot_mappings`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0008`

## Description
Builds slot-mapping tensors by KV-cache group and by layer, optionally slicing them for ubatches before creating the forward context.

## Current approach
Every forward pass constructs slot_mappings_by_gid, fills padded tails with -1, rebuilds a layer-name-to-slot-mapping dict by walking every KV-cache group and layer, and creates per-ubatch sliced dicts when DBO is active.

## Estimated impact explanation
The hot path is per step and layer-count dependent. Optimizing it removes host overhead for large models and improves median TPOT, with the largest gains when DBO or full CUDA graph padding is active.

## Evolve rationale
The per-layer dict rebuild scales with model layer count and sits between input prep and forward execution. Headroom is in caching the static layer-to-group mapping, reusing immutable per-layer views, skipping fill_ when there is no padded tail, and precomputing ubatch layer mappings. Correctness oracle: per-layer slot_mapping tensors and attention metadata should match current behavior in v1 attention and GPU model-runner tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace per-step slot_mappings_by_layer dict with a lazy Mapping proxy keyed by precomputed layer→gid
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py:3741-3813, drop the eager dict construction at lines 3798-3802 and the per-ubatch sliced-dict construction at lines 3804-3810. Instead, build a static frozen `layer_name_to_gid: dict[str, int]` exactly once at the point where `kv_cache_config` is finalized (alongside an existing `_kv_cache_groups`-style attribute), and on the hot path return a lightweight `LayerSlotMappingView(collections.abc.Mapping)` instance constructed in O(1) per step that holds two references: (a) the static `layer_name_to_gid` table and (b) the per-step `slot_mappings_by_gid: dict[int, Tensor]`. Its `__getitem__`/`get` does `slot_mappings_by_gid[layer_name_to_gid[name]]`, which is what the sole consumer (`vllm/model_executor/layers/attention/attention.py:659`, plus the analogous `mla_attention.py:536/1006` and `extract_hidden_states.py:56` sites that all do a single `.get(layer_name)`) already performs. For the DBO ubatch case, replace the inner double loop at lines 3807-3810 with a `UBatchSlicedSlotMappingView` that wraps `slot_mappings_by_gid` plus a `token_slice` and slices lazily on first access per (gid, slice), memoizing the sliced tensor on the proxy so each gid is sliced at most `num_ubatches` times per step instead of `num_layers × num_ubatches` dict insertions. Because consumers only call `dict.get` and `isinstance(_, dict)` (attention.py:656), relax that assertion to `isinstance(_, Mapping)` (or expose `_is_dict_like`) — the only hot-path operation needed on the proxy is `.get`. Also gate the `slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)` inside `_get_slot_mapping` on `num_tokens_unpadded < num_tokens_padded` so the no-padding common case avoids a CUDA kernel launch per group per step. Net effect: construction cost on the per-step path drops from O(num_kv_groups + num_layers × (1 + num_ubatches)) Python-level dict insertions to O(num_kv_groups) plus a couple of object allocations, which is the dominant Python overhead in `_get_slot_mappings` for 80+ layer models in DBO mode.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's own `evolve_rationale` lists four ideas: caching the static layer-to-group mapping, reusing immutable per-layer views, skipping `fill_` on no-tail, and precomputing ubatch layer mappings — all of which still assume a materialized `dict[str, Tensor]` is built each step. This proposal goes further by removing that materialization entirely on the hot path via a Mapping proxy that defers indexing to consumer access, plus a memoized lazy-slicing proxy for the DBO ubatch case (which, with N layers × K ubatches, is the most expensive arm of the function and is not specifically addressed by the rationale's `precomputing ubatch layer mappings` since precomputation still produces K full dicts per step). The `fill_` gating piece is the one overlap with the rationale and is included only as a small no-cost adjacent fix; the structural change is the proxy.

---

### 2. Source layer slot mappings from attention metadata instead of ForwardContext
- **Agent:** codex

**Detailed description.**

In vllm/v1/worker/gpu_model_runner.py:_get_slot_mappings, keep building only slot_mappings_by_gid for _build_attention_metadata and stop constructing the duplicate slot_mappings_by_layer dict/list returned for ForwardContext. Then update the KV-cache update consumers to use the per-layer attention metadata they already resolve: get_attention_context() can set layer_slot_mapping = getattr(attn_metadata, "slot_mapping", None), MLA direct-call/update paths can use attn_metadata.slot_mapping after resolving attn_metadata for self.layer_name, and extract_hidden_states can do the same. DBO remains covered because split_attn_metadata already slices CommonAttentionMetadata.slot_mapping per ubatch before each ubatch ForwardContext is created. This removes the O(num_layers) layer dict and O(num_layers * num_ubatches) sliced dict work without adding a Mapping proxy or preserving a second slot-mapping data path.

**Novelty rationale.**

There are no deep_research_proposals. Agent A proposes keeping the ForwardContext slot_mapping contract but making it lazy via Mapping proxies and a static layer-to-gid table. This proposal is different: it removes the duplicate ForwardContext layer mapping entirely and reuses the already-built per-layer attention metadata as the source of truth for do_kv_cache_update. The candidate rationale mentions caching/reusing/precomputing mappings, but not collapsing the layer mapping into existing attention metadata.

---
