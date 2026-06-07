# FlashAttnMLAMetadataBuilder.decode_schedule

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/backends/mla/flashattn_mla.py`](vllm/v1/attention/backends/mla/flashattn_mla.py) (lines 106–252)
- **Symbol:** `FlashAttnMLAMetadataBuilder.decode_schedule`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0012`

## Description
FlashAttention MLA metadata scheduling region: reorder threshold, FA3 scheduler enablement, CUDA graph split cap, and decode scheduler metadata construction/copy.

## Current approach
Sets reorder_batch_threshold = 512 for processing small prefills through the decode path. __init__ enables FA3 AOT scheduling when get_flash_attn_version() == 3 and preallocates scheduler metadata for full CUDA graphs, with max_num_splits from attention_config.flash_attn_max_num_splits_for_cuda_graph or 1 in batch-invariant mode. _build_decode computes max_query_len from query_start_loc_cpu, selects max_num_splits = 0/self.max_num_splits/1, calls get_scheduler_metadata, and copies metadata into a persistent CUDA graph buffer when needed.

## Estimated impact explanation
DeepSeek-style multi-turn workloads often use MLA and mix single-token decode with short extends. Better threshold and scheduler/split policy can improve TTFT for short prefills and median TPOT for decode, but the external FlashAttention kernel still owns much of the runtime.

## Evolve rationale
This mirrors the FlashAttention AOT policy for the MLA path, but with MLA-specific shape, split, and small-prefill-as-decode choices. reorder_batch_threshold, FA3 enablement, max_num_splits selection, and scheduler metadata reuse are concrete policy knobs around equivalent math. Correctness oracle: tests/v1/attention/test_mla_backends.py validates FlashAttnMLA outputs across decode/prefill cases, tests/kernels/attention/test_flashmla.py and CUDA graph tests cover related MLA scheduler behavior, and backend-selection tests exercise the configured backend path.

## Deep research proposals

### 1. Load-balanced split/threshold policy for FlashAttn MLA decode scheduler metadata
- **Finding:** `find-0004` — *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*
- **Source URL:** <https://openreview.net/forum?id=RXPofAsL8F>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/mla/flashattn_mla.py (FlashAttnMLAMetadataBuilder.decode_schedule, lines 106-252), replace the largely static scheduling policy (reorder_batch_threshold = 512; max_num_splits chosen from a fixed {0, self.max_num_splits, 1}) with a load-balanced, batch-aware policy applied before get_scheduler_metadata while still copying into the preallocated CUDA-graph buffer. Concretely: (1) in _build_decode, derive a load-balance signal from query_start_loc_cpu / seq_lens_cpu (e.g. variance or max/mean ratio of per-request token counts and KV lengths, plus current batch size relative to self.max_num_splits) and pick max_num_splits along a small calibrated curve instead of the current 0/all/1 trichotomy, so highly skewed multi-turn agent batches get more splits and uniform decode batches stay at 0 splits to preserve kernel efficiency; (2) make reorder_batch_threshold a function of recent batch composition (short-prefill vs single-token decode share) rather than a hard 512, retaining the existing decode-path semantics but routing more or fewer short prefills through decode based on observed mix; (3) keep FA3 AOT enablement and the persistent CUDA-graph scheduler metadata buffer untouched so CUDA Graph capture and replay still see fixed shapes — only the chosen max_num_splits value (bounded by self.max_num_splits) and the contents written into that buffer change per build. No kernel changes; the FlashAttention kernel call site and metadata layout in __init__ are preserved.

**Proposal rationale.**

The finding's central, transferable claim — that load-balanced scheduling can adapt to request dynamism while remaining CUDA-Graph compatible — directly targets the gap in this candidate: max_num_splits and reorder_batch_threshold are picked with almost no awareness of intra-batch skew, even though multi-turn agentic workloads (the stated caller objective) produce highly heterogeneous mixes of single-token decodes and short extends that are exactly what FlashInfer-style load balancing is designed for. Because the candidate already preallocates scheduler metadata into a fixed CUDA-graph buffer, a load-aware choice of splits and threshold can be inserted without breaking graph capture, matching the finding's compatibility constraint. This addresses median TPOT (better split balance for skewed decode batches) and TTFT for short prefills (smarter routing through the decode path) while leaving the external FlashAttention MLA kernel and FA3 AOT path intact.

---

### 2. Adaptive max_num_splits for MLA decode based on KV length and effective batch
- **Finding:** `find-0005` — *Flash-Decoding for long-context inference – PyTorch Blog*
- **Source URL:** <https://pytorch.org/blog/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/mla/flashattn_mla.py around lines 106-252, replace the current ternary max_num_splits policy in FlashAttnMLAMetadataBuilder._build_decode (max_num_splits = 0 outside CUDA graph capture, self.max_num_splits during capture, 1 in batch-invariant mode) with a Flash-Decoding-style adaptive heuristic that derives splits from the runtime decode shape passed to get_scheduler_metadata. Concretely: (1) compute an effective decode batch size from query_start_loc_cpu/num_decodes and a representative KV length from seq_lens (e.g., max or mean of decode rows); (2) when occupancy is low (small effective batch * num_kv_heads relative to SM count) and KV length is large, raise max_num_splits toward self.max_num_splits to expose the KV-length parallelization dimension that Flash-Decoding describes; when batch is already large enough to saturate SMs or KV is short, keep max_num_splits at 0/1 to avoid unnecessary partial-output merges. Keep the CUDA-graph path safe by clamping the chosen value to self.max_num_splits (the value used to preallocate the persistent scheduler-metadata buffer) so the existing copy_ into the preallocated buffer at lines ~230-252 still fits, and keep the batch-invariant override at 1. The reorder_batch_threshold = 512 and FA3 AOT enablement in __init__ remain unchanged; only the per-step split selection becomes workload-aware.

**Proposal rationale.**

The candidate already routes through FlashAttention's split-KV decode (get_scheduler_metadata + max_num_splits is precisely the Flash-Decoding interface), but its current policy is shape-agnostic: it picks 0, a fixed preallocated cap, or 1 with no reference to KV length or effective decode batch. The finding's central claim is that adding the KV-sequence-length parallelization dimension is most valuable exactly when long contexts meet small effective batch sizes — the regime that matches the caller's multi-turn agentic workload (long accumulated KV per turn, modest concurrent decodes). Since the MLA decode path is the bottleneck for median TPOT in this regime and the scheduler-metadata buffer is already sized to self.max_num_splits, an adaptive selection can convert idle SMs into useful KV-parallel work without changing kernels, ABI, or CUDA-graph capture invariants. The same mechanism is neutral or near-neutral when batch is already large, limiting downside.

---

## Agent proposals

### 1. Bucket-keyed reuse of decode scheduler metadata to elide get_scheduler_metadata and the H2D copy on stable multi-turn decodes
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/mla/flashattn_mla.py, inside FlashAttnMLAMetadataBuilder._build_decode (lines ~106-252), wrap the `get_scheduler_metadata(...)` call and the subsequent `copy_` into the persistent CUDA-graph scheduler-metadata buffer with a tiny per-builder cache (size 2-4, LRU) keyed by (num_decodes, chosen max_num_splits, fingerprint_of_bucketed_seq_lens). The fingerprint is computed by rounding each decode row's KV length up to a coarse bucket (e.g. 32 or 64 tokens) and hashing the resulting tuple, optionally including max_query_len bucketed similarly. On a cache hit, skip the host-side `get_scheduler_metadata` invocation and skip the `copy_` into the preallocated GPU buffer — that buffer already holds the metadata produced for the matching key, so the next FlashAttention kernel launch will read identical scheduler tiles. On a miss (first build, batch composition change, or KV crossed a bucket boundary), call `get_scheduler_metadata` as today and `copy_` into the buffer, then store the key. Keep all existing invariants intact: the persistent buffer in __init__ is unchanged, FA3 AOT enablement is unchanged, the chosen `max_num_splits` is still bounded by `self.max_num_splits` (so the buffer sizing still fits), batch-invariant mode still forces 1, and the FlashAttention kernel itself still receives the *true* (un-bucketed) seq_lens / query_start_loc for indexing — only the work-partition descriptor is reused. This makes the cache safe (correct outputs always; at worst slightly stale tile balance until the next miss) and CUDA-graph compatible (capture replays the same buffer, same shapes). Hit rate is expected to be high in multi-turn agent decode loops, where consecutive steps differ only by KV growth of 1 and stay within the same bucket for tens of steps.

**Novelty rationale.**

Both existing deep_research_proposals (find-0004, find-0005) modify *which value* of max_num_splits and reorder_batch_threshold is selected based on batch skew or KV length / effective batch occupancy. Neither addresses the per-step CPU + H2D overhead of producing and uploading scheduler metadata: get_scheduler_metadata is still called every build and the result is still copy_-ed into the CUDA-graph buffer every build under both proposals. This proposal is orthogonal — it reuses metadata across consecutive decode steps via bucketed-key caching, eliminating the call and the copy on hits regardless of which split policy is in effect. It composes additively with either find-0004 or find-0005 (they pick the value, this one caches the resulting descriptor) and targets a different bottleneck (host-side scheduling latency in tight TPOT loops) than the kernel-side work-balance gains those proposals describe.

---

### 2. Bypass FA3 AOT metadata on fast-build MLA decodes
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/backends/mla/flashattn_mla.py`, make `FlashAttnMLAMetadataBuilder` honor the existing `fast_build` path used by `build_for_drafting()`, mirroring the non-MLA FlashAttention builder. Thread `fast_build` from `MLACommonMetadataBuilder.build()` into `_build_decode` and `_schedule_decode` (or override the subclass build path), compute `aot_schedule = self.fa_aot_schedule and not fast_build and not envs.VLLM_BATCH_INVARIANT`, and return `scheduler_metadata=None` when that is false. Keep the current `max_num_splits` selection, including the full-CUDA-graph cap and batch-invariant `1`, so kernel split behavior and workspace sizing remain unchanged; only `get_scheduler_metadata(...)` and the persistent-buffer copy/zero are skipped for fast-build or batch-invariant MLA decode metadata. Add a focused regression that monkeypatches `get_scheduler_metadata` and verifies `build_for_drafting()` for `FlashAttnMLAMetadataBuilder` does not call it while normal decode still does when FA3 AOT scheduling is enabled.

**Novelty rationale.**

The deep-research proposals change which `max_num_splits` or reorder threshold is selected based on workload shape; this proposal does not alter those policies. Agent A proposes a bucketed reuse cache for normal repeated decode metadata; this proposal instead uses the existing `fast_build`/batch-invariant semantic modes to avoid producing AOT scheduler metadata at all, so it applies even with no cache history and composes with any future split policy or metadata cache.

---
