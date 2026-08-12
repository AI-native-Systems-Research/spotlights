# use_cascade_attention

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flash_attn.py`](vllm/v1/attention/backends/flash_attn.py) (lines 1586–1661)
- **Symbol:** `use_cascade_attention`
- **Kind:** function
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_attention-0001`

## Description
Decides whether FlashAttention cascade attention is used for a batch with a shared prefix.

## Current approach
Uses fixed gates for common_prefix_len < 256 and num_reqs < 8, then compares a rough CTA-count model with hardcoded 128-token Q and KV tile assumptions. The source explicitly marks the two gates as TODOs to tune.

## Estimated impact explanation
Multi-turn agentic batches often share long conversation prefixes. Better routing can reduce median TTFT by using cascade when shared-prefix work dominates and avoiding cascade overhead when it does not.

## Evolve rationale
The branch selects between numerically equivalent attention implementations. Tunable constructs are the 256-token prefix gate, 8-request gate, and CTA/tile performance model; correctness oracle is output equality against non-cascade FlashAttention plus existing cascade attention tests.

## Deep research proposals

### 1. Refine cascade-vs-FlashDecoding routing using Flash-Decoding's split-KV parallelization model
- **Finding:** `find-vllm_v1_attention-0002` — *Flash-Decoding for Long-Context Inference*
- **Source URL:** <https://princeton-nlp.github.io/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Update the Flash-Decoding branch of `use_cascade_attention` in vllm/v1/attention/backends/flash_attn.py (lines 1637-1661) to model the extra KV-sequence-length parallelization dimension that Flash-Decoding actually exploits, rather than treating prefix work as a serial multiplier (`flash_decoding_ctas *= num_prefix_tiles`). Concretely: (1) introduce a `kv_split` factor that estimates how many KV-length splits Flash-Decoding would launch given `common_prefix_len`, `num_sms`, `num_reqs * num_kv_heads`, and the head-group CTA count — mirroring the adaptive split heuristic Flash-Decoding uses to saturate SMs when batch is small and context is long; (2) replace the `flash_decoding_ctas *= num_prefix_tiles` line with a cost that divides prefix tiles by `kv_split` and adds a small LSE reduction/merge term proportional to `kv_split * num_reqs * num_kv_heads`; (3) keep the cascade side unchanged since cascade already amortizes the shared prefix across requests. Re-tune the `common_prefix_len < 256` and `num_reqs < 8` gates against this refined model on multi-turn agentic traces so cascade is preferred exactly when shared-prefix work dominates the split-KV benefit of Flash-Decoding.

**Proposal rationale.**

The candidate's performance model explicitly compares cascade attention against FlashDecoding but models FlashDecoding as if it serialized over `num_prefix_tiles`, ignoring the very parallelization dimension that makes Flash-Decoding fast — splitting along KV sequence length and merging via LSE (the finding's core contribution). This directly underestimates Flash-Decoding's throughput on the long-shared-prefix / small-batch decode regime that dominates multi-turn agentic workloads, so the current model is biased toward returning True (use cascade) exactly when Flash-Decoding would in fact win, and biased toward False when batch is small (the `num_reqs < 8` gate) even though that is precisely where Flash-Decoding's split-KV shines and cascade's prefix reuse would also help. Incorporating an adaptive split-KV term plus a merge cost gives the two branches a comparable cost surface, which is the transferable idea from the finding and addresses the tunable-construct gap flagged in the candidate's evolve_rationale. The correctness oracle is unchanged (both branches are numerically equivalent up to the LSE merge), so the risk is limited to routing regressions caught by existing cascade tests plus TTFT/TPOT benchmarks on agentic traces.

---

### 2. Replace fixed cascade gates with an occupancy-aware routing policy
- **Finding:** `find-vllm_v1_attention-0005` — *Multi-Head, Multi-Query, and Group-Query Attention*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flash_attn.py:1586-1661 (use_cascade_attention), keep the correctness/support gates (alibi, sliding window, local attention, DCP) but replace the hard-coded shortcuts `common_prefix_len < 256` and `num_reqs < 8` with an occupancy-driven decision that mirrors TensorRT-LLM's generation-phase multi-block heuristic. Concretely: (a) compute the FlashDecoding occupancy as `flash_decoding_waves = cdiv(num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size) * num_prefix_tiles, num_sms)` and the cascade occupancy analogously, and only enable cascade when FlashDecoding is under-occupying the SMs (e.g. `flash_decoding_waves < THRESH` and prefix work dominates) or when cascade's projected wave count is materially lower; (b) derive the 256-token and 8-request thresholds from the same model rather than as separate constants, so they scale with `num_sms`, `num_kv_heads`, and `num_queries_per_kv`; (c) parameterize `q_tile_size`/`kv_tile_size` so they reflect what flash-attn actually picks for the current head_dim/dtype (64/128/256) instead of always 128. The unified oracle is output equality against non-cascade FlashAttention plus the existing cascade tests, with TTFT/TPOT benchmarked on multi-turn shared-prefix batches.

**Proposal rationale.**

The finding's core transferable idea is that split/multi-block attention should be selected by an occupancy model driven by batch size, head count, and SM count rather than by fixed constants. This directly addresses the two TODO-marked constants and the rough 128/128 tile assumption in use_cascade_attention, which are the tunable constructs the candidate flags. On the target multi-turn agentic workload, shared prefixes are long and per-decode queries are short — exactly the regime where an occupancy-aware policy chooses cascade for prefix-dominated work and avoids it when FlashDecoding already saturates the GPU, plausibly improving median TTFT/TPOT without changing numerics.

---

## Agent proposals

### 1. Route cascade vs Flash-Decoding by HBM-bytes and L2 residency of the shared prefix
- **Agent:** claude

**Detailed description.**

Rework the performance model in `use_cascade_attention` at vllm/v1/attention/backends/flash_attn.py:1637-1661 to be memory-bandwidth-oriented rather than CTA-count-oriented, since decode-phase attention on long shared prefixes is HBM-bandwidth-bound, not compute-bound. Concretely: (1) compute `prefix_kv_bytes = common_prefix_len * num_kv_heads * head_dim * dtype_bytes * 2` (K and V). Cascade reads this once for the whole batch, so `cascade_prefix_bytes = prefix_kv_bytes`; Flash-Decoding reads it `num_reqs` times but with cache reuse, so `fd_prefix_bytes = prefix_kv_bytes * effective_reuse_factor(num_reqs, prefix_kv_bytes, l2_size)` where `effective_reuse_factor` is `num_reqs` when `prefix_kv_bytes > l2_size` (each request refetches from HBM) and decays toward 1 when `prefix_kv_bytes << l2_size / num_concurrent_ctas` (later requests hit L2). Approximate this piecewise: `reuse = 1 + (num_reqs - 1) * min(1.0, prefix_kv_bytes / max(l2_size - working_set_slack, 1))`. (2) Add per-request suffix and query traffic that is common to both branches (cancels out, but include for clarity when the ratio is close). (3) Divide total bytes by an HBM-bandwidth proxy (or leave in raw bytes since both branches use the same GPU) and pick cascade when `cascade_prefix_bytes < fd_prefix_bytes`. (4) Obtain `l2_size` and `num_sms` from the existing device-query path used to fetch `num_sms` (extend that helper to also return L2 size, cached per-device). (5) Re-derive the 256-token and 8-request early-exit gates from the same bytes model: cascade is only worth its scheduling overhead when `cascade_prefix_bytes >= scheduling_overhead_bytes_equivalent` and when `fd_prefix_bytes - cascade_prefix_bytes` exceeds a small fixed savings floor. The correctness oracle is unchanged (output equality against non-cascade FlashAttention plus existing cascade tests); validation is TTFT/TPOT on multi-turn agentic traces with prefix lengths spanning below-L2, near-L2, and above-L2 regimes.

**Novelty rationale.**

The two existing deep_research_proposals both model the decision in the compute domain: proposal 1 refines Flash-Decoding's split-KV parallelism and LSE merge cost, and proposal 2 replaces the constants with an SM-occupancy / wave-count model. Neither models HBM bytes moved or L2 cache reuse of the shared prefix across requests — yet on the target multi-turn agentic workload (long shared prefix, batch size 8-32, decode phase with `query_lens == 1`) attention is HBM-bandwidth-bound and the *decisive* asymmetry between cascade and Flash-Decoding is that cascade streams the prefix from HBM once per batch while Flash-Decoding re-streams it per request modulo L2 reuse. An occupancy-only model (proposal 2) will happily route to Flash-Decoding whenever it saturates SMs, missing the case where every one of those SMs is stalled on HBM; a split-KV model (proposal 1) improves Flash-Decoding's compute estimate but still ignores the bandwidth axis. The L2-residency piecewise reuse factor and the `prefix_kv_bytes` comparison are the transferable idea here and are absent from both prior proposals, so this is complementary rather than overlapping.

---

### 2. Make cascade routing aware of non-decode query lengths
- **Agent:** codex

**Detailed description.**

Change `use_cascade_attention` in `vllm/v1/attention/backends/flash_attn.py:1586-1661` so the `if not use_flash_decoding: return True` branch is no longer unconditional. Add a query-length-aware path for mixed/chunked prefill batches where `np.all(query_lens == 1)` is false: compute `num_query_tokens = int(np.sum(query_lens))`, `q_tiles = cdiv(num_query_tokens, q_tile_size)`, and only use cascade when the shared-prefix reuse is large enough to offset the extra cascade prefix/suffix launches and merge work. A conservative first policy would keep the current `True` for small decode-like totals, but disable cascade when `q_tiles * num_query_heads` already produces enough CTAs to occupy the GPU and `num_query_tokens / num_reqs` is high, because normal FlashAttention then has ample Q-side parallelism while cascade adds an additional prefix pass. Add targeted tests or benchmarks with heterogeneous `query_lens` such as `[1, 1, 64, 128]` and chunked-prefill-heavy batches to confirm this does not regress TTFT/TPOT outside pure decode.

**Novelty rationale.**

The existing deep research proposals focus on the Flash-Decoding comparison: split-KV modeling, SM occupancy, and tile sizing. Agent A adds an HBM/L2 byte model for the shared prefix, again aimed at the long-prefix decode case where Flash-Decoding rereads prefix KV per request. This proposal targets a separate blind spot: the non-Flash-Decoding branch currently returns `True` for every supported batch after the fixed gates, ignoring `query_lens` entirely. Mixed decode plus chunked prefill can have enough Q-side work to make ordinary FlashAttention efficient while cascade pays extra launch and merge overhead, so a query-length-aware guard is complementary rather than duplicative.

---
