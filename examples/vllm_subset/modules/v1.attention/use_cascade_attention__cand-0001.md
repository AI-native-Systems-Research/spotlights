# use_cascade_attention

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/backends/flash_attn.py`](vllm/v1/attention/backends/flash_attn.py) (lines 1053–1128)
- **Symbol:** `use_cascade_attention`
- **Kind:** function
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0001`

## Description
Heuristic that decides whether the FlashAttention backend should split shared-prefix work into cascade attention.

## Current approach
Hard-gates on common_prefix_len < 256, unsupported attention variants, num_reqs < 8, and dcp_world_size > 1. For FlashDecoding-shaped decode batches it then compares a rough CTA/wave model using fixed q_tile_size = kv_tile_size = 128.

## Estimated impact explanation
Multi-turn agentic batches commonly share large prefixes across requests. A bad gate either misses KV-bandwidth savings, hurting TTFT, or pays unnecessary split/merge overhead, hurting median TPOT.

## Evolve rationale
This is a pure policy gate between algebraically equivalent attention paths. The thresholds 256 and 8 and the fixed tile-size performance model are explicitly heuristic and self-contained. Correctness oracle: tests/kernels/attention/test_cascade_flash_attn.py and tests/v1/attention/test_attention_backends.py compare cascade-on/off outputs within tolerance; performance oracle is TTFT/TPOT on prefix-sharing batches.

## Deep research proposals

### 1. Replace independent prefix/batch thresholds with a Hydragen-style shared-prefix cost model
- **Finding:** `find-0002` — *Hydragen: High-Throughput LLM Inference with Shared Prefixes*
- **Source URL:** <https://arxiv.org/abs/2402.05099>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flash_attn.py:1053-1128 (use_cascade_attention), replace the current pair of independent hard gates (common_prefix_len < 256 and num_reqs < 8) with a single cost-model gate inspired by Hydragen's shared-prefix vs unique-suffix decomposition. Concretely: (1) compute an estimated savings term proportional to num_reqs * common_prefix_len (KV bytes saved by reading the shared prefix once instead of num_reqs times) and an overhead term proportional to the split/merge cost (a small constant plus a function of the average unique-suffix length); (2) enable cascade when savings/overhead exceeds 1, instead of when both raw counts exceed fixed cutoffs. Keep the existing hard-fail conditions (unsupported attention variants, dcp_world_size > 1) unchanged. For the FlashDecoding-shaped decode branch (the CTA/wave block with q_tile_size = kv_tile_size = 128), fold the same product term into the wave model so a long shared prefix can compensate for a small batch, and a large batch can compensate for a moderately short prefix. Expose the model coefficients as module-level constants so they can be tuned without restructuring the gate.

**Proposal rationale.**

The current heuristic uses independent thresholds (>=256 prefix tokens AND >=8 requests) that under-fire on exactly the multi-turn agentic batches the caller targets: small batches sharing very long system prompts/tool schemas, or larger branchy batches sharing shorter prefixes. Hydragen's framing - 'computes attention over the shared prefix and unique suffixes separately' - makes the relevant quantity the product of batch size and prefix length (KV-read amortization) versus unique-suffix work, not either one in isolation. Substituting a product-form cost model for the two scalar cutoffs is a direct, conservative application of the finding to the candidate's gate, addresses the explicit 'self-contained heuristic thresholds' weakness called out in evolve_rationale, and is testable against the existing cascade-on/off equivalence tests plus a TTFT/TPOT measurement on prefix-sharing batches.

---

### 2. Replace fixed cascade-gate thresholds with a load-balance-aware estimator
- **Finding:** `find-0004` — *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*
- **Source URL:** <https://openreview.net/forum?id=RXPofAsL8F>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/v1/attention/backends/flash_attn.py:1053-1128` (`use_cascade_attention`), replace the hard-coded `common_prefix_len < 256`, `num_reqs < 8`, and the fixed `q_tile_size = kv_tile_size = 128` CTA/wave model with a load-balance-aware decision inspired by FlashInfer's scheduling. Concretely: (1) compute per-request suffix KV lengths already available in the metadata and estimate the work imbalance between the shared-prefix pass (one big KV span across all queries) and the per-request suffix pass; (2) gate cascade-on when the estimated wall time of (prefix + merge + suffix) is lower than the non-cascade baseline given current SM occupancy and the actual suffix-length distribution, rather than against fixed thresholds; (3) keep the dcp_world_size and unsupported-variant hard gates and ensure the new estimator is a pure function of metadata sizes (no kernel launches, no host-device sync) so CUDA Graph compatibility on the decode path is preserved. The thresholds 256 and 8 become emergent from the estimator rather than constants.

**Proposal rationale.**

The candidate is a policy gate whose two main weaknesses are (a) fixed cut-offs that do not reflect actual batch shape and (b) a tile-size performance model that ignores suffix-length skew, which is exactly what hurts on multi-turn agentic batches with shared prefixes and uneven tail lengths. FlashInfer's contribution that 'load-balanced scheduling adjusts to dynamism of user requests while maintaining compatibility with CUDAGraph' targets the same gap: dynamic, skewed batches need a balance-aware decision, not fixed thresholds, and must remain graph-capturable. Porting that idea as a metadata-only estimator inside `use_cascade_attention` keeps the change scoped to the gate (algebraically equivalent kernels on either side, so the existing cascade-on/off correctness tests in `tests/kernels/attention/test_cascade_flash_attn.py` still serve as the oracle) while plausibly improving TTFT on prefix-heavy batches and avoiding TPOT regressions on small/skewed batches the current num_reqs<8 rule rejects too coarsely.

---

### 3. Replace fixed num_reqs/tile-size gate with Flash-Decoding occupancy model for cascade decision
- **Finding:** `find-0005` — *Flash-Decoding for long-context inference – PyTorch Blog*
- **Source URL:** <https://pytorch.org/blog/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flash_attn.py:1053-1128 (use_cascade_attention), replace the hard `num_reqs < 8` cutoff and the fixed q_tile_size=kv_tile_size=128 CTA/wave estimate with an occupancy-aware model inspired by Flash-Decoding. Concretely: (1) Drop the fixed `num_reqs < 8` short-circuit and instead compute an effective-parallelism estimate `parallel_units = num_reqs * num_kv_heads` (the natural parallelism of the non-cascade path) and compare it against a target occupancy `target_ctas = num_sms * waves_target`. (2) When `parallel_units < target_ctas` and `common_prefix_len` is large (the regime Flash-Decoding identifies as decode-underutilized), the cascade path — which adds parallelism along the shared-prefix KV-length dimension and merges partial outputs via log-sum-exp — should be preferred; gate cascade-on in this regime even at smaller `num_reqs`. (3) For the CTA/wave comparison currently using fixed 128/128 tiles, use the actual q_tile/kv_tile sizes that FlashAttention will pick for the given (head_dim, dtype, sm) configuration so the estimate matches the kernel that will actually launch. Keep the existing supported-variants and dcp_world_size>1 hard gates unchanged, and keep `common_prefix_len < 256` as a floor (since shared-prefix KV-length parallelism only helps once the prefix is long enough for a useful number of KV splits). Validate with tests/kernels/attention/test_cascade_flash_attn.py and tests/v1/attention/test_attention_backends.py for correctness, and measure TTFT/TPOT on multi-turn agentic batches with long shared prefixes and small effective batch sizes.

**Proposal rationale.**

The candidate's current heuristic uses a fixed `num_reqs < 8` cutoff and a fixed-tile CTA/wave model to decide between cascade and non-cascade attention; both are acknowledged as crude. Flash-Decoding directly identifies the regime that makes cascade most valuable — long contexts with small effective batch sizes that leave `batch * num_kv_heads` parallelism below GPU saturation — and prescribes adding KV-sequence-length as a new parallelism dimension, exactly what cascade does on the shared prefix. Reframing the gate as an occupancy comparison (`num_reqs * num_kv_heads` vs `num_sms * waves_target`) using the kernel's actual tile sizes addresses a concrete gap in the heuristic: it captures why `num_reqs < 8` is a proxy (low parallelism) rather than a fundamental boundary, and it lets the gate adapt to long-prefix multi-turn workloads where cascade should win even at modest batch sizes — the workload the candidate explicitly targets.

---

## Agent proposals

### 1. Profile-guided autotuning of the cascade gate via startup microbenchmarks plus online shadow timing
- **Agent:** claude

**Detailed description.**

Replace the hand-coded constants in vllm/v1/attention/backends/flash_attn.py:1053-1128 (use_cascade_attention) with an empirically calibrated decision function rather than a different analytical model. Concretely: (1) On engine init, after CUDA-graph warmup, run a one-time microbenchmark that times both cascade_attention and the non-cascade FlashAttention path on a small grid of representative shapes — (num_reqs, common_prefix_len, mean/p95 suffix_len, num_kv_heads, num_queries_per_kv, head_dim, dtype) — using the same kernels production will launch (vllm_flash_attn_version, dcp config). Fit a small classifier (e.g., isotonic regression on a 1-D cost score, or a linear boundary in log-space across 4–6 features) and persist it keyed by GPU SKU + FA version + dtype to ~/.cache/vllm so subsequent processes skip the calibration. (2) Replace `common_prefix_len < 256`, `num_reqs < 8`, and the q_tile=kv_tile=128 CTA/wave block with a single call into this fitted predictor; keep the unsupported-variant and dcp_world_size>1 hard gates unchanged. (3) Optionally enable bounded online refinement outside CUDA-graph capture: when the batch lands within a configurable margin of the fitted boundary, occasionally re-run the unchosen path under CUDA events on a low cadence (e.g., 1-in-N forwards), feed the measured Δlatency into a per-bucket EMA, and let the predictor drift. This is orthogonal to which features the predictor consumes — it can be parameterized using a Hydragen-style product term, suffix-skew estimator, or Flash-Decoding occupancy ratio — but the constants and the boundary itself come from measurement, not derivation. Validate correctness with tests/kernels/attention/test_cascade_flash_attn.py and tests/v1/attention/test_attention_backends.py (the gate flips don't change kernel outputs); validate performance with TTFT/TPOT on multi-turn agentic batches and confirm calibration cost is amortized across the run.

**Novelty rationale.**

The three existing deep_research_proposals all replace the gate with a different *analytical* model — Hydragen-style savings/overhead ratio (find-0002), FlashInfer-style load-balance estimator (find-0004), and Flash-Decoding occupancy formula (find-0005) — each with its own hand-picked coefficients (cost-model constants, target_ctas/waves_target, suffix-skew weights). None proposes that the constants and boundary should be *measured* on the deployment hardware via a startup microbenchmark plus optional online shadow-timing, nor does any address the drift-between-FA/CUDA-versions/SKUs failure mode where a fixed-coefficient analytical model gradually mis-fits. This proposal is orthogonal and composable: it can adopt any of those three analytical forms as its feature parameterization, but its core claim — that the gate should be data-driven rather than derived — is distinct from all three.

---

### 2. Lift the DCP hard gate with rank-local cascade attention
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/backends/flash_attn.py:1053-1128`, stop treating `dcp_world_size > 1` as an unconditional cascade disable once the DCP path can split the shared context locally. Concretely, compute the rank-local common-prefix length with the same partitioning logic used by `get_dcp_local_seq_lens`, build DCP metadata for a local shared-prefix span plus per-request local suffix spans, and update `_forward_with_dcp` so its context-attention stage runs two local FlashAttention calls: one shared-prefix call with batch size 1 and `causal=False`, and one per-request suffix-context call with `causal=False`. Merge those local prefix/suffix states with `merge_attn_states`, then feed the merged local context state into the existing `dcp_combine` LSE reduction and the existing final merge with query-token attention. Keep the ALiBi/sliding-window/local-attention hard gates, and fall back to the current single DCP context-attention path when the rank-local shared prefix is too short. Validate with DCP-enabled cascade correctness coverage and TTFT/TPOT runs on long-context shared-prefix batches using `decode_context_parallel_size > 1`.

**Novelty rationale.**

The listed deep-research proposals and Claude proposal all keep the `dcp_world_size > 1` hard gate unchanged while changing the non-DCP decision model through analytical costs, occupancy estimates, or calibration. This proposal targets a different limitation: making cascade attention legal and useful under decode context parallelism by changing the DCP context-attention dataflow, not by retuning thresholds or replacing the non-DCP heuristic.

---
