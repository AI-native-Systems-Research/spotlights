# ARCCachePolicy.touch

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/policies/arc.py`](vllm/v1/kv_offload/cpu/policies/arc.py) (lines 75–101)
- **Symbol:** `ARCCachePolicy.touch`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0002`

## Description
ARC touch handling promotes T1 hits, refreshes T2 hits, and adapts target_t1_size on B1/B2 ghost hits.

## Current approach
Materializes list(keys) and iterates in reverse. Ready T1 hits move to T2, T2 hits move to MRU, and ghost hits adjust target_t1_size using max(1, len(B_other) / len(B_self)) without smoothing or momentum.

## Estimated impact explanation
Touch-time adaptation influences the future eviction target and therefore primary-cache hit rate, especially during bursty agent turns. It is meaningful, but downstream eviction decisions cap the immediate TTFT/TPOT effect.

## Evolve rationale
The promotion rule and ghost-hit adaptation formula are independent ARC tuning levers with deterministic state transitions. Correctness oracle: get/insert/remove/touch semantics, disjoint T1/T2/B1/B2 membership, capacity bounds, and existing ARC manager tests.

## Deep research proposals

### 1. Bias ARC touch promotion and ghost-hit adaptation with agent-supplied workflow priority
- **Finding:** `find-vllm_v1_kv_offload-0002` — *Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo*
- **Source URL:** <https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend ARCCachePolicy.touch (vllm/v1/kv_offload/cpu/policies/arc.py:75-101) to consult a per-request priority signal read from req_context.kv_transfer_params (e.g. a 'block_priority' or 'retention_hint' field, cached on req_context via set_state/get_state on first touch of the request). Use the hint in two places within the existing touch flow: (1) T1 hit path — for a 'high' priority key, promote T1->T2 even on first ready touch (as today) but additionally record a small persistent boost so the block does not fall back to T1 via subsequent policy pressure; for a 'low' priority key, keep it in T1 without promoting to T2 so it becomes an eviction candidate sooner. (2) Ghost-hit path (B1/B2) — scale the existing delta = max(1, len(B_other)/len(B_self)) by a bounded priority multiplier (e.g. clamp(hint_weight, 0.25, 4.0)) before updating target_t1_size, so ghost hits on workflow-critical blocks pull target_t1_size harder toward the partition that would have retained them, while ghost hits on low-value blocks contribute less. All updates continue to respect min(..., cache_capacity) and max(..., 0) bounds, and disjoint T1/T2/B1/B2 membership is preserved because only ordering / target_t1_size and (optionally) a boost annotation stored on BlockStatus change. When no hint is present the method behaves exactly as today, keeping existing ARC manager tests green.

**Proposal rationale.**

The finding shows a production agentic-inference system (NVIDIA Dynamo) exposing prefetch/pin/evict-first controls so the harness can retain blocks by workflow value rather than pure recency, citing 'lower-priority blocks are evicted first' as the mechanism. ARCCachePolicy.touch is the natural insertion point because it already receives req_context and is the single place that both promotes hits (T1->T2, T2->MRU) and adapts the T1/T2 balance via ghost hits — the two levers that determine what evict() will later select. The current implementation ignores req_context entirely, so multi-turn agentic workloads (the stated caller workload) cannot bias retention toward blocks a subsequent tool-call turn will need, even when the harness knows this. Wiring the hint into promotion and into a bounded scaling of the ghost-hit delta gives the module a workflow-value signal without changing ARC's state machine or capacity/disjointness invariants, addressing the gap that evolve_rationale flags (touch-time adaptation drives future eviction target and hit rate) and directly targeting the median-TTFT/TPOT objective under bursty multi-turn traffic.

---

## Agent proposals

### 1. Dampen ARC ghost-hit adaptation per touch batch with EMA-smoothed target_t1_size
- **Agent:** claude

**Detailed description.**

Modify ARCCachePolicy.touch in vllm/v1/kv_offload/cpu/policies/arc.py:75-101 so that (a) all B1 and B2 ghost hits within a single touch() call are aggregated before target_t1_size is updated, and (b) the aggregated update is applied through an EMA-style smoothing step rather than raw additive deltas. Concretely: keep the existing reversed iteration and T1/T2 promotion logic unchanged, but replace the two per-key ghost-hit branches with counters (b1_hits, b2_hits) plus per-branch move_to_end bookkeeping. After the loop, compute a single signed pressure using the ARC ratio evaluated once at loop entry (b2_len/b1_len for B1 hits, b1_len/b2_len for B2 hits), scale by log1p(b1_hits) and log1p(b2_hits) respectively so a burst of ghost hits in the same turn contributes sub-linearly instead of compounding N times, and blend into target_t1_size via target_t1_size = clamp(alpha * proposed + (1-alpha) * target_t1_size, 0, cache_capacity) with a small alpha (e.g., 0.25) exposed as a class-level constant. Preserve every existing invariant: min(..., cache_capacity) and max(..., 0) bounds, disjoint T1/T2/B1/B2 membership (only counters and move_to_end are affected), the T1 non-ready special case, and identical externally-observable semantics of get/insert/remove. Existing ARC tests remain green because single-key touches reduce to the current formula in expectation (log1p(1)=ln 2 ≈ 0.69 folded into the alpha constant tuning), and the smoothing constant defaults such that behavior on any touch() with a single ghost hit stays within one delta of today's implementation.

**Novelty rationale.**

Distinct from find-vllm_v1_kv_offload-0002: that proposal consumes an external workflow-priority hint from req_context and scales the per-key ghost-hit delta by it. This proposal introduces no external signal and instead attacks the gap evolve_rationale explicitly names ('without smoothing or momentum'): per-batch aggregation of ghost-hits + EMA blending into target_t1_size. In multi-turn agentic prefills a single touch() covers a long shared prefix, so today's implementation compounds the additive delta once per matched key and can jerk target_t1_size far more than the underlying access-pattern change warrants; batch aggregation with log1p dampening plus alpha-smoothing stabilizes target_t1_size across turns. The two proposals are composable (priority could scale the aggregated pressure) but neither implies the other.

---

### 2. Avoid copying reversible touch key batches
- **Agent:** codex

**Detailed description.**

Update `ARCCachePolicy.touch` in `vllm/v1/kv_offload/cpu/policies/arc.py:75-101` to preserve the existing reverse-order ARC state transitions while skipping `list(keys)` when the caller already supplies a reversible collection. Add a small helper or local branch such as `if isinstance(keys, Reversible): key_iter = reversed(keys) else: key_iter = reversed(list(keys))`, importing `Reversible` from `collections.abc`. This keeps generator support and all T1/T2/B1/B2 behavior unchanged, but avoids an O(n) allocation and full pre-copy for common call sites that pass lists, slices, or other collections of offload keys. Add a focused unit test with a custom reversible collection whose iteration would fail if materialized through forward iteration, plus an assertion that ARC membership and `target_t1_size` match the current list-backed behavior.

**Novelty rationale.**

This is not covered by the deep-research proposal, which adds workflow-priority hints and changes promotion/adaptation policy, and it is not covered by Agent A's proposal, which aggregates and smooths ghost-hit updates. This proposal targets the separate implementation detail called out in the candidate's current approach: unconditional `list(keys)` materialization. It changes touch-time CPU and memory overhead without introducing a new retention signal or altering ARC's adaptive formula.

---
