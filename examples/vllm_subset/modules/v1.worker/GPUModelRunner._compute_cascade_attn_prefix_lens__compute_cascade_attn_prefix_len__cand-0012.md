# GPUModelRunner._compute_cascade_attn_prefix_lens/_compute_cascade_attn_prefix_len

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 2367–2499)
- **Symbol:** `GPUModelRunner._compute_cascade_attn_prefix_lens/_compute_cascade_attn_prefix_len`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0012`

## Description
Computes per-KV-group cascade-attention prefix lengths and decides whether shared-prefix cascade attention should be enabled for each attention group.

## Current approach
The outer loop walks every KV-cache group and attention group. The inner helper computes common_prefix_len from num_common_prefix_blocks, caps it by min(num_computed_tokens), rounds down to block_size, derives sliding/local-attention flags, and delegates a binary decision to attn_metadata_builder.use_cascade_attention with no hysteresis or memoization.

## Estimated impact explanation
This policy can change which attention kernels run. Better gating can move both prefill-heavy TTFT and decode TPOT for shared-prefix multi-turn workloads, so the potential upside is larger than a pure Python micro-optimization.

## Evolve rationale
Cascade attention can be beneficial for batches sharing long prefixes, which is common for agentic sessions with repeated system/developer prompts. Headroom is in tuning the prefix clipping and backend threshold policy, caching decisions when prefix/batch shape is stable, and adding hysteresis to avoid step-to-step flips. Correctness oracle: logits parity between cascade and non-cascade paths within tolerance in cascade-attention e2e/kernel tests, plus unchanged attention metadata when the decision remains false.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Self-calibrating cascade gating via periodic A/B latency probing in stable batch regimes
- **Agent:** claude

**Detailed description.**

In `GPUModelRunner._compute_cascade_attn_prefix_lens` (vllm/v1/worker/gpu_model_runner.py:2367-2401), augment the gating with an online A/B probe that self-calibrates the cascade decision against observed forward-pass latency, rather than relying solely on the static threshold inside `attn_metadata_builder.use_cascade_attention`.

Concretely:

1. Add a small per-(kv_cache_group, attn_group) state dict on `GPUModelRunner` keyed by a stability fingerprint = (num_kv_cache_groups, num_reqs, num_common_prefix_blocks[gid], rounded query-length histogram bucket, kv_cache_spec.block_size, sliding/local flags). The fingerprint is cheap to compute from arguments already in scope at line 2391–2397.

2. For each fingerprint, maintain (a) the static decision the existing heuristic returned, (b) an EMA of measured forward-pass latency under cascade=ON and cascade=OFF, (c) a `steps_since_probe` counter, and (d) a confidence count.

3. When the same fingerprint has been observed for K consecutive steps (e.g., K=8) — i.e., the multi-turn agentic batch has settled into steady-state decode — every M-th step (e.g., M=64, stochastically jittered) override the static heuristic to the *opposite* decision and tag the step. After the model forward returns, sample a CUDA-event-based latency for the attention region (or use the runner's existing forward-time accounting) and update the EMA for that arm.

4. Use the EMA delta to derive an effective decision: choose cascade ON iff `ema_off - ema_on > guard_band`, with `guard_band` proportional to the EMA noise floor. When confidence is low (cold start) fall back to the static heuristic.

5. Reset/decay state when the fingerprint changes for >1 step (handled implicitly by keying on fingerprint) and bound the dict size with an LRU.

Why it helps the stated workload: multi-turn agentic batches reach a stable decode regime where the static `use_cascade_attention` heuristic — which is a backend-defined threshold over query_lens / num_query_heads / num_kv_heads / num_sms / dcp_world_size — may be miscalibrated for this exact (model, hardware, batch shape, prefix length) point. Median TPOT in steady-state decode is exactly where steady-state fingerprints repeat tens of thousands of times, so even a small online correction compounds. TTFT benefits when the same calibration carries over to recurring prefill-with-shared-system-prompt patterns.

Files touched: only `vllm/v1/worker/gpu_model_runner.py` (state on the runner, ~30 lines around lines 2367-2401 plus a tiny helper to record post-forward latency, and a probe-aware wrapper around line 2487's `use_cascade_attention` call). No backend changes; this layers on top of existing `attn_metadata_builder.use_cascade_attention`.

Correctness: the override only chooses between two paths the system already produces correct outputs on, so logits-parity oracles for cascade are unchanged. The only added invariant is that latency measurements must exclude steps with mismatched fingerprints (handled by the fingerprint key).

Guarding: behind a default-off env flag (e.g., `VLLM_CASCADE_ONLINE_CALIBRATE`) so it does not affect deterministic benchmarking until validated.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, but the evolve_rationale enumerates four directions: (1) tuning prefix clipping, (2) tuning the backend threshold policy, (3) caching decisions when prefix/batch shape is stable, and (4) hysteresis against step-to-step flips. This proposal is none of those: it is *active online exploration* — the runner deliberately deviates from the static heuristic on a small fraction of steps to measure the actual latency of both arms and feeds that signal back into a self-calibrating gate. (1)/(2) are static tuning, (3) memoizes the same decision, and (4) is debouncing. Periodic A/B probing with EMA-weighted latency feedback is structurally different: it discovers the locally-correct decision per (model, hardware, batch shape, prefix length) point rather than guessing it offline, which is precisely the regime where multi-turn agentic decode steady states amortize the probing cost over many repeated fingerprints.

---

### 2. Recompute cascade prefixes from the actually scheduled batch
- **Agent:** codex

**Detailed description.**

In `GPUModelRunner._compute_cascade_attn_prefix_lens` / `_compute_cascade_attn_prefix_len` (`vllm/v1/worker/gpu_model_runner.py:2367-2499`), add a scheduled-batch common-prefix fallback that derives the longest shared KV block prefix from `self.input_batch.block_table[kv_cache_gid]` after `_update_states` has removed unscheduled requests from the persistent batch. The scheduler-provided `num_common_prefix_blocks` is computed over all requests with allocated KV cache, and `vllm/v1/core/kv_cache_manager.py` documents that this can be 0 when unscheduled allocated requests do not share the prefix even though all scheduled requests do. The runner is the right place to recover that lost opportunity because it already has only the current scheduled request rows, their block tables, and `num_computed_tokens_cpu[:num_reqs]`.

Concretely, add a small helper near the candidate region, e.g. `_compute_scheduled_common_prefix_len(kv_cache_gid, num_reqs) -> int`, that scans the CPU block table rows for the current scheduled batch and counts identical leading block IDs up to the minimum populated row length. Convert that count to tokens using the block table's effective block size, then feed the resulting token length into the existing safety logic: cap by `num_computed_tokens.min()`, round down to `kv_cache_spec.block_size`, reject sliding/local/ALiBi/DCP through the same `attn_metadata_builder.use_cascade_attention` call, and return 0 if the backend says no. Use the scheduled-batch prefix only when it exceeds the scheduler prefix, so the existing scheduler result remains the fast/common path.

This directly targets multi-turn agentic workloads where many currently decoded requests share a long system/developer prompt, but unrelated paused or temporarily unscheduled requests still hold KV blocks and force the scheduler's global common-prefix result to 0. Enabling cascade for the scheduled cohort in those steps should improve median TPOT and can also improve TTFT for batched shared-prompt prefill without changing the attention backend or the cascade kernel semantics. Validation should include a unit test with three allocated requests where only two scheduled rows share a prefix, asserting that `_compute_cascade_attn_prefix_lens` now returns a nonzero prefix for the scheduled two, plus existing cascade/non-cascade logits parity tests.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. This is not Agent A's online A/B calibration: it does not measure latency, override heuristics, maintain EMA state, or learn a better threshold. It also is not threshold tuning, memoization, or hysteresis. The change addresses a different source of false negatives: the scheduler's common-prefix value is computed over all allocated KV-cache owners, while the runner executes only the scheduled batch. Recomputing the prefix over scheduled rows can expose cascade opportunities that the static heuristic never sees.

---
