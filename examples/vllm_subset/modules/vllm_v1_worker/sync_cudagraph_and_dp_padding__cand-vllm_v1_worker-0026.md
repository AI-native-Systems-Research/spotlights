# sync_cudagraph_and_dp_padding

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/dp_utils.py`](vllm/v1/worker/gpu/dp_utils.py) (lines 16–83)
- **Symbol:** `sync_cudagraph_and_dp_padding`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0026`

## Description
New GPU runner DP coordination for cudagraph mode and token padding across data-parallel ranks.

## Current approach
Uses a CPU 3 x dp_size tensor, dist.all_reduce on the CPU group, then derives all-zero, minimum cudagraph mode, max token count, and uniform-token agreement through scalar tensor operations and .item() calls.

## Estimated impact explanation
The CPU collective avoids GPU sync but still adds per-step coordination latency in DP serving; reducing scalar post-processing or combining it with scheduler metadata improves median TPOT at scale.

## Evolve rationale
The CPU all_reduce tensor layout and scalar post-processing define the per-step DP synchronization policy. DP padding/cudagraph invariants are covered by DP execution tests and the dispatch_cg_and_sync_dp call path in GPUModelRunner.execute_model.

## Deep research proposals

### 1. Amortize DP cudagraph/padding sync across multi-step decode windows
- **Finding:** `find-vllm_v1_worker-0001` — *[RFC]: Multi-Step Scheduling*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/6854>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend sync_cudagraph_and_dp_padding in vllm/v1/worker/gpu/dp_utils.py (lines 16-83) to operate at a multi-step granularity rather than once per decode token. Instead of building the CPU 3 x dp_size tensor, invoking dist.all_reduce on the CPU group, and materializing all-zero / min-cudagraph-mode / max-token / uniform-token decisions via .item() on every step, cache the agreed cudagraph mode and padded token count for a window of N lookahead decode steps whenever the scheduler indicates the DP-visible state (running set, uniform-token property, cudagraph eligibility) is stable across that window. During the window, dispatch_cg_and_sync_dp in GPUModelRunner.execute_model reuses the cached decision without a CPU collective or scalar host syncs, only re-running the full all_reduce + post-processing on window boundaries or when a rank detects a state change that invalidates the cached agreement. Combine the residual sync with existing scheduler metadata (piggyback on the step-boundary control message) so DP coordination no longer contributes a fresh CPU all_reduce per token.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out per-step scalar post-processing and coordination latency as the TPOT bottleneck; the finding's core technique - amortizing per-step Python/host overheads across N decode steps by delaying scheduler/output synchronization until lookahead slots are exhausted - maps directly onto this function's cost model. The finding's supporting evidence ("amortize all these overheads over n-steps at a time") specifically targets the class of per-step host-side coordination the candidate performs, addressing the gap that today every decode token pays a CPU all_reduce + multiple .item() syncs even when the DP-visible state is unchanged. This is a concrete, transferable change (introduce a multi-step cached agreement with invalidation) rather than a topical restatement.

---

## Agent proposals

### 1. Replace SUM 3xdp_size all_reduce with packed MIN+MAX collectives that fold reductions into the wire
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu/dp_utils.py:16-83, change sync_cudagraph_and_dp_padding's coordination primitive so the collective itself computes the min-cudagraph-mode / max-num-tokens / uniform-token-agreement decisions, removing the SUM-based scatter layout and three of the four `.item()` host syncs per DP step.

Concretely, replace the 3 x dp_size int32 CPU tensor initialized with zeros and reduced with SUM by two length-1 int64 CPU tensors constructed per rank:
  packed_min = pack(cg_mode.value, -num_tokens_as_i32, uniform_token_count or SENTINEL_MAX)
  packed_max = pack(cg_mode.value, num_tokens_as_i32, uniform_token_count or SENTINEL_MIN)
using a fixed bit layout (e.g., high 8 bits cg_mode, middle 32 bits num_tokens/-num_tokens, low 24 bits uniform_token_count with 0xFFFFFF / 0 sentinels). Issue exactly two collectives on the DP CPU group: dist.all_reduce(packed_min, op=MIN) and dist.all_reduce(packed_max, op=MAX). Unpack the two scalar results to obtain simultaneously: synced_cg_mode = MIN(cg_mode) (drives the 'any rank wants eager => all eager' rule), synced_num_tokens = MAX(num_tokens), the all-zero case (packed_max's num_tokens field == 0), and uniform-token agreement (packed_min.uniform == packed_max.uniform and != sentinel). Only when downstream code still needs the per-rank num_tokens vector for return (num_tokens_across_dp handed back to callers) do we materialize it -- and only then via a single additional all_gather of a length-1 int32 into a preallocated dp_size buffer, which is bandwidth-equivalent to the current SUM tensor's useful payload but requires zero pre-zeroing and no scatter-through-SUM idiom.

Benefits, specific to this candidate:
  - Wire payload for the decision fields drops from 3*4*dp_size bytes to a fixed 16 bytes, independent of dp_size -- meaningful for large DP fan-out multi-turn agentic serving.
  - The `.min().item()`, `.max().item()`, and `torch.all(uniform_token_counts_across_dp == synced_uniform_token_count).item()` reductions collapse into two scalar unpacks (no tensor reductions, no host syncs beyond the two collectives themselves).
  - The `torch.all(num_tokens_across_dp == 0).item()` early-exit becomes a single equality check on the unpacked MAX result.
  - The SUM-with-zero-init scatter idiom (which silently corrupts if two ranks ever write the same column, e.g., due to a resend bug) is replaced with idempotent MIN/MAX ops that are correct under retries.

Keep the num_tokens_across_dp return value's shape/semantics so dispatch_cg_and_sync_dp callers in GPUModelRunner.execute_model don't change, but produce it via the single all_gather only when the caller actually consumes it (skip when the early-exit / eager paths return None today).

**Novelty rationale.**

The existing deep_research_proposal (find-vllm_v1_worker-0001) is about temporal amortization -- caching an agreed decision across N decode steps and skipping the collective entirely on cached steps. This proposal is orthogonal and operates on the *per-step* primitive itself: even when a sync is unavoidable (window boundary, state change, first step), it makes that single sync cheaper by (a) folding min/max/uniform-equality directly into MIN/MAX collective ops instead of doing them as post-hoc scalar tensor reductions, (b) shrinking the wire payload from O(dp_size) to O(1) for the decision fields, and (c) eliminating three of the four `.item()` host syncs per invocation. It does not overlap with amortization -- the two techniques compose cleanly (cheaper primitive underneath a coarser cache), and neither subsumes the other.

---

### 2. Ignore idle DP ranks when agreeing on cudagraph mode and uniform-token padding
- **Agent:** codex

**Detailed description.**

Update `sync_cudagraph_and_dp_padding` in `vllm/v1/worker/gpu/dp_utils.py` so ranks with `num_tokens == 0` do not force active ranks into eager mode or invalidate uniform-token agreement. After the existing all-reduce, keep the current all-zero early return, then derive an `active_mask = num_tokens_across_dp > 0`. Compute `synced_cg_mode` from `cg_mode_across_dp[active_mask].min()` instead of all ranks, and compute `synced_uniform_token_count` only across `uniform_token_counts_across_dp[active_mask]`, treating idle ranks as don't-care. Active ranks that truly need eager still contribute `CUDAGraphMode.NONE` with `num_tokens > 0`, so the eager fallback rule remains intact. Idle ranks can then dispatch with local `num_reqs == 0` and the synced padded token count, allowing them to participate in the same padded cudagraph execution instead of making the whole DP group run eager. Add focused DP tests for mixed active/idle ranks: one where active ranks share a uniform token count and still get a graph-capable descriptor, one where active ranks disagree and fall back to `uniform_token_count=None`, and one where an active rank requests eager and still disables cudagraph globally.

**Novelty rationale.**

The deep_research_proposal amortizes this synchronization across multiple decode steps; it does not change the per-step policy for how idle ranks affect cudagraph eligibility. Agent A's proposal changes the collective encoding and folds reductions into MIN/MAX operations; it still describes computing agreement over the same fields and does not address the semantic issue that zero-token ranks currently contribute `CUDAGraphMode.NONE` and `uniform_token_count=0`. This proposal is a policy change specific to mixed active/idle DP steps, which are common in multi-turn agentic serving and can directly improve TPOT by preserving cudagraph execution under DP load imbalance.

---
