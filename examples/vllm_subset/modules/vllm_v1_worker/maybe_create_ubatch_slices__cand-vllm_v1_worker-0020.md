# maybe_create_ubatch_slices

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/ubatch_utils.py`](vllm/v1/worker/ubatch_utils.py) (lines 63–114)
- **Symbol:** `maybe_create_ubatch_slices`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0020`

## Description
Chooses token split points and materializes legacy microbatch request/token slices for DBO.

## Current approach
Defaults to uniform token-count split_point = num_tokens_padded // num_ubatches and maps token split points back to request slices with np.searchsorted.

## Estimated impact explanation
Cost-balanced slices can improve DBO overlap, moving TPOT for mixed prefill/decode agent batches where equal token counts do not imply equal runtime.

## Evolve rationale
The split_point rule is a runtime scheduling heuristic. The sum-of-token invariant at line 112, tests/v1/attention/test_attention_splitting.py, and DBO coverage validate the produced slices.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Compute-cost-weighted DBO split using per-request query length and KV prefix
- **Agent:** claude

**Detailed description.**

Replace the uniform token-count split in `maybe_create_ubatch_slices` (vllm/v1/worker/ubatch_utils.py:63-114) with a cost-weighted split that balances estimated per-microbatch compute rather than raw token counts. Concretely: (1) accept an optional `num_computed_tokens` array (KV prefix length per request, already available on `CommonAttentionMetadata._num_computed_tokens_cpu` at the call sites in gpu_model_runner.py:4386 and 6055) plus a boolean `uniform_decode` hint; (2) when `split_point is None` and `uniform_decode is False`, compute a per-request cost estimate `c_i = alpha * q_i + beta * q_i * (kv_i + q_i/2)` where `q_i = num_scheduled_tokens[i]` and `kv_i = num_computed_tokens[i]` — the linear term captures MLP/projection FLOPs proportional to query tokens and the quadratic term captures causal attention FLOPs (q * average KV length seen, including the request's own newly scheduled tokens); (3) build `cu_cost = np.cumsum(c_i)` and pick token split points by locating the request boundary nearest to `cu_cost[-1] * k / num_ubatches` for `k in 1..num_ubatches-1` via `np.searchsorted(cu_cost, target)`, then converting that request boundary back to a token index using the existing `cu_num_tokens` array; (4) keep the current uniform token behavior as a fast path when `uniform_decode is True` (all `q_i` equal → cost weighting collapses to token weighting) or when `num_computed_tokens` is unavailable, so decode-only DBO paths pay zero extra work and preserve the existing `assert sum(s.num_tokens ...) == num_tokens_padded` invariant. `alpha` and `beta` can start as fixed constants (e.g. `alpha=1.0`, `beta=1/hidden_size`) and be exposed on `ParallelConfig` alongside `dbo_prefill_token_threshold` for later tuning. Update `tests/v1/attention/test_attention_splitting.py` with mixed prefill/decode fixtures (one long prefill + many decodes) to lock in that the split point falls after the long prefill rather than mid-way through the token axis. This targets TPOT for mixed agentic batches where a uniform token split assigns a heavy prefill request and its quadratic attention cost entirely to one microbatch, stalling the other stream.

**Novelty rationale.**

There are no existing deep_research_proposals listed on this candidate, so no overlap is possible. The specific mechanism proposed — cost weighting using q_i * (kv_i + q_i/2) to model causal-attention FLOPs, cumulative-cost searchsorted, reuse of `_num_computed_tokens_cpu` from `CommonAttentionMetadata`, and a `uniform_decode` fast-path exit — is a concrete, code-level design distinct from the candidate's evolve_rationale, which only observes at a high level that equal token counts do not imply equal runtime without specifying how to weight or where the KV-prefix input would come from.

---

### 2. Split DBO on actual tokens before padding
- **Agent:** codex

**Detailed description.**

Change `maybe_create_ubatch_slices` in `vllm/v1/worker/ubatch_utils.py:63-114` so the default `split_point is None` path computes uniform split points from the unpadded scheduled token total (`cu_num_tokens[-1]` / `np.sum(num_scheduled_tokens)`) rather than `num_tokens_padded`. Keep `_pad_out_ubatch_slices` as the only place that extends the final microbatch to `num_tokens_padded`. This requires moving the `cu_num_tokens` construction before default split-point selection, then using something like `actual_tokens // num_ubatches` for the first split while preserving the final padded invariant. Add a focused test where `num_scheduled_tokens.sum()` is much smaller than `num_tokens_padded` but still large enough not to trigger the existing empty-last-ubatch abort, e.g. actual 130 padded 256 with two ubatches; the expected unpadded slices should be roughly balanced over real work instead of `[0:128]` and `[128:130]` plus padding. This targets mixed agentic batches under cudagraph/DP padding, where the current padded-token split can leave one stream doing almost all real attention/MLP work while the other mostly processes padding, worsening DBO overlap and TPOT.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. This is also distinct from Agent A's proposal: Agent A changes the heuristic to a KV-prefix-aware compute-cost model and request-boundary placement. This proposal keeps the existing uniform token heuristic, but fixes the input domain from padded tokens to actual scheduled tokens so padding does not distort microbatch balance; it is a narrower padding-aware scheduling change not covered by the cost-weighted design.

---
