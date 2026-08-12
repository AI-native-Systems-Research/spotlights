# Triton MLA decode split policy

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/mla/triton_mla.py`](vllm/v1/attention/backends/mla/triton_mla.py) (lines 35–47)
- **Symbol:** `Triton MLA decode split policy`
- **Kind:** config_block
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0025`

## Description
Chooses the number of KV splits for Triton MLA decode and reserves matching workspace capacity.

## Current approach
Uses _MIN_WORK_PER_SPLIT=512 and _SPLIT_OCCUPANCY_MULTIPLIER=2; _compute_num_kv_splits rounds max_seq_len // 512 up to a power of two and caps it at sm_count * 2. Batch-invariant mode later forces one split in forward_mqa.

## Estimated impact explanation
This split-KV policy directly trades parallelism against reduction overhead in Triton MLA decode. MLA agentic serving can be decode-heavy, so better split counts can move median TPOT while preserving outputs.

## Evolve rationale
Concrete tunables are _MIN_WORK_PER_SPLIT, _SPLIT_OCCUPANCY_MULTIPLIER, power-of-two rounding, and the SM-count cap. Correctness oracle is Triton MLA output equality across split counts and exact workspace shape agreement with _reserve_attn_logits_workspace.

## Deep research proposals

### 1. Adapt Triton MLA num_kv_splits to per-decode batch and actual seq_len using Flash-Decoding occupancy heuristic
- **Finding:** `find-vllm_v1_attention-0002` — *Flash-Decoding for Long-Context Inference*
- **Source URL:** <https://princeton-nlp.github.io/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Change the split-KV policy in vllm/v1/attention/backends/mla/triton_mla.py (lines 35-47 and its call site in forward_mqa) so that _compute_num_kv_splits considers the actual runtime shape (batch B, q_num_heads H, per-request max_seq_len), not just the static model-level max_seq_len. Concretely: (1) Replace the current formula ideal_splits = next_pow2(max_seq_len // _MIN_WORK_PER_SPLIT) with a Flash-Decoding-style target where the split count is chosen so that B * H * splits saturates roughly sm_count * _SPLIT_OCCUPANCY_MULTIPLIER, i.e. splits ~= max(1, next_pow2((sm_count * _SPLIT_OCCUPANCY_MULTIPLIER) // max(1, B * H))), still bounded below by the current _MIN_WORK_PER_SPLIT rule (do not split when per-split work would drop under 512 KV tokens) and above by the current SM-count cap. (2) Continue to use max_model_len for the workspace reservation in _reserve_attn_logits_workspace (worst-case upper bound), but pass the batch-aware split count into decode_attention_fwd at runtime in forward_mqa so short-context or large-batch decode does not over-split and pay reduction overhead, while small-batch long-context decode gains the extra sequence-length parallelism Flash-Decoding describes. (3) Preserve the batch-invariant single-split override in forward_mqa and the power-of-two rounding, so kernel-instantiation count is unchanged.

**Proposal rationale.**

The finding is exactly the technique the candidate implements: split KV along the sequence dimension and merge partial LSE states. The candidate's current policy, however, is derived only from the static max_model_len and sm_count, which is the classic gap Flash-Decoding calls out — the right knob is total parallel work relative to SM occupancy, and that depends on the current batch size and heads. Adapting the split count to (B * H, actual seq_len) directly targets the described TPOT regime (small batch, long context in multi-turn agentic decode) without touching numerics, since Triton MLA output equality across split counts is the stated correctness oracle and the workspace reservation is still sized at the worst case so no runtime growth is triggered.

---

### 2. Make Triton MLA decode split policy occupancy-aware over batch and heads
- **Finding:** `find-vllm_v1_attention-0005` — *Multi-Head, Multi-Query, and Group-Query Attention*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the fixed `_MIN_WORK_PER_SPLIT=512` / `_SPLIT_OCCUPANCY_MULTIPLIER=2` heuristic in `vllm/v1/attention/backends/mla/triton_mla.py` (lines 35-47) with an occupancy-aware policy for `_compute_num_kv_splits` that also considers the decode batch size `B` and the per-request query head count `q_num_heads * dcp_world_size` already computed for workspace reservation. Concretely, estimate the number of persistent blocks the decode kernel would already launch without splitting (roughly `B * q_num_heads / heads_per_block`, where `heads_per_block` matches the Triton kernel's head tile) and only increase `num_kv_splits` beyond 1 when that base block count is well below `sm_count * _SPLIT_OCCUPANCY_MULTIPLIER`. When occupancy is already saturated (large `B` or many heads), clamp splits toward 1 to avoid the split-reduction overhead; when occupancy is low (small `B`, short sequences relative to SMs), keep the current power-of-two growth up to `sm_count * multiplier`. Keep the power-of-two rounding and the SM-count cap as the outer envelope so `_reserve_attn_logits_workspace` still sees a deterministic worst-case `max_splits` (computed at `B = max_num_seqs`) and workspace shape invariants with `current_workspace_manager().get_simultaneous(...)` are preserved. `forward_mqa`'s batch-invariant path still forces one split. Validate by checking Triton MLA output equality across the new split counts against the old policy and confirming the reserved workspace shape is a superset of any runtime request.

**Proposal rationale.**

The candidate's current split policy depends only on `max_seq_len` and `sm_count` and ignores decode batch size and query-head count, so on decode-heavy multi-turn agentic workloads it can over-split at large B (wasting SMs on reductions) or under-split at small B (leaving SMs idle). The finding describes exactly this pattern from TensorRT-LLM's generation-phase multi-block attention: enable multi-block only when low occupancy makes one-block-per-head inefficient, guided by batch size, head count, SM count, and internal heuristics. Transferring that occupancy-aware gating to `_compute_num_kv_splits` addresses the concrete tunables called out in `evolve_rationale` while keeping the correctness oracle (output equality, workspace shape agreement with `_reserve_attn_logits_workspace`) intact, and directly targets the median TPOT reduction goal in the caller context.

---

## Agent proposals

### 1. Use intra-batch seq_len distribution (p90 of decode.seq_lens) instead of max_seq_len when choosing Triton MLA num_kv_splits
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/mla/triton_mla.py, change the input to _compute_num_kv_splits at its forward_mqa call site (line 261-263) from attn_metadata.max_seq_len to a distribution-aware 'effective' seq_len derived from the per-request tensor attn_metadata.decode.seq_lens (already available on device, see lines 287+). Concretely: (1) On the decode path, once per step, compute effective_seq_len = min(max_seq_len, int(torch.quantile(seq_lens.float(), 0.90).item())) — or an approximation via a sorted index (seq_lens_sorted[int(0.9 * B)]) to avoid the torch.quantile CPU sync on the hot path — and cache it on attn_metadata so repeated MLA layers reuse it. (2) Pass effective_seq_len into _compute_num_kv_splits so num_kv_splits = min(next_pow2(max(1, effective_seq_len // _MIN_WORK_PER_SPLIT)), sm_count * _SPLIT_OCCUPANCY_MULTIPLIER). (3) Leave _reserve_attn_logits_workspace unchanged: it continues to size the shared workspace off model_config.max_model_len, so the reserved (B, q_num_heads, max_splits, kv_lora_rank+1) buffer is always a superset of the runtime request (workspace shape invariant preserved). (4) Keep the VLLM_BATCH_INVARIANT single-split override and the power-of-two rounding so kernel instantiation count is unchanged. The motivation is that _compute_num_kv_splits' current lower-bound rule (_MIN_WORK_PER_SPLIT=512 applied to max_seq_len) implicitly assumes every request in the batch has roughly max_seq_len KV rows. In mixed agentic decode batches (one long summarizer + many short chat turns), this over-splits: short requests end up with seq_len_i << num_kv_splits * 512, so many of their KV splits do near-zero work but the stage-2 reduce kernel still pays O(B * H * num_kv_splits * (kv_lora_rank+1)) traffic across all requests. Using p90 (or a similar upper-tail quantile) keeps enough splits for the genuinely long-context requests to benefit from sequence-dim parallelism while eliminating the empty-split reduction tax for the short majority. Correctness oracle: Triton MLA output equality across split counts (as noted in evolve_rationale); the reduction merges partial LSEs so any legal num_kv_splits produces the same output up to floating-point associativity, and effective_seq_len <= max_model_len keeps the reserved workspace a valid superset.

**Novelty rationale.**

Both existing deep_research_proposals (finding_id find-vllm_v1_attention-0002 and find-vllm_v1_attention-0005) adapt the split count based on (batch size B, query head count H) versus SM occupancy — Flash-Decoding and TRT-LLM multi-block gating respectively — but neither changes the seq_len input to _compute_num_kv_splits: both keep attn_metadata.max_seq_len (or an equivalent single scalar) as the sequence-dimension driver. This proposal is orthogonal: it targets the *intra-batch distribution* of decode.seq_lens, not the (B, H) occupancy dimension. On a skewed decode batch (one long-context request with many short-turn requests, common in multi-turn agentic serving), the existing proposals' occupancy math would still see max_seq_len and choose a split count that leaves most requests' extra splits doing near-zero KV work while still paying stage-2 reduction cost. Using p90 of the per-request seq_lens closes exactly that gap. The two levers compose freely — an implementation could combine (B, H) occupancy gating from proposal #1/#2 with this seq_len quantile without conflict — so this is a genuinely additive knob, not a rephrasing.

---

### 2. Gate split-KV on actual decode token count to avoid splitting tiny steps
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/backends/mla/triton_mla.py`, add a runtime lower-bound guard at the `forward_mqa` call site so `_compute_num_kv_splits` is only allowed to return more than one split when the current decode workload has enough non-padding KV work to amortize the second-stage reduction. Concretely, derive `total_kv_tokens = int(attn_metadata.decode.seq_lens.sum().item())` or reuse an already-materialized decode token-count field if one exists nearby, and clamp `num_kv_splits = 1` when `total_kv_tokens < B * _MIN_WORK_PER_SPLIT` or when the average decode context length is below `_MIN_WORK_PER_SPLIT`. Keep `_reserve_attn_logits_workspace` unchanged and still size it from `model_config.max_model_len`, keep the batch-invariant single-split override, and otherwise fall back to the existing power-of-two split policy. Validate with output equality against the old policy across short, medium, and long `decode.seq_lens`, plus a workspace-shape assertion that the reserved logits buffer remains a superset of all runtime split counts. This targets median TPOT for multi-turn agentic batches where many decode steps have short accumulated context: the current max-seq-based policy can split even when the step’s total useful KV rows are too small for split-KV to pay for its reduce pass.

**Novelty rationale.**

The two deep-research proposals make the split count occupancy-aware over batch size, head count, SM count, and the static or per-request max sequence length. Agent A’s proposal changes the sequence-length scalar to an intra-batch upper-tail quantile such as p90. This proposal is a different guard: it uses the aggregate amount of useful KV work in the current decode step, especially average or summed `decode.seq_lens`, as an amortization threshold for the fixed stage-2 reduction overhead. A p90 sequence length can still be above 512 because of a minority of longer requests, and an occupancy heuristic can still request extra splits for low `B * H`; this guard would keep a single split when the total work across the batch is too small to make split-KV profitable. It is therefore an additive runtime profitability check rather than another occupancy formula or another choice of max-versus-quantile sequence length.

---
