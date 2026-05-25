# decode_attention_fwd.launch_config

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/ops/triton_decode_attention.py`](vllm/v1/attention/ops/triton_decode_attention.py) (lines 195–645)
- **Symbol:** `decode_attention_fwd.launch_config`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0011`

## Description
Triton decode attention stage1/stage2 launch-configuration region for MHA, GQA/MQA/MLA, and split reduction.

## Current approach
MHA uses BLOCK = 64 on non-HIP and 8 on HIP, with num_warps 4 or reduced for grouped KV. Grouped GQA/MQA/MLA uses BLOCK = 32, HIP Lk>=576 falls back to 16, Lk 576/288 get special BLOCK_DMODEL/BLOCK_DPE splits, BLOCK_H = 16, num_warps = 4, and HIP-specific compiler kwargs. Stage2 reduction uses grid (batch, head_num), BLOCK_DV = next_power_of_2(Lv), num_warps = 4, and num_stages = 2.

## Estimated impact explanation
This path is used by TRITON_MLA and other Triton decode users; long-context agentic decode spends most TPOT here. Per-device and per-shape launch tuning can improve SM occupancy and reduce wasted split-reduction work, directly lowering median TPOT.

## Evolve rationale
After a backend chooses num_kv_splits, this owned launch region fixes tile width, head grouping, warp count, stage count, and HIP compiler options for the actual decode kernels. Those are exactly the runtime launch parameters that trade occupancy, memory traffic, and reduction overhead without changing outputs. Correctness oracle: tests/kernels/attention/test_triton_decode_attention.py covers normal and fp8 decode against references, and tests/v1/attention/test_mla_backends.py covers MLA backend integration.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Shape-aware BLOCK_H + cached autotune over (BLOCK_N, num_warps, num_stages) for grouped decode stage1
- **Agent:** claude

**Detailed description.**

In `_decode_grouped_att_m_fwd` (vllm/v1/attention/ops/triton_decode_attention.py:445-536), replace the hard-coded `BLOCK_H = 16` with a shape-aware value: `BLOCK_H = min(16, max(8, triton.next_power_of_2(max(kv_group_num, 1))))`. Today, when `kv_group_num` is 4 or 8 (common Llama-3.x GQA, Mistral, Qwen2 — workload-dominant for agentic decode), the kernel allocates a 16-wide H tile but `VALID_BLOCK_H` and `mask_h` mask off half the lanes, paying register pressure and dot-product work for masked rows. Lowering BLOCK_H to next_pow2(kv_group_num) keeps every lane productive, halves the per-program tile size, and roughly doubles the number of resident programs per SM/CU on the head_num/BLOCK_H grid axis. The internal masking logic at lines 301-304 and 391 already correctly handles `kv_group_num <= BLOCK_H`, so correctness is unchanged.

In the same launch site, wrap the four launch knobs `(BLOCK_N, BLOCK_H, num_warps, num_stages)` plus the HIP `extra_kargs` in a small process-wide cache keyed by `(Lk, Lv, kv_group_num, page_size, is_hip, is_mla, dtype)`. On first launch for a given key, evaluate ~6 curated configs (e.g. BLOCK_N in {16,32,64}; num_warps in {2,4,8}; num_stages in {1,2,3}; HIP `waves_per_eu` in {1,2}) using `do_bench` against a one-shot warm-up of stage1, store the winner, and reuse it for the rest of the run. Apply the same pattern to `_decode_att_m_fwd` (lines 195-258) where `BLOCK = 64`/`8` and `num_warps = 4`/`2`/`1` are also fixed, and to `_decode_softmax_reducev_fwd` stage2 (lines 606-645) where `num_warps = 4` is fixed regardless of `BLOCK_DV` — the optimal warp count for `BLOCK_DV=128` (MLA Lv) and `BLOCK_DV=512` is unlikely to be the same.

The shape key is small (a handful of unique tuples per deployment), the search runs once per shape per process, and the cache is process-local so it does not perturb numerical determinism inside a run. Validate against the existing oracles: `tests/kernels/attention/test_triton_decode_attention.py` (normal + fp8 decode) and `tests/v1/attention/test_mla_backends.py` (MLA backend integration). Benchmark long-context agentic decode on Llama-3.1-8B-Instruct (kv_group_num=8, Lk=Lv=128) and DeepSeek-V3 MLA (Lk=576, Lv=512) at seq_len in {8k,32k,64k}; expected gains are most pronounced at long context where stage1 dominates TPOT and BLOCK_H wastage is amortized over many KV iterations.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, so by construction this is novel relative to that set. The proposal is also distinct from the candidate's own `evolve_rationale`, which only frames *what* could be tuned in the launch region — it does not propose right-sizing BLOCK_H to kv_group_num (the specific masking-waste insight) or a per-shape, one-shot cached autotune harness around the grouped/MHA/stage2 launch sites.

---

### 2. Bypass stage2 for single-split decode
- **Agent:** codex

**Detailed description.**

Add a direct-output path when `num_kv_splits == 1` in `vllm/v1/attention/ops/triton_decode_attention.py`. In the MHA and grouped stage1 launches, add a `DIRECT_OUT` constexpr variant that passes `o`, `lse`, and their strides into `_fwd_kernel_stage1` / `_fwd_grouped_kernel_stage1`; when direct mode is enabled, store `acc / e_sum` directly to `o` and `e_max + tl.log(e_sum)` directly to `lse` instead of writing the single partial result into `attn_logits`. Then have `decode_attention_fwd_normal` and `decode_attention_fwd_grouped` skip `_decode_softmax_reducev_fwd` for `num_kv_splits == 1`. This removes one Triton launch plus the intermediate `attn_logits` read/write for short-context decode and batch-invariant MLA, where stage2 is only copying a single partial result rather than reducing multiple splits. Validate by extending `tests/kernels/attention/test_triton_decode_attention.py` to cover `num_kv_splits in [1, 8]` for normal, grouped, paged, and fp8 paths, and by running the MLA backend tests with the batch-invariant path that already forces one split.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes shape-aware `BLOCK_H` and cached autotuning of launch parameters, including stage2 warp count; it does not propose changing the dataflow or eliminating stage2 when no split reduction is needed. This proposal is a separate single-split fast path that reduces launch count and memory traffic rather than tuning tile sizes, warps, stages, or HIP compiler kwargs.

---
