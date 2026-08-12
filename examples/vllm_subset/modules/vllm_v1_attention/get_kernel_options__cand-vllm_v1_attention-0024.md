# get_kernel_options

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flex_attention.py`](vllm/v1/attention/backends/flex_attention.py) (lines 1422–1483)
- **Symbol:** `get_kernel_options`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0024`

## Description
Selects PyTorch FlexAttention kernel options, especially BLOCK_M and BLOCK_N, from logical block sizes, dtype, direct-build mode, and shared-memory capacity.

## Current approach
For direct build it forwards block_m/block_n directly. Otherwise it uses preferred_block=32 for fp32 and 64 for other dtypes, enforces divisibility with gcd, lowers both candidates when shared memory is below 144 KiB, clamps to a lower bound of 16, and forces FlexAttention through FORCE_USE_FLEX_ATTENTION.

## Estimated impact explanation
BLOCK_M and BLOCK_N control FlexAttention kernel occupancy, memory use, and compilation choices. Better per-device and per-shape values can improve TTFT for compiled prefills and median TPOT for repeated decode on the Flex backend.

## Evolve rationale
Concrete tunables are preferred_block, block_lower_bound, the 144 KiB shared-memory threshold, and the ensure_divisible policy for BLOCK_M/BLOCK_N. Correctness oracle is FlexAttention output equality for legal kernel options plus existing FlexAttention backend tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Split BLOCK_M/BLOCK_N selection by decode vs prefill regime with shared-memory-aware sizing
- **Agent:** claude

**Detailed description.**

In `get_kernel_options` at vllm/v1/attention/backends/flex_attention.py:1422-1483, replace the single `preferred_block=32/64` heuristic with a regime-aware selection driven by the query shape and head dimension. Detect decode vs prefill from the incoming `query` (dim -2 length relative to `block_m`, i.e. effective Q tokens per doc): when Q per doc is small (decode / speculative decode), pick BLOCK_M as `max(block_lower_bound, next_pow2(effective_q_per_doc))` — typically 16 — and expand BLOCK_N up to 128 (still passed through `ensure_divisible(candidate, block_n)`), since long-context decode is dominated by KV-tile iteration and small BLOCK_M wastes fewer rows. For prefill (long Q), keep the current 32/64 preferred_block for BLOCK_M and scale BLOCK_N by head_dim: allow 128 when `head_dim <= 128` and shared memory permits, otherwise fall back to 64. Replace the single hard-coded 144 KiB threshold with an explicit per-tile shared-memory estimate: for BF16/FP16 use `2 * (BLOCK_M + BLOCK_N) * head_dim` bytes (4x for FP32), and pick the largest divisible (BLOCK_M, BLOCK_N) pair that fits under `device_props.shared_memory_per_block_optin` (with the existing ROCm 64 KiB fallback). Keep the `use_direct_build` fast path untouched, keep `FORCE_USE_FLEX_ATTENTION=True`, keep `ensure_divisible` and `block_lower_bound=16` as invariants, and continue to allow the caller's `self.block_m` / `self.block_n` overrides at lines 1394-1397 to take final precedence. To thread the regime signal without changing the public shape, either (a) extend the signature to accept `attn_metadata` (already available at the call site line 1390) and read decode vs prefill from it, or (b) infer from `query.shape[-2]` relative to `block_m`. This targets multi-turn agentic workloads where median TPOT is dominated by long-KV decode iterations and where the current single-heuristic block size leaves FlexAttention kernel occupancy on the table.

**Novelty rationale.**

The candidate has no listed deep_research_proposals, so any concrete tuning proposal is novel by construction. Beyond that, this proposal introduces two axes the current code explicitly ignores: (1) a decode-vs-prefill regime split for BLOCK_M/BLOCK_N (current code uses one preferred_block for both, ill-suited to short-Q decode common in multi-turn agentic serving), and (2) an explicit per-tile shared-memory cost model parameterized by head_dim and dtype instead of a single 144 KiB threshold. Both changes preserve the existing correctness invariants (`ensure_divisible`, `block_lower_bound`, `FORCE_USE_FLEX_ATTENTION`, direct-build path).

---
