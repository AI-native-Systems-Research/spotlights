# RopeState.prepare_positions and _prepare_rope_positions_kernel

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/mm/rope.py`](vllm/v1/worker/gpu/mm/rope.py) (lines 110–214)
- **Symbol:** `RopeState.prepare_positions and _prepare_rope_positions_kernel`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0032`

## Description
New GPU runner M-RoPE/XD-RoPE position preparation from staged prefill positions, deltas, and per-request query ranges.

## Current approach
Launches one Triton program per request with BLOCK_SIZE=1024 and an inner static loop over RoPE dimensions, even when decode queries are one token, and reads staged UVA-backed prefill positions for prefill rows.

## Estimated impact explanation
The cost is concentrated in multimodal agent workloads using M-RoPE/XD-RoPE; adaptive tiling or batching removes wasted lanes in one-token decode and improves multimodal median TPOT.

## Evolve rationale
The prepare_positions launch site and _prepare_rope_positions_kernel BLOCK_SIZE=1024 tiling are concrete targets. tests/v1/worker/test_mrope_prompt_embeds.py, tests/v1/worker/test_gpu_model_runner_mm_gather.py, and multimodal RoPE tests validate generated position IDs.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Split M-RoPE/XD-RoPE position preparation into a fused decode kernel plus per-request prefill kernel
- **Agent:** claude

**Detailed description.**

In vllm/v1/worker/gpu/mm/rope.py (RopeState.prepare_positions, lines 110-131, and _prepare_rope_positions_kernel, lines 166-214), replace the single 'one Triton program per request with BLOCK_SIZE=1024' launch with a bifurcated launch driven by an is_prefill mask computed once on the host (or in a small preamble kernel) from prefill_lens vs num_computed_tokens.

For decode requests (query_len == 1, which is the steady state on multi-turn agentic workloads dominating median TPOT), launch a single flat 1D kernel over the concatenated decode token range: one program covers a BLOCK of decode tokens (e.g., BLOCK=256), each thread computes `orig_pos = num_computed[req] + 0` and writes `orig_pos + delta` for M-RoPE or `orig_pos` for XD-RoPE across all NUM_DIMS in a single vectorized store. This eliminates the wasted 1023 masked-off lanes per decode request today (query_len=1 with BLOCK_SIZE=1024) and collapses N tiny grid programs into ceil(N_decode/BLOCK) programs, cutting kernel launch/dispatch overhead which is a real fraction of TPOT at small batch sizes typical of agent turns.

For prefill requests, keep a per-request kernel but pick BLOCK_SIZE adaptively (e.g., next_power_of_two(min(max_query_len, 1024)) computed on the host from the max of prefill query_lens) so short chunked-prefill segments do not waste an entire 1024-lane tile. The prefill loads still come from the UVA-backed prefill_positions with the existing stride math.

Dispatch shape: compute two host-side index tensors (decode_batch_ids, prefill_batch_ids) once from `(num_computed >= prefill_lens)` and launch two kernels back-to-back on the same stream. The decode kernel takes a flat query_start_loc slice for decode rows; the prefill kernel takes prefill rows only, so the inner `if is_prefill` branch and the `tl.load(prefill_delta_ptr)` for decode rows (unused, wasted memory traffic) both go away.

Correctness is covered by tests/v1/worker/test_mrope_prompt_embeds.py, tests/v1/worker/test_gpu_model_runner_mm_gather.py, and the multimodal RoPE tests. Benchmark median TPOT on a multi-turn agent trace (e.g., Qwen2-VL-7B M-RoPE, batch=8-32, decode-heavy) via `vllm bench serve` before/after.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete change is novel by construction. The specific idea — splitting the prepare_positions launch by decode-vs-prefill into a fused flat decode kernel (removing the 1023 wasted lanes and per-request grid dimension on the hot path) plus an adaptively-tiled prefill kernel — is more targeted than the generic 'adaptive tiling or batching' phrasing in the candidate's evolve_rationale: it identifies the specific structural inefficiency (per-request grid + fixed 1024 tile + branch on is_prefill inside the tile loop) and the concrete two-kernel replacement, including host-side is_prefill partitioning and elimination of the delta load on the decode path.

---

### 2. Add a no-op fast path when staged RoPE positions are already current
- **Agent:** codex

**Detailed description.**

In `vllm/v1/worker/gpu/mm/rope.py`, make `RopeState.prepare_positions` skip launching `_prepare_rope_positions_kernel` for requests whose prepared position rows are already valid for the current query range. Track a small per-request cache key alongside `mrope_positions`/`mrope_positions_cpu`, such as `(req_id or persistent batch slot generation, query_start_loc, query_len, num_computed_tokens, mrope_delta, is_xdrope)`, and only include rows with a changed key in the kernel grid. This targets multi-turn agent workloads where scheduler iterations often revisit the same active multimodal requests while only a subset advances or where prepare_positions is called for empty/unchanged multimodal state. For unchanged rows, reuse the existing `mrope_positions` contents and avoid both the Triton launch work and the UVA-backed `prefill_positions` reads. The implementation should invalidate the key whenever the request occupying a batch slot changes, when prefill positions are restaged, or when the RoPE mode/delta changes; tests should cover slot reuse and a repeated `prepare_positions` call producing identical output without rewriting stale data.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes restructuring execution by splitting decode and prefill into specialized kernels and adapting tile sizes, which still recomputes every included request each call. This proposal is orthogonal: it adds request/slot-level validity tracking so unchanged RoPE rows are not launched or recomputed at all, including prefill-position UVA reads, and focuses on idempotent repeated calls and slot invalidation rather than decode-vs-prefill kernel specialization.

---
