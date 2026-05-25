# _get_tile_size

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/ops/triton_unified_attention.py`](vllm/v1/attention/ops/triton_unified_attention.py) (lines 477–502)
- **Symbol:** `_get_tile_size`
- **Kind:** function
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0002`

## Description
Selects the Triton TILE_SIZE used by kernel_unified_attention for prefill and decode, including the Gemma3 signature helper it depends on.

## Current approach
_is_gemma3_attention special-cases sliding_window == 1024 with head_size in {128, 256}; _get_tile_size returns 32 for that case, 32 for prefill, 32 for fp8 element_size < 2, and 16 otherwise. It ignores num_queries_per_kv, number of heads, sequence shape, GPU, and cache layout.

## Estimated impact explanation
TRITON_ATTN decode is sensitive to tile size, and the current selector covers only dtype, prefill/decode, and one model signature. Better per-(head_size, num_kv_heads, GPU) tile choices can reduce median TPOT on decode-heavy agentic batches.

## Evolve rationale
TILE_SIZE controls loop trip count, memory locality, register pressure, and occupancy in the unified Triton attention kernel. This is a compact closed-form heuristic that can be replaced by a per-shape or per-device tuning table. Correctness oracle: tests/kernels/attention/test_triton_unified_attention.py and tests/v1/attention/test_attention_backends.py validate outputs; tile size should only affect performance within numerical tolerance.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Make _get_tile_size GQA-aware via a register-budget formula
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/ops/triton_unified_attention.py:477-502, extend _get_tile_size to take num_queries_per_kv (already available at the call site in kernel_unified_attention) and pick TILE_SIZE so that the inner-loop register footprint TILE_SIZE * num_queries_per_kv * head_size stays inside a per-dtype budget. Concretely: keep the existing Gemma3 special case, but otherwise compute TILE_SIZE = clamp(prev_pow2(REG_BUDGET // (num_queries_per_kv * head_size)), 16, 64), with REG_BUDGET ≈ 8192 elements for bf16/fp16 prefill, 4096 for bf16/fp16 decode, and 16384 for fp8 (element_size < 2). This collapses the four current branches into one closed-form rule but actually responds to GQA group size, which dominates register pressure on modern decode-heavy models (Llama-3 70B ratio 8, Qwen2/Mistral ratio 4-8) — the regime that matters most for median TPOT on agentic workloads. Add a tiny module-level dict for explicit overrides (e.g., (head_size=128, num_queries_per_kv=8, dtype=bf16, decode) -> 16) so empirically-tuned points can pin values without reverting the whole heuristic. Validate with tests/kernels/attention/test_triton_unified_attention.py and tests/v1/attention/test_attention_backends.py for correctness, then microbench decode tiles for the GQA shapes above to confirm TPOT improvement.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, and the candidate's current_approach explicitly notes that num_queries_per_kv is one of the inputs ignored by the present selector. This proposal adds GQA-group-size as a first-class input via a register-budget closed form plus a small override table — a single targeted change distinct from a generic autotuner sweep or a per-GPU lookup table.

---

### 2. Use the decode tile size for 2D decode launches
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/ops/triton_unified_attention.py`, decouple the `TILE_SIZE` choice from the 2D-vs-3D launch decision. Today `unified_attention` computes both `TILE_SIZE_PREFILL` and `TILE_SIZE_DECODE`, but then every non-3D launch uses `TILE_SIZE_PREFILL`; decode-only batches fall back to the 2D kernel whenever `num_seqs > seq_threshold_3D`, the 3D buffers are unavailable, or batch invariance is enabled, so bf16/fp16 decode in those common large-batch cases silently gets `TILE_SIZE=32` instead of `_get_tile_size(..., is_prefill=False)` returning 16. Change the selection to be based on the actual query shape, e.g. `is_prefill = max_seqlen_q > 1`, and pass the resulting `tile_size` to the main kernel; keep passing that same decode tile size into `reduce_segments` when `use_3d` is true. This preserves the current Gemma3 and fp8 behavior, but makes the existing decode heuristic apply to both 2D and 3D decode paths. Validate with the existing unified attention correctness tests, and microbench a decode-only batch above `seq_threshold_3D` because that is the path where median TPOT should move.

**Novelty rationale.**

There are no deep_research_proposals to avoid. Agent A proposes adding `num_queries_per_kv` to `_get_tile_size` and deriving a GQA/register-budget tile formula; this proposal does not change the tile formula at all. It fixes the call-site policy so the current decode tile choice is actually used for 2D decode fallbacks, a distinct issue caused by coupling tile selection to `use_3d` rather than prefill/decode shape.

---
