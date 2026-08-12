# GPUModelRunner._determine_batch_execution_and_padding

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py) (lines 4025–4137)
- **Symbol:** `GPUModelRunner._determine_batch_execution_and_padding`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0009`

## Description
Selects legacy cudagraph mode, batch descriptor, DP microbatching, and DP padding each step.

## Current approach
Dispatches cudagraph selection, optionally coordinates across DP ranks, then re-dispatches after synced token counts; DP mode reads num_tokens_across_dp[dp_rank].item() when padding is active.

## Estimated impact explanation
DP-enabled deployments pay this per step; removing redundant dispatch and D2H scalar reads improves median TPOT, especially when data-parallel agent traffic keeps all ranks active.

## Evolve rationale
The dispatch key and DP padding result are policy decisions that can be memoized or represented without host syncs. Correctness is covered by tests/v1/cudagraph/, tests/v1/worker/test_gpu_batch_ordering.py, and tests/v1/worker/test_gpu_warmup_blocks.py.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Eliminate per-step D2H sync and redundant dispatch by using pinned host buffer for DP token counts and memoizing dispatch keys
- **Agent:** claude

**Detailed description.**

In `GPUModelRunner._determine_batch_execution_and_padding` (vllm/v1/worker/gpu_model_runner.py:4025-4137), make two targeted changes that reduce per-step latency on DP-enabled agentic workloads:

1. Replace the `.item()` D2H scalar read at line 4112 (`num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())`) with a pinned-host readback. Coordinate the DP all-reduce inside `coordinate_batch_across_dp` to write its result into a persistent pinned-host (CPU) tensor (via `torch.empty(..., pin_memory=True)` allocated once on the runner and reused). This lets the local rank's slot be indexed by an ordinary Python int without forcing a full device sync on the compute stream. On many NCCL configurations the reduction can be issued on a dedicated comm stream and consumed on host after a lightweight event-wait, avoiding the compute-stream stall that `.item()` currently induces at every step. Because `dp_rank` is a fixed integer for the process, only that one slot needs to be read; keep `num_tokens_across_dp` as a device tensor for downstream consumers but read the scalar from the pinned buffer.

2. Memoize the two `dispatch_cudagraph(...)` calls with an LRU cache keyed by the tuple `(num_tokens_padded, has_lora, uniform_decode, num_active_loras, disable_full, valid_modes_frozen, invalid_modes_frozen, force_eager)`. The dispatcher's decision is a pure function of these inputs and the (immutable-after-warmup) captured cudagraph plan, so a small `functools.lru_cache`-style dict on the runner (e.g., `self._cudagraph_dispatch_cache: dict[tuple, tuple[CUDAGraphMode, BatchDescriptor]]`) removes redundant work in both the pre-DP dispatch and the post-DP re-dispatch. The cache is bounded by the finite set of captured `(num_tokens, uniform_decode, has_lora, num_active_loras)` combinations — well under 1k entries in practice. Invalidate on cudagraph re-capture / LoRA registration events by clearing the dict where those already happen.

Additionally, when `data_parallel_size > 1` but `num_tokens_across_dp` returns `None` (all-agree fast path), skip re-dispatch entirely (the current code already does this via the outer `if`, but ensure the pre-DP dispatch result is not recomputed).

Validation: tests/v1/cudagraph/, tests/v1/worker/test_gpu_batch_ordering.py, tests/v1/worker/test_gpu_warmup_blocks.py cover correctness. Measure median TPOT with a multi-turn agentic workload on DP=2/4 (e.g., vllm bench serve with multi-turn dataset) before/after; expect improvement most visible when decode-only steps dominate and per-step dispatch overhead is a larger fraction of TPOT.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so nothing overlaps. The proposal is specifically actionable at the two identified hot spots (the `.item()` D2H sync at line 4112 and the double `dispatch_cudagraph` calls at 4080/4114), and pairs them into a single coherent change: pinned-host readback to remove the compute-stream stall plus a bounded memoization dict keyed on the exact dispatcher inputs — both targeted at per-step median TPOT on the multi-turn agentic workload named in the caller context.

---

### 2. Drop CG padding when DP synchronization downgrades the step to eager
- **Agent:** codex

**Detailed description.**

In `GPUModelRunner._determine_batch_execution_and_padding` (`vllm/v1/worker/gpu_model_runner.py:4025-4137`), add a fast path for the case where `coordinate_batch_across_dp` returns `synced_cudagraph_mode == CUDAGraphMode.NONE.value` and `should_ubatch` is false. Today `dp_utils._post_process_dp_padding` returns each rank's already-padded token count even when DP padding is not required, so a rank that initially selected a CUDA graph but is downgraded because another rank needs eager mode re-dispatches eager using its CG-padded token count. That preserves correctness but can execute unnecessary padding tokens on the downgraded ranks.

Change the runner-side policy so this downgraded-eager/no-ubatch case re-dispatches `CUDAGraphMode.NONE` with the original local `num_tokens` after sequence-parallel padding, not the earlier CUDA-graph padded `num_tokens_padded`; also set `num_tokens_across_dp` to `None` or otherwise avoid passing a DP-padding vector downstream for this case. The existing DP coordination still synchronizes the eager downgrade across ranks, but it stops carrying over graph-shape padding once graph execution has been disabled for the step. Validate with a DP test where one rank's batch is graphable and another's is forced eager, asserting the final descriptor on the graphable rank uses the minimal eager token count rather than the captured graph size, plus the existing `tests/v1/cudagraph/` and `tests/v1/worker/test_gpu_batch_ordering.py`.

**Novelty rationale.**

There are no deep_research proposals to overlap with. Agent A's proposal targets removing a per-step `.item()` sync with a pinned host buffer and memoizing `dispatch_cudagraph` decisions; this proposal instead changes the DP downgrade semantics so eager fallback does not inherit CUDA-graph padding at all. It is a distinct policy optimization that reduces actual padded compute in mixed-rank DP steps rather than reducing host dispatch overhead.

---
