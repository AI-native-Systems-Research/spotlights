# _select_swap_blocks_fn

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/gpu_worker.py`](vllm/v1/kv_offload/cpu/gpu_worker.py) (lines 35–58)
- **Symbol:** `_select_swap_blocks_fn`
- **Kind:** function
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_kv_offload-0004`

## Description
Selects the CPU/GPU block-transfer implementation: C++ DMA path or Triton batched-copy kernel.

## Current approach
Hard-coded dispatch: GPU-to-CPU always uses DMA; CPU-to-GPU uses Triton only when Triton is available, the platform is not XPU/ROCm, all page sizes are 8-byte-aligned, and max page size is below THRESHOLD_BYTES. Triton chunk size is min(next_power_of_2(max_page_size), 8192).

## Estimated impact explanation
CPU-to-GPU promotion runs on cache hits from the CPU tier. Better crossover selection can raise transfer bandwidth and reduce the promotion stall on the TTFT critical path for cache-hit-heavy agentic turns.

## Evolve rationale
The direction gate, platform gate, page-size threshold, alignment rule, and 8192-byte chunk cap are measurable dispatch heuristics. Correctness oracle: byte-identical output between DMA and Triton for the same descriptor arrays plus existing swap-block tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Make CPU->GPU dispatch and Triton grid size batch-size aware at call time
- **Agent:** claude

**Detailed description.**

Change `_select_swap_blocks_fn` at `vllm/v1/kv_offload/cpu/gpu_worker.py:35-58` from a one-shot init-time selector into a lightweight per-call dispatcher for the CPU->GPU direction (GPU->CPU keeps the DMA fast path as today). The current logic freezes the choice using static page-size/alignment/platform gates; but the Triton implementation in `swap_blocks_triton.py` already has a runtime cliff — it falls back to `ops.swap_blocks_batch` when `n < MIN_N=16`, and its launch uses a hard-coded `NUM_SMS=12` regardless of actual `n`. In multi-turn agentic workloads, CPU->GPU promotion batches are highly bimodal: short suffix continuations produce a handful of descriptors (below MIN_N — pure overhead through the wrapper), while full-history rehydrations produce hundreds (where 12 SMs undershoot occupancy). Concretely: (1) When the static gates pass, return a closure that on each call inspects `src_addrs.numel()` and the aggregate byte volume. Below MIN_N descriptors OR total bytes below a small floor (e.g., 128KB, where PCIe-batched DMA wins on setup cost), route directly to `ops.swap_blocks_batch` and skip the pointer H2D copy the Triton path performs. Above MIN_N and below THRESHOLD_BYTES per descriptor, invoke the Triton kernel with grid size `min(next_pow2(cdiv(n, 4)), sm_count)` derived from `torch.cuda.get_device_properties(0).multi_processor_count` cached once — replacing the fixed 12. (2) Push the `n < MIN_N` fallback out of `swap_blocks_batch` in `swap_blocks_triton.py` and into this dispatcher so the fallback avoids one Python indirection on the hot path. (3) Cache the closure state (device SM count, chosen chunk size) on the handler so per-call overhead is a numel read plus two integer comparisons. Correctness oracle stays byte-identical to DMA; new microbench varies (n, page_size) over {1,4,8,16,32,64,128,256} x {2KB, 8KB, 16KB, 32KB} to fit the crossover.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals (none). The candidate's evolve_rationale enumerates static, init-time dispatch heuristics (direction gate, platform gate, page-size threshold, alignment rule, 8192-byte chunk cap). This proposal introduces an orthogonal runtime dimension not present in that enumeration: dispatching per call based on descriptor count `n` and aggregate bytes, and dynamically sizing the Triton launch grid from the device's SM count instead of the hard-coded 12. It also relocates the existing `MIN_N` fallback out of the Triton wrapper to eliminate a wasted call on small batches, which is a code-motion change the static-gate framing does not cover.

---

### 2. Use a hybrid CPU->GPU path for mixed page-size KV groups
- **Agent:** codex

**Detailed description.**

Change `_select_swap_blocks_fn` in `vllm/v1/kv_offload/cpu/gpu_worker.py:35-58` so heterogeneous KV groups do not fall back wholesale to `ops.swap_blocks_batch` just because one `page_size_bytes` fails the Triton gates. Today `max(page_sizes) >= THRESHOLD_BYTES` or any unaligned page disables Triton for every descriptor in the transfer, which is pessimistic for hybrid/HMA layouts where most descriptors may still be small 8-byte-aligned pages. Compute the eligible page-size set at handler initialization; if all are eligible, keep the current Triton selection, if none are eligible, keep DMA, and if mixed, return a CPU->GPU wrapper that partitions descriptors by `sizes` into eligible contiguous spans and ineligible spans, issuing Triton for eligible spans and DMA for the rest on the same transfer stream. Add a correctness test using mixed page sizes, including one above `THRESHOLD_BYTES` and one below it, verifying byte-identical output to the all-DMA oracle and that eligible descriptors still take the Triton wrapper when Triton/platform gates pass.

**Novelty rationale.**

There are no listed deep_research_proposals. Agent A’s proposal changes dispatch based on runtime descriptor count, aggregate byte volume, and device SM count; it still assumes the static page-size gate decides whether the whole transfer is Triton-eligible. This proposal targets a different blind spot: mixed per-descriptor page-size eligibility, avoiding the current all-or-nothing fallback when only some KV cache references violate the Triton threshold or alignment rule.

---
