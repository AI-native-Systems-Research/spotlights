# NUM_SMS/THRESHOLD_BYTES/MIN_N

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/swap_blocks_triton.py`](vllm/v1/kv_offload/cpu/swap_blocks_triton.py) (lines 12–21)
- **Symbol:** `NUM_SMS/THRESHOLD_BYTES/MIN_N`
- **Kind:** config_block
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0005`

## Description
Triton swap-block tuning constants define SM count, descriptor-size crossover, and minimum descriptor batch size.

## Current approach
Fixed H100 PCIe Gen5 constants: NUM_SMS = 12, THRESHOLD_BYTES = 28 * 1024, and MIN_N = 16, applied uniformly across devices, page sizes, and workloads.

## Estimated impact explanation
Non-H100 devices and different host links may have different DMA/Triton crossovers. Tuning these values can improve CPU-to-GPU promotion latency, though the effect is limited to the Triton branch selected by _select_swap_blocks_fn.

## Evolve rationale
These constants directly control launch geometry and DMA fallback for the owned Triton transfer path. Correctness oracle: transfer byte equality is unchanged; only path choice and timing vary, validated by swap-block tests and microbenchmarks across page sizes and batch counts.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Online one-shot autotune of NUM_SMS/THRESHOLD_BYTES/MIN_N at handler init, with soft per-call MIN_N
- **Agent:** claude

**Detailed description.**

Replace the H100-baked constants at vllm/v1/kv_offload/cpu/swap_blocks_triton.py:19-21 with a `SwapTuning` dataclass populated by a short (<20 ms) autotune that runs once during CPU->GPU handler construction in gpu_worker.py (around the existing call to `_select_swap_blocks_fn` at line 218). The autotune uses the KV cache page_size_bytes actually present in `kv_cache_groups_data_refs`, allocates a pinned scratch descriptor set via the existing `_new_descriptor_buffers` helper, and times both `ops.swap_blocks_batch` (C++ cuMemcpyBatchAsync) and the Triton `_swap_blocks_kernel` across n in {4, 16, 64, 256} and NUM_SMS candidates in {4, 8, 12, 16, min(device_sm_count, 32)} on a dedicated stream with CUDA events (same timing pattern already used for `Transfer.start_event`/`end_event`). The best (n, num_sms) points fit a piecewise crossover, producing per-handler `autotuned_num_sms`, `autotuned_threshold_bytes`, `autotuned_min_n`, and a `autotuned_bytes_floor = crossover_n * page_size`. Cache the result on disk keyed by `(torch.cuda.get_device_properties(0).uuid, page_size_bytes, torch.__version__, triton.__version__)` under XDG_CACHE_HOME so subsequent process starts skip the sweep. Then change `_select_swap_blocks_fn` in gpu_worker.py:35-58 to return a closure that decides per call: `use_triton = (n >= autotuned_min_n) and (n * max_page >= autotuned_bytes_floor)`; falling below either bound routes to `ops.swap_blocks_batch`. This turns the current hard `n < MIN_N` cliff at swap_blocks_triton.py:60 into a payload-aware predicate, which matters for multi-turn agentic prompts where the CPU->GPU promotion at turn start frequently has n between 8 and 20 blocks and currently always takes the DMA path even when Triton would win at the deployed link. The `swap_blocks_batch(...)` signature stays the same; `bytes_per_chunk` continues to be resolved from `triton.next_power_of_2(max(page_sizes))`. Correctness is unchanged (still byte-equal transfers via one of two existing implementations); only path selection and launch geometry vary, and the existing swap-block tests suffice as the oracle.

**Novelty rationale.**

No deep_research_proposals are listed on this candidate, so no direct overlap is possible. The natural deep-research direction here would be a static device->constants lookup table (add rows for H100 SXM, B200, GH200, MI300X, etc.). This proposal is materially different: it measures the crossover on the actual deployed hardware/host-link/page-size combination at process start rather than shipping SKU-specific numbers, and it converts MIN_N from a hard threshold into a per-call payload-aware predicate. Those two elements (runtime auto-tune keyed on device UUID + page size, and a soft MIN_N evaluated per invocation using bytes_floor = n * max_page) are orthogonal to any table-lookup approach and specifically address the small-n regime that dominates multi-turn agentic TTFT at turn boundaries.

---

### 2. Add runtime overrides for swap-block Triton tuning constants
- **Agent:** codex

**Detailed description.**

Replace the module-level literals in `vllm/v1/kv_offload/cpu/swap_blocks_triton.py` for `NUM_SMS`, `THRESHOLD_BYTES`, and `MIN_N` with values read from explicit environment/config overrides, falling back to the current H100-tuned defaults. For example, expose `VLLM_KV_OFFLOAD_TRITON_NUM_SMS`, `VLLM_KV_OFFLOAD_TRITON_THRESHOLD_BYTES`, and `VLLM_KV_OFFLOAD_TRITON_MIN_N`, validate them at import or handler initialization, and log the resolved values once. This lets production deployments tune the Triton-vs-DMA crossover for their GPU, PCIe/NVLink topology, page size, and multi-turn agentic traffic without carrying a local patch or paying any startup benchmarking cost. The change is behavior-preserving by default and can be validated with existing byte-equality swap-block tests plus a small benchmark sweep comparing median TTFT/TPOT under default values versus deployment-provided overrides.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposed an online one-shot autotuner with per-handler cached measurements and a soft per-call `MIN_N` predicate. This proposal is different: it does not benchmark, cache, or dynamically infer thresholds. It adds a deterministic operator-controlled override surface around the existing constants so deployments can apply known-good values from offline benchmarking or fleet-specific rollout experiments while retaining the current defaults.

---
