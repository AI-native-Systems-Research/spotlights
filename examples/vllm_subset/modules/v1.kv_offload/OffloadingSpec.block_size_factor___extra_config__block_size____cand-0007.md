# OffloadingSpec.block_size_factor / extra_config["block_size"]

[← v1.kv_offload](../v1.kv_offload.md)

- **File:** [`vllm/v1/kv_offload/base.py`](vllm/v1/kv_offload/base.py) (lines 351–367)
- **Symbol:** `OffloadingSpec.block_size_factor / extra_config["block_size"]`
- **Kind:** config_block
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0007`

## Description
Config-defined offloaded-block granularity that maps GPU KV blocks into larger CPU offload blocks through block_size_factor.

## Current approach
OffloadingSpec defaults block_size_factor to 1. If kv_connector_extra_config["block_size"] is provided, it requires a single GPU block size, asserts the offloaded block size is a multiple of the GPU block size, and sets block_size_factor = offloaded_block_size // gpu_block_size. CPUOffloadingSpec uses that factor to compute num_blocks, and CpuGpuOffloadingHandlers/SingleDirectionOffloadingHandler use it to map CPU blocks to GPU sub-blocks during transfers.

## Estimated impact explanation
The impact is medium because it changes a central granularity but is workload dependent: better sizing can reduce transfer-prep overhead and improve prefix-cache residency for multi-turn agentic workloads, moving TTFT on cache-warm turns and TPOT under offload pressure, while poor sizing can waste CPU capacity or CPU<->GPU bandwidth.

## Evolve rationale
This block-size policy directly trades off CPU-cache capacity, offload metadata size, transfer batch size, and partial-block overfetch. Headroom includes workload-aware defaults, adaptive or model-aware block-size selection, and heuristics that choose coarser CPU blocks for stable shared prefixes while avoiding wasted bandwidth for short or sparsely reused prompts. Correctness oracles are the divisibility/alignment assertions, existing offloading connector tests with custom block_size, and GPU-worker round-trip tests across block_size_factor and unaligned group offsets.

## Deep research proposals

### 1. Workload-aware CPU offload block_size_factor tuned for multi-turn agentic prefix reuse
- **Finding:** `find-0003` — *SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends*
- **Source URL:** <https://www.lmsys.org/blog/2025-09-10-sglang-hicache/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend OffloadingSpec in vllm/v1/kv_offload/base.py (lines 351-367) so that block_size_factor is selected with awareness of the agentic, prefix-reuse-heavy workload rather than being a single static multiple of the GPU block size. Concretely: (1) keep the current kv_connector_extra_config["block_size"] override path and its divisibility assertion as the explicit user-facing contract, but (2) when no override is supplied, replace the hard-coded default of 1 with a workload-aware default that picks a larger offloaded block (e.g. a small multiple such as 4-16 GPU blocks) when the deployment signals a multi-turn / shared-prefix workload, and falls back to 1 for short, low-reuse traffic. Surface this choice through a new optional extra_config knob (e.g. "block_size_policy": "static" | "prefix_optimized") and/or a documented heuristic that consumes existing scheduler/connector signals about prefix-cache hit rate. Downstream, CPUOffloadingSpec.get_num_new_blocks and the CpuGpuOffloadingHandlers / SingleDirectionOffloadingHandler transfer logic continue to use block_size_factor unchanged, so the change is localized to how the factor is chosen, not how it is consumed. Following HiCache's design point that the CPU memory layout should be decoupled from the GPU layout and tuned for reuse, the larger CPU block becomes the natural unit for tracking shared multi-turn prefixes, while the GPU block size and existing alignment invariants are preserved.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out workload-aware defaults and heuristics that pick coarser CPU blocks for stable shared prefixes while avoiding waste on short prompts; HiCache provides an external data point that decoupling CPU layout from GPU layout and optimizing reuse metadata (rather than treating the CPU tier as a flat block store) is what unlocks gains on multi-turn agentic workloads, which matches this caller's stated objective of reducing median TPOT and TTFT under multi-turn traffic. The finding does not change the divisibility/alignment contract that the existing assertions enforce, so the correctness oracles listed in the candidate (offloading-connector tests with custom block_size and round-trip tests across block_size_factor) remain directly applicable. The proposal is therefore concrete and transferable: it reuses the existing block_size_factor mechanism but replaces a static default with a workload-aware policy informed by HiCache's hierarchical-cache design.

---

## Agent proposals

### 1. Auto-tune block_size_factor via a startup PCIe transfer micro-benchmark
- **Agent:** claude

**Detailed description.**

Augment OffloadingSpec in vllm/v1/kv_offload/base.py (lines 351-367) with a hardware-aware default for block_size_factor that is chosen by a one-time, lightweight PCIe transfer micro-benchmark at engine startup, rather than the current static default of 1 (or any workload-derived heuristic). Concretely: (1) preserve the existing kv_connector_extra_config["block_size"] override and its divisibility assertion as the user-facing contract; (2) when no override and no explicit policy is given, run a short calibration that performs cudaMemcpyAsync transfers of varying sizes corresponding to candidate offloaded-block sizes (gpu_block_size * k for k in {1, 2, 4, 8, 16, 32}), measures effective H2D and D2H bandwidth and per-transfer launch overhead on this host's PCIe/NVLink topology, and selects the smallest k whose bandwidth is within a configurable margin (e.g. 90%) of the asymptotic peak — i.e. the knee of the bandwidth-vs-transfer-size curve. The chosen k becomes block_size_factor; an extra_config knob (e.g. "block_size_calibration": "auto"|"off"|<int>) lets users disable or pin the result. The downstream consumers — CPUOffloadingSpec.get_num_new_blocks, CpuGpuOffloadingHandlers, SingleDirectionOffloadingHandler — keep using block_size_factor unchanged, and the existing divisibility assertion still enforces correctness. Calibration cost is bounded (a few ms, run once), and results can be cached keyed by (gpu_uuid, gpu_block_size, kv_dtype, num_layers) under VLLM_CACHE_ROOT to amortize across restarts. This grounds the granularity choice in the actual transfer-prep-overhead vs. bandwidth tradeoff that the candidate's evolve_rationale highlights, independently of any prefix-reuse signal.

**Novelty rationale.**

The existing deep_research_proposal (find-0003) selects block_size_factor from workload signals (multi-turn / prefix-cache hit rate) and exposes a "block_size_policy": "static"|"prefix_optimized" knob. This proposal is orthogonal and complementary: it picks the factor from a hardware-side signal — measured PCIe transfer bandwidth and per-transfer launch overhead on the actual deployment host — rather than from workload behavior. It addresses the "transfer batch size" and "transfer-prep overhead" axes of the candidate's evolve_rationale that the workload-aware proposal does not directly target, works even before any traffic has been observed (no warmup window of mispredicted reuse), and would compose with find-0003 (e.g. workload policy can scale up from the hardware-derived floor). The micro-benchmark-driven autotune mechanism, the bandwidth-knee selection rule, and the (gpu_uuid, gpu_block_size, kv_dtype) cache key are not present in the existing proposal.

---

### 2. Add fine-grained tail offload for coarse CPU blocks
- **Agent:** codex

**Detailed description.**

Extend the `OffloadingSpec.block_size_factor` policy in `vllm/v1/kv_offload/base.py` around lines 351-367 with an opt-in extra_config flag such as `store_partial_tail_blocks`. Keep the configured coarse `block_size_factor` for stable prefix blocks, but allow the scheduler to store the most recent completed GPU/hash blocks even when they do not yet fill a whole offloaded CPU block. Concretely, for `block_size_factor > 1`, store a tail entry keyed by the last completed sub-block hash for the group, transfer only the valid GPU sub-blocks using the existing `GPULoadStoreSpec.group_sizes` and `block_indices` machinery, and let lookup/load return hits at GPU-block granularity for that tail. When the tail later grows into a full coarse offloaded block, promote or replace the partial entry with the normal coarse key to avoid duplicate residency. Keep this limited to full GPU/hash blocks, not token-partial blocks, so the current divisibility and alignment invariants remain intact. Add scheduler and worker tests with `block_size_factor=4` where 1-3 completed GPU blocks at the end of a multi-turn request are reusable on the next turn, plus a promotion test for the partial-to-full transition.

**Novelty rationale.**

The deep_research_proposal changes how a single static `block_size_factor` is chosen from workload signals; agent A changes how that same single factor is chosen from a startup transfer benchmark. This proposal does not choose a different factor. It changes the all-or-nothing full-offloaded-block policy by adding a fine-grained tail tier that composes with either selection strategy, specifically addressing short multi-turn appendages that would otherwise miss CPU reuse until enough tokens accumulate to fill the coarse block.

---
