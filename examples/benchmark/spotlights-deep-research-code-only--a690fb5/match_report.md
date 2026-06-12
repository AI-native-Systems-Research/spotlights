# Match report — spotlights-deep-research-code-only--a690fb5

- **Window**: `2025-12-02__2026-06-03`
- **Judge model**: `sonnet` (prompt v1)
- **Findings file**: `C:\projects\vs_code\spotlights\runs\spotlights\code-only-a690fb5\findings.json`
- **Scored at**: 2026-06-10T15:58:14.268175+00:00
- **Ground truth**: **624 PRs** in the filtered view (out of **5329** merged PRs scraped, 11.7%). Findings are graded against the diffs of these view PRs.

### What we measure

For each finding, we collect candidate PRs in the view whose diff touches the finding's file, classify each as **Tier 1** (diff hunk lands *inside the finding's exact symbol* — function/method/class) or **Tier 2** (same file, different function), then ask the LLM judge to label each (finding, candidate) pair. Verdicts:

- **`same_idea`** — diff makes essentially the same change as the finding (strongest evidence).
- **`related`** — same code touched, related but distinct change.
- **`neighborhood`** — same file, unrelated code (informative noise; the file is a hotspot).
- **`no_match`** — diff has nothing to do with the finding.

All verdict counts are **(finding, candidate-PR) pair** totals summed across findings — one PR can appear under several findings, and a finding can have many candidate PRs.

## Summary

| Metric | Value |
|---|---:|
| Findings | 10 |
| Findings judged (≥1 candidate in view) | 9 |
| **Weighted score** (score `v1`) | **0.330** |
| `same_idea` (strict) | 2 (20%) |
| `same_idea` or `related` (loose) | 5 (50%) |
| Tier-2 yield | 3 (30%) |

**Score weights** — T1 = hunk inside the finding's symbol, T2 = same file. Per-finding score = best (verdict, tier) weight; overall = mean across findings.

| Verdict | T1 weight | T2 weight |
|---|---:|---:|
| `same_idea`     | 1.00 | 0.70 |
| `related`       | 0.60 | 0.40 |
| `neighborhood`  | 0.05 | 0.05 |
| `no_match`      | 0.00 | 0.00 |

### Per-finding verdict breakdown

| Finding | Tier-1 | Tier-2 | `same_idea` | `related` | `neighborhood` | `no_match` |
|---|---:|---:|---:|---:|---:|---:|
| cand-0001 (`expand_block_ids`) | 0 | 8 | 1 | 0 | 7 | 0 |
| cand-0002 (`CpuGpuOffloadingHandler.transfer_async`) | 6 | 2 | 2 | 4 | 1 | 1 |
| cand-0003 (`CpuGpuOffloadingHandler.get_finished`) | 1 | 7 | 0 | 2 | 6 | 0 |
| cand-0004 (`ARCOffloadingManager.prepare_store`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0005 (`ARCOffloadingManager.touch`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0006 (`LRUOffloadingManager.prepare_store`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0007 (`CPUOffloadingSpec.get_manager.eviction_policy_dispatch`) | 0 | 4 | 0 | 1 | 3 | 0 |
| cand-0008 (`CPUBackend.allocate_blocks`) | 0 | 0 | 0 | 0 | 0 | 0 |
| cand-0009 (`OffloadingSpec.offloaded_block_size_default`) | 0 | 4 | 0 | 2 | 2 | 0 |
| cand-0010 (`LRUOffloadingManager.touch`) | 0 | 1 | 0 | 0 | 1 | 0 |
| **total** | **7** | **29** | **3** | **9** | **23** | **1** |

## Per-finding

### **[hit]** `cand-0001` — `expand_block_ids` — score **0.70**

- File: `vllm/v1/kv_offload/worker/cpu_gpu.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 8

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #37206 | T2 | `same_idea` | compute_sub_block_ptrs uses base_ptr + block_ids[:,None]*row_stride + sub_offsets[None,:], replacing the Python loop |
| #27942 | T2 | `neighborhood` | — |
| #29870 | T2 | `neighborhood` | — |
| #31916 | T2 | `neighborhood` | — |
| #37853 | T2 | `neighborhood` | — |
| #38453 | T2 | `neighborhood` | — |
| #38460 | T2 | `neighborhood` | — |
| #39182 | T2 | `neighborhood` | — |

### **[hit]** `cand-0002` — `CpuGpuOffloadingHandler.transfer_async` — score **1.00**

- File: `vllm/v1/kv_offload/worker/cpu_gpu.py`
- Tier-1 candidates: 6  ·  Tier-2 candidates: 2

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #37206 | T1 | `same_idea` | transfer_async swaps expand_block_ids+broadcast for vectorized compute_sub_block_ptrs writing flat src/dst pointer arrays directly |
| #38460 | T1 | `same_idea` | transfer_async replaces per-layer ops.swap_blocks loop with one ops.swap_blocks_batch over flat (src,dst,size) pointer arrays |
| #27942 | T1 | `related` | transfer_async records start/end events and stores Transfer dataclass for per-call timing — orthogonal perf telemetry, not the dispatch reduction |
| #29870 | T1 | `related` | transfer_async adds self._transfer_events[job_id]=event and a new wait() — preemption bug fix, not the per-layer batching |
| #37853 | T1 | `related` | transfer_async loop now zips self.tensor_block_size_in_bytes (renamed); constructor refactored for canonical KV caches, not perf |
| #38453 | T1 | `related` | transfer_async replaces single-loop construction with per-group loop populating all_src/all_dst/all_sizes for swap_blocks_batch |
| #39182 | T2 | `neighborhood` | — |
| #31916 | T2 | `no_match` | — |

### **[hit]** `cand-0003` — `CpuGpuOffloadingHandler.get_finished` — score **0.60**

- File: `vllm/v1/kv_offload/worker/cpu_gpu.py`
- Tier-1 candidates: 1  ·  Tier-2 candidates: 7

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #27942 | T2 | `related` | get_finished rewritten to emit TransferResult with start/end-event timing — metrics, not polling-cost reduction |
| #29870 | T1 | `related` | get_finished: adds `del self._transfer_events[job_id]`; new wait() — preemption handling, not per-stream FIFO |
| #31916 | T2 | `neighborhood` | — |
| #37206 | T2 | `neighborhood` | — |
| #37853 | T2 | `neighborhood` | — |
| #38453 | T2 | `neighborhood` | — |
| #38460 | T2 | `neighborhood` | — |
| #39182 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0004` — `ARCOffloadingManager.prepare_store` — score **0.05**

- File: `vllm/v1/kv_offload/arc_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0005` — `ARCOffloadingManager.touch` — score **0.05**

- File: `vllm/v1/kv_offload/arc_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0006` — `LRUOffloadingManager.prepare_store` — score **0.05**

- File: `vllm/v1/kv_offload/lru_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |

### **[hit]** `cand-0007` — `CPUOffloadingSpec.get_manager.eviction_policy_dispatch` — score **0.40**

- File: `vllm/v1/kv_offload/cpu.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 4

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #35342 | T2 | `related` | cpu.py get_manager: wraps manager in FilterReusedOffloadingManager when store_threshold>=2 |
| #24498 | T2 | `neighborhood` | — |
| #31916 | T2 | `neighborhood` | — |
| #36610 | T2 | `neighborhood` | — |

### **[no candidates]** `cand-0008` — `CPUBackend.allocate_blocks` — score **0.00**

- File: `vllm/v1/kv_offload/backends/cpu.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 0
- Skip reason: `no_file_match`

_No candidate PRs touch this file in the filtered view._

### **[hit]** `cand-0009` — `OffloadingSpec.offloaded_block_size_default` — score **0.40**

- File: `vllm/v1/kv_offload/spec.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 4

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #24498 | T2 | `related` | spec.py __init__: adds kv_cache_config param; doesn't change offloaded_block_size default |
| #36610 | T2 | `related` | spec.py __init__: restructures gpu/offloaded block_size into per-group tuple + block_size_factor |
| #31916 | T2 | `neighborhood` | — |
| #37853 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0010` — `LRUOffloadingManager.touch` — score **0.05**

- File: `vllm/v1/kv_offload/lru_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |
