# Match report — spotlights-code-only-skip3-4--a690fb5

- **Window**: `2025-12-02__2026-06-03`
- **Judge model**: `sonnet` (prompt v1)
- **Findings file**: `C:\projects\vs_code\spotlights\runs\spotlights\code-only-a690fb5\findings.json`
- **Scored at**: 2026-06-14T17:04:26.550682+00:00
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
| Findings judged (≥1 candidate in view) | 10 |
| **Weighted score** (score `v1`) | **0.445** |
| `same_idea` (strict) | 3 (30%) |
| `same_idea` or `related` (loose) | 6 (60%) |
| Tier-2 yield | 2 (20%) |

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
| cand-0001 (`expand_block_ids`) | 0 | 8 | 1 | 2 | 5 | 0 |
| cand-0002 (`CpuGpuOffloadingHandler.transfer_async`) | 6 | 2 | 1 | 5 | 2 | 0 |
| cand-0003 (`LRUOffloadingManager.prepare_store`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0004 (`ARCOffloadingManager.prepare_store`) | 0 | 1 | 0 | 0 | 0 | 1 |
| cand-0005 (`ARCOffloadingManager.touch`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0006 (`CPUOffloadingSpec.get_manager.eviction_policy_dispatch`) | 0 | 4 | 0 | 1 | 3 | 0 |
| cand-0007 (`OffloadingSpecFactory.register_spec('CPUOffloadingSpec')`) | 0 | 3 | 1 | 1 | 1 | 0 |
| cand-0008 (`LRUOffloadingManager.touch`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0009 (`OffloadingSpec.__init__.offloaded_block_size`) | 0 | 4 | 0 | 2 | 2 | 0 |
| cand-0010 (`CpuGpuOffloadingHandler.get_finished`) | 1 | 7 | 0 | 2 | 5 | 1 |
| **total** | **7** | **32** | **3** | **13** | **21** | **2** |

## Per-finding

### **[hit]** `cand-0001` — `expand_block_ids` — score **0.70**

- File: `vllm/v1/kv_offload/worker/cpu_gpu.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 8

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #37206 | T2 | `same_idea` | Replaces expand_block_ids loop with vectorized: all_ptrs = base_ptr + block_ids[:,None]*row_stride + sub_offsets[None,:] |
| #38453 | T2 | `related` | Calls compute_sub_block_ptrs per group with skip_count; same vectorized helper, refactored callsite |
| #38460 | T2 | `related` | Builds flat pointer arrays vectorized: all_src[start:end] = base_ptrs[t_idx] + src_block_ids * bsz (still calls expand_block_ids) |
| #27942 | T2 | `neighborhood` | — |
| #29870 | T2 | `neighborhood` | — |
| #31916 | T2 | `neighborhood` | — |
| #37853 | T2 | `neighborhood` | — |
| #39182 | T2 | `neighborhood` | — |

### **[hit]** `cand-0002` — `CpuGpuOffloadingHandler.transfer_async` — score **1.00**

- File: `vllm/v1/kv_offload/worker/cpu_gpu.py`
- Tier-1 candidates: 6  ·  Tier-2 candidates: 2

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #38460 | T1 | `same_idea` | cpu_gpu.py transfer_async: replaces per-layer swap_blocks loop with single ops.swap_blocks_batch call |
| #27942 | T1 | `related` | cpu_gpu.py transfer_async: adds start/end events + Transfer dataclass for timing metrics, no batching change |
| #29870 | T1 | `related` | cpu_gpu.py transfer_async: adds _transfer_events dict and wait() for preemption; hot path unchanged |
| #37206 | T1 | `related` | cpu_gpu.py transfer_async: replaces expand_block_ids with vectorized compute_sub_block_ptrs + pinned mmap |
| #37853 | T1 | `related` | cpu_gpu.py transfer_async: handler signature refactor for CanonicalKVCaches; per-layer loop preserved |
| #38453 | T1 | `related` | cpu_gpu.py transfer_async: multi-group support, still uses batched swap_blocks_batch |
| #31916 | T2 | `neighborhood` | — |
| #39182 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0003` — `LRUOffloadingManager.prepare_store` — score **0.05**

- File: `vllm/v1/kv_offload/lru_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0004` — `ARCOffloadingManager.prepare_store` — score **0.00**

- File: `vllm/v1/kv_offload/arc_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `no_match` | — |

### **[file-only]** `cand-0005` — `ARCOffloadingManager.touch` — score **0.05**

- File: `vllm/v1/kv_offload/arc_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |

### **[hit]** `cand-0006` — `CPUOffloadingSpec.get_manager.eviction_policy_dispatch` — score **0.40**

- File: `vllm/v1/kv_offload/cpu.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 4

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #35342 | T2 | `related` | get_manager wraps dispatched manager in FilterReusedOffloadingManager — extends construction, not eviction |
| #24498 | T2 | `neighborhood` | — |
| #31916 | T2 | `neighborhood` | — |
| #36610 | T2 | `neighborhood` | — |

### **[hit]** `cand-0007` — `OffloadingSpecFactory.register_spec('CPUOffloadingSpec')` — score **1.00**

- File: `vllm/v1/kv_offload/factory.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 3

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #40020 | T1 | `same_idea` | register_spec('TieringOffloadingSpec', ...) — new pipeline variant added at the extension site |
| #24498 | T2 | `related` | create_spec gains kv_cache_config arg; spec_cls(config, kv_cache_config) — same factory, different function |
| #36610 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0008` — `LRUOffloadingManager.touch` — score **0.05**

- File: `vllm/v1/kv_offload/lru_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |

### **[hit]** `cand-0009` — `OffloadingSpec.__init__.offloaded_block_size` — score **0.60**

- File: `vllm/v1/kv_offload/spec.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 4

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #24498 | T1 | `related` | Adds kv_cache_config param to OffloadingSpec.__init__; does not touch offloaded_block_size derivation |
| #36610 | T1 | `related` | Replaces offloaded_block_size with hash_block_size + per-group gpu_block_size tuple + block_size_factor |
| #31916 | T2 | `neighborhood` | — |
| #37853 | T2 | `neighborhood` | — |

### **[hit]** `cand-0010` — `CpuGpuOffloadingHandler.get_finished` — score **0.60**

- File: `vllm/v1/kv_offload/worker/cpu_gpu.py`
- Tier-1 candidates: 1  ·  Tier-2 candidates: 7

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #27942 | T2 | `related` | get_finished rewritten to emit TransferResult with timing metrics; still head-of-deque poll |
| #29870 | T1 | `related` | get_finished: del self._transfer_events[job_id]; new wait() method using _transfer_events dict |
| #37206 | T2 | `neighborhood` | — |
| #37853 | T2 | `neighborhood` | — |
| #38453 | T2 | `neighborhood` | — |
| #38460 | T2 | `neighborhood` | — |
| #39182 | T2 | `neighborhood` | — |
| #31916 | T2 | `no_match` | — |
