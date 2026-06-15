# Match report — spotlights-full--a690fb5--rerun

- **Window**: `2025-12-02__2026-06-03`
- **Judge model**: `sonnet` (prompt v1)
- **Findings file**: `C:\projects\vs_code\spotlights\runs\spotlights\full-a690fb5\findings.json`
- **Scored at**: 2026-06-15T11:51:56.058120+00:00
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
| Findings | 9 |
| Findings judged (≥1 candidate in view) | 9 |
| **Weighted score** (score `v1`) | **0.528** |
| `same_idea` (strict) | 4 (44%) |
| `same_idea` or `related` (loose) | 6 (67%) |
| Tier-2 yield | 2 (22%) |

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
| cand-0001 (`ARCOffloadingManager.prepare_store`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0002 (`ARCOffloadingManager.touch`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0003 (`LRUOffloadingManager.prepare_store`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0004 (`expand_block_ids`) | 0 | 8 | 1 | 1 | 6 | 0 |
| cand-0005 (`CpuGpuOffloadingHandler.transfer_async`) | 6 | 2 | 2 | 4 | 2 | 0 |
| cand-0006 (`CpuGpuOffloadingHandler.get_finished`) | 1 | 7 | 0 | 2 | 6 | 0 |
| cand-0007 (`CPUOffloadingSpec.get_manager eviction_policy selector`) | 0 | 4 | 1 | 2 | 1 | 0 |
| cand-0008 (`OffloadingSpecFactory.create_spec/register_spec`) | 0 | 3 | 1 | 2 | 0 | 0 |
| cand-0009 (`OffloadingSpec.__init__ offloaded_block_size`) | 0 | 4 | 0 | 2 | 2 | 0 |
| **total** | **7** | **31** | **5** | **13** | **20** | **0** |

## Per-finding

### **[file-only]** `cand-0001` — `ARCOffloadingManager.prepare_store` — score **0.05**

- File: `vllm/v1/kv_offload/arc_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0002` — `ARCOffloadingManager.touch` — score **0.05**

- File: `vllm/v1/kv_offload/arc_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0003` — `LRUOffloadingManager.prepare_store` — score **0.05**

- File: `vllm/v1/kv_offload/lru_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `neighborhood` | — |

### **[hit]** `cand-0004` — `expand_block_ids` — score **0.70**

- File: `vllm/v1/kv_offload/worker/cpu_gpu.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 8

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #37206 | T2 | `same_idea` | compute_sub_block_ptrs replaces expand_block_ids loop with vectorized block_ids[:,None]*row_stride + sub_offsets[None,:] |
| #38460 | T2 | `related` | transfer_async still calls expand_block_ids; reworks pointer assembly around it but doesn't vectorize the loop |
| #27942 | T2 | `neighborhood` | — |
| #29870 | T2 | `neighborhood` | — |
| #31916 | T2 | `neighborhood` | — |
| #37853 | T2 | `neighborhood` | — |
| #38453 | T2 | `neighborhood` | — |
| #39182 | T2 | `neighborhood` | — |

### **[hit]** `cand-0005` — `CpuGpuOffloadingHandler.transfer_async` — score **1.00**

- File: `vllm/v1/kv_offload/worker/cpu_gpu.py`
- Tier-1 candidates: 6  ·  Tier-2 candidates: 2

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #38453 | T1 | `same_idea` | transfer_async builds flat all_src/all_dst/all_sizes across groups+tensors and dispatches via swap_blocks_batch |
| #38460 | T1 | `same_idea` | replaces per-layer ops.swap_blocks loop with single ops.swap_blocks_batch over flat pointer arrays |
| #27942 | T1 | `related` | transfer_async records start_event/end_event for timing metrics; perf hot path unchanged |
| #29870 | T1 | `related` | transfer_async records job_id->event in _transfer_events; adds wait(); not a perf change |
| #37206 | T1 | `related` | replaces expand_block_ids with vectorized compute_sub_block_ptrs and adds shared mmap pinned region |
| #37853 | T1 | `related` | refactors constructor for hybrid CanonicalKVCaches; transfer_async loop body unchanged |
| #31916 | T2 | `neighborhood` | — |
| #39182 | T2 | `neighborhood` | — |

### **[hit]** `cand-0006` — `CpuGpuOffloadingHandler.get_finished` — score **0.60**

- File: `vllm/v1/kv_offload/worker/cpu_gpu.py`
- Tier-1 candidates: 1  ·  Tier-2 candidates: 7

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #27942 | T2 | `related` | get_finished rewritten with Transfer dataclass and start/end timing events |
| #29870 | T1 | `related` | get_finished: del self._transfer_events[job_id]; new wait() syncs events by job_id |
| #31916 | T2 | `neighborhood` | — |
| #37206 | T2 | `neighborhood` | — |
| #37853 | T2 | `neighborhood` | — |
| #38453 | T2 | `neighborhood` | — |
| #38460 | T2 | `neighborhood` | — |
| #39182 | T2 | `neighborhood` | — |

### **[hit]** `cand-0007` — `CPUOffloadingSpec.get_manager eviction_policy selector` — score **1.00**

- File: `vllm/v1/kv_offload/cpu.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 4

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #35342 | T1 | `same_idea` | get_manager: wraps selected manager with FilterReusedOffloadingManager admission policy |
| #24498 | T2 | `related` | get_manager: replaces num_cpu_blocks with derived num_blocks from cpu_bytes_to_use |
| #36610 | T2 | `related` | get_manager: switches to per-group gpu_block_size and block_size_factor for backend sizing |
| #31916 | T2 | `neighborhood` | — |

### **[hit]** `cand-0008` — `OffloadingSpecFactory.create_spec/register_spec` — score **0.70**

- File: `vllm/v1/kv_offload/factory.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 3

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #40020 | T2 | `same_idea` | register_spec('TieringOffloadingSpec', 'vllm.v1.kv_offload.tiering.spec', ...) — tiered spec via the factory |
| #24498 | T2 | `related` | create_spec gains kv_cache_config param; passes it to spec_cls(config, kv_cache_config) |
| #36610 | T2 | `related` | create_spec tightens kv_cache_config from KVCacheConfig\|None to required KVCacheConfig |

### **[hit]** `cand-0009` — `OffloadingSpec.__init__ offloaded_block_size` — score **0.60**

- File: `vllm/v1/kv_offload/spec.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 4

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #24498 | T1 | `related` | OffloadingSpec.__init__ adds kv_cache_config param; doesn't alter offloaded_block_size default/assert |
| #36610 | T1 | `related` | Restructures __init__ block-size logic into hash/gpu(per-group)/offloaded with block_size_factor; still static default |
| #31916 | T2 | `neighborhood` | — |
| #37853 | T2 | `neighborhood` | — |
