# Match report — signal-pipeline-otel-4b61266--label-relaxed

- **Window**: `2025-12-02__2026-06-03`
- **Judge model**: `sonnet` (prompt v1)
- **Findings file**: `C:\projects\vs_code\spotlights\runs\signal-pipeline\with-telemetry-4b612664\report\findings.json`
- **Scored at**: 2026-06-09T07:31:50.990340+00:00
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
| Findings | 5 |
| Findings judged (≥1 candidate in view) | 5 |
| **Weighted score** (score `v1`) | **0.230** |
| `same_idea` (strict) | 0 (0%) |
| `same_idea` or `related` (loose) | 2 (40%) |
| Tier-2 yield | 1 (20%) |

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
| cand-0001 (`OffloadingConnector`) | 0 | 8 | 0 | 1 | 7 | 0 |
| cand-0002 (`OffloadingConnectorScheduler.get_num_new_matched_tokens`) | 2 | 8 | 0 | 1 | 9 | 0 |
| cand-0003 (`NgramProposer`) | 0 | 1 | 0 | 0 | 1 | 0 |
| cand-0004 (`Scheduler.schedule (WAITING-queue admission loop)`) | 0 | 8 | 0 | 0 | 8 | 0 |
| cand-0005 (`KVCacheManager.get_computed_blocks (prefix_cache_stats.recor`) | 0 | 5 | 0 | 0 | 5 | 0 |
| **total** | **2** | **30** | **0** | **2** | **30** | **0** |

## Per-finding

### **[hit]** `cand-0001` — `OffloadingConnector` — score **0.40**

- File: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 8

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T2 | `related` | OffloadingConnectorScheduler.get_num_new_matched_tokens: delay loads when blocks already in-flight (perf in same connector) |
| #24498 | T2 | `neighborhood` | — |
| #27577 | T2 | `neighborhood` | — |
| #27942 | T2 | `neighborhood` | — |
| #29870 | T2 | `neighborhood` | — |
| #30419 | T2 | `neighborhood` | — |
| #30761 | T2 | `neighborhood` | — |
| #31916 | T2 | `neighborhood` | — |

### **[hit]** `cand-0002` — `OffloadingConnectorScheduler.get_num_new_matched_tokens` — score **0.60**

- File: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- Tier-1 candidates: 2  ·  Tier-2 candidates: 8

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T1 | `related` | get_num_new_matched_tokens adds early-return paths (None) for in-flight blocks; not the empty-manager skip |
| #24498 | T2 | `neighborhood` | — |
| #27577 | T2 | `neighborhood` | — |
| #27942 | T2 | `neighborhood` | — |
| #29870 | T2 | `neighborhood` | — |
| #30419 | T2 | `neighborhood` | — |
| #30761 | T2 | `neighborhood` | — |
| #31916 | T2 | `neighborhood` | — |
| #32064 | T2 | `neighborhood` | — |
| #36610 | T1 | `neighborhood` | — |

### **[file-only]** `cand-0003` — `NgramProposer` — score **0.05**

- File: `vllm/v1/spec_decode/ngram_proposer.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 1

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #25954 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0004` — `Scheduler.schedule (WAITING-queue admission loop)` — score **0.05**

- File: `vllm/v1/core/sched/scheduler.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 8

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #24322 | T2 | `neighborhood` | — |
| #26813 | T2 | `neighborhood` | — |
| #27170 | T2 | `neighborhood` | — |
| #28284 | T2 | `neighborhood` | — |
| #29821 | T2 | `neighborhood` | — |
| #30145 | T2 | `neighborhood` | — |
| #30166 | T2 | `neighborhood` | — |
| #30522 | T2 | `neighborhood` | — |

### **[file-only]** `cand-0005` — `KVCacheManager.get_computed_blocks (prefix_cache_stats.record call site)` — score **0.05**

- File: `vllm/v1/core/kv_cache_manager.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 5

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #26813 | T2 | `neighborhood` | — |
| #30166 | T2 | `neighborhood` | — |
| #35758 | T2 | `neighborhood` | — |
| #37307 | T2 | `neighborhood` | — |
| #44165 | T2 | `neighborhood` | — |
