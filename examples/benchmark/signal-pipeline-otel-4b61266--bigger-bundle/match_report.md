# Match report — signal-pipeline-otel-4b61266--bigger-bundle

- **Window**: `2025-12-02__2026-06-03`
- **Judge model**: `sonnet` (prompt v1)
- **Findings file**: `C:\projects\vs_code\spotlights\runs\signal-pipeline\with-telemetry-4b612664-bigger\report\findings.json`
- **Scored at**: 2026-06-15T12:28:38.740947+00:00
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
| **Weighted score** (score `v1`) | **0.540** |
| `same_idea` (strict) | 1 (20%) |
| `same_idea` or `related` (loose) | 5 (100%) |
| Tier-2 yield | 3 (60%) |

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
| cand-0001 (`CPUOffloadingSpec`) | 0 | 4 | 1 | 1 | 2 | 0 |
| cand-0002 (`OffloadingConnectorScheduler.get_num_new_matched_tokens`) | 2 | 8 | 0 | 2 | 8 | 0 |
| cand-0003 (`PrefixCacheStats.record`) | 0 | 5 | 0 | 1 | 4 | 0 |
| cand-0004 (`OutputProcessor.do_tracing`) | 1 | 6 | 0 | 1 | 6 | 0 |
| cand-0005 (`GPUModelRunner._model_forward`) | 0 | 8 | 0 | 1 | 7 | 0 |
| **total** | **3** | **31** | **1** | **6** | **27** | **0** |

## Per-finding

### **[hit]** `cand-0001` — `CPUOffloadingSpec` — score **0.70**

- File: `vllm/v1/kv_offload/cpu.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 4

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #35342 | T2 | `same_idea` | FilterReusedOffloadingManager wraps manager when store_threshold>=2 to suppress unused GPU→CPU stores |
| #24498 | T2 | `related` | self.num_blocks = int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block — resizes pool, doesn't gate instantiation |
| #31916 | T2 | `neighborhood` | — |
| #36610 | T2 | `neighborhood` | — |

### **[hit]** `cand-0002` — `OffloadingConnectorScheduler.get_num_new_matched_tokens` — score **0.60**

- File: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- Tier-1 candidates: 2  ·  Tier-2 candidates: 8

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #29087 | T1 | `related` | get_num_new_matched_tokens: returns (None, False) to delay reqs whose hit-blocks are already being loaded |
| #36610 | T1 | `related` | OffloadingConnectorScheduler.__init__: derives offloaded_block_size from per-group gpu_block_size & factor |
| #24498 | T2 | `neighborhood` | — |
| #27577 | T2 | `neighborhood` | — |
| #27942 | T2 | `neighborhood` | — |
| #29870 | T2 | `neighborhood` | — |
| #30419 | T2 | `neighborhood` | — |
| #30761 | T2 | `neighborhood` | — |
| #31916 | T2 | `neighborhood` | — |
| #32064 | T2 | `neighborhood` | — |

### **[hit]** `cand-0003` — `PrefixCacheStats.record` — score **0.40**

- File: `vllm/v1/metrics/stats.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 5

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #33290 | T2 | `related` | Adds PromptTokenStats with per-source breakdown (local_compute / local_cache_hit / external_kv_transfer) — alternative metric, not a fix to PrefixCacheStats.record |
| #34510 | T2 | `neighborhood` | — |
| #37160 | T2 | `neighborhood` | — |
| #37460 | T2 | `neighborhood` | — |
| #38709 | T2 | `neighborhood` | — |

### **[hit]** `cand-0004` — `OutputProcessor.do_tracing` — score **0.60**

- File: `vllm/v1/engine/output_processor.py`
- Tier-1 candidates: 1  ·  Tier-2 candidates: 6

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #27987 | T1 | `related` | do_tracing: span.set_attribute(GEN_AI_REQUEST_ID, req_state.external_req_id) — touches span attrs, not code.* |
| #28284 | T2 | `neighborhood` | — |
| #32056 | T2 | `neighborhood` | — |
| #32975 | T2 | `neighborhood` | — |
| #37460 | T2 | `neighborhood` | — |
| #39568 | T2 | `neighborhood` | — |
| #39917 | T2 | `neighborhood` | — |

### **[hit]** `cand-0005` — `GPUModelRunner._model_forward` — score **0.40**

- File: `vllm/v1/worker/gpu_model_runner.py`
- Tier-1 candidates: 0  ·  Tier-2 candidates: 8

| PR | Tier | Verdict | Citation |
|---:|:---:|---|---|
| #24322 | T2 | `related` | Adds DraftModelProposer routing in __init__/propose_draft_token_ids — enables draft-model speculative decoding the finding cites as a mitigation |
| #20859 | T2 | `neighborhood` | — |
| #25954 | T2 | `neighborhood` | — |
| #27532 | T2 | `neighborhood` | — |
| #28284 | T2 | `neighborhood` | — |
| #29821 | T2 | `neighborhood` | — |
| #29870 | T2 | `neighborhood` | — |
| #30145 | T2 | `neighborhood` | — |
