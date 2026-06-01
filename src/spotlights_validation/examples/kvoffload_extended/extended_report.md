# Extended Validation Plan — Discovery Report

**Target version:** vllm v0.18.0
**Base plan:** `src/spotlights_validation/examples/kvoffload/validation_plan.json`
**Extended plan:** `src/spotlights_validation/examples/kvoffload_extended/validation_plan.json`

The extended plan starts from the base kvoffload artifacts and adds entries
discovered from vllm GitHub issues and PRs. All paths were verified against the
`v0.18.0` git tag; entries whose test files postdate the release were excluded.

---

## Added entries

18 harness entries, 10 workloads, and 18 plan entries were added on top of the
base plan across two discovery passes. `unit-kv-cache-coordinator` was promoted
from "excluded" to a skipped entry in the plan itself (see index 12).

The entries from the **first discovery pass** are grouped below by kind and
priority. See [Second discovery pass](#second-discovery-pass-2026-06-01) for
entries added on 2026-06-01.

---

### Correctness — Priority 2 (halt on failure)

#### index 3 — `gh_cache_pollution_prevention`

**Name:** correctness tests — prefix-cache pollution prevention (invalid block eviction)
**Source:** PR [vllm-project/vllm#26813](https://github.com/vllm-project/vllm/pull/26813)
**Components:** `kv_cache`, `scheduler`
**Invoke:** `pytest -v tests/v1/kv_connector/unit/test_cache_pollution_prevention.py`
**Estimated duration:** 120 s

Validates that when KV blocks fail to sync-load from the offload tier, the
invalid blocks — and all subsequent blocks whose content depended on them —
are evicted from the prefix cache hash table. Without this eviction, a future
request could match and reuse a cached block whose backing data was never
successfully loaded, producing corrupt outputs. Test and feature introduced
in PR #26813 ("KV Load Failure Recovery/Abort Configuration").

> **Note — source_ref corrected (2026-05-26):** this entry was initially
> mis-attributed to issue [#42948](https://github.com/vllm-project/vllm/issues/42948).
> Issue #42948 describes a distinct bug — `_maybe_evict_cached_block` running
> unconditionally during block allocation, destroying single-storage cache keys
> for hybrid MLA+SWA groups even when the pool has free space — for which no
> test exists in v0.18.0. See the Excluded entries table for the gap entry.

**Workload — `gh_wl_deepseek_v4_multi_turn_prefix`** (agentic, source: [#42948](https://github.com/vllm-project/vllm/issues/42948))
DeepSeek-V4-Flash A→B→A multi-turn pattern. Originally attributed to this
entry by the discovery process due to the shared "cache pollution" label;
the workload exercises the #42948 regression (0% hit rate after an
interleaved request destroys the first block's cache key) rather than the
failed-load eviction path this test covers. The workload is retained here
pending creation of a dedicated #42948 test entry.

---

#### index 4 — `gh_invalid_blocks_correctness`

**Name:** correctness tests — invalid block handling (sync recompute, sync fail, async recompute)
**Source:** issue [vllm-project/vllm#42085](https://github.com/vllm-project/vllm/issues/42085)
**Components:** `kv_offload`, `kv_cache`, `scheduler`
**Invoke:** `pytest -v tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py`
**Estimated duration:** 180 s

Covers three failure modes for KV blocks that arrive back from the offload
tier in an invalid state: synchronous recompute (block re-prefilled inline),
synchronous failure (request aborted), and asynchronous recompute (block
re-queued). The linked issue is a `popleft_n` assertion failure triggered when
a second long-context request exhausts the GPU block pool while the first is
still offloading.

---

### Unit — Priority 3 (halt on failure)

#### index 8 — `gh_error_propagation`

**Name:** unit tests — KV connector error propagation
**Source:** issue [vllm-project/vllm#39491](https://github.com/vllm-project/vllm/issues/39491)
**Components:** `kv_offload`, `distributed.kv_transfer`
**Invoke:** `pytest -v tests/v1/kv_connector/unit/test_error_propagation.py`
**Estimated duration:** 120 s

Verifies that errors raised inside a KV connector (e.g. a failed GPU→CPU
transfer) propagate correctly up to the scheduler and result in a clean
request abort rather than a silent hang or an unhandled exception that kills
the worker process.

---

### Correctness — Priority 3 (no halt)

#### index 13 — `gh_reset_prefix_cache_e2e`

**Name:** correctness tests — prefix cache reset end-to-end
**Source:** PR [vllm-project/vllm#41956](https://github.com/vllm-project/vllm/pull/41956)
**Components:** `kv_cache`, `scheduler`
**Invoke:** `pytest -v tests/v1/core/test_reset_prefix_cache_e2e.py`
**Estimated duration:** 600 s

End-to-end test that exercises the prefix-cache reset path: blocks that were
evicted to NVMe or CPU storage must be correctly reloaded and their hashes
re-verified on the next matching request. Without this, a reset can leave
stale hash entries that cause false hits against blocks that no longer exist
on the offload tier.

**Workload — `wl-prefix-heavy`** (batch-inference)
High prefix-reuse batch: repeated requests sharing a long common prefix
over multiple eviction cycles, run against `benchmarks/benchmark_prefix_caching.py`.

---

### Integration — Priority 6 (no halt)

#### index 25 — `gh_nixl_disagg_accuracy`

**Name:** integration tests — NixL disaggregated prefill/decode accuracy
**Source:** issue [vllm-project/vllm#33689](https://github.com/vllm-project/vllm/issues/33689)
**Components:** `kv_offload`, `distributed.kv_transfer`
**Invoke:** `pytest -v tests/v1/kv_connector/nixl_integration/test_disagg_accuracy.py`
**Estimated duration:** 1800 s

Validates output correctness in a PD-disaggregated setup where KV blocks are
transferred from a prefill node to a decode node via NixL. The linked issue
reports output divergence between disaggregated and non-disaggregated mode
under sustained load.

**Workload — `wl-large-kv-decode`** (batch-inference)
Decode under memory pressure: core scenario for disaggregated offload
correctness validation (large KV blocks, multiple concurrent requests).

---

### Benchmarks — Priority 8–9 (skipped, no halt)

Both benchmark entries are currently skipped pending `output_template`
configuration. They will be re-enabled once metric extraction is wired up.

#### index 29 — `gh_benchmark_latency`

**Name:** benchmark — latency (TTFT and TPOT vs. baseline pre-change)
**Source:** issue [vllm-project/vllm#32604](https://github.com/vllm-project/vllm/issues/32604)
**Components:** `scheduler`, `kv_cache`, `worker`
**Invoke:** `python benchmarks/benchmark_latency.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 512 --output-len 128 --num-iters 50`
**Estimated duration:** 1800 s
**Status:** skipped — no output_template defined yet

Regression check for TTFT and TPOT. The linked issue reported a >10% drop in
output token throughput after a KV offload configuration change. A >10% drop
in output token throughput compared to the no-offload baseline is a hard fail.

**Workload — `gh_wl_decode_speed_regression_check`** (batch-inference)
Standard 512-token prompt decode: checks that output token throughput does not
regress by more than 10% vs. the no-offload baseline.

---

#### index 31 — `gh_benchmark_throughput`

**Name:** benchmark — throughput (tokens/s vs. baseline pre-change)
**Source:** issue [vllm-project/vllm#32604](https://github.com/vllm-project/vllm/issues/32604)
**Components:** `scheduler`, `kv_cache`, `worker`
**Invoke:** `python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 1000 --input-len 512 --output-len 128`
**Estimated duration:** 3600 s
**Status:** skipped — no output_template defined yet

Token throughput regression check. Covers two workloads: a standard decode
baseline and an SWA + eager SimpleCPUOffloading scenario derived from
[#42571](https://github.com/vllm-project/vllm/issues/42571) (double-free fix
overhead check).

**Workload — `wl-standard-decode`** (batch-inference)
1000-prompt standard decode: reference throughput baseline.

**Workload — `gh_wl_sliding_window_offload`** (batch-inference, source: [#42571](https://github.com/vllm-project/vllm/issues/42571))
SWA + eager SimpleCPUOffloading: validates that the double-free fix for SWA
evictions does not regress throughput.

---

## Second discovery pass (2026-06-01)

A second discovery pass was run with expanded search terms (`kv_connector offload`,
`SimpleCPUOffload`, `OffloadingConnector`, `swap_blocks kv`, `cpu_kv_cache`).
Maintainer engagement was verified on all source issues via the GitHub comments API.

### New entries added

#### index 33 — `gh_basic_cpu_offload_correctness` (Priority 2, halt on failure)

**Name:** correctness tests — CPU offload output determinism under concurrency
**Source:** issue [vllm-project/vllm#31210](https://github.com/vllm-project/vllm/issues/31210)
**Components:** `kv_offload`, `kv_cache`, `worker`
**Invoke:** `pytest -v tests/basic_correctness/test_cpu_offload.py`
**Estimated duration:** 900 s

Validates that KV blocks restored from CPU memory contain correct data under
high concurrency. Issue #31210 reports wrong/garbled generation output with
OffloadingConnector + chunked prefill under 7 concurrent clients with 96
max_num_seqs. Confirmed by `robertgshaw2-redhat` (maintainer) and `orozery`
(core contributor). Currently excluded from `correctness-basic` due to
`cudaErrorDevicesUnavailable` on exclusive-process GPU hosts — this entry
enables it as a separate, selectively-runnable test.

**Workload — `gh_wl_high_concurrency_correctness`** (batch-inference, source: [#31210](https://github.com/vllm-project/vllm/issues/31210))
96 seqs, 4000-token input, 180-token output, OffloadingConnector with 50000
CPU blocks, `VLLM_BATCH_INVARIANT=1` for determinism. Same prompt must
produce identical output across runs.

---

#### index 34 — `gh_prefetch_offload_correctness` (Priority 3, skipped)

**Name:** correctness tests — prefetch KV offload path
**Source:** issue [vllm-project/vllm#33689](https://github.com/vllm-project/vllm/issues/33689)
**Components:** `kv_offload`, `kv_cache`
**Invoke:** `pytest -v tests/basic_correctness/test_prefetch_offload.py`
**Estimated duration:** 600 s
**Status:** skipped — requires gated `meta-llama/Llama-3.2-1B-Instruct` model access

Exercises the prefetch offload path where KV blocks are speculatively loaded
from CPU before the scheduler needs them. Currently excluded from
`correctness-basic` due to HF token gating.

---

#### index 35 — `gh_cpu_offloading_e2e` (Priority 2, halt on failure)

**Name:** integration tests — CPU offloading end-to-end (full model path)
**Source:** issue [vllm-project/vllm#31210](https://github.com/vllm-project/vllm/issues/31210)
**Components:** `kv_offload`, `kv_cache`, `scheduler`, `worker`
**Invoke:** `pytest -v tests/v1/kv_offload/test_cpu_offloading.py`
**Estimated duration:** 1200 s

End-to-end CPU offloading test that runs a full model forward pass with the
offloading connector active. Deselected from the base `unit-kv-offload-tiering`
entry (which uses `-k 'not test_cpu_offloading'`). Exercises the complete
store→evict→reload cycle including GPU→CPU transfer, CPU block management,
and CPU→GPU reload. Relevant to issues #31210 (wrong generation), #41515
(HMA fails on subsequent request), and PR #36636 (hybrid model fix).

**Workload — `gh_wl_hma_hybrid_model_offload`** (agentic, source: [#41515](https://github.com/vllm-project/vllm/issues/41515))
Hybrid attention model (Qwen3.5-27B style) with HMA + KV offload. Multi-turn
chat where subsequent requests fail. Validates SupportsHMA correctness.

---

#### index 36 — `gh_nixl_edge_cases` (Priority 5, no halt)

**Name:** stress tests — NixL integration edge cases (memory pressure, error recovery)
**Source:** issue [vllm-project/vllm#42085](https://github.com/vllm-project/vllm/issues/42085)
**Components:** `kv_offload`, `distributed.kv_transfer`, `kv_cache`
**Invoke:** `pytest -v tests/v1/kv_connector/nixl_integration/test_edge_cases.py`
**Estimated duration:** 1800 s

Stress tests targeting edge cases in the NixL KV transfer integration: memory
pressure (GPU block pool exhaustion), error recovery (transfer failures
mid-flight), and concurrent access patterns. Relevant to multiple bugs:
#42085 (block pool exhaustion), #42371 (requests stuck), #40259 (CUDA crash).

**Workload — `wl-large-kv-decode`** (batch-inference)
Decode under memory pressure: 128 concurrent requests with 16K KV cache.

**Workload — `gh_wl_dcp_offloading`** (batch-inference, source: [#41549](https://github.com/vllm-project/vllm/pull/41549))
DCP + OffloadingConnector: validates num_blocks accounting with
`decode_context_parallel_size > 1` (TP=4, DCP=2).

---

### Annotations added to existing entries

- **`unit-kv-connector-offloading`** (`test_offloading_connector.py`): added
  `source_refs` for PR [#36636](https://github.com/vllm-project/vllm/pull/36636)
  (HMA support), issue [#41515](https://github.com/vllm-project/vllm/issues/41515)
  (HMA subsequent request failure), issue [#35507](https://github.com/vllm-project/vllm/issues/35507)
  (block_hashes assertion), and PR [#41549](https://github.com/vllm-project/vllm/pull/41549)
  (DCP/PCP support).

### New workload entries

| workload_id | class | source | description |
|---|---|---|---|
| `gh_wl_high_concurrency_correctness` | batch-inference | [#31210](https://github.com/vllm-project/vllm/issues/31210) | 96 seqs, 4K input, OffloadingConnector, VLLM_BATCH_INVARIANT=1 determinism check |
| `gh_wl_hma_hybrid_model_offload` | agentic | [#41515](https://github.com/vllm-project/vllm/issues/41515) | Hybrid model (Qwen3.5-27B) + HMA + multi-turn chat |
| `gh_wl_dcp_offloading` | batch-inference | [#41549](https://github.com/vllm-project/vllm/pull/41549) | DCP (TP=4, DCP=2) + OffloadingConnector + prefix caching |

---

## Base-plan sync (2026-06-01)

The following changes were propagated from the base `kvoffload/validation_plan.json`
to keep the extended plan consistent:

- **Hardcoded paths removed:** all `/proj/...` absolute paths replaced with
  `$ROOT_DIR`-relative references (benchmark invoke, output_template source).
- **`before_run` hooks added:** benchmark entries (indices 21–23) and the NixL
  integration entry (index 24) now run `kill_vllm.sh` before starting to avoid
  port conflicts with stale vLLM processes.
- **LRU/ARC baseline benchmarks added:** two new entries (indices 22–23) run the
  multi-turn benchmark with `--policy lru` and `--policy arc` respectively,
  providing baselines for the evolved-policy comparison.
- **`unit-kv-cache-coordinator` promoted:** moved from the Excluded table to a
  skipped entry (index 12) with an explicit `skip_reason`, matching the base plan.
- **`integration-engine` invoke updated:** added `--ignore` flags for
  `test_abort_final_step.py` and `test_output_processor.py`, plus
  `-k 'not test_skip_tokenizer_initialization'` to exclude tests requiring gated
  HF models. `estimated_duration` increased to 1800 s.
- **NixL integration invoke updated:** switched from `pytest` to
  `bash tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` (starts the
  full server stack); `output_format` changed to `custom`.
- **Stress test invoke updated:** switched from `pytest` to the
  `run_nixl_stress.sh` wrapper script; `output_format` changed to `custom`.
- **Benchmark output_template simplified:** metric paths now point to the
  `collect_summary.py`-generated `summary.json` (flat keys: `ttft_ms_mean`,
  `tpot_ms_mean`, `cpu_hit_rate`, `evictions`) instead of the old nested
  `best_program_info.json` structure.
- **All indices renumbered** to account for the 3 inserted entries (coordinator,
  LRU, ARC). Total entry count: 32.

---

## Excluded entries

8 entries were identified during initial discovery but excluded because their
test paths do not exist in the vllm `v0.18.0` tag. (`unit-kv-cache-coordinator`
was previously in this list but is now included as a skipped entry at index 12.)

| id | missing path | reason |
|---|---|---|
| `unit-simple-kv-offload` | `tests/v1/simple_kv_offload/` | entire directory absent in v0.18.0 |
| `gh_simple_kv_offload_scheduler_regression` | `tests/v1/simple_kv_offload/test_scheduler.py` | same missing directory |
| `gh_offloading_connector_scheduler_unit` | `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` | `offloading_connector/` subdir added after v0.18.0 |
| `gh_offloading_connector_worker_unit` | `tests/v1/kv_connector/unit/offloading_connector/test_worker.py` | same missing subdir |
| `gh_offloading_connector_metrics` | `tests/v1/kv_connector/unit/offloading_connector/test_metrics.py` | same missing subdir |
| `gh_offloading_connector_worker_metadata` | `tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py` | same missing subdir |
| `gh_bidirectional_kv_transfer` | `tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py` | introduced in PR #43097, merged after v0.18.0 |
| `gh_nixl_multi_connector_edge_cases` | `tests/v1/kv_connector/nixl_integration/test_multi_connector_edge_cases.py` | path absent; only `test_multi_connector.py` exists at unit level |
| `gh_hybrid_prefix_cache_eviction` | *(no test file)* | issue [#42948](https://github.com/vllm-project/vllm/issues/42948): `_maybe_evict_cached_block` destroys single-storage cache keys for hybrid MLA+SWA groups unconditionally; no test exists in v0.18.0 — the closest coverage is `test_maybe_evict_cached_block` in `tests/v1/core/test_prefix_caching.py` (PR #21400) which only exercises the basic two-block-same-hash case |

### Version-gated entries (second discovery pass, 2026-06-01)

The following bug fixes and features have relevant test files that were added
**after** the v0.18.0 release. They are documented here for future reference
when upgrading past v0.18.0.

| PR/Issue | Title | Missing path | Notes |
|---|---|---|---|
| [#42959](https://github.com/vllm-project/vllm/pull/42959) | Prevent offloading stale sliding window blocks | `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` | By `orozery` (core contributor). Fixes wrongful KV data offloading when SWA block is freed and re-allocated before store completes. |
| [#41777](https://github.com/vllm-project/vllm/pull/41777) | Flush final KV block on request finish | `tests/v1/simple_kv_offload/test_scheduler.py` | Eager mode silently drops last full KV block when computed in same step as request finish. |
| [#42612](https://github.com/vllm-project/vllm/pull/42612) | Deduplicate blocks in BlockPool.free_blocks() | `tests/v1/core/test_block_pool_double_free.py` | Merged fix for SWA double-free causing `num_free_blocks` over-count. |
| [#43946](https://github.com/vllm-project/vllm/pull/43946) | Eviction-triggered store for OffloadingConnector | `tests/v1/kv_offload/cpu/test_manager.py` | Opt-in lazy store mode (`KV_OFFLOAD_LAZY_STORE=1`); test paths all post-v0.18.0. |
| [#43689](https://github.com/vllm-project/vllm/pull/43689) | SharedOffloadRegion align blocks to page-size | `tests/v1/kv_offload/test_fs_tier.py` | Required for O_DIRECT filesystem offloading. |
| [#41228](https://github.com/vllm-project/vllm/pull/41228) | Scheduler-side sliding window group support | `tests/v1/kv_connector/unit/offloading_connector/` | Part of HMA+offloading series; tests in follow-up PRs. |

---

## Conclusions

### Coverage summary

The extended plan now contains **36 plan entries** (vs. 25 in base), **42 harness
entries** (vs. 24), and **15 workload definitions** (vs. 6). The additions cover:

- **Correctness under concurrency** (#31210, #40259): CPU offload output
  determinism when KV blocks are actively being evicted/restored.
- **HMA + offloading** (#36636, #41515): hybrid attention models where
  OffloadingConnector must implement SupportsHMA correctly.
- **DCP/PCP compatibility** (#41549): decode context parallelism with offloading.
- **Prefetch offload** (#33689): speculative KV block preloading path.
- **NixL edge cases** (#42085, #42371): memory pressure and error recovery in
  disaggregated prefill/decode setups.

### Issue #42948 — no test coverage at v0.18.0

The only test touching `_maybe_evict_cached_block` is
[`tests/v1/core/test_prefix_caching.py:1852`](https://github.com/vllm-project/vllm/blob/v0.18.0/tests/v1/core/test_prefix_caching.py#L1852)
(`test_maybe_evict_cached_block`), added in PR #21400. It covers only the
basic two-block-same-hash case and does not exercise the #42948 scenario
(hybrid MLA+SWA groups, single-storage entries being unconditionally
destroyed during `get_new_blocks` with free blocks available). No commit in
the v0.18.0 repo references issue #42948.

The gap entry `gh_hybrid_prefix_cache_eviction` and the associated workload
`gh_wl_deepseek_v4_multi_turn_prefix` are therefore correctly flagged as
unmet — there is no test to point them at yet.

### Upgrade path

6 version-gated entries (listed in the table above) will become available when
upgrading past v0.18.0. The most impactful are #42959 (stale SWA blocks) and
#42612 (BlockPool double-free), both of which fix data-corruption paths that
are exercisable but untestable at the current pin.
