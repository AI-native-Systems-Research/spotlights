# Extended Validation Plan — Discovery Report

**Target version:** vllm v0.18.0
**Base plan:** `artifacts/kvoffload/validation_plan.json`
**Extended plan:** `artifacts/kvoffload_extended/validation_plan.json`

The extended plan starts from the base kvoffload artifacts and adds entries
discovered from vllm GitHub issues and PRs. All paths were verified against the
`v0.18.0` git tag; entries whose test files postdate the release were excluded.

---

## Added entries

7 entries were added on top of the base plan. They are grouped below by kind and
priority.

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

#### index 12 — `gh_reset_prefix_cache_e2e`

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

#### index 22 — `gh_nixl_disagg_accuracy`

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

#### index 26 — `gh_benchmark_latency`

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

#### index 28 — `gh_benchmark_throughput`

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

## Excluded entries

9 entries were identified during initial discovery but excluded because their
test paths do not exist in the vllm `v0.18.0` tag.

| id | missing path | reason |
|---|---|---|
| `unit-simple-kv-offload` | `tests/v1/simple_kv_offload/` | entire directory absent in v0.18.0 |
| `gh_simple_kv_offload_scheduler_regression` | `tests/v1/simple_kv_offload/test_scheduler.py` | same missing directory |
| `gh_offloading_connector_scheduler_unit` | `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` | `offloading_connector/` subdir added after v0.18.0 |
| `gh_offloading_connector_worker_unit` | `tests/v1/kv_connector/unit/offloading_connector/test_worker.py` | same missing subdir |
| `gh_offloading_connector_metrics` | `tests/v1/kv_connector/unit/offloading_connector/test_metrics.py` | same missing subdir |
| `gh_offloading_connector_worker_metadata` | `tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py` | same missing subdir |
| `gh_bidirectional_kv_transfer` | `tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py` | introduced in PR #43097, merged after v0.18.0 |
| `unit-kv-cache-coordinator` | `tests/v1/core/test_kv_cache_coordinator.py` | test file absent (source exists, no test in v0.18.0) |
| `gh_nixl_multi_connector_edge_cases` | `tests/v1/kv_connector/nixl_integration/test_multi_connector_edge_cases.py` | path absent; only `test_multi_connector.py` exists at unit level |
| `gh_hybrid_prefix_cache_eviction` | *(no test file)* | issue [#42948](https://github.com/vllm-project/vllm/issues/42948): `_maybe_evict_cached_block` destroys single-storage cache keys for hybrid MLA+SWA groups unconditionally; no test exists in v0.18.0 — the closest coverage is `test_maybe_evict_cached_block` in `tests/v1/core/test_prefix_caching.py` (PR #21400) which only exercises the basic two-block-same-hash case |

---

## Conclusions

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
