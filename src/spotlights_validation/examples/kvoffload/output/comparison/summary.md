# Validation Results Comparison

## Regressions (passed in baseline, failed in evolved)

None

## Fixes (failed in baseline, passed in evolved)

None

## Tests Comparison

| run | id | status | invoke | pass | failed | skipped |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | unit-kv-offload-tiering | pass | pytest -v tests/v1/kv_offload/ -k 'not test_cpu_offloading' | 22 | 0 | 0 |
| baseline | unit-kv-connector-offloading | pass | pytest -v tests/v1/kv_connector/unit/test_offloading_connector.py | 14 | 0 | 0 |
| baseline | unit-kv-connector-lifecycle | pass | pytest -v tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py | 1 | 0 | 0 |
| baseline | unit-kv-cache-layout | pass | pytest -v tests/v1/kv_connector/unit/test_kv_cache_layout.py | 2 | 0 | 0 |
| baseline | unit-kv-load-failure-recovery | pass | pytest -v tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py | 11 | 0 | 0 |
| baseline | unit-kv-cache-manager | pass | pytest -v tests/v1/core/test_prefix_caching.py | 49 | 0 | 0 |
| baseline | unit-single-type-kv-cache-manager | pass | pytest -v tests/v1/core/test_single_type_kv_cache_manager.py | 7 | 0 | 0 |
| baseline | unit-kv-cache-utils | pass | pytest -v tests/v1/core/test_kv_cache_utils.py | 48 | 0 | 0 |
| baseline | unit-kv-cache-coordinator | skipped | pytest -v tests/v1/core/test_kv_cache_coordinator.py |  |  |  |
| baseline | unit-cache-kernels | fail | pytest -v tests/kernels/test_cache_kernels.py -k 'not test_gather_cache_oob' | 0 | 1 | 0 |
| baseline | unit-scheduler | fail | pytest -v tests/v1/core/test_scheduler.py | 92 | 1 | 1 |
| baseline | unit-async-scheduler | pass | pytest -v tests/v1/core/test_async_scheduler.py | 8 | 0 | 0 |
| baseline | unit-worker | pass | pytest -v tests/v1/worker/ | 62 | 0 | 1 |
| baseline | integration-scheduler-e2e | pass | pytest -v tests/v1/core/test_scheduler_e2e.py | 2 | 0 | 0 |
| baseline | integration-engine | fail | pytest -v tests/v1/engine/ --ignore=tests/v1/engine/test_async_llm.py --ignore=tests/v1/engine/test_engine_core_client.py --ignore=tests/v1/engine/test_abort_final_step.py --ignore=tests/v1/engine/test_output_processor.py -k 'not test_skip_tokenizer_initialization' | 23 | 1 | 1 |
| baseline | correctness-basic | pass | pytest -v tests/basic_correctness/ -k 'not meta-llama and not tiering and not test_cumem and not test_cpu_offload and not Gemma2 and not test_prefetch_offload' | 2 | 0 | 8 |
| baseline | integration-kv-connector-nixl | fail | bash tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh | 0 | 1 | 0 |
| baseline | stress-kv-offload-memory-pressure | fail | bash -lc 'bash "$ROOT_DIR/spotlights/src/spotlights_validation/examples/kvoffload/scripts/run_nixl_stress.sh"' | 0 | 1 | 0 |
| evolved | unit-kv-offload-tiering | pass | pytest -v tests/v1/kv_offload/ -k 'not test_cpu_offloading' | 22 | 0 | 0 |
| evolved | unit-kv-connector-offloading | pass | pytest -v tests/v1/kv_connector/unit/test_offloading_connector.py | 14 | 0 | 0 |
| evolved | unit-kv-connector-lifecycle | pass | pytest -v tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py | 1 | 0 | 0 |
| evolved | unit-kv-cache-layout | pass | pytest -v tests/v1/kv_connector/unit/test_kv_cache_layout.py | 2 | 0 | 0 |
| evolved | unit-kv-load-failure-recovery | pass | pytest -v tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py | 11 | 0 | 0 |
| evolved | unit-kv-cache-manager | pass | pytest -v tests/v1/core/test_prefix_caching.py | 49 | 0 | 0 |
| evolved | unit-single-type-kv-cache-manager | pass | pytest -v tests/v1/core/test_single_type_kv_cache_manager.py | 7 | 0 | 0 |
| evolved | unit-kv-cache-utils | pass | pytest -v tests/v1/core/test_kv_cache_utils.py | 48 | 0 | 0 |
| evolved | unit-kv-cache-coordinator | skipped | pytest -v tests/v1/core/test_kv_cache_coordinator.py |  |  |  |
| evolved | unit-cache-kernels | fail | pytest -v tests/kernels/test_cache_kernels.py -k 'not test_gather_cache_oob' | 0 | 1 | 0 |
| evolved | unit-scheduler | fail | pytest -v tests/v1/core/test_scheduler.py | 92 | 1 | 1 |
| evolved | unit-async-scheduler | pass | pytest -v tests/v1/core/test_async_scheduler.py | 8 | 0 | 0 |
| evolved | unit-worker | pass | pytest -v tests/v1/worker/ | 62 | 0 | 1 |
| evolved | integration-scheduler-e2e | pass | pytest -v tests/v1/core/test_scheduler_e2e.py | 2 | 0 | 0 |
| evolved | integration-engine | fail | pytest -v tests/v1/engine/ --ignore=tests/v1/engine/test_async_llm.py --ignore=tests/v1/engine/test_engine_core_client.py --ignore=tests/v1/engine/test_abort_final_step.py --ignore=tests/v1/engine/test_output_processor.py -k 'not test_skip_tokenizer_initialization' | 23 | 1 | 1 |
| evolved | correctness-basic | pass | pytest -v tests/basic_correctness/ -k 'not meta-llama and not tiering and not test_cumem and not test_cpu_offload and not Gemma2 and not test_prefetch_offload' | 2 | 0 | 8 |
| evolved | integration-kv-connector-nixl | fail | bash tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh | 0 | 1 | 0 |
| evolved | stress-kv-offload-memory-pressure | fail | bash -lc 'bash "$ROOT_DIR/spotlights/src/spotlights_validation/examples/kvoffload/scripts/run_nixl_stress.sh"' | 0 | 1 | 0 |

## Benchmark Comparison

| run | id | status | policy | ttft_ms_mean | tpot_ms_mean | cpu_hit_rate | evictions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-lru | benchmark-multi-turn-kv-offload-lab-lru | pass | lru | 133.315 | 12.268 | 0.05419577462406333 | 326646.0 |
| baseline-arc | benchmark-multi-turn-kv-offload-lab-arc | pass | arc | 142.374 | 12.41 | 0.15140693877677106 | 252944.0 |
| evolved | benchmark-multi-turn-kv-offload-lab | pass | evolved | 137.957 | 12.294 | 0.34273712083248264 | 114168.0 |
