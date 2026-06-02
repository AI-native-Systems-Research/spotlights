| row | index | id | status | priority | halt_on_failure | invoke | pass | failed | skipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | unit-kv-offload-tiering | pass | 1 | True | pytest -v tests/v1/kv_offload/ -k 'not test_cpu_offloading' | 22 | 0 | 0 |
| baseline | 2 | unit-kv-connector-offloading | pass | 1 | True | pytest -v tests/v1/kv_connector/unit/test_offloading_connector.py | 14 | 0 | 0 |
| baseline | 3 | unit-kv-connector-lifecycle | pass | 2 | True | pytest -v tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py | 1 | 0 | 0 |
| baseline | 4 | unit-kv-cache-layout | pass | 2 | True | pytest -v tests/v1/kv_connector/unit/test_kv_cache_layout.py | 2 | 0 | 0 |
| baseline | 5 | unit-kv-load-failure-recovery | pass | 2 | True | pytest -v tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py | 11 | 0 | 0 |
| baseline | 6 | unit-kv-cache-manager | pass | 3 | True | pytest -v tests/v1/core/test_prefix_caching.py | 49 | 0 | 0 |
| baseline | 7 | unit-single-type-kv-cache-manager | pass | 3 | True | pytest -v tests/v1/core/test_single_type_kv_cache_manager.py | 7 | 0 | 0 |
| baseline | 8 | unit-kv-cache-utils | pass | 3 | False | pytest -v tests/v1/core/test_kv_cache_utils.py | 48 | 0 | 0 |
| baseline | 9 | unit-kv-cache-coordinator | skipped | 3 | False | pytest -v tests/v1/core/test_kv_cache_coordinator.py |  |  |  |
| baseline | 10 | unit-cache-kernels | fail | 4 | False | pytest -v tests/kernels/test_cache_kernels.py -k 'not test_gather_cache_oob' | 0 | 1 | 0 |
| baseline | 11 | unit-scheduler | fail | 4 | False | pytest -v tests/v1/core/test_scheduler.py | 92 | 1 | 1 |
| baseline | 12 | unit-async-scheduler | pass | 4 | False | pytest -v tests/v1/core/test_async_scheduler.py | 8 | 0 | 0 |
| baseline | 13 | unit-worker | pass | 4 | False | pytest -v tests/v1/worker/ | 62 | 0 | 1 |
| baseline | 14 | integration-scheduler-e2e | pass | 5 | False | pytest -v tests/v1/core/test_scheduler_e2e.py | 2 | 0 | 0 |
| baseline | 15 | integration-engine | fail | 5 | False | pytest -v tests/v1/engine/ --ignore=tests/v1/engine/test_async_llm.py --ignore=tests/v1/engine/test_engine_core_client.py --ignore=tests/v1/engine/test_abort_final_step.py --ignore=tests/v1/engine/test_output_processor.py -k 'not test_skip_tokenizer_initialization' | 23 | 1 | 1 |
| baseline | 16 | correctness-basic | pass | 6 | True | pytest -v tests/basic_correctness/ -k 'not meta-llama and not tiering and not test_cumem and not test_cpu_offload and not Gemma2 and not test_prefetch_offload' | 2 | 0 | 8 |
| baseline | 20 | integration-kv-connector-nixl | fail | 6 | False | bash tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh | 0 | 1 | 0 |
| baseline | 21 | stress-kv-offload-memory-pressure | fail | 7 | False | bash -lc 'bash "$ROOT_DIR/spotlights/src/spotlights_validation/examples/kvoffload/scripts/run_nixl_stress.sh"' | 0 | 1 | 0 |
| evolved | 1 | unit-kv-offload-tiering | pass | 1 | True | pytest -v tests/v1/kv_offload/ -k 'not test_cpu_offloading' | 22 | 0 | 0 |
| evolved | 2 | unit-kv-connector-offloading | pass | 1 | True | pytest -v tests/v1/kv_connector/unit/test_offloading_connector.py | 14 | 0 | 0 |
| evolved | 3 | unit-kv-connector-lifecycle | pass | 2 | True | pytest -v tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py | 1 | 0 | 0 |
| evolved | 4 | unit-kv-cache-layout | pass | 2 | True | pytest -v tests/v1/kv_connector/unit/test_kv_cache_layout.py | 2 | 0 | 0 |
| evolved | 5 | unit-kv-load-failure-recovery | pass | 2 | True | pytest -v tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py | 11 | 0 | 0 |
| evolved | 6 | unit-kv-cache-manager | pass | 3 | True | pytest -v tests/v1/core/test_prefix_caching.py | 49 | 0 | 0 |
| evolved | 7 | unit-single-type-kv-cache-manager | pass | 3 | True | pytest -v tests/v1/core/test_single_type_kv_cache_manager.py | 7 | 0 | 0 |
| evolved | 8 | unit-kv-cache-utils | pass | 3 | False | pytest -v tests/v1/core/test_kv_cache_utils.py | 48 | 0 | 0 |
| evolved | 9 | unit-kv-cache-coordinator | skipped | 3 | False | pytest -v tests/v1/core/test_kv_cache_coordinator.py |  |  |  |
| evolved | 10 | unit-cache-kernels | fail | 4 | False | pytest -v tests/kernels/test_cache_kernels.py -k 'not test_gather_cache_oob' | 0 | 1 | 0 |
| evolved | 11 | unit-scheduler | fail | 4 | False | pytest -v tests/v1/core/test_scheduler.py | 92 | 1 | 1 |
| evolved | 12 | unit-async-scheduler | pass | 4 | False | pytest -v tests/v1/core/test_async_scheduler.py | 8 | 0 | 0 |
| evolved | 13 | unit-worker | pass | 4 | False | pytest -v tests/v1/worker/ | 62 | 0 | 1 |
| evolved | 14 | integration-scheduler-e2e | pass | 5 | False | pytest -v tests/v1/core/test_scheduler_e2e.py | 2 | 0 | 0 |
| evolved | 15 | integration-engine | fail | 5 | False | pytest -v tests/v1/engine/ --ignore=tests/v1/engine/test_async_llm.py --ignore=tests/v1/engine/test_engine_core_client.py --ignore=tests/v1/engine/test_abort_final_step.py --ignore=tests/v1/engine/test_output_processor.py -k 'not test_skip_tokenizer_initialization' | 0 | 1 | 0 |
| evolved | 16 | correctness-basic | fail | 6 | True | pytest -v tests/basic_correctness/ -k 'not meta-llama and not tiering and not test_cumem and not test_cpu_offload and not Gemma2 and not test_prefetch_offload' | 0 | 2 | 8 |
| evolved | 20 | integration-kv-connector-nixl | not_executed | 6 | False | bash tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh |  |  |  |
| evolved | 21 | stress-kv-offload-memory-pressure | not_executed | 7 | False | bash -lc 'bash "$ROOT_DIR/spotlights/src/spotlights_validation/examples/kvoffload/scripts/run_nixl_stress.sh"' |  |  |  |
