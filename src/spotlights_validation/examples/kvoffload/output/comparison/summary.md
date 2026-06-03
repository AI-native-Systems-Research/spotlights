# Validation Results Comparison

## Regressions (passed in baseline, failed in evolved)

None

## Fixes (failed in baseline, passed in evolved)

None

## Tests Comparison

| run | id | status | pass | failed | skipped |
| --- | --- | --- | --- | --- | --- |
| baseline | unit-kv-offload-tiering | pass | 22 | 0 | 0 |
| evolved | unit-kv-offload-tiering | pass | 22 | 0 | 0 |
| baseline | unit-kv-connector-offloading | pass | 14 | 0 | 0 |
| evolved | unit-kv-connector-offloading | pass | 14 | 0 | 0 |
| baseline | unit-kv-connector-lifecycle | pass | 1 | 0 | 0 |
| evolved | unit-kv-connector-lifecycle | pass | 1 | 0 | 0 |
| baseline | unit-kv-cache-layout | pass | 2 | 0 | 0 |
| evolved | unit-kv-cache-layout | pass | 2 | 0 | 0 |
| baseline | unit-kv-load-failure-recovery | pass | 11 | 0 | 0 |
| evolved | unit-kv-load-failure-recovery | pass | 11 | 0 | 0 |
| baseline | unit-kv-cache-manager | pass | 49 | 0 | 0 |
| evolved | unit-kv-cache-manager | pass | 49 | 0 | 0 |
| baseline | unit-single-type-kv-cache-manager | pass | 7 | 0 | 0 |
| evolved | unit-single-type-kv-cache-manager | pass | 7 | 0 | 0 |
| baseline | unit-kv-cache-utils | pass | 48 | 0 | 0 |
| evolved | unit-kv-cache-utils | pass | 48 | 0 | 0 |
| baseline | unit-kv-cache-coordinator | skipped |  |  |  |
| evolved | unit-kv-cache-coordinator | skipped |  |  |  |
| baseline | unit-cache-kernels | fail | 0 | 1 | 0 |
| evolved | unit-cache-kernels | fail | 0 | 1 | 0 |
| baseline | unit-scheduler | fail | 92 | 1 | 1 |
| evolved | unit-scheduler | fail | 92 | 1 | 1 |
| baseline | unit-async-scheduler | pass | 8 | 0 | 0 |
| evolved | unit-async-scheduler | pass | 8 | 0 | 0 |
| baseline | unit-worker | pass | 62 | 0 | 1 |
| evolved | unit-worker | pass | 62 | 0 | 1 |
| baseline | integration-scheduler-e2e | pass | 2 | 0 | 0 |
| evolved | integration-scheduler-e2e | pass | 2 | 0 | 0 |
| baseline | integration-engine | fail | 13 | 11 | 1 |
| evolved | integration-engine | fail | 13 | 11 | 1 |
| baseline | correctness-basic | pass | 2 | 0 | 8 |
| evolved | correctness-basic | pass | 2 | 0 | 8 |
| baseline | integration-kv-connector-nixl | fail | 0 | 1 | 0 |
| evolved | integration-kv-connector-nixl | fail | 0 | 1 | 0 |
| baseline | stress-kv-offload-memory-pressure | fail | 0 | 1 | 0 |
| evolved | stress-kv-offload-memory-pressure | fail | 0 | 1 | 0 |

## Benchmark Comparison

| run | id | status | policy | ttft_ms_mean | tpot_ms_mean | cpu_hit_rate | evictions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-lru | benchmark-multi-turn-kv-offload-lab-lru | pass | lru | 146.642 | 12.395 | 0.05017457689135302 | 354688.0 |
| baseline-arc | benchmark-multi-turn-kv-offload-lab-arc | pass | arc | 142.829 | 12.362 | 0.13696339198976895 | 271294.0 |
| evolved | benchmark-multi-turn-kv-offload-lab | pass | evolved | 129.996 | 12.269 | 0.3296662150198481 | 132417.0 |
