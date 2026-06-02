# Validation Results Comparison

## Regressions (passed in baseline, failed in evolved)

- **`correctness-basic`** (validation entry)
  - RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
  - torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
Search for `cudaErrorDevicesUnavailable' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

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
| baseline | integration-engine | fail | 23 | 1 | 1 |
| evolved | integration-engine | fail | 0 | 1 | 0 |
| baseline | correctness-basic | pass | 2 | 0 | 8 |
| evolved | correctness-basic | fail | 0 | 2 | 8 |
| baseline | integration-kv-connector-nixl | fail | 0 | 1 | 0 |
| evolved | integration-kv-connector-nixl | not_executed |  |  |  |
| baseline | stress-kv-offload-memory-pressure | fail | 0 | 1 | 0 |
| evolved | stress-kv-offload-memory-pressure | not_executed |  |  |  |

## Benchmark Comparison

| run | id | status | policy | ttft_ms_mean | tpot_ms_mean | cpu_hit_rate | evictions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-lru | benchmark-multi-turn-kv-offload-lab-lru | pass | lru | 146.444 | 12.413 | 0.04290750951308535 | 356470.0 |
| baseline-arc | benchmark-multi-turn-kv-offload-lab-arc | pass | arc | 139.909 | 12.369 | 0.13933070445211726 | 236548.0 |
| evolved | benchmark-multi-turn-kv-offload-lab | pass | evolved | 131.538 | 12.48 | 0.34762372972256556 | 120870.0 |
