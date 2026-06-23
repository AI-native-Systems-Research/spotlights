# Workload Relevance Ranking for ContextualSessionManager Benchmarking

## Change Under Test

The OpenEvolve exp9 best output (`ContextualSessionManager`) — a session-aware KV offloading eviction policy replacing LRU/ARC with:

- **Session merging**: extends existing conversation chains on cache hits
- **Hierarchical scoring**: recency + cumulative hits + position bonus (favoring prefix roots)
- **Admission gate**: rejects low-value one-shot content
- **Tail-first eviction**: prunes most-recent blocks first, preserving prefix heads

Best run metrics: `cpu_hit_rate = 44.7%`, `ttft_ratio = 1.79x`, `throughput_ratio ≈ 0.98`.

## Ranking Criteria

> "Can this workload help us to properly benchmark the performance impact in terms of TTFT and CPU Hit Rate?"

## Rankings

| Rank | Score | Workload ID | Rationale |
|------|-------|-------------|-----------|
| 1 | **10/10** | `gh_wl_deepseek_v4_multi_turn_prefix` | A→B→A pattern with shared prefix is the *exact* scenario session merging + position bonus is built for. Directly measures whether prefix blocks survive between interleaved requests — CPU hit rate and TTFT on the second A request are the primary signals. |
| 2 | **9/10** | `wl-agentic` | Multi-turn with retained KV across turns. Session merging accumulates hits per conversation, hierarchical scoring protects active sessions. Directly exercises the core design intent. TTFT on later turns depends on reload. |
| 3 | **9/10** | `wl-prefix-heavy` | High prefix reuse with eviction pressure. Position bonus (30000 / (1 + pos/8)) massively favors prefix heads. Measures whether shared roots stay cached and how fast TTFT is on reuse after eviction. |
| 4 | **8/10** | `gh_wl_hma_hybrid_model_offload` | Multi-turn (3 turns/session) with prefix caching enabled. Tests session merging across turns in hybrid-attention context. Good for CPU hit rate on subsequent turns. |
| 5 | **8/10** | `wl-large-kv-decode` | Core offload scenario — 128 requests exceeding VRAM forces eviction/reload cycles. High eviction pressure means the scoring function is constantly exercised. TTFT of new requests depends on whether eviction policy chose wisely. |
| 6 | **7/10** | `gh_wl_decode_speed_regression_check` | Directly measures throughput with/without offload. While focused on output token throughput, TTFT is measurable from the same run and the regression threshold gives a clear pass/fail signal for the new policy's overhead. |
| 7 | **6/10** | `wl-mixed-prefill-decode` | Mixed batch stresses block allocation and the admission gate. TTFT of newly-arriving requests during active decode sessions tests whether the policy correctly prioritizes. Moderate signal. |
| 8 | **6/10** | `gh_wl_sustained_load_requests_stuck` | 50 concurrent requests with prefix caching. If the deadlock is fixed, this exercises high-concurrency steady state where hit rate matters. However, the primary signal is liveness, not perf. |
| 9 | **5/10** | `wl-long-context` | Large single-request KV blocks will trigger offload, but unique content per request means low reuse — the session manager's scoring won't differentiate much. TTFT is dominated by prefill length, not reload. |
| 10 | **5/10** | `gh_wl_high_concurrency_correctness` | High concurrency + determinism check. Useful as a correctness gate (wrong output = corrupted reload), but the primary metric is correctness not TTFT/hit rate. |
| 11 | **4/10** | `gh_wl_small_block_cpu_gpu_transfer` | Tests PCIe transfer throughput (DMA vs Triton). Measures the *data plane* speed, not the *control plane* policy decisions. Hit rate is irrelevant; TTFT is dominated by transfer bandwidth, not eviction quality. |
| 12 | **3/10** | `gh_wl_long_context_block_pool_exhaustion` | Correctness bug (assert on pool exhaustion). Only 5 sequential requests, no reuse pattern. Near-zero signal for hit rate or TTFT benchmarking. |
| 13 | **3/10** | `gh_wl_sliding_window_offload` | Correctness scenario (double-free). SWA blocks re-entering windows test correctness not policy quality. No multi-turn or prefix reuse pattern to exercise scoring. |
| 14 | **2/10** | `gh_wl_dcp_offloading` | Block accounting correctness with TP=4. Requires multi-GPU, tests an assertion fix. Zero signal for eviction policy performance. |
| 15 | **2/10** | `wl-standard-decode` | No memory pressure — 64 requests × 1K KV fits in GPU. Offloading never triggers, so CPU hit rate is undefined and TTFT is unaffected by the policy. Only useful as a no-regression baseline. |

## Summary

The **top-3 workloads** (`gh_wl_deepseek_v4_multi_turn_prefix`, `wl-agentic`, `wl-prefix-heavy`) directly exercise the ContextualSessionManager's core innovations — session merging, position-biased scoring, and prefix preservation under eviction pressure. These give the strongest signal for both CPU hit rate and TTFT.

The **bottom tier** are correctness/liveness scenarios or workloads that never trigger the eviction policy.

## Recommended Benchmark Suite (score >= 7)

1. `gh_wl_deepseek_v4_multi_turn_prefix` — A-B-A prefix survival
2. `wl-agentic` — multi-turn session continuity
3. `wl-prefix-heavy` — prefix reuse under pressure
4. `gh_wl_hma_hybrid_model_offload` — multi-turn hybrid model
5. `wl-large-kv-decode` — sustained eviction pressure
6. `gh_wl_decode_speed_regression_check` — throughput/TTFT regression gate

## Invoke Commands

### 1. `gh_wl_deepseek_v4_multi_turn_prefix` (10/10)

```bash
# Server
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark
python benchmarks/benchmark_serving.py \
  --model deepseek-ai/DeepSeek-V3-0324 \
  --num-prompts 20 \
  --sharegpt-pattern "A-B-A" \
  --shared-prefix-tokens 512 \
  --unique-suffix-tokens 128 \
  --request-rate inf
```

### 2. `wl-agentic` (9/10)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark (multi-turn agentic pattern)
python benchmarks/benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --multi-turn \
  --num-prompts 50 \
  --request-rate 5 \
  --max-concurrency 16
```

### 3. `wl-prefix-heavy` (9/10)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark
python benchmarks/benchmark_prefix_caching.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching
```

### 4. `gh_wl_hma_hybrid_model_offload` (8/10)

```bash
# Server
vllm serve Qwen/Qwen3.5-27B \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}' \
  --override-pooler-config '{"hybrid_kv_cache_manager": true}'

# Benchmark
python benchmarks/benchmark_serving.py \
  --model Qwen/Qwen3.5-27B \
  --num-prompts 10 \
  --input-len 512 \
  --output-len 256 \
  --multi-turn \
  --turns-per-session 3
```

### 5. `wl-large-kv-decode` (8/10)

```bash
# Server (memory-pressure: 128 concurrent × 16K KV exceeds VRAM)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}' \
  --max-num-seqs 128

# Benchmark
python benchmarks/benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 128 \
  --input-len 16384 \
  --output-len 256 \
  --request-rate inf
```

### 6. `gh_wl_decode_speed_regression_check` (7/10)

```bash
# Server (with offload)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 64 \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector"}'

# Benchmark
python benchmarks/benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 100 \
  --input-len 256 \
  --output-len 1024 \
  --request-rate inf

# Compare: repeat without --kv-transfer-config for baseline.
# Regression signal: >10% drop in output_token_throughput.
```

### 7. `wl-mixed-prefill-decode` (6/10)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-chunked-prefill \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark (mixed prefill+decode via concurrent arrivals)
python benchmarks/benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 200 \
  --input-len 2048 \
  --output-len 512 \
  --request-rate 10
```

### 8. `gh_wl_sustained_load_requests_stuck` (6/10)

```bash
# Server
vllm serve Qwen/Qwen3-235B-A22B \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","cpu_bytes_to_use":34359738368}'

# Benchmark
python benchmarks/benchmark_serving.py \
  --model Qwen/Qwen3-235B-A22B \
  --num-prompts 200 \
  --max-concurrency 50 \
  --request-rate inf
```

### 9. `wl-long-context` (5/10)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 131072 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark
python benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### 10. `gh_wl_high_concurrency_correctness` (5/10)

```bash
# Server
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 9000 \
  --max-num-seqs 96 \
  --enable-chunked-prefill \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":{"num_cpu_blocks":50000}}'

# Benchmark (determinism check: compare output for identical prompts)
python benchmarks/benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 500 \
  --input-len 4000 \
  --output-len 180 \
  --max-concurrency 7
```

### 11. `gh_wl_small_block_cpu_gpu_transfer` (4/10)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --block-size 16 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark (small blocks, high request rate)
python benchmarks/benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 512 \
  --input-len 128 \
  --output-len 64 \
  --request-rate 50
```

### 12. `gh_wl_long_context_block_pool_exhaustion` (3/10)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 32768 \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector"}' \
  --override-pooler-config '{"hybrid_kv_cache_manager": true}'

# Benchmark (sequential long-context requests)
python benchmarks/benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 5 \
  --input-len 32768 \
  --output-len 64 \
  --request-rate inf
```

### 13. `gh_wl_sliding_window_offload` (3/10)

```bash
# Server (SWA model required, e.g. Mistral)
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --enable-chunked-prefill \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector"}'

# Benchmark
python benchmarks/benchmark_serving.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --num-prompts 32 \
  --input-len 4096 \
  --output-len 256 \
  --request-rate inf
```

### 14. `gh_wl_dcp_offloading` (2/10)

```bash
# Server (requires TP >= 4)
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --tensor-parallel-size 4 \
  --block-size 16 \
  --enforce-eager \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","decode_context_parallel_size":2,"kv_offloading_size":8}'

# Benchmark
python benchmarks/benchmark_serving.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-prompts 10 \
  --input-len 600 \
  --output-len 128 \
  --request-rate inf
```

### 15. `wl-standard-decode` (2/10)

```bash
# Server (no offload — baseline only)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 64

# Benchmark
python benchmarks/attention_benchmarks/benchmark.py \
  --config benchmarks/attention_benchmarks/configs/standard_attention.yaml
```
