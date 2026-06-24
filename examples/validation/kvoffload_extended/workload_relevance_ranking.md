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
| 1 | **10/10** | [`gh_wl_deepseek_v4_multi_turn_prefix`](#1-gh_wl_deepseek_v4_multi_turn_prefix-1010) | Synthetic A.1→A.2→B→A.3 sequence with a shared prefix is the *exact* scenario session merging + position bonus is built for. Directly measures whether prefix blocks survive an interleaved request B — CPU hit rate and TTFT on the final A.3 request are the primary signals. |
| 2 | **9/10** | [`wl-agentic`](#2-wl-agentic-910) | Multi-turn with retained KV across turns. Session merging accumulates hits per conversation, hierarchical scoring protects active sessions. Directly exercises the core design intent. TTFT on later turns depends on reload. |
| 3 | **9/10** | [`wl-prefix-heavy`](#3-wl-prefix-heavy-910) | High prefix reuse with eviction pressure. Position bonus (30000 / (1 + pos/8)) massively favors prefix heads. Measures whether shared roots stay cached and how fast TTFT is on reuse after eviction. |
| 4 | **8/10** | [`gh_wl_hma_hybrid_model_offload`](#4-gh_wl_hma_hybrid_model_offload-810) | Multi-turn (3 turns/session) with prefix caching enabled. Tests session merging across turns in hybrid-attention context. Good for CPU hit rate on subsequent turns. |
| 5 | **8/10** | [`wl-large-kv-decode`](#5-wl-large-kv-decode-810) | Core offload scenario — 128 requests exceeding VRAM forces eviction/reload cycles. High eviction pressure means the scoring function is constantly exercised. TTFT of new requests depends on whether eviction policy chose wisely. |
| 6 | **7/10** | [`gh_wl_decode_speed_regression_check`](#6-gh_wl_decode_speed_regression_check-710) | Directly measures throughput with/without offload. While focused on output token throughput, TTFT is measurable from the same run and the regression threshold gives a clear pass/fail signal for the new policy's overhead. |
| 7 | **6/10** | [`wl-mixed-prefill-decode`](#7-wl-mixed-prefill-decode-610) | Mixed batch stresses block allocation and the admission gate. TTFT of newly-arriving requests during active decode sessions tests whether the policy correctly prioritizes. Moderate signal. |
| 8 | **6/10** | [`gh_wl_sustained_load_requests_stuck`](#8-gh_wl_sustained_load_requests_stuck-610) | 50 concurrent requests with prefix caching. If the deadlock is fixed, this exercises high-concurrency steady state where hit rate matters. However, the primary signal is liveness, not perf. |
| 9 | **5/10** | [`wl-long-context`](#9-wl-long-context-510) | Large single-request KV blocks will trigger offload, but unique content per request means low reuse — the session manager's scoring won't differentiate much. TTFT is dominated by prefill length, not reload. |
| 10 | **5/10** | [`gh_wl_high_concurrency_correctness`](#10-gh_wl_high_concurrency_correctness-510) | High concurrency + determinism check. Useful as a correctness gate (wrong output = corrupted reload), but the primary metric is correctness not TTFT/hit rate. |
| 11 | **4/10** | [`gh_wl_small_block_cpu_gpu_transfer`](#11-gh_wl_small_block_cpu_gpu_transfer-410) | Tests PCIe transfer throughput (DMA vs Triton). Measures the *data plane* speed, not the *control plane* policy decisions. Hit rate is irrelevant; TTFT is dominated by transfer bandwidth, not eviction quality. |
| 12 | **3/10** | [`gh_wl_long_context_block_pool_exhaustion`](#12-gh_wl_long_context_block_pool_exhaustion-310) | Correctness bug (assert on pool exhaustion). Only 5 sequential requests, no reuse pattern. Near-zero signal for hit rate or TTFT benchmarking. |
| 13 | **3/10** | [`gh_wl_sliding_window_offload`](#13-gh_wl_sliding_window_offload-310) | Correctness scenario (double-free). SWA blocks re-entering windows test correctness not policy quality. No multi-turn or prefix reuse pattern to exercise scoring. |
| 14 | **2/10** | [`gh_wl_dcp_offloading`](#14-gh_wl_dcp_offloading-210) | Block accounting correctness with TP=4. Requires multi-GPU, tests an assertion fix. Zero signal for eviction policy performance. |
| 15 | **2/10** | [`wl-standard-decode`](#15-wl-standard-decode-210) | No memory pressure — 64 requests × 1K KV fits in GPU. Offloading never triggers, so CPU hit rate is undefined and TTFT is unaffected by the policy. Only useful as a no-regression baseline. |

## Summary

The **top-3 workloads** (`gh_wl_deepseek_v4_multi_turn_prefix`, `wl-agentic`, `wl-prefix-heavy`) directly exercise the ContextualSessionManager's core innovations — session merging, position-biased scoring, and prefix preservation under eviction pressure. These give the strongest signal for both CPU hit rate and TTFT.

The **bottom tier** are correctness/liveness scenarios or workloads that never trigger the eviction policy.

## Recommended Benchmark Suite (score >= 7)

1. `gh_wl_deepseek_v4_multi_turn_prefix` — A.1→A.2→B→A.3 prefix survival
2. `wl-agentic` — multi-turn session continuity
3. `wl-prefix-heavy` — prefix reuse under pressure
4. `gh_wl_hma_hybrid_model_offload` — multi-turn hybrid model
5. `wl-large-kv-decode` — sustained eviction pressure
6. `gh_wl_decode_speed_regression_check` — throughput/TTFT regression gate

## Invoke Commands

### 1. `gh_wl_deepseek_v4_multi_turn_prefix` (10/10)

vLLM ref: [issue #42948](https://github.com/vllm-project/vllm/issues/42948)

```bash
# Server (per issue #42948 repro)
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 4096
```

The "A-B-A" workload is **synthetically generated**, not a recorded trace and
not a `vllm bench serve` dataset. The repro is a self-contained client that
procedurally builds two ~250K-token prompts sharing only the chat-template
prelude, then issues them in an A→A→B→A sequence and reads the prefix-cache
hit counters around each call:

```python
import urllib.request, json, hashlib, time

def long_prompt(seed_phrase, target_tokens=250_000):
    # filler chunks = MD5(seed_phrase + i) padded with "x"*100
    chunks, i = [], 0
    while sum(len(c) for c in chunks) // 4 < target_tokens:
        chunks.append(hashlib.md5(f"{seed_phrase}{i}".encode()).hexdigest() + "x" * 100)
        i += 1
    return "".join(chunks)

PROMPT_A = long_prompt("alpha")   # shared prelude + divergent user content
PROMPT_B = long_prompt("bravo")   # different prefix from the first user token

def send(prompt, label):
    body = json.dumps({
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3, "temperature": 0.0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions", body,
        {"Content-Type": "application/json"})
    t0 = time.time()
    urllib.request.urlopen(req).read()
    print(label, f"{time.time()-t0:.2f}s")
    # also scrape prefix_cache_hits_total / queries_total from /metrics here

send(PROMPT_A, "A.1"); time.sleep(1)
send(PROMPT_A, "A.2"); time.sleep(1)   # warm: should be ~100% prefix-cache hit
send(PROMPT_B, "B");   time.sleep(1)   # different prefix, cold
send(PROMPT_A, "A.3")                  # should be ~100% hit; bug makes it 0%
```

The signal: with the soft-pin fix (PRs #42985, #43191) the final `A.3` regains
a near-100% prefix-cache hit; without it, B's arrival evicts A's prefix blocks
and `A.3` reports 0%. Deterministic, runs in <5 minutes.

### 2. `wl-agentic` (9/10)

vLLM ref: [vllm/v1/kv_offload/](https://github.com/vllm-project/vllm/tree/main/vllm/v1/kv_offload)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark (multi-turn agentic pattern)
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --multi-turn \
  --num-prompts 50 \
  --request-rate 5 \
  --max-concurrency 16
```

### 3. `wl-prefix-heavy` (9/10)

vLLM ref: [benchmark_prefix_caching.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_prefix_caching.py)

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

vLLM ref: [issue #41515](https://github.com/vllm-project/vllm/issues/41515)

```bash
# Server
vllm serve Qwen/Qwen3.5-27B \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}' \
  --override-pooler-config '{"hybrid_kv_cache_manager": true}'

# Benchmark
vllm bench serve \
  --model Qwen/Qwen3.5-27B \
  --num-prompts 10 \
  --input-len 512 \
  --output-len 256 \
  --multi-turn \
  --turns-per-session 3
```

### 5. `wl-large-kv-decode` (8/10)

vLLM ref: [standard_attention.yaml](https://github.com/vllm-project/vllm/blob/main/benchmarks/attention_benchmarks/configs/standard_attention.yaml)

```bash
# Server (memory-pressure: 128 concurrent × 16K KV exceeds VRAM)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}' \
  --max-num-seqs 128

# Benchmark
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 128 \
  --input-len 16384 \
  --output-len 256 \
  --request-rate inf
```

### 6. `gh_wl_decode_speed_regression_check` (7/10)

vLLM ref: [issue #32604](https://github.com/vllm-project/vllm/issues/32604)

```bash
# Server (with offload)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 64 \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector"}'

# Benchmark
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 100 \
  --input-len 256 \
  --output-len 1024 \
  --request-rate inf

# Compare: repeat without --kv-transfer-config for baseline.
# Regression signal: >10% drop in output_token_throughput.
```

### 7. `wl-mixed-prefill-decode` (6/10)

vLLM ref: [standard_attention.yaml](https://github.com/vllm-project/vllm/blob/main/benchmarks/attention_benchmarks/configs/standard_attention.yaml)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-chunked-prefill \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark (mixed prefill+decode via concurrent arrivals)
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 200 \
  --input-len 2048 \
  --output-len 512 \
  --request-rate 10
```

### 8. `gh_wl_sustained_load_requests_stuck` (6/10)

vLLM ref: [issue #42371](https://github.com/vllm-project/vllm/issues/42371)

```bash
# Server
vllm serve Qwen/Qwen3-235B-A22B \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","cpu_bytes_to_use":34359738368}'

# Benchmark
vllm bench serve \
  --model Qwen/Qwen3-235B-A22B \
  --num-prompts 200 \
  --max-concurrency 50 \
  --request-rate inf
```

### 9. `wl-long-context` (5/10)

vLLM ref: [benchmark_long_document_qa_throughput.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_long_document_qa_throughput.py)

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

vLLM ref: [issue #31210](https://github.com/vllm-project/vllm/issues/31210)

```bash
# Server
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 9000 \
  --max-num-seqs 96 \
  --enable-chunked-prefill \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":{"num_cpu_blocks":50000}}'

# Benchmark (determinism check: compare output for identical prompts)
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 500 \
  --input-len 4000 \
  --output-len 180 \
  --max-concurrency 7
```

### 11. `gh_wl_small_block_cpu_gpu_transfer` (4/10)

vLLM ref: [PR #42212](https://github.com/vllm-project/vllm/pull/42212)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --block-size 16 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark (small blocks, high request rate)
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 512 \
  --input-len 128 \
  --output-len 64 \
  --request-rate 50
```

### 12. `gh_wl_long_context_block_pool_exhaustion` (3/10)

vLLM ref: [issue #42085](https://github.com/vllm-project/vllm/issues/42085)

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 32768 \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector"}' \
  --override-pooler-config '{"hybrid_kv_cache_manager": true}'

# Benchmark (sequential long-context requests)
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 5 \
  --input-len 32768 \
  --output-len 64 \
  --request-rate inf
```

### 13. `gh_wl_sliding_window_offload` (3/10)

vLLM ref: [issue #42571](https://github.com/vllm-project/vllm/issues/42571)

```bash
# Server (SWA model required, e.g. Mistral)
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --enable-chunked-prefill \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector"}'

# Benchmark
vllm bench serve \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --num-prompts 32 \
  --input-len 4096 \
  --output-len 256 \
  --request-rate inf
```

### 14. `gh_wl_dcp_offloading` (2/10)

vLLM ref: [PR #41549](https://github.com/vllm-project/vllm/pull/41549)

```bash
# Server (requires TP >= 4)
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --tensor-parallel-size 4 \
  --block-size 16 \
  --enforce-eager \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","decode_context_parallel_size":2,"kv_offloading_size":8}'

# Benchmark
vllm bench serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-prompts 10 \
  --input-len 600 \
  --output-len 128 \
  --request-rate inf
```

### 15. `wl-standard-decode` (2/10)

vLLM ref: [standard_attention.yaml](https://github.com/vllm-project/vllm/blob/main/benchmarks/attention_benchmarks/configs/standard_attention.yaml)

```bash
# Server (no offload — baseline only)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 64

# Benchmark
python benchmarks/attention_benchmarks/benchmark.py \
  --config benchmarks/attention_benchmarks/configs/standard_attention.yaml
```
