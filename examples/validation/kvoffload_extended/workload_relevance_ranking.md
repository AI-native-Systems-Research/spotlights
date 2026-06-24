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

This is a **benchmarking** criterion, not a correctness one. A workload scores high only
if it produces a *graded* TTFT / CPU-hit-rate signal (a distribution over many requests),
exercises the **`kv_offload`** tier where this policy actually lives, and ideally
differentiates a *smart* eviction policy from a dumb one. Two properties matter and are
often conflated:

- **Eviction pressure** — memory must overflow so the policy actually fires.
- **Reuse structure** — requests must share blocks (prefix or multi-turn) so the *hit rate*
  reflects whether the policy kept the **right** blocks, not just *that* it evicted.

A workload needs **both** to benchmark policy *quality*. Pressure alone measures
overhead; reuse alone (without pressure) never triggers the policy. Binary
pass/fail bug repros, liveness/deadlock checks, and determinism checks are
**correctness** scenarios and are scored low regardless of how much offload machinery
they touch.

## Rankings

| Rank | Score | Workload ID | Rationale |
|------|-------|-------------|-----------|
| 1 | **10/10** | [`wl-agentic`](#1-wl-agentic-1010) | Multi-turn sessions with retained KV across turns + `OffloadingConnector` + 50 prompts at concurrency 16. The only workload combining sustained pressure, real reuse structure, **and** a large request count → a genuine TTFT / hit-rate distribution. Directly exercises session merging and hierarchical scoring. The truest benchmark of the policy. |
| 2 | **9/10** | [`wl-large-kv-decode`](#2-wl-large-kv-decode-910) | 128 requests × 16K KV exceeding VRAM at `request-rate inf` → sustained, heavy offload eviction. Best measure of TTFT **overhead** under pressure across a large sample. Caveat: no explicit reuse pattern, so it measures eviction *throughput/overhead* well but eviction *quality* (hit rate) poorly — add structured reuse to make it top-tier on hit rate too. |
| 3 | **8/10** | [`wl-prefix-heavy`](#3-wl-prefix-heavy-810) | High prefix reuse — the position-bonus / prefix-preservation logic is built for exactly this. Strong hit-rate signal **if** the run is sized so memory pressure actually forces offload eviction; `benchmark_prefix_caching.py` can otherwise fit in GPU and never trigger the policy. Pressure-dependent. |
| 4 | **7/10** | [`gh_wl_decode_speed_regression_check`](#4-gh_wl_decode_speed_regression_check-710) | Clean throughput/overhead regression gate with a defined >10% pass/fail threshold — a graded perf signal, not a bug repro. Decode-heavy (input 256 / output 1024) so it measures TPOT/throughput more than TTFT, and has no reuse → little hit-rate signal. Good cost-of-offload gate. |
| 5 | **6/10** | [`wl-mixed-prefill-decode`](#5-wl-mixed-prefill-decode-610) | Concurrent prefill+decode (200 prompts at rate 10) stresses block allocation and the admission gate; TTFT of newly-arriving requests during active decode is a real, graded signal. Moderate reuse, moderate pressure. |
| 6 | **6/10** | [`wl-standard-decode`](#6-wl-standard-decode-610) | No memory pressure → the policy never fires, so it carries **zero** signal about the policy in isolation. Scored mid-pack only because it is the **required no-offload baseline**: every TTFT/hit-rate number above is meaningless without it (impact = delta vs baseline). Necessary, not sufficient. |
| 7 | **5/10** | [`wl-long-context`](#7-wl-long-context-510) | Large single-request KV blocks trigger offload, but unique content per request means near-zero reuse — the scoring function can't differentiate good vs bad eviction. TTFT is dominated by prefill length, not reload. Pressure without reuse. |
| 8 | **4/10** | [`gh_wl_deepseek_v4_multi_turn_prefix`](#8-gh_wl_deepseek_v4_multi_turn_prefix-410) | **Correctness regression repro for #42948, not a benchmark.** Signal is binary (≈100% vs 0% hit) over only 3–4 requests — no distribution. Exercises the **GPU prefix-cache** path (`_maybe_evict_cached_block`); its `components_exercised` omit `kv_offload`, so it may not touch the CPU tier this policy governs at all. Heavy setup (DeepSeek-V4 + TP2). Elegant concept illustration, poor benchmark. |
| 9 | **4/10** | [`gh_wl_small_block_cpu_gpu_transfer`](#9-gh_wl_small_block_cpu_gpu_transfer-410) | Tests PCIe transfer throughput (DMA vs Triton) — the *data plane*, not the *control-plane* policy decisions. Hit rate is irrelevant; TTFT is dominated by transfer bandwidth, not eviction quality. |
| 10 | **4/10** | [`gh_wl_hma_hybrid_model_offload`](#10-gh_wl_hma_hybrid_model_offload-410) | Primarily a **correctness** scenario: validates `SupportsHMA` and that subsequent turns don't fail/corrupt. Only 10 requests / 3 turns → thin perf signal, on a heavy hybrid model. The reuse pattern is right, but the sample and intent are correctness-shaped. |
| 11 | **3/10** | [`gh_wl_sustained_load_requests_stuck`](#11-gh_wl_sustained_load_requests_stuck-310) | Scheduler **deadlock** repro (#42371). Primary signal is liveness (does it hang), not TTFT/hit rate. Touches concurrency surface but yields no graded perf measurement of the policy. |
| 12 | **3/10** | [`gh_wl_high_concurrency_correctness`](#12-gh_wl_high_concurrency_correctness-310) | Determinism/correctness check (`VLLM_BATCH_INVARIANT=1`, identical-output assertion). The primary metric is correctness, not TTFT/hit rate. |
| 13 | **2/10** | [`gh_wl_long_context_block_pool_exhaustion`](#13-gh_wl_long_context_block_pool_exhaustion-210) | Correctness bug (assert on pool exhaustion). Only 5 sequential requests, no reuse pattern. Near-zero signal for hit rate or TTFT benchmarking. |
| 14 | **2/10** | [`gh_wl_sliding_window_offload`](#14-gh_wl_sliding_window_offload-210) | Correctness scenario (double-free). SWA blocks re-entering windows test correctness, not policy quality. No multi-turn or prefix reuse pattern to exercise scoring. |
| 15 | **2/10** | [`gh_wl_dcp_offloading`](#15-gh_wl_dcp_offloading-210) | Block-accounting correctness with TP=4. Requires multi-GPU, tests an assertion fix. Zero signal for eviction-policy performance. |

## Summary

The **top tier** (`wl-agentic`, `wl-large-kv-decode`, `wl-prefix-heavy`) are the only
workloads that combine offload-tier eviction pressure with a graded, large-sample
TTFT / CPU-hit-rate signal — they directly exercise session merging, position-biased
scoring, and prefix preservation under pressure. `gh_wl_decode_speed_regression_check`
adds a clean overhead-regression gate, and `wl-standard-decode` is the indispensable
baseline that makes the others' deltas interpretable.

The **demoted middle** (`gh_wl_deepseek_v4_multi_turn_prefix`, `gh_wl_hma_hybrid_model_offload`)
look attractive because they showcase the *concept* of prefix survival, but they are
binary bug/correctness repros over tiny request counts — and the DeepSeek one likely
exercises the GPU prefix-cache tier rather than the CPU offload tier this policy governs.

The **bottom tier** are correctness/liveness/determinism scenarios or data-plane transfer
tests that never produce a graded policy-quality signal.

## Recommended Benchmark Suite

**Core (graded policy signal, score ≥ 7):**

1. `wl-agentic` — multi-turn session continuity (pressure + reuse + large N) — *primary*
2. `wl-large-kv-decode` — sustained eviction pressure (**add structured reuse** to also measure hit-rate quality)
3. `wl-prefix-heavy` — prefix reuse under pressure (size it so offload actually triggers)
4. `gh_wl_decode_speed_regression_check` — throughput/overhead regression gate

**Required baselines (without these, the numbers above are uninterpretable):**

- `wl-standard-decode` — no-offload reference point
- **LRU / ARC policy runs** — the change *replaces* LRU/ARC, so the suite must run the
  same core workloads under `--policy lru` and `--policy arc` (extended-plan indices 18–19)
  to report a real TTFT / hit-rate *delta* attributable to `ContextualSessionManager`.

## Invoke Commands

### 1. `wl-agentic` (10/10)

vLLM ref: [vllm/v1/kv_offload/](https://github.com/vllm-project/vllm/tree/main/vllm/v1/kv_offload)

**Score rationale:** The single best benchmark here. It is the only workload that
simultaneously satisfies all three requirements — sustained offload pressure, genuine
reuse structure (KV retained across turns of the same session), and enough requests
(50 prompts, concurrency 16) to yield a real TTFT / hit-rate *distribution* rather than a
point reading. Session merging accumulates hits per conversation and hierarchical scoring
protects active sessions, so this directly measures the policy's core design intent. TTFT
on later turns depends on whether the policy kept earlier-turn blocks resident — exactly
the quality signal we want.

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

### 2. `wl-large-kv-decode` (9/10)

vLLM ref: [standard_attention.yaml](https://github.com/vllm-project/vllm/blob/main/benchmarks/attention_benchmarks/configs/standard_attention.yaml)

**Score rationale:** 128 requests × 16K KV deliberately overflow VRAM at `request-rate inf`,
producing the heaviest sustained eviction pressure of any workload — the scoring function
runs constantly, and TTFT-under-pressure is measured across a large sample. The one caveat
the workload misses: **it has no explicit reuse pattern**, so it measures eviction
*throughput / overhead* well but eviction *quality* (hit rate) poorly — with unique content
per request, a smart policy and a dumb one look identical because there is nothing to reuse.
Add a structured reuse component (shared prefixes or repeated prompts) to also exercise
the hit-rate dimension; as written it is a top-tier **overhead** benchmark.

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

### 3. `wl-prefix-heavy` (8/10)

vLLM ref: [benchmark_prefix_caching.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_prefix_caching.py)

**Score rationale:** High prefix reuse is the textbook case for the position bonus
(`30000 / (1 + pos/8)`) and tail-first eviction — it directly measures whether shared
roots stay cached and how fast TTFT recovers on reuse after eviction. Strong hit-rate
signal, but **pressure-dependent**: `benchmark_prefix_caching.py` can fit entirely in GPU
memory, in which case offload eviction never triggers and the policy is never tested. Size
the run (more/longer prompts, tighter `--gpu-memory-utilization`) so the offload tier is
actually exercised — otherwise it silently degrades into a no-pressure baseline.

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

### 4. `gh_wl_decode_speed_regression_check` (7/10)

vLLM ref: [issue #32604](https://github.com/vllm-project/vllm/issues/32604)

**Score rationale:** A clean, graded perf-regression gate — it directly measures output
token throughput with vs. without offload and has a defined >10% pass/fail threshold, which
fits the benchmarking criterion squarely (it is a measurement, not a binary bug repro).
Capped at 7 because it is decode-heavy (input 256 / output 1024), so it speaks to
TPOT/throughput far more than to **TTFT**, and it carries no reuse pattern, so it yields
little **hit-rate** signal. Best used as the "cost of turning offload on" gate.

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

### 5. `wl-mixed-prefill-decode` (6/10)

vLLM ref: [standard_attention.yaml](https://github.com/vllm-project/vllm/blob/main/benchmarks/attention_benchmarks/configs/standard_attention.yaml)

**Score rationale:** A mixed batch (200 prompts at rate 10) stresses block allocation and
the admission gate while requests are simultaneously prefilling and decoding. TTFT of
newly-arriving requests during active decode sessions is a real, graded signal and tests
whether the admission gate correctly prioritizes high-value content. Mid-tier because both
the eviction pressure and the reuse structure are moderate rather than sustained — it
exercises the policy but doesn't stress it.

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

### 6. `wl-standard-decode` (6/10)

vLLM ref: [standard_attention.yaml](https://github.com/vllm-project/vllm/blob/main/benchmarks/attention_benchmarks/configs/standard_attention.yaml)

**Score rationale:** In isolation this workload carries **zero** signal about the policy:
64 requests × 1K KV fit comfortably in GPU, offloading never triggers, CPU hit rate is
undefined, and TTFT is unaffected by eviction. It is scored 6 — not 2 — purely because it
is the **mandatory no-offload baseline**: every TTFT and hit-rate number from the workloads
above is only meaningful as a *delta* against this reference. Necessary for benchmarking
the impact, but it does not itself benchmark the policy. Always run it; never report it alone.

```bash
# Server (no offload — baseline only)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 64

# Benchmark
python benchmarks/attention_benchmarks/benchmark.py \
  --config benchmarks/attention_benchmarks/configs/standard_attention.yaml
```

### 7. `wl-long-context` (5/10)

vLLM ref: [benchmark_long_document_qa_throughput.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_long_document_qa_throughput.py)

**Score rationale:** Large single-request KV blocks do trigger offload, so there is genuine
pressure — but the content is unique per request, so reuse is near zero and the scoring
function has nothing to differentiate. The policy cannot demonstrate that it kept the
*right* blocks when no block is ever reused. TTFT is dominated by prefill length, not
reload latency, further muting the policy's contribution. Pressure without reuse → weak
quality signal.

```bash
# Server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 131072 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector"}'

# Benchmark
python benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### 8. `gh_wl_deepseek_v4_multi_turn_prefix` (4/10)

vLLM ref: [issue #42948](https://github.com/vllm-project/vllm/issues/42948)

**Score rationale (demoted from 10):** This is a **correctness regression repro, not a
benchmark.** Three problems against the benchmarking criterion: (1) its signal is *binary* —
with the soft-pin fix the final `A.3` regains ≈100% prefix-cache hit, without it 0% — over
only 3–4 requests, so there is no distribution to measure; (2) it targets the **GPU
prefix-cache** eviction path (`_maybe_evict_cached_block`), and its `components_exercised`
omit `kv_offload`, so it likely never touches the CPU offload tier where
`ContextualSessionManager` operates — it reads `prefix_cache_hits_total`, a different
counter than the `cpu_hit_rate` we care about; (3) it requires DeepSeek-V4-Flash + TP2,
impractical for routine benchmarking. It is an elegant illustration of the *concept* the
policy targets (prefix survival across an interleaved request), which is why it's tempting —
but conceptual fit is not a graded performance measurement. Keep it as a **correctness
gate**, not in the benchmark suite.

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
and `A.3` reports 0%. Deterministic, runs in <5 minutes. **Note this is a
pass/fail signal on the GPU prefix-cache path — useful as a correctness gate,
weak as a TTFT/CPU-hit-rate benchmark.**

### 9. `gh_wl_small_block_cpu_gpu_transfer` (4/10)

vLLM ref: [PR #42212](https://github.com/vllm-project/vllm/pull/42212)

**Score rationale:** Tests PCIe transfer throughput (cuMemcpyBatchAsync → Triton fast path
vs. DMA) for small payloads — this is the **data plane**, not the control-plane policy.
Hit rate is irrelevant to it, and TTFT is dominated by transfer bandwidth rather than by
which blocks the eviction policy chose to keep. It measures *how fast a block moves*, not
*whether the policy moved the right block*. Orthogonal to `ContextualSessionManager`.

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

### 10. `gh_wl_hma_hybrid_model_offload` (4/10)

vLLM ref: [issue #41515](https://github.com/vllm-project/vllm/issues/41515)

**Score rationale (demoted from 8):** Its reuse pattern (multi-turn, 3 turns/session) is
the right *shape*, but its purpose and signal are **correctness**, not performance: it
validates that `OffloadingConnector` implements `SupportsHMA` and that subsequent turns
don't fail or corrupt state on hybrid (FullAttention + Mamba/GatedDeltaNet) models. At only
10 requests / 3 turns on a heavy hybrid model it produces a thin, noisy perf sample — the
real question it answers is "does turn 2 work," which is a correctness gate. Useful for
that; weak as a graded TTFT/hit-rate benchmark.

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

### 11. `gh_wl_sustained_load_requests_stuck` (3/10)

vLLM ref: [issue #42371](https://github.com/vllm-project/vllm/issues/42371)

**Score rationale (demoted from 6):** This is a scheduler **deadlock** repro — the root
cause is async KV-load requests in `WAITING_FOR_REMOTE_KVS` not being counted against
`max_num_seqs`, producing Running=0 / Waiting=N with zero throughput. Its primary signal is
**liveness** (does the engine hang), which is binary and correctness-flavored, not a graded
TTFT/hit-rate measurement. It touches the high-concurrency offload surface but yields no
usable perf number for the policy until/unless the deadlock is fixed — and even then,
liveness is what it proves.

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

### 12. `gh_wl_high_concurrency_correctness` (3/10)

vLLM ref: [issue #31210](https://github.com/vllm-project/vllm/issues/31210)

**Score rationale (demoted from 5):** A determinism/correctness check — it runs with
`VLLM_BATCH_INVARIANT=1` and asserts that identical prompts produce identical output, to
catch corrupted blocks restored from CPU under concurrency. The metric is correctness
(wrong output = corrupted reload), not TTFT or hit rate. It exercises the offload path
heavily but produces no graded performance signal about eviction *quality*.

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

### 13. `gh_wl_long_context_block_pool_exhaustion` (2/10)

vLLM ref: [issue #42085](https://github.com/vllm-project/vllm/issues/42085)

**Score rationale:** A correctness bug repro (assert failure in `popleft_n` on GPU block
pool exhaustion). Only 5 sequential requests with no reuse pattern → near-zero signal for
hit rate or TTFT benchmarking. Belongs in the correctness plan, not the benchmark suite.

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

### 14. `gh_wl_sliding_window_offload` (2/10)

vLLM ref: [issue #42571](https://github.com/vllm-project/vllm/issues/42571)

**Score rationale:** A correctness scenario (KV block double-free when an SWA block
re-enters the attention window and appears twice in the CPU offload list). It validates a
bug fix, not policy performance — there is no multi-turn or prefix reuse pattern to exercise
the scoring function. Correctness gate only.

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

### 15. `gh_wl_dcp_offloading` (2/10)

vLLM ref: [PR #41549](https://github.com/vllm-project/vllm/pull/41549)

**Score rationale:** Block-accounting correctness with Decode Context Parallelism — without
the fix the engine dies with an `AssertionError` at `scheduler.py:269`. It requires TP ≥ 4
(multi-GPU) and tests an assertion fix, producing zero signal for eviction-policy
performance. Pure correctness, heaviest setup cost of the bottom tier.

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
