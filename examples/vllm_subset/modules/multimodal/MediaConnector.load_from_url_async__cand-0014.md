# MediaConnector.load_from_url_async

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/media/connector.py`](vllm/multimodal/media/connector.py) (lines 321–354)
- **Symbol:** `MediaConnector.load_from_url_async`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0014`

## Description
Asynchronous URL loader that checks the media cache, downloads HTTP bytes, stores cache entries, and decodes media on a global thread pool.

## Current approach
Concurrent requests for the same uncached URL each perform their own cache check, HTTP fetch, cache write, and media_io.load_bytes decode. Cache file I/O and CPU media decoding share global_thread_pool, and the cache write is awaited before scheduling decode for the downloaded bytes.

## Estimated impact explanation
For multi-turn or batched agentic workloads that reuse the same media URL, request coalescing can eliminate duplicate downloads and decodes. That reduces media TTFT under concurrency, though single uncached requests remain dominated by network and decoder cost.

## Evolve rationale
The async HTTP branch at lines 331-354 is an owned scheduling policy. Headroom includes an in-flight per-URL future to coalesce concurrent misses, separating blocking cache I/O from decode work, and scheduling decode without waiting for cache eviction when the returned bytes are already available. Oracle: tests/multimodal/media/test_connector.py and fetch_*_async callers must observe the same media_io.load_bytes result or exception for each URL, with allowed-domain checks, timeout propagation, redirect policy, and cache semantics preserved.

## Deep research proposals

### 1. Coalesce concurrent same-URL media fetches with a singleflight in-flight map
- **Finding:** `find-0006` — *singleflight package - golang.org/x/sync/singleflight - Go Packages*
- **Source URL:** <https://pkg.go.dev/golang.org/x/sync/singleflight>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Apply Go's singleflight duplicate-call-suppression pattern to MediaConnector.load_from_url_async at vllm/multimodal/media/connector.py:321-354. Add a per-connector in-flight registry mapping URL -> asyncio.Future (guarded by an asyncio.Lock or per-key futures dict) that tracks an ongoing miss-path resolution. On entry to the HTTP branch, after the initial cache lookup miss, the first caller for a URL atomically inserts a pending Future into the registry and proceeds to (a) async_get_bytes, (b) cache write via run_in_executor, and (c) media_io.load_bytes decode via run_in_executor; the resolved decoded value (or exception) is set on the Future and the entry removed. Concurrent callers that find an existing entry await the same Future and receive the identical result/exception, eliminating duplicate downloads, cache writes, and decode invocations. The allowed-domain check runs before registry lookup so disallowed URLs still raise per-call; fetch_timeout, redirect policy, and cache semantics are preserved because only one underlying fetch occurs. Optionally schedule decode without awaiting the cache write (issue both run_in_executor calls and gather them) so duplicate waiters are unblocked as soon as decode completes rather than after eviction-bound cache I/O. The same singleflight key shape can be reused later for hash/processor-cache misses keyed by multimodal hash.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'an in-flight per-URL future to coalesce concurrent misses' as headroom; the finding describes exactly that mechanism (singleflight: 'If a duplicate comes in, the duplicate caller waits for the original to complete and receives the same results.'). Multi-turn agentic workloads frequently re-reference the same media URL across concurrent turns, so deduplicating the miss path directly attacks the duplicated-download/decode cost that dominates media TTFT under concurrency, while preserving the oracle (same media_io.load_bytes result/exception, allowed-domain checks, timeout, redirect policy, cache semantics) because only the first caller drives the side effects.

---

## Agent proposals

### 1. Split global_thread_pool into separate I/O and CPU-decode executors to remove head-of-line blocking
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/media/connector.py, replace the single global_thread_pool (defined at line 36 and used throughout load_from_url_async at lines 334-365) with two purpose-built executors: (1) a `cache_io_pool` ThreadPoolExecutor sized generously (e.g. max(32, 4 * cpu_count)) for short blocking disk I/O — `_get_cached_bytes`, `_put_cached_bytes`, and the `_load_file_url` stat/read path; and (2) a `decode_pool` ThreadPoolExecutor sized to roughly cpu_count (or a new env var VLLM_MEDIA_DECODE_THREAD_COUNT) for CPU-bound work — `media_io.load_bytes` and `_load_data_url` (which decodes inline base64). Update load_from_url_async at lines 334-353 so the cache-hit branch reads cached bytes on cache_io_pool then dispatches `media_io.load_bytes` on decode_pool; the miss branch dispatches `_put_cached_bytes` on cache_io_pool and `media_io.load_bytes` on decode_pool. This separation matters because today a burst of concurrent image/video decodes (each holding a slot for hundreds of milliseconds) starves quick cache lookups behind the same FIFO queue, inflating TTFT even for already-cached media; conversely, slow disk evictions on cache_put occupy slots that decode work could otherwise consume. Sizing the CPU pool to physical cores also prevents oversubscription thrash in libjpeg-turbo / pyav decoders that are themselves multi-threaded. Preserve the existing oracle: same `media_io.load_bytes` result/exception per URL, allowed-domain check unchanged, fetch_timeout and redirect policy untouched, cache semantics identical. Keep `global_thread_pool` as a thin alias (or remove all uses) and register `atexit.register` shutdowns for both new pools. Tests in tests/multimodal/media/test_connector.py continue to pass; add a test that submits N concurrent slow-decode tasks plus one cache-hit lookup and asserts the cache-hit path completes without waiting for the decode queue to drain.

**Novelty rationale.**

The listed deep_research_proposal (find-0006) focuses exclusively on singleflight coalescing of duplicate URL fetches and only briefly mentions parallelizing the cache-write and decode `run_in_executor` calls via gather. It does not address the executor topology — both proposed parallel calls still target the same `global_thread_pool`, so head-of-line blocking between bursty CPU decodes and short cache I/O remains. Pool separation is a complementary, orthogonal optimization that the candidate's evolve_rationale explicitly calls out ('separating blocking cache I/O from decode work') and that the existing proposal does not implement; it benefits even single-caller workloads where singleflight has no effect (every request is unique), e.g. a multi-turn agent presenting a fresh URL each turn while concurrent users hit the cache.

---

### 2. Bypass cache executor work when media caching is disabled
- **Agent:** codex

**Detailed description.**

In `vllm/multimodal/media/connector.py::MediaConnector.load_from_url_async`, guard the HTTP cache lookup and cache write executor calls behind `self._media_cache_dir is not None`. When `VLLM_MEDIA_CACHE` is unset, skip the `_get_cached_bytes` `run_in_executor` call before `async_get_bytes`, and skip the `_put_cached_bytes` `run_in_executor` call after download; dispatch only `media_io.load_bytes(data)` for decode. Keep the existing lookup/write order unchanged when caching is enabled so TTL, LRU, and cache-hit behavior are preserved. Add a targeted async connector test with `_media_cache_dir = None` and monkeypatched `_get_cached_bytes`/`_put_cached_bytes` counters or sentinels to assert the disabled-cache path performs no cache executor work while still returning the decoded media.

**Novelty rationale.**

The deep_research proposal covers coalescing concurrent misses and optionally overlapping cache write with decode, but it still assumes every request enters the cache miss machinery. Claude's proposal separates cache I/O and decode executors, but still routes cache operations through executor infrastructure. This proposal removes two no-op executor round trips entirely for the default cache-disabled configuration, benefiting unique-URL and single-caller workloads where singleflight has no effect and executor splitting still leaves unnecessary scheduling overhead.

---
