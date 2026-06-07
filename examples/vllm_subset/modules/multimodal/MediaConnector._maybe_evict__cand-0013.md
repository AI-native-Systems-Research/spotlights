# MediaConnector._maybe_evict

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/media/connector.py`](vllm/multimodal/media/connector.py) (lines 197–229)
- **Symbol:** `MediaConnector._maybe_evict`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0013`

## Description
Evicts expired and least-recently-used files from the opt-in media download cache after cached HTTP downloads are written.

## Current approach
Every put scans the entire cache directory, stats each non-hidden file, collects expired entries, sorts all live entries by mtime, then unlinks entries until total_size is within the byte budget.

## Estimated impact explanation
This only affects deployments using VLLM_MEDIA_CACHE, but for URL-heavy agentic workloads the cache-miss write path blocks media loading executor time. Lower eviction overhead reduces media TTFT and avoids contention when many repeated media URLs are cached.

## Evolve rationale
The full directory scan and entries.sort at lines 204-224 are the cache eviction policy. Headroom includes maintaining cached total size and an mtime heap/index, batching expiry cleanup, avoiding stat calls for files unaffected by the new write, and moving eviction work off the request-critical path when the cache is under budget. Oracle: tests/multimodal/media/test_connector.py media-cache tests must preserve TTL expiration, LRU eviction order, exclude behavior for the just-written file, and the configured max-size invariant.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Deduplicate concurrent evictions with non-blocking flock and switch the scan to os.scandir
- **Agent:** claude

**Detailed description.**

In `MediaConnector._maybe_evict` (vllm/multimodal/media/connector.py:197-229), wrap the body in a non-blocking advisory file lock (`fcntl.flock` with `LOCK_EX | LOCK_NB`) on a sentinel file inside `_media_cache_dir` (e.g. `.evict.lock`). If the lock cannot be acquired, return immediately — another worker thread or process is already scanning, sorting, and unlinking, so a second pass would do duplicate stat()/sort() work and racy unlinks against the file the first pass already saw. This directly addresses thundering-herd behavior under URL-heavy agentic loads where many cache-miss puts arrive concurrently from the multimodal loader pool.

In the same edit, replace the `cache_dir.iterdir()` + `f.stat()` loop (lines 204-219) with `os.scandir(cache_dir)` and `entry.stat()` on the `DirEntry`. On macOS/Linux, `scandir` returns entries with the dirent already populated and exposes the file name without constructing a `Path` object per iteration; `entry.stat()` reuses the open directory file descriptor, halving the syscall cost of the hot scan loop while preserving exact mtime/size semantics. Wrap entries that survive into `Path` objects only at unlink time (or call `os.unlink(entry.path)` directly).

Keep the existing TTL filter, the `exclude` short-circuit, the descending mtime sort, and the size-budget loop unchanged — the eviction policy and oracle invariants (TTL expiration, LRU order, exclude-just-written, max-size invariant in `tests/multimodal/media/test_connector.py`) all remain bit-identical when the lock is held. The lock file itself starts with `.` so the existing `f.name.startswith('.')` guard already excludes it from the scan.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The candidate's `evolve_rationale` lists in-memory size/mtime indices, batched expiry cleanup, skipping stats for unaffected files, and skipping eviction when under budget — all of which target the *single-caller* cost of one `_maybe_evict` invocation. This proposal is orthogonal: it removes *redundant* invocations across concurrent callers (worker threads / sibling processes sharing `VLLM_MEDIA_CACHE`) via a non-blocking advisory lock, which the rationale does not mention and which an in-memory accumulator inside one process cannot achieve. The `os.scandir` swap is a concrete syscall-level change at the exact lines cited (204-219) that the rationale also does not enumerate.

---

### 2. Short-circuit under-budget eviction and replace full LRU sort with a min-heap
- **Agent:** codex

**Detailed description.**

In `MediaConnector._maybe_evict` (`vllm/multimodal/media/connector.py:197-229`), keep the existing full scan and TTL handling, but change the post-scan eviction path so it first unlinks expired files and returns immediately when `total_size <= self._media_cache_max_bytes`. Only when the cache is actually over budget should it build or heapify eviction candidates by `st_mtime` and pop the oldest entries until the size budget is satisfied. This preserves the TTL-first policy, the exclusion of the just-written file, and exact LRU eviction order, while avoiding the unconditional `entries.sort(...)` cost on the common path where a write does not push the cache over budget. Add a focused regression test that writes several live entries while the cache remains below the configured max size and asserts no LRU candidate is removed, plus keep the existing LRU/TTL tests as the behavioral oracle.

**Novelty rationale.**

There are no existing deep_research_proposals for this candidate. Agent A proposes concurrency deduplication with a non-blocking `flock` and replacing directory iteration with `os.scandir`; it explicitly keeps the existing descending mtime sort unchanged. This proposal targets a different cost center inside the same method: eliminating the unconditional full sort when no size eviction is needed, and limiting ordering work to the over-budget path.

---
