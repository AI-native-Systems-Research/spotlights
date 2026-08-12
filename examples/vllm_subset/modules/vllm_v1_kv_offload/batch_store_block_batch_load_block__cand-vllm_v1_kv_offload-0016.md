# batch_store_block/batch_load_block

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/tiering/fs/io.py`](vllm/v1/kv_offload/tiering/fs/io.py) (lines 168–222)
- **Symbol:** `batch_store_block/batch_load_block`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0016`

## Description
Filesystem batch I/O implementation validates offsets, dispatches to the C extension when available, and falls back to serial Python per-block store/load loops.

## Current approach
Each batch validates every offset, then the C path materializes memoryview slices and temp paths before one extension call. The Python fallback loops over paths serially, stops on the first load error, and annotates load failures with num_succeeded for partial promotion recovery.

## Estimated impact explanation
FS-backed secondary-to-primary loads sit on the TTFT path when reused blocks spill beyond CPU primary memory. Reducing Python loop overhead or improving batch I/O can lower promotion latency, with gains bounded by filesystem bandwidth and the optional C path.

## Evolve rationale
The _HAS_FSIO_C dispatch, memoryview-slice materialization, serial fallback loops, first-error stop rule, and partial-load accounting are owned runtime choices for the FS tier. Correctness oracle: tests/v1/kv_offload/tiering/test_fs_tier.py, byte-identical store/load round trips, atomic temp-file replacement semantics, and exact num_succeeded preservation on partial load failure.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Parallelize the Python fallback in batch_store_block / batch_load_block with a bounded ThreadPoolExecutor
- **Agent:** claude

**Detailed description.**

In vllm/v1/kv_offload/tiering/fs/io.py:168-222, the Python fallback (taken whenever vllm.fs_io_C is not importable) is a strictly serial per-block loop. Because os.open / os.readv / os.write / os.replace all release the GIL, the fallback leaves most of the achievable batch throughput on the table on the FS->primary promotion path that gates TTFT. Change the two fallback branches as follows:

1. batch_load_block fallback (lines 214-222): replace the serial `for i, (path, offset) in enumerate(zip(paths, offsets))` loop with a bounded ThreadPoolExecutor (workers = min(8, len(paths)), created lazily and reused via a module-level singleton or contextlib.closing per call to avoid per-batch pool-creation overhead). Submit one future per (path, offset, index) that calls the existing `_load_block`. Wait for completions; on first exception, cancel pending futures and compute `num_succeeded` as the largest prefix k such that indices 0..k-1 all completed successfully, so partial-load recovery in FsTier.get_finished_jobs stays byte-for-byte identical to today's contract (this preserves the existing test invariant in tests/v1/kv_offload/tiering/test_fs_tier.py that num_succeeded counts blocks before the failing one). Attach it to the raised OSError exactly as the serial path does.

2. batch_store_block fallback (lines 188-190): mirror the change with the same executor, but propagate exceptions directly (stores today just raise on first error; no num_succeeded needed).

3. Leave the _HAS_FSIO_C dispatch branches (lines 183-187, 210-213) untouched — the C extension already batches internally.

4. Guard the executor size behind an env knob (e.g. VLLM_FS_KV_IO_FALLBACK_WORKERS) defaulting to 8, so single-core / container-constrained deployments can pin it to 1 and recover today's serial behavior. Document the O_DIRECT interaction: with O_DIRECT on Linux the kernel does not readahead, which is exactly why cross-block parallelism (not readahead) is the correct lever for the fallback.

Correctness oracle is unchanged: byte-identical round trips, atomic temp-file rename semantics, exact num_succeeded preservation on partial load failure. tests/v1/kv_offload/tiering/test_fs_tier.py should still pass; add one case that induces failure at block k in a batch of >k blocks with the executor active and asserts num_succeeded == k.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any well-scoped proposal is by definition not covered. This proposal is also specifically narrowed to the *fallback* branch of the _HAS_FSIO_C dispatch (which the candidate description explicitly flags as a serial Python loop) and is orthogonal to the outer DualQueueThreadPool that parallelizes across jobs rather than within a batch.

---

### 2. Add duplicate-offset coalescing to batch_load_block before dispatch
- **Agent:** codex

**Detailed description.**

In vllm/v1/kv_offload/tiering/fs/io.py:batch_load_block, pre-process the incoming (path, offset) batch to detect repeated source offsets for the same filesystem block file and avoid issuing redundant disk reads for aliases within the same batch. Build a stable mapping from each unique (path, offset) pair to the first destination index, call the existing C-extension path or Python fallback only for the unique pairs, then copy the loaded bytes from the first destination slice into any duplicate destination slices before returning. Preserve current validation and error behavior by validating all original offsets first, and on failure from the unique-load call translate its num_succeeded back to the original prefix semantics expected by FsTier.get_finished_jobs: duplicates whose representative completed count as succeeded only if they appear before the first failed original item. This targets multi-turn agentic reuse where request batches may ask for the same spilled KV block more than once, reducing promotion reads without changing on-disk format or atomic store semantics. Add a focused test in tests/v1/kv_offload/tiering/test_fs_tier.py or the nearest fs/io test that loads a batch containing duplicate offsets, asserts byte-identical destinations, and injects a later load failure to verify translated num_succeeded remains the original prefix count.

**Novelty rationale.**

There are no deep_research_proposals listed for this candidate. Agent A's proposal parallelizes the existing Python fallback loop and leaves the C-extension dispatch untouched; this proposal instead reduces the amount of work before either dispatch by coalescing duplicate load requests, and it applies equally to the C and Python paths. It is therefore not a duplicate of bounded fallback threading.

---
