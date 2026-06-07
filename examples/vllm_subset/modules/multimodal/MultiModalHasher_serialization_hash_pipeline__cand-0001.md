# MultiModalHasher serialization/hash pipeline

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/hasher.py`](vllm/multimodal/hasher.py) (lines 52–162)
- **Symbol:** `MultiModalHasher serialization/hash pipeline`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0001`

## Description
Serializes heterogeneous multimodal kwargs into a stable byte stream and feeds that stream into the configured hash function for cache keys and shared-field grouping.

## Current approach
A per-type isinstance ladder converts PIL images, MediaWithBytes, torch.Tensor, np.ndarray, scalars, dicts, and tuples into many small byte or memoryview chunks. Tensor handling calls .cpu() before numpy conversion, bfloat16 tensors are forced contiguous, non-contiguous ndarrays use .tobytes(), recursive dict/list handling repeatedly encodes f-string keys, and hash_kwargs calls hasher.update() for every yielded chunk after sorting kwargs.

## Estimated impact explanation
This runs before HF processing for every multimodal cache lookup and is reused by shared-field batching. Its cost scales with image/video byte volume, so reducing copies and Python chunk overhead directly lowers media TTFT and improves repeated-media multi-turn cache-hit paths.

## Evolve rationale
The optimization unit is serialize_item/iter_item_to_bytes plus the hash_kwargs update loop at lines 158-160. Headroom includes reducing avoidable tensor/ndarray copies, packing adjacent small chunks before hasher.update(), pre-encoding repeated structural keys, and caching per-item digests where the MultiModalKwargsItem is immutable for the cache lifetime. Oracle: tests/multimodal/test_hasher.py and multimodal cache key behavior require the digest from hash_kwargs to remain bit-identical for every supported input type and VLLM_MM_HASHER_ALGORITHM setting.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Use blake3 multi-threaded mode for large multimodal buffers
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/hasher.py:154-162, hash_kwargs always creates a single-threaded blake3 hasher and feeds it many small chunks via .update(). For media TTFT the dominant cost is hashing image/video bytes (PIL pixel arrays produced at line 68, MediaWithBytes.original_bytes at line 84, and tensor.numpy() buffers at lines 102/106), each easily reaching multi-MB to tens-of-MB. blake3's Python binding supports built-in parallelism via the max_threads constructor argument (and AUTO sentinel) — when fed a single large bytes/memoryview, it splits chunks across threads and substantially outperforms serial update on large inputs.

Proposed change, scoped to this file:
1. In _get_hasher_factory at lines 22-47, when algorithm == 'blake3', return a factory that constructs blake3(max_threads=blake3.AUTO). Keep sha256/sha512 unchanged. Because hexdigest() output of blake3 is independent of how update() is sliced and of max_threads, this preserves the bit-identical digest invariant required by tests/multimodal/test_hasher.py.
2. In hash_kwargs at lines 154-162, add a 'large-buffer fast path': within the inner loop, if a yielded chunk is a memoryview/bytes whose nbytes exceeds a threshold (e.g. 256 KiB), call hasher.update(chunk) directly so blake3 can internally parallelize that single big update; smaller chunks continue to go through the existing loop. This isolates the parallelism win to the cases where it actually amortizes thread spin-up — namely image pixel arrays, MediaWithBytes.original_bytes, and tensor numpy buffers — while leaving small-chunk traffic (keys, scalars, shapes) on the cheap serial path.
3. Guard the change with the existing VLLM_MM_HASHER_ALGORITHM check so FIPS algorithms are unaffected, and fall back to single-threaded blake3 if the installed blake3 wheel does not accept max_threads (older versions) by catching TypeError once at factory time and memoizing the working factory in _get_hasher_factory's lru_cache.

This directly attacks the wall-clock cost of hashing large media blobs that runs before HF processing on every multimodal cache lookup, which is the bottleneck for media TTFT in repeated/multi-turn agentic workloads.

**Novelty rationale.**

The candidate has no existing deep_research_proposals. The candidate's evolve_rationale focuses exclusively on the serialization side (reducing tensor/ndarray copies, packing adjacent small chunks before update, pre-encoding repeated f-string keys, and caching per-item digests on immutable MultiModalKwargsItem). It does not touch the hash primitive itself. This proposal is orthogonal: it leaves serialize_item/iter_item_to_bytes byte-for-byte identical and instead exploits blake3's built-in multi-threaded update path on the large contiguous buffers (PIL pixel arrays, MediaWithBytes.original_bytes, tensor.numpy()). Parallelizing the cryptographic work via blake3(max_threads=AUTO) and a size-thresholded fast path is a separate axis of speedup from anything listed in evolve_rationale.

---

### 2. Preserve MediaWithBytes wrappers on the normal hash path
- **Agent:** codex

**Detailed description.**

Route multimodal item hashing through the raw item accessor even when no user UUIDs are supplied, so `MultiModalHasher.serialize_item()` actually receives `MediaWithBytes` objects and takes the existing `original_bytes` branch at `vllm/multimodal/hasher.py:77-84`. Today `ProcessorInputs.get_mm_hashes()` uses `data_items.get_all_items_for_hash()` only in the UUID-aware branch, but the common no-UUID branch iterates `data_items`, which unwraps `MediaWithBytes` into a plain PIL image. That forces the hasher down the slower `np.asarray(image)` pixel path at `hasher.py:67-75` instead of hashing the already-available encoded bytes. Change the no-UUID branch to iterate `data_items.get_all_items_for_hash()` as well, then add a regression test with an `ImageMediaIO.load_bytes()` image showing that hashing the processor input matches hashing the `MediaWithBytes` wrapper, not the unwrapped PIL image. This reduces pre-processor TTFT for URL/base64/file images by avoiding decoded-pixel array materialization and often hashing far fewer bytes, while using the synchronization-preserving wrapper the hasher already supports.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A focuses on making the hash primitive faster with BLAKE3 multi-threading for large buffers. This proposal is a different axis: it fixes the caller path so the hasher avoids creating and hashing the large decoded PIL pixel buffer in the first place when original encoded bytes are already available. It does not change BLAKE3 construction, chunk thresholds, or large-buffer update behavior.

---
