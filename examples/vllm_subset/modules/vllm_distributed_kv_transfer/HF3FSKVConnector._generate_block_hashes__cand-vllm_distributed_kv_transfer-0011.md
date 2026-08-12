# HF3FSKVConnector._generate_block_hashes

[← vllm/distributed/kv_transfer](../vllm_distributed_kv_transfer.md)

- **File:** [`vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py) (lines 955–982)
- **Symbol:** `HF3FSKVConnector._generate_block_hashes`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_distributed_kv_transfer-0011`

## Description
Builds HF3FS block hash chains by looping over block-sized token slices and computing one prefix hash per full block.

## Current approach
Pure Python loop over range(0, len(token_ids), self._block_size). Each iteration slices token_ids, calls _compute_prefix_hash, conditionally appends the hash, and advances previous_hash sequentially; lookup, save, and load paths repeat this work.

## Estimated impact explanation
Hash generation is on the TTFT-critical prefix lookup path and scales with prompt length, which is common in multi-turn agentic conversations with long shared history.

## Evolve rationale
Long prompts can require thousands of per-block hash operations, and chunked prefill can repeat the same prefix chain across lookup/save/load. Caching request hash chains, avoiding list slices, or replacing stringified-list MD5 with a faster stable binary hash over token arrays are local algorithm changes preserving the returned block_hashes list. Oracle: tests/v1/kv_connector/unit/test_hf3fs_connector.py can assert deterministic block hashes and unchanged lookup/save/load behavior.

## Deep research proposals

### 1. Replace stringified-list MD5 with xxHash XXH3/XXH128 over binary token buffers
- **Finding:** `find-vllm_distributed_kv_transfer-0010` — *xxHash - Extremely fast hash algorithm*
- **Source URL:** <https://github.com/Cyan4973/xxHash>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py, change _compute_prefix_hash (lines 1007-1012) and its callsite inside _generate_block_hashes (lines 955-982) so that each per-block prefix hash is computed with xxhash (XXH3_128 or XXH3_64) over a compact binary representation of the block tokens, rather than hashlib.md5 over f"{previous_hash}_{token_ids}".encode(). Concretely: (1) import xxhash; (2) for each block, feed the previous hash bytes plus the block's token ids as a raw byte buffer (e.g. array.array('q', token_ids[start_idx:end_idx]).tobytes() or numpy.asarray(token_ids, dtype=np.int64).view(np.uint8).tobytes()) into a single xxhash.xxh3_128() (or xxh3_64) update-and-hexdigest call, using the previous block's hash bytes as the seed/prefix so the chain remains deterministic; (3) return the hex digest so downstream lookup/save/load keys retain their string shape and block_hashes list contract stays identical. Because xxhash is stable and portable, cross-process HF3FS keys stay consistent across workers. Keep the surrounding loop in _generate_block_hashes (block_size stride, start_block_id filter, max_blocks_count early exit) untouched. tests/v1/kv_connector/unit/test_hf3fs_connector.py can assert deterministic block_hashes and unchanged lookup/save/load behavior after the swap.

**Proposal rationale.**

The candidate's hot path spends most of its per-block cost in two places the finding directly targets: Python's list.__repr__ to build combined_string = f"{previous_hash}_{token_ids}", and MD5 over the resulting text-encoded bytes. Both scale with block_size per block and with num_blocks per request, so a long agentic multi-turn prompt with thousands of blocks pays this cost on every lookup/save/load. xxHash's XXH3 is explicitly designed for RAM-speed throughput and strong small-input behavior at block_size-sized inputs, and operating on a binary token buffer removes the O(block_size) repr formatting step entirely. This is a local, drop-in algorithmic change that preserves the returned block_hashes list contract required by the candidate's oracle, and it addresses the exact TTFT-critical prefix-hash overhead called out in the candidate's evolve_rationale.

---

### 2. Adopt APC-style faster block hash and cached chain in HF3FSKVConnector._generate_block_hashes
- **Finding:** `find-vllm_distributed_kv_transfer-0013` — *[RFC] Automatic Prefix Caching*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/2614>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Apply the APC RFC's block-level hash identity (hash(previous_hash, tokens_in_block)) and its P1 'faster hash function' priority to HF3FSKVConnector._generate_block_hashes at vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:955-982 and its helper _compute_prefix_hash at lines 1007-1012. Concretely: (1) Replace the stringified-list MD5 in _compute_prefix_hash — currently `hashlib.md5(f"{previous_hash}_{token_ids}".encode()).hexdigest()` — with a fast stable binary hash over the block's tokens combined with the parent block's hash digest (e.g., feed the previous hash's raw bytes plus a numpy/array-based bytes view of the int token slice into a fast hasher such as xxhash or blake2b-digest-size-16), avoiding the O(block_size) Python repr formatting on every block. (2) In _generate_block_hashes, iterate without re-slicing token_ids each iteration (use an array view or index arithmetic on a bytes/np.ndarray of the tokens) and thread the previous block's hash forward exactly as the APC design describes, so identical prefixes across lookup/save/load calls produce identical chains. (3) Mirror the APC RFC's block hash metadata by caching the computed hash chain per (request_id, token_ids identity) so that within one request the lookup, save, and load paths all reuse the same chain instead of recomputing thousands of MD5s. Preserve the returned list[str] contract of _generate_block_hashes and its use of start_block_id / max_blocks_count truncation so callers and the tests/v1/kv_connector/unit/test_hf3fs_connector.py oracle continue to see deterministic, unchanged block hashes and unchanged lookup/save/load behavior.

**Proposal rationale.**

The APC RFC directly names the two levers that bind this candidate's hot path: (a) 'every block in the KV cache can be uniquely identified by hash(prefix tokens, tokens in this block)' — exactly the chain _generate_block_hashes builds — and (b) 'Faster hash function' is listed as P1 future work. The current implementation uses `f"{previous_hash}_{token_ids}"` MD5, which forces a Python repr of the whole int list plus a cryptographic digest per block; on long agentic prompts this is thousands of blocks on the TTFT-critical prefix lookup path and is repeated across lookup/save/load. The finding therefore contributes two concrete, transferable ideas — a faster non-cryptographic binary hash over token bytes and block-metadata reuse — that map cleanly onto this method while keeping the returned block_hashes list identity stable so the existing HF3FS oracle test still passes.

---

## Agent proposals

### 1. Bound _generate_block_hashes work in get_num_new_matched_tokens with an exponential-probe existence check
- **Agent:** claude

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py, refactor the interaction between _generate_block_hashes (lines 955-982) and get_num_new_matched_tokens (lines 695-728) so that per-block hashing is bounded by the actual external-cache hit length rather than by num_tokens_to_check. Concretely, replace the current 'compute all hashes, then batch_key_exists over all of them' pattern (lines 717-726) with an exponential-then-binary probe: (1) extend _generate_block_hashes with a resumable form that accepts a starting previous_hash and start_idx and returns the last block's hash so a chain can be grown incrementally; (2) in get_num_new_matched_tokens, iteratively generate the next chunk of block hashes (starting at 1, doubling: 1, 2, 4, 8, 16, ... capped at num_tokens_to_check/_block_size), call batch_key_exists on just that chunk, and stop the doubling as soon as any probe returns a miss; (3) when the miss chunk is reached, binary-search inside that chunk to locate the first missing block exactly by continuing the chain from the last known-hit prefix; (4) reuse the fully materialized chain that this probe already produced (up to matched_blocks + 1) by storing it on RequestSchedulingState so wait_for_save (line 631) and start_load_kv (line 653) do not recompute the same prefix chain a second and third time - they slice from the cached chain, extending it only if their block window exceeds what get_num_new_matched_tokens materialized. Keep the returned list[str] contract of _generate_block_hashes intact for backward compatibility of other callers, and preserve start_block_id / max_blocks_count semantics so the tests/v1/kv_connector/unit/test_hf3fs_connector.py oracle continues to see deterministic block hashes and unchanged lookup/save/load behavior.

**Novelty rationale.**

The two existing deep_research_proposals both stay strictly inside _compute_prefix_hash / _generate_block_hashes and change only the hash primitive (MD5 to xxhash over binary buffers) plus optional per-request chain memoization. Neither touches the caller pattern in get_num_new_matched_tokens, which is where the true waste on the TTFT-critical path lives: on a fresh multi-turn prompt of thousands of blocks whose external cache hit is only a few hundred blocks deep, the current code hashes every block up to num_tokens_to_check before ever asking the metadata service whether any of them exist. An exponential-probe / staged-existence strategy shortens the hashed-prefix length to O(matched_blocks * log) regardless of hash-function speed, and reusing the produced chain across the get_num_new_matched_tokens -> wait_for_save -> start_load_kv lifecycle addresses a different axis of duplication than proposal #2's per-request identity cache inside one _generate_block_hashes call. The two ideas compose but do not overlap.

---

### 2. Stop hashing the same prefix twice by passing matched block hashes through scheduler metadata
- **Agent:** codex

**Detailed description.**

In vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py, make get_num_new_matched_tokens return or persist the exact block_hashes it already computed for the matched prefix, then have start_load_kv and wait_for_save consume that materialized list instead of calling _generate_block_hashes again for overlapping ranges. A narrow implementation is to add an HF3FS-specific field on the request's scheduling/connector metadata that stores the full block_hashes list generated at lines 717-726, along with the block size and token-count watermark it corresponds to. Then update the later load/save paths to slice that list for start_block_id/max_blocks_count semantics, only extending it via _generate_block_hashes when the later path needs blocks beyond the stored watermark. This preserves _generate_block_hashes' list[str] contract and hash values, but removes repeated chain construction across the common lookup -> load/save lifecycle for the same request.

**Novelty rationale.**

The deep_research proposals mention changing the hash primitive, avoiding slices, and a broad per-(request_id, token_ids identity) cache inside or around _generate_block_hashes. Agent A proposes a larger exponential-probe lookup algorithm and includes chain reuse as part of that redesign. This proposal is narrower and independent: keep the existing eager batch_key_exists lookup behavior and the existing hash function, but explicitly pass the already computed block_hashes from get_num_new_matched_tokens to the subsequent HF3FS load/save callers through scheduler metadata. It avoids a different, immediately observable duplication without requiring a new probing strategy or hash implementation.

---
