# BlockTables.gather_block_tables/compute_slot_mappings

[← v1.worker](../v1.worker.md)

- **File:** [`vllm/v1/worker/gpu/block_table.py`](vllm/v1/worker/gpu/block_table.py) (lines 115–283)
- **Symbol:** `BlockTables.gather_block_tables/compute_slot_mappings`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0018`

## Description
Gathers modular-runner block-table rows into forward-order storage and computes per-KV-group slot mappings, including padded block-table rows and padded token tails for CUDA graph execution.

## Current approach
gather_block_tables launches _gather_block_tables_kernel over (num_kv_cache_groups, num_reqs_padded) with BLOCK_SIZE=1024 and returns tuple slices. compute_slot_mappings launches _compute_slot_mappings_kernel over (num_groups, num_reqs + 1) with TRITON_BLOCK_SIZE=1024; the last batch program pads from actual_num_tokens to max_num_batched_tokens, while request programs recompute block indices/offsets and context-parallel locality.

## Estimated impact explanation
The path is per forward and scales with KV-cache groups, padded requests, and scheduled tokens. Faster gather/slot-map preparation should reduce median TPOT and prefill TTFT in the modular runner, especially when CUDA graph padding or multiple KV-cache groups are active.

## Evolve rationale
These owned kernels run in the modular GPU runner prepare_attn path every forward. Headroom is in jointly scheduling gather and slot-mapping work, specializing CP_SIZE == 1, tuning block sizes separately for short decode and long prefill, skipping padding when no padded tail/rows exist, and reducing tuple/dict slicing overhead for stable shapes. Correctness oracle: exact gathered input_block_tables, num_blocks usage, slot_mappings values, PAD_SLOT_ID placement, and downstream attention logits parity in modular GPU model-runner and block-table tests.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Skip gather_block_tables via dirty-row tracking for stable agentic decode batches
- **Agent:** claude

**Detailed description.**

Add an incremental fast-path to BlockTables.gather_block_tables (vllm/v1/worker/gpu/block_table.py:115-133) that exploits the observation that, in multi-turn agentic decoding, the running batch composition (idx_mapping) is typically stable across consecutive forwards and num_blocks per request changes only on the rare block-boundary crossing (roughly once every block_size decode tokens per request). Concretely:

1. Cache, on the BlockTables instance, the most recently-applied gather's idx_mapping (as a small CPU int32 tensor, since the runner constructs it on host) and a snapshot of num_blocks (which already lives in UvaBackedTensor.np, so this is a cheap CPU array copy).
2. In append_block_ids and apply_staged_writes, mark a small CPU dirty set of (group, req_idx) pairs whose source block_tables row was actually mutated since the last gather. (apply_staged_writes already iterates StagedWriteTensors; piggyback on that.)
3. At the top of gather_block_tables, compare the new idx_mapping to the cached one. If they match exactly and the dirty set is small (say <= K rows, e.g. K=8), do the gather as a tight Python loop of `input_block_tables[group][batch_idx, start:end].copy_(block_tables[group].gpu[req_idx, start:end])` for each dirty (group, req_idx), where start = num_blocks_at_last_gather[group, req_idx] and end = num_blocks_now[group, req_idx]. These are typically 0–1 element copies, so the entire 'gather' becomes a handful of small index assignments and the full _gather_block_tables_kernel launch is skipped. Padded rows beyond num_reqs in input_block_tables remain zero from the previous step (still zero), so no zero-fill is needed either.
4. If idx_mapping changed or the dirty set is large, fall back to today's _gather_block_tables_kernel and update the caches.

Correctness oracle: input_block_tables[group][:num_reqs_padded] must remain bit-identical to today's output; that property follows from the dirty set being a superset of all source-side mutations and idx_mapping-equality guaranteeing each batch_idx still maps to the same req_idx as last call. Compute_slot_mappings is unchanged — it reads source block_tables, not input_block_tables, so nothing downstream is sensitive to whether gather ran.

Measurable wins for this candidate: on stable agentic decode steps (the common case once a multi-turn batch warms up), the entire _gather_block_tables_kernel launch, its grid setup, and the tuple slicing become a no-op or a few-element copy. This is per-forward overhead that scales with num_kv_cache_groups * num_reqs_padded, so eliminating it on the hot path should noticeably reduce median TPOT, with no impact on prefill TTFT (where the dirty set is large and the path falls back).

**Novelty rationale.**

There are no listed deep_research_proposals for this candidate, so novelty must be judged against the candidate's own evolve_rationale. That rationale enumerates: joint scheduling of gather+slot-mapping, CP_SIZE==1 kernel specialization, decode-vs-prefill block-size tuning, skipping the slot-mapping padding tail when empty, and reducing Python tuple/dict slicing overhead. All five are kernel-internal or Python-glue optimizations that still launch _gather_block_tables_kernel every forward. This proposal is categorically different: it skips the gather kernel entirely on stable-batch decode steps by tracking source-side dirtiness and reusing the persistent input_block_tables across forwards, which exploits the multi-turn agentic workload's batch-composition stability — a property the rationale does not consider.

---

### 2. Add a vectorized all-decode slot-mapping fast path
- **Agent:** codex

**Detailed description.**

In vllm/v1/worker/gpu/block_table.py:142-167, add a decode-only branch of BlockTables.compute_slot_mappings that the modular GPU model runner can enable when input_batch.num_tokens == input_batch.num_reqs, meaning one scheduled token per request and any extra num_tokens_padded is only CUDA graph padding. Dispatch a new Triton kernel that maps many requests per program, for example BLOCK_REQS=128 or 256, instead of today's (num_groups, num_reqs + 1) grid where each request program uses a 1024-lane token loop for a single decode token. For each lane, load req_state_idx = idx_mapping[batch_idx], position = positions[batch_idx], compute block index, block offset, CP locality, and slot id exactly as _compute_slot_mappings_kernel does, then store slot_mappings[group_id, batch_idx]. Preserve PAD_SLOT_ID handling for the padded tail with a small tail-fill path, and keep the existing request-loop kernel for prefill and mixed-length batches. Validate by bit-comparing slot_mappings against the current kernel for CP_SIZE=1, CP_SIZE>1, multiple KV cache groups, block-boundary positions, and padded CUDA graph decode shapes, then checking attention parity through prepare_attn.

**Novelty rationale.**

There are no deep_research_proposals to duplicate. Claude's proposal skips or incrementally updates gather_block_tables using stable idx_mapping and dirty block-table rows; this proposal does not cache or skip gather and remains useful even when request membership changes. The candidate rationale mentions CP_SIZE specialization, block-size tuning for short decode, padding skips, gather/slot joint scheduling, and slice overhead. This is different because it changes the slot-mapping work decomposition for the all-decode shape from one mostly idle 1024-lane program per request to one program per block of requests, rather than merely tuning the existing request-loop kernel.

---
