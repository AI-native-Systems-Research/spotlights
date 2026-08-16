# MambaManager.allocate_new_blocks

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 1532–1651)
- **Symbol:** `MambaManager.allocate_new_blocks`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0014`

## Description
Mamba align-mode block allocation path covering running-state blocks, speculative-block reuse, partial-hit CoW, and producer partial-tail handoff bookkeeping.

## Current approach
Branches on align mode, mutates req_blocks with null padding and recycled speculative blocks, calls block_pool.get_new_blocks for the delta, handles partial-hit CoW by moving cache hashes or applying a local copy, records pending copies/offloads, and updates per-request align-mode state.

## Estimated impact explanation
Only Mamba align-mode models use this path, but for them it runs during allocation and shapes recurrent-state reuse. Cleaner reuse and lower Python overhead reduce TPOT and follow-up-turn TTFT.

## Evolve rationale
Dense policy bundle for Mamba state reuse and CoW promotion. Headroom in recycling more speculative blocks, batching hash moves, replacing per-index nulling with range operations, and tightening the predictor/allocation contract with get_num_blocks_to_allocate to avoid over-allocation drift. Correctness oracle: Mamba align tests plus a reference model over num_tokens, num_tokens_main_model, speculative blocks, and partial-hit cases where returned blocks, req_blocks, moved hashes, and pending copies match exactly.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Lazy speculative-block allocation with unused-tail reclamation in Mamba align mode
- **Agent:** claude

**Detailed description.**

Change MambaManager.allocate_new_blocks (vllm/v1/core/single_type_kv_cache_manager.py:1532-1651) so that the `num_speculative_blocks` tail is not eagerly reserved on every allocation call. Two coordinated edits:

1) On the first-prefill branch (`not blocks_allocated`), instead of packing `num_new_blocks = 1 + self.num_speculative_blocks + int(has_partial_hit)` into a single `block_pool.get_new_blocks` call, split into: (a) the running-state + CoW blocks needed for this step (`1 + int(has_partial_hit)`), and (b) a `num_speculative_blocks` reservation drawn from a per-manager free-list of previously-released speculative blocks first, falling back to `block_pool.get_new_blocks` only for the shortfall. Maintain this free-list (`self._recycled_spec_blocks: deque[KVCacheBlock]`) populated by (2).

2) Add a reclamation hook invoked by the scheduler at the end of each step (or lazily at the top of `allocate_new_blocks` for a given request): when `num_required_blocks <= len(req_blocks) and not has_partial_hit` (the current early-return on line 1561-1562), if the request's speculative tail was fully unused for `K` consecutive steps (tracked via a small per-request int in `self._spec_unused_steps`), pop the tail speculative blocks off `req_blocks`, decrement their ref_cnt, and push them onto `self._recycled_spec_blocks` for reuse by any other Mamba-align request in the same manager. Reset the counter whenever a step actually consumes speculative blocks (i.e. hits the `blocks_allocated` recycling branch and produces a nonzero delta).

Also, replace the two Python-level loops that build the null-block padding (lines 1582-1588) and rotate speculative blocks (lines 1592-1599) with slice assignments: `req_blocks.extend([self._null_block] * (num_skipped_blocks - prev_block_len))` becomes `req_blocks[prev_block_len:prev_block_len] = self._null_slab[:num_skipped_blocks - prev_block_len]` where `_null_slab` is a preallocated list of `_null_block` references sized to the max sequence length in blocks; and the speculative-recycling loop is rewritten as a single `req_blocks.extend(req_blocks[start:end])` followed by `req_blocks[start:end] = self._null_slab[:end-start]`. This removes per-index Python overhead on the hot allocator path.

Tighten `get_num_blocks_to_allocate` (lines ~1500-1530) so its estimate for `blocks_allocated` requests accounts for speculative blocks drawn from `_recycled_spec_blocks` (they are free-list hits, not new pool allocations), keeping the predictor/allocator contract exact and avoiding the drift the candidate description calls out.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any concrete change is novel by default. Beyond that, this proposal targets a specific behavior the candidate only vaguely alludes to ("num_required_blocks might be less than len(req_blocks) if blocks are over-allocated at last round") by introducing an explicit per-manager speculative-block free-list plus a K-step unused-tail reclamation policy — a mechanism, not just a rationale. It also converts the two per-index Python loops into slab-backed slice assignments, a distinct micro-optimization the evolve_rationale mentions abstractly ("replacing per-index nulling with range operations") but does not spell out as a preallocated-slab approach, and it explicitly wires the predictor (`get_num_blocks_to_allocate`) to the new free-list so the two stay consistent. The reclamation angle in particular reduces block-pool pressure for multi-turn agentic workloads where the same request survives many turns with intermittently-unused speculative tails, which directly improves TTFT for follow-up turns by preserving more prefix-cached blocks.

---

### 2. Add a memory-pressure fallback that drops partial-hit CoW preservation
- **Agent:** codex

**Detailed description.**

Change the Mamba align-mode partial-hit path in `MambaManager.allocate_new_blocks` (`vllm/v1/core/single_type_kv_cache_manager.py:1532-1651`) so a partial prefix-cache hit does not always require the extra CoW block when that extra block is the only reason the request cannot be scheduled. Add an explicit fallback mode, wired through `get_num_blocks_to_allocate`/the coordinator, that reports both the preferred allocation count (`base + 1` for CoW preservation) and the minimum allocation count (`base`) for partial-hit requests. If the scheduler has enough blocks for the minimum but not the preferred count, admit the request in fallback mode. In `allocate_new_blocks`, when fallback mode is active for `partial_hit`, remove the source block's prefix-cache hashes instead of moving them to a new `cow_block`, clear any `_producer_partial_tail_reqs` entry for that request, do not append `_pending_cow_copies` or `_pending_partial_tail_offloads`, and let the running request keep/overwrite the existing append-only worker block-table entry. Add focused tests covering: preferred path still preserves/moves hashes and queues the copy/offload; fallback path invalidates the partial hash, allocates one fewer block, schedules the request, and leaves no stale cache entry that a same-step or later request can hit.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal is about speculative-tail lazy allocation, tail reclamation, recycled speculative-block free lists, slice-based null/speculative block movement, and keeping the predictor exact for that free list. This proposal targets a different allocation pressure point: the mandatory extra CoW block for partial prefix-cache hits. It deliberately trades preservation of a partial cache/offload opportunity for immediate admission only under memory pressure, which is not covered by Agent A's speculative-block reclamation mechanism.

---
