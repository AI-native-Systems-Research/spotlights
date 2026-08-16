# MambaManager.reachable_block_mask

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py) (lines 1359–1414)
- **Symbol:** `MambaManager.reachable_block_mask`
- **Kind:** config_block
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_core-0022`

## Description
Mamba sparse state-snapshot retention mask that selects segment-boundary and reachable-boundary recurrent-state blocks to register in the prefix cache.

## Current approach
Returns dense caching when retention is disabled, otherwise builds a boolean mask, marks one state per retention segment, and always marks replay/shared-prefix boundary states.

## Estimated impact explanation
Scoped to Mamba models, but recurrent-state retention strongly affects reuse for long multi-turn histories. Better snapshot selection improves prefix hit rate and reduces TTFT on later agent turns.

## Evolve rationale
Mamba-side runtime heuristic controlled by retention_interval. Headroom in adaptive per-depth retention, connector-aware boundary selection, and lower-overhead mask generation for long prompts. Correctness oracle: all reachable_boundaries must map to retained boundary blocks, retention_interval <= block_size must be equivalent to dense retention, and find_longest_cache_hit must never depend on a masked-out state for a legal retained boundary.

## Deep research proposals

### 1. Workflow-aware boundary prioritization for Mamba sparse retention
- **Finding:** `find-vllm_v1_core-0001` — *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows*
- **Source URL:** <https://www.alphaxiv.org/abs/2507.07400>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend MambaManager.reachable_block_mask (vllm/v1/core/single_type_kv_cache_manager.py:1359-1414) to accept prioritized reachable boundaries rather than a flat Sequence[int]. Concretely, change the `reachable_boundaries` parameter (or add a companion parameter) to carry a per-boundary steps-to-execution priority derived from the agent workflow, and use those priorities to guide retention when the state-block budget is tight. Two composable changes: (1) When retention_interval > 0 and the number of unique reachable boundary blocks exceeds a budget, keep boundaries with the lowest steps-to-execution first (soon-to-be-reused) and demote or drop far-future boundaries — instead of the current unconditional 'always mark every reachable boundary'. (2) Bias segment-boundary retention (the per_segment loop at lines 1398-1402) toward segments that contain or lead into a high-priority boundary, so nearby state snapshots that will feed an imminent replay/junction survive eviction. The prioritization signal must flow in from the caller (find_longest_cache_hit / prefix cache manager) that already knows the request's replay boundary and shared-prefix junctions; extending that plumbing to carry a steps-to-execution weight is the main integration point. Retain the existing invariants stated in evolve_rationale: retention_interval <= block_size still returns None (dense), and any boundary that ends up masked out must not be depended on by find_longest_cache_hit.

**Proposal rationale.**

KVFlow's core mechanism — assign each cache node a steps-to-execution priority derived from an Agent Step Graph and preferentially retain soon-needed prefixes — maps directly onto this candidate's already-existing `reachable_boundaries` concept. The candidate's evolve_rationale explicitly identifies 'adaptive per-depth retention' and 'connector-aware boundary selection' as headroom, and the caller objective is median TTFT/TPOT on a multi-turn agentic workload — exactly the regime KVFlow targets (paused agent sessions whose recurrent state is evicted before reuse). Current code treats all reachable boundaries as equal (all always retained) and treats all segment boundaries as equal (fixed stride); under memory pressure that spreads retention thinly across far-future reuse points and dilutes the benefit for imminent turns. Adding a workflow-derived priority into the mask decision lets the same retention budget concentrate on state blocks whose reuse is next, improving prefix hit rate on the next agent turn and reducing TTFT, without changing the correctness oracle (retained set is a strict subset of the current retained set plus prioritized ordering).

---

### 2. Score Mamba state retention by predicted-reuse × compute-savings-per-byte (Marconi-style)
- **Finding:** `find-vllm_v1_core-0004` — *Marconi: Prefix Caching for the Era of Hybrid LLMs*
- **Source URL:** <https://huggingface.co/papers/2411.19379>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend MambaManager.reachable_block_mask in vllm/v1/core/single_type_kv_cache_manager.py (lines 1359-1414) to replace the fixed-stride segment mask with a Marconi-style utility score. Instead of unconditionally marking one state per retention_interval-sized segment plus the reachable boundaries, compute for each candidate boundary block a score = predicted_reuse_likelihood * compute_savings / memory_footprint, and keep blocks whose score exceeds a threshold (or top-K under a global byte budget). Concretely: (a) keep the existing reachable_boundaries as guaranteed-admit (they are proven reuse points, so their predicted_reuse is 1.0); (b) for interior segment boundaries, derive predicted_reuse from lightweight signals already available at retention-mask construction time (e.g., depth in the request's prompt, distance to the nearest reachable boundary, and, if plumbable, a hit-count or last-hit-age from the prefix cache's block metadata for the corresponding attention prefix hash); (c) compute compute_savings as the token count that would be replayed from this state on a hit (i.e., block_size * reuse_likelihood-weighted tail length); (d) divide by the constant per-block state size to get savings-per-byte. Retain blocks with the highest scores until either the retention_interval budget or an explicit byte cap is met. When retention_interval is None keep the current dense-return short-circuit; when it is 0 keep only reachable_boundaries as today. Preserve the correctness oracle: every entry in reachable_boundaries must still be marked True, and retention_interval <= block_size must still fall through to dense caching.

**Proposal rationale.**

The candidate's own comment already labels the reachable-boundary policy 'Marconi-style APC', but the current mask selects segment states by uniform stride only, ignoring the paper's core contribution: admission/eviction by predicted reuse and compute-savings-per-memory-footprint. Under the caller's multi-turn agentic workload, recurrent-state snapshots are large and unevenly valuable — states near frequently reused prefixes should be retained even if they fall off a uniform stride, and states in low-reuse regions can be dropped to free budget. Replacing the uniform-stride branch with a utility score directly addresses the candidate's evolve_rationale gap around 'adaptive per-depth retention' and 'connector-aware boundary selection', and plausibly improves prefix hit rate (thus median TTFT and TPOT on later turns) without changing correctness because reachable_boundaries remain guaranteed-admit.

---

### 3. Add connector-aware commit policy to Mamba retention mask selection
- **Finding:** `find-vllm_v1_core-0008` — *[RFC]: Semantic KV Cache Reuse Interface*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/44223>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend MambaManager.reachable_block_mask in vllm/v1/core/single_type_kv_cache_manager.py (lines 1359-1414) to consult a per-request cache-commit policy (e.g. EXACT_COMMIT vs REQUEST_ONLY) when constructing the retained-state boolean mask. Recurrent-state blocks whose provenance is an external/approximate donor (semantic reuse or connector-provided KV) would be marked reachable for the current request path but excluded from the retention mask that registers boundary snapshots into the exact prefix-cache map. Concretely: pass the commit-policy signal (or an is_exact_source predicate) into reachable_block_mask alongside reachable_boundaries and retention_interval; when marking segment-boundary and replay/shared-prefix boundary states, retain only those whose source is EXACT_COMMIT, and treat REQUEST_ONLY boundaries as usable-but-not-registered. Keep the existing invariants intact: retention_interval <= block_size remains equivalent to dense retention for exact sources, and all reachable_boundaries still map to a usable block for find_longest_cache_hit.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out headroom in connector-aware boundary selection, and its correctness oracle requires that find_longest_cache_hit never depend on a masked-out state for a legal retained boundary. The RFC's cache-commit policy is the missing upstream primitive that lets the mask distinguish 'reachable for this request' from 'safe to commit to the exact prefix-cache map.' For multi-turn agentic workloads, connector-provided or semantic donor states could contribute to prefix hits on later turns (reducing TTFT) without polluting the exact-match cache, which is exactly the safety gap the finding identifies. This is a concrete, transferable idea: thread a policy bit through the mask construction rather than treating every reachable boundary as commit-worthy.

---

## Agent proposals

### 1. Co-residency-gated Mamba retention: only keep boundary states whose attention block is (or will remain) cached
- **Agent:** claude

**Detailed description.**

Modify MambaManager.reachable_block_mask in vllm/v1/core/single_type_kv_cache_manager.py:1359-1414 so that segment-boundary retention consults the sibling attention group's prefix-cache residency at the same block index. Rationale grounded in the file itself: find_longest_cache_hit at lines 1332-1354 walks block_hashes right-to-left and requires block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids) to return a hit on the attention side before the Mamba state at boundary i can be used; a retained recurrent snapshot at a block index whose attention counterpart has been evicted is dead weight that still costs a state-block slot. Concretely: (a) thread an optional co_resident_predicate: Callable[[int], bool] (or a precomputed bit-vector aligned to [start_block, end_block)) into reachable_block_mask, supplied by the caller that already holds the block_pool and the request's block_hashes for the attention group; (b) in the segment-boundary loop at lines 1398-1402, before setting mask[i] = True for a per_segment stride hit, test the predicate for that absolute block index and skip when it returns False; (c) leave reachable_boundaries at lines 1408-1412 untouched — they are proven reuse points and must remain guaranteed-admit per the correctness oracle; (d) when the predicate is None, preserve today's behavior byte-for-byte; (e) when the predicate signals that a stride slot is not co-resident, advance to the next block within the same segment that is co-resident (short bounded scan of at most per_segment - 1 blocks) so the segment still contributes one usable snapshot when any exists — this keeps the 'one state per segment' invariant when possible without spending the state slot on a boundary that cannot produce a hit. Preserve the retention_interval None / <= block_size dense short-circuits at lines 1381-1383 and 1395-1397 unchanged. The predicate can be implemented cheaply by the caller as a lookup against the attention group's already-computed block_hashes plus block_pool.get_cached_block, or as an approximate residency bit sourced from the attention group's LRU metadata; the mask code stays agnostic to which.

**Novelty rationale.**

None of the three deep_research_proposals address the coupling between the Mamba retention mask and the attention prefix cache's own residency at the same block index. Proposal 1 (KVFlow) prioritizes by workflow steps-to-execution — a forward-looking signal from the caller's step graph, not a check against the attention cache's current state. Proposal 2 (Marconi) scores boundaries by a utility function of predicted_reuse × compute_savings / bytes; its 'hit-count or last-hit-age from the prefix cache' is a scalar reuse-likelihood input, not a per-block co-residency gate that reflects the hard requirement that find_longest_cache_hit needs an attention-side hit at index i for the Mamba state at i to be usable. Proposal 3 (connector commit policy) filters by provenance (EXACT vs REQUEST_ONLY) — a source-quality signal orthogonal to whether the attention counterpart is currently resident. This proposal directly encodes the file-level invariant visible at lines 1332-1354: a retained Mamba boundary is only reachable when its paired attention block is a hit, and it composes cleanly with any of the three existing proposals (priority ordering, utility scoring, or commit filtering can all run on top of the co-residency-gated candidate set).

---

### 2. Use a sparse retained-index representation for Mamba retention masks
- **Agent:** codex

**Detailed description.**

Change `MambaManager.reachable_block_mask` in `vllm/v1/core/single_type_kv_cache_manager.py:1359-1414` so sparse retention does not always allocate and fill a dense `list[bool]` of length `end_block - start_block`. Introduce a compact representation for sparse mode, for example a small `RetainedBlockMask` dataclass containing `start_block`, `end_block`, `per_segment` or explicit retained relative indices, plus `__contains__`/iteration helpers. For `retention_interval == 0`, return only the reachable boundary indices; for `retention_interval > block_size`, represent the periodic segment-boundary pattern analytically and union in reachable boundaries. Keep `None` as the dense-cache sentinel for `retention_interval is None`, `alignment_tokens is None`, and `per_segment <= 1`. Update the immediate callers that currently consume `list[bool] | None` to accept the sparse mask abstraction when deciding which Mamba state blocks to register in the prefix cache. This preserves the selected retained blocks exactly, including guaranteed reachable boundaries, while avoiding O(number_of_prompt_blocks) allocation and writes for long multi-turn histories where only a small periodic subset is retained.

**Novelty rationale.**

The existing deep_research_proposals change which blocks are retained: workflow priority, Marconi-style utility scoring, or connector commit filtering. Agent A's proposal also changes selection by gating retained Mamba states on attention-cache co-residency. This proposal is orthogonal: it keeps the current retention policy byte-for-byte but changes the mask representation and caller consumption to remove dense-mask construction overhead for long prompts. It directly targets the candidate's lower-overhead mask-generation headroom without overlapping the prior proposals' prioritization, scoring, provenance, or residency ideas.

---
