# HybridKVCacheCoordinator.find_longest_cache_hit

[← vllm/v1/core](../vllm_v1_core.md)

- **File:** [`vllm/v1/core/kv_cache_coordinator.py`](vllm/v1/core/kv_cache_coordinator.py) (lines 685–817)
- **Symbol:** `HybridKVCacheCoordinator.find_longest_cache_hit`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_core-0001`

## Description
Fixed-point reconciliation of prefix-cache hit length across heterogeneous KV cache groups including full attention, sliding-window attention, chunked-local attention, and Mamba.

## Current approach
Iterates over sorted attention_groups, invokes each group's classmethod finder, tracks EAGLE verification in a Python set, memoizes the downward-closed full-attention hit, and repeats until curr_hit_length stops shrinking. The simple hybrid case of one full-attention group plus one other group exits after one pass.

## Estimated impact explanation
Hybrid prefix-cache lookup sits on TTFT for newly admitted, resumed, and preempted requests. Multi-turn agent workloads repeatedly submit long shared prefixes, so reducing reconciliation probes directly reduces median TTFT.

## Evolve rationale
Central hybrid prefix-cache admission loop. Headroom remains in attention-group ordering, extending the simple-hybrid short-circuit to larger converged group sets, caching per-group results keyed by candidate hit length, and reducing Python traversal overhead. Correctness oracle: existing hybrid cache-hit tests plus invariants that returned hit_length is monotonically non-increasing across iterations, alignment-valid, and accepted by every group at return.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Reorder attention groups by empirical shrink probability and skip groups that already accepted the current candidate length
- **Agent:** claude

**Detailed description.**

Modify HybridKVCacheCoordinator.find_longest_cache_hit (vllm/v1/core/kv_cache_coordinator.py:685-817) to (1) statically reorder self.attention_groups once at construction so the group most likely to shrink the candidate (typically sliding-window or chunked-local, which have strictly narrower admissibility than full-attention) is probed FIRST after the downward-closed full-attention lookup, and (2) skip re-probing any non-full-attention group in subsequent fixed-point iterations when hit_length_by_group[first_group_id] == curr_hit_length AND that group's finder is monotone in max_length (which holds for sliding-window, chunked-local, and Mamba finders whose acceptance set at length L is a superset of their acceptance at L' > L). Concretely: after the full-attention short-circuit block (line 738-745), add an analogous downward-closed skip for groups tagged as monotone via a new manager_cls.hit_is_downward_closed classvar (True for FullAttention/SlidingWindow/ChunkedLocal/Mamba finders, since a hit at length L trivially implies a hit at any L' <= L that is alignment-valid). Reordering means the first pass converges to the true fixed point in most hybrid configurations (e.g., Gemma-2, Llama-4-scout sliding+full, Jamba full+mamba) so the outer while loop exits after one iteration for the general N-group case, not just the current 2-group is_simple_hybrid short-circuit at line 795. Preserve existing invariants: EAGLE verification set is cleared whenever any group actually shrinks curr_hit_length; the trailing full-attention truncation at lines 798-808 is unchanged; num_uncached_common_prefix_tokens computation at line 813 is unchanged because longest_hit_length is still tracked across all probes.

**Novelty rationale.**

There are no deep_research_proposals listed on this candidate, so any concrete idea is novel by construction. The proposal is distinct from the candidate's own evolve_rationale hints (which mention 'attention-group ordering' and 'extending the simple-hybrid short-circuit' only abstractly) because it specifies (a) a static ordering derived from a per-manager-class monotonicity/downward-closedness flag rather than a runtime heuristic, and (b) a per-iteration skip predicate keyed on hit_length_by_group equality that generalizes the existing full-attention short-circuit at line 738-745 to every monotone group, rather than caching per-group results by candidate length as the rationale suggested.

---

### 2. Split hybrid lookup into length-only reconciliation and one final block materialization pass
- **Agent:** codex

**Detailed description.**

Refactor `HybridKVCacheCoordinator.find_longest_cache_hit` in `vllm/v1/core/kv_cache_coordinator.py:685-817` so the fixed-point loop asks each manager for only the candidate hit length during reconciliation, then materializes `KVCacheBlock` lists exactly once after the final `hit_length` is known. Concretely, add a lightweight manager path such as `find_longest_cache_hit_length(...)` or a `materialize_blocks=False` option that runs the same hash probes but avoids constructing and mutating `computed_blocks` lists on provisional candidates that may be discarded by a later group. The coordinator would keep `hit_length_by_group` and `longest_hit_length` in the loop, including the existing EAGLE margin/drop semantics, then perform one final per-group materializing call at the reconciled boundary (using the same EAGLE drop behavior and asserting the returned length still accepts the final boundary). This targets long multi-turn prefixes where sliding-window, chunked-local, and Mamba finders currently allocate `[null_block] * max_num_blocks`, append cached blocks, and later overwrite or truncate those lists across fixed-point iterations even though only the scalar length is needed until convergence.

**Novelty rationale.**

There are no listed deep_research_proposals, so this does not duplicate one. It is also distinct from agent A's proposal: agent A changes group ordering and skips re-probes for monotone/downward-closed groups, while this proposal keeps the reconciliation order/skip policy unchanged and reduces per-probe allocation and Python list work by separating scalar length discovery from final block-list construction.

---
