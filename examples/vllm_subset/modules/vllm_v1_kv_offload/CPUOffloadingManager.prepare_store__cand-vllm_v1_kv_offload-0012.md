# CPUOffloadingManager.prepare_store

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/manager.py`](vllm/v1/kv_offload/cpu/manager.py) (lines 166–236)
- **Symbol:** `CPUOffloadingManager.prepare_store`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0012`

## Description
Prepares CPU-tier stores by applying admission filters, computing eviction needs, evicting victims, allocating blocks, and inserting pending writes.

## Current approach
Optionally filters by a fixed store_threshold, then filters already-stored keys, computes an all-or-nothing eviction budget, asks the cache policy for exactly that many victims, and returns None if the whole batch cannot fit.

## Estimated impact explanation
Better store admission can avoid wasting primary capacity and reduce all-or-nothing store failures. It improves future primary hit rate and lowers promotion pressure, behind but complementary to the eviction policies.

## Evolve rationale
Admission, filter fusion, partial acceptance under eviction pressure, and score-based store decisions are local policy levers. Correctness oracle: PrepareStoreOutput invariants, existing CPU manager tests, keys_to_store subset of requested keys, capacity bounds, and evicted blocks being idle.

## Deep research proposals

### 1. Workflow-value-aware admission and partial-batch store in prepare_store
- **Finding:** `find-vllm_v1_kv_offload-0002` — *Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo*
- **Source URL:** <https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend `CPUOffloadingManager.prepare_store` (vllm/v1/kv_offload/cpu/manager.py:166-236) to consult per-request workflow metadata carried on `req_context` (RequestOffloadingContext), so admission and eviction decisions reflect predicted future value rather than only ref-count recency. Concretely: (1) Admission — replace the single fixed `store_threshold` gate at line 173 with a per-key admission score that combines the existing count-based prior with request-provided hints (e.g. `retain=True` for prefixes marked as reusable across tool calls in a multi-turn agent, `prefetch_priority` for blocks the harness expects to reload after a tool call, `store_priority` scalar). Keys below the score cut-off are dropped from `keys_to_store` and counted in `stores_skipped_in_current_batch`; keys marked `retain` bypass the count threshold entirely. (2) Retention on eviction — plumb a retention/pin signal from `req_context` through to the `protected` set built at line 198, so blocks the harness flagged as high-value are excluded from `self._policy.evict(...)` victim candidates in addition to the current in-batch protection. (3) Partial acceptance — when `num_blocks_to_evict > self._num_evictable_cache_blocks` (line 190) or `self._policy.evict` returns `None` (line 200), instead of returning `None` for the whole batch, rank `keys_to_store` by the admission score, drop the lowest-scored tail until the remaining set fits within `_num_evictable_cache_blocks` (and the policy can produce that many victims), and proceed with the trimmed subset — preserving the `PrepareStoreOutput` invariants (`keys_to_store` remains a subset of the caller's requested keys, capacity bounds hold, and evicted blocks are still idle policy-selected victims). Existing behavior is the default when no metadata is supplied (score falls back to the count-based prior; empty retention set; all-or-nothing when scores are uniform), keeping current CPU manager tests unchanged.

**Proposal rationale.**

The candidate's `evolve_rationale` explicitly names admission, partial acceptance under eviction pressure, and score-based store decisions as the local policy levers to evolve, and the caller context is a multi-turn agentic workload — precisely the setting the Dynamo blog targets. The finding contributes three concrete, transferable ideas the current code lacks: request-metadata-driven admission (today only a fixed integer ref-count threshold), retention/pin hints to bias eviction away from high-value blocks (today the `protected` set only reflects the current input batch), and priority-aware partial acceptance so an agent's high-value prefixes are still admitted when the full batch would fail eviction (today `prepare_store` returns `None` and stores nothing). `prepare_store` already receives `req_context`, so the metadata channel exists without an interface change, and the correctness oracle listed on the candidate (PrepareStoreOutput invariants, keys_to_store subset, capacity bounds, evicted blocks idle) is preserved by the proposed changes.

---

### 2. TinyLFU-based admission with per-candidate victim comparison in prepare_store
- **Finding:** `find-vllm_v1_kv_offload-0005` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://paperity.org/p/377179711/tinylfu-a-highly-efficient-cache-admission-policy>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the current fixed-threshold admission gate in CPUOffloadingManager.prepare_store (vllm/v1/kv_offload/cpu/manager.py:166-236) with a TinyLFU-style admission decision that uses an approximate frequency sketch instead of the exact OrderedDict `self.counts` tracker. Concretely:

1. Add a Count-Min-sketch-with-aging frequency estimator (small fixed-size 4-bit counters plus a periodic halving/reset step keyed off a sample counter) as an alternative/supplement to `self.counts`. Update it on every `lookup` (lines 112-127) instead of the OrderedDict, giving bounded memory beyond `max_tracker_size` and cheap O(1) frequency estimates for arbitrary keys.
2. In `prepare_store`, after computing `keys_to_store` (line 176) and determining `num_blocks_to_evict` (line 186), when eviction is required (line 189) query the cache policy for candidate victims and compare each victim's estimated frequency against the corresponding incoming candidate's estimated frequency. Reject (drop from `keys_to_store`) any incoming candidate whose sketch-estimated frequency is not strictly greater than the victim it would displace. Count rejected keys via `stores_skipped_in_current_batch`.
3. This enables partial acceptance: instead of returning `None` when eviction can't cover the whole batch (line 192 / line 201), the admission gate shrinks `keys_to_store` to the subset worth admitting under the current victim frequency distribution, then re-runs eviction/allocation on the reduced set. Preserve the existing PrepareStoreOutput invariants (`keys_to_store` subset of requested keys, capacity bounds, evicted blocks idle) and the existing `protected = set(keys)` guard (line 198) so already-cached blocks are not evicted.
4. Expose the sketch width/depth and aging period as constructor args alongside `store_threshold` / `max_tracker_size` (lines 48-49), and keep `store_threshold` as a hard floor applied before the TinyLFU comparison so the existing behavior is a special case (threshold=1, sketch disabled).

**Proposal rationale.**

The candidate's `evolve_rationale` explicitly names admission, filter fusion, partial acceptance under eviction pressure, and score-based store decisions as the local policy levers to exercise, and its `current_approach` is a fixed `store_threshold` gate (line 173) plus an all-or-nothing eviction budget (lines 189-192). TinyLFU's core contribution — using a compact frequency sketch to compare an incoming item's recent frequency against the victim's before admission — maps directly onto both gaps: it replaces the fixed threshold with a victim-relative score, and turning the all-or-nothing eviction into a per-key admission decision naturally unlocks partial acceptance. For the multi-turn agentic workload noted in caller context, repeated shared prefixes create long-tail high-frequency blocks that a sketch-based admission gate can preferentially keep resident in the CPU tier, reducing store thrash under capacity pressure and improving future primary hit rate — exactly the medium-impact TTFT/TPOT lever described in `estimated_impact_explanation`. The finding contributes a concrete algorithm (approximate frequency sketch + admission comparison) not currently present; the existing `counts` OrderedDict is a simplified precursor, so this is a substantive extension rather than a restatement.

---

## Agent proposals

### 1. Prefix-contiguity-aware admission and tail-trim partial acceptance in prepare_store
- **Agent:** claude

**Detailed description.**

Extend `CPUOffloadingManager.prepare_store` (vllm/v1/kv_offload/cpu/manager.py:166-236) to treat the `keys` argument as an ordered prefix chain rather than an unordered bag, and add two structural admission rules on top of the existing scalar `store_threshold` gate:

1. **Contiguity gate.** After the already-stored filter at line 176 produces `keys_to_store`, walk the caller-provided key sequence in order and drop any candidate whose immediate prefix predecessor in the chain is neither already stored (`self._policy.get(prev) is not None`) nor itself an admitted earlier entry in `keys_to_store`. This eliminates "holes" where an interior or tail block would be stored behind an unstored predecessor — such blocks are unreachable on lookup (the connector's left-to-right prefix scan stops at the first miss) and merely waste primary capacity. Increment `stores_skipped_in_current_batch` by the number dropped. Ordering is derived from the existing `OffloadKey` chain relation (each key already carries its prefix hash / position in the sequence used at line 176), so no new signature is needed.

2. **Tail-trim partial acceptance.** Replace the all-or-nothing early return at line 192 / line 201 with a chain-tail trim: when `num_blocks_to_evict > self._num_evictable_cache_blocks` or `self._policy.evict(num_blocks_to_evict, protected)` returns `None`, iteratively drop the *last* entry of `keys_to_store` (the deepest tail block in the prefix chain) and recompute `num_blocks_to_evict` until either the batch fits within `_num_evictable_cache_blocks` and the policy can supply that many victims, or the admitted set is empty (fall back to returning the empty-output branch at lines 178-183). Trimming from the tail preserves the reachable prefix, so a partial admission still yields useful lookup hits on the next turn, unlike a random-subset trim which can strand mid-chain blocks.

3. **Invariants preserved.** `keys_to_store` remains a subset of the caller's requested keys, `_num_evictable_cache_blocks` bookkeeping is unchanged (still driven by `self._policy.evict`'s idle-victim guarantee), the `protected = set(keys)` guard at line 198 is unchanged, and the empty-batch branch at lines 178-183 is reused. When the input is a single contiguous prefix chain (the common case), the contiguity gate is a no-op and behavior matches the current implementation apart from partial acceptance. Existing CPU manager tests remain green because they exercise contiguous single-request inputs.

Surface a single knob on `__init__` (e.g. `partial_accept: bool = False`) that gates the tail-trim behavior so operators can opt in without changing the current all-or-nothing default.

**Novelty rationale.**

The two listed deep_research_proposals both operate on *per-key scalar scores* — one uses workflow/retention metadata from `req_context`, the other uses a TinyLFU count-min-sketch frequency estimate — and their partial-acceptance mechanism ranks/rejects keys individually by that scalar. Neither observes that KV blocks form an ordered prefix chain and that admission decisions on such a chain are structurally coupled: a stored tail block behind an unstored predecessor is unreachable on lookup and is dead storage regardless of its scalar score. This proposal contributes two structural admission rules absent from both prior proposals — a contiguity gate that drops orphaned interior/tail candidates, and a chain-tail trim that preserves the reachable prefix under partial acceptance. It is orthogonal to and composes with either existing proposal (workflow-priority or TinyLFU can select *which* chain to keep; contiguity-awareness selects *what fraction of that chain* is worth keeping), so it is not covered by, subsumed by, or a restatement of either finding.

---

### 2. Add group-completeness admission for multi-KV-group chunks
- **Agent:** codex

**Detailed description.**

Extend `CPUOffloadingManager.prepare_store` in `vllm/v1/kv_offload/cpu/manager.py:166-236` with a structural filter that treats all `OffloadKey`s sharing the same block hash as one logical offloaded chunk across KV cache groups. After the existing `store_threshold` gate and before computing `num_blocks_to_evict`, group the original input keys by `get_offload_block_hash(key)` and record the group IDs present in the caller's batch with `get_offload_group_idx(key)`. For each hash, keep newly admitted keys only if every group ID from the original batch for that hash is either already present in `self._policy` or remains in `keys_to_store`; otherwise drop the newly admitted siblings for that hash and increment `stores_skipped_in_current_batch`. This prevents storing a chunk for only some KV groups when the lookup path later needs all relevant groups to agree before it can raise the request's loadable token boundary. Add a focused CPU manager test that submits two group keys with the same block hash, forces one group below the threshold or otherwise filtered out, and asserts the other group is skipped rather than allocated; also cover the case where the missing sibling is already stored so the new sibling is still accepted.

**Novelty rationale.**

The deep research proposals are scalar admission policies: workflow metadata scoring, retention hints, TinyLFU victim comparison, and score-ranked partial acceptance. Agent A's proposal is prefix-chain structural admission: avoid holes in the ordered token prefix and trim from the chain tail. This proposal is a different structural invariant over KV cache groups for the same logical chunk hash. It does not choose keys by score and does not reason about predecessor/successor prefix reachability; it prevents cross-group partial stores that cannot improve complete-chunk or partial-tail lookup unless the sibling groups are also available.

---
