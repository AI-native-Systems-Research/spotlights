# _get_group_hash/group_and_batch_mm_items

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/utils.py`](vllm/multimodal/utils.py) (lines 135–208)
- **Symbol:** `_get_group_hash/group_and_batch_mm_items`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0011`

## Description
Groups consecutive MultiModalKwargsItem objects into compatible batches and reduces each group into batched tensor kwargs.

## Current approach
For every item, group_and_batch_mm_items builds a sorted tuple of (key, _get_group_hash(elem)). _get_group_hash hashes every MultiModalSharedField value with MultiModalHasher.hash_kwargs, then itertools.groupby computes group sizes before a second pass batches each run.

## Estimated impact explanation
This runs while batching multimodal kwargs across requests. Avoiding repeated hashing and tuple materialization reduces scheduling overhead and can improve median TPOT in multi-turn workloads where shared media fields recur across concurrent requests.

## Evolve rationale
The shared-field hash call at line 139 and group_id materialization at lines 187-194 are the optimization unit. Headroom includes memoizing shared-field hashes on the elem or field data for the duration of a request, reusing hashes already computed for the processor cache, and replacing the group_ids list plus groupby pass with a single linear run detector. Oracle: tests/multimodal/test_utils.py must preserve group boundaries, group_size values, and grouped tensor data exactly for mixed field sets and shared-data splits.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Identity-first adjacent comparison to skip hashing shared fields
- **Agent:** claude

**Detailed description.**

Replace the upfront `group_ids` materialization in `group_and_batch_mm_items` (vllm/multimodal/utils.py:187-194) with a single forward scan that decides group boundaries by comparing each item to its immediate predecessor. For each item, first compare the sorted key tuple; if the keys differ, start a new group. If the keys match, walk the shared-field elements and short-circuit on Python object identity (`elem.data is prev_elem.data`) before falling back to `MultiModalHasher.hash_kwargs`. In `_get_group_hash` (lines 135-139), the only purpose of the hash is pairwise equality between adjacent items, so identity checks are sound and strictly cheaper. In multi-turn agentic workloads the scheduler typically batches consecutive `MultiModalKwargsItem` objects sourced from the same request (or from a processor cache hit that returns the same shared `MultiModalSharedField` payload by reference); in those cases `is` succeeds and no tensor hashing occurs at all. Only the inter-request boundary pays for one hash per shared field, and even there the hash can be cached on the `MultiModalFieldElem` (or on `field.data`'s `id`) for the lifetime of the call. The change deletes the second `groupby` pass entirely — group sizes are emitted incrementally as the scan advances start_idx — preserving the (group_size, batched_kwargs) yield contract that tests/multimodal/test_utils.py asserts. Concretely, the rewrite is roughly: maintain `prev_keys` and a parallel list of `prev_shared_data` (only for elems whose field is `MultiModalSharedField`); on mismatch, flush `_batch_mm_items(items[start:i])` and reset. This eliminates the O(n) intermediate `group_ids` list, the per-item sorted tuple allocation pair, and — most importantly — the per-shared-field `hash_kwargs` cost that today runs even for items that are bit-identical by reference.

**Novelty rationale.**

The candidate has no existing deep_research_proposals, but its `evolve_rationale` enumerates three optimization angles: (1) memoizing shared-field hashes on the elem/field, (2) reusing hashes already computed by the processor cache, and (3) replacing the `group_ids` list + `groupby` pass with a linear run detector. This proposal is orthogonal to (1) and (2) because it avoids hashing entirely on the hot path via Python object-identity comparison of `elem.data` against the previous adjacent item — which is sound for the pairwise-equality use of the hash and is the common case when the scheduler batches same-request or processor-cache-hit items by reference. Memoizing or reusing cached hashes still computes/looks up a hash; identity short-circuiting does not. It overlaps with (3) only in that it also fuses the two passes, but the novel core — identity-before-hash for `MultiModalSharedField` — is not mentioned anywhere in the rationale.

---

### 2. Accumulate batch elems during group scanning
- **Agent:** codex

**Detailed description.**

Refactor `group_and_batch_mm_items` in `vllm/multimodal/utils.py` so the grouping pass also builds the reducer input for the current run. Instead of first materializing all `group_ids`, then computing `group_sizes`, then slicing `items[start_idx:start_idx + group_size]` and rewalking every item inside `_batch_mm_items`, maintain `current_group_id`, `current_size`, and a `defaultdict[str, list[MultiModalFieldElem]]` for the active group. For each item, compute the same compatibility signature as today, flush the accumulated elems through a small helper equivalent to `_batch_mm_items` when the signature changes, then append the item’s elems to the new/current accumulator. This preserves exact group boundaries and `reduce_data` behavior while removing the `group_sizes` list, the item slice allocation per group, and the second full traversal of each grouped item’s kwargs. It is especially useful when multimodal items have several kwargs, because every key currently gets visited once for grouping and again for batching.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes an identity-first adjacent comparator to avoid shared-field hashing and mentions deleting the `groupby` pass, but it still flushes by calling `_batch_mm_items(items[start:i])`, which slices the original sequence and walks the group again. This proposal targets that remaining batching-side overhead by accumulating the exact `reduce_data` inputs during the grouping scan; it is orthogonal to whether the group boundary check uses current hashes, cached hashes, or Agent A’s identity short-circuit.

---
