# ARCCachePolicy.evict

[← v1.kv_offload](../v1.kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/policies/arc.py`](vllm/v1/kv_offload/cpu/policies/arc.py) (lines 97–156)
- **Symbol:** `ARCCachePolicy.evict`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0005`

## Description
ARC eviction routine that atomically selects n victims, choosing T1 or T2 from the adaptive target, skipping protected/ref-counted entries, moving selected keys to ghost lists, and trimming ghosts to cache_capacity.

## Current approach
For each requested victim it restarts an OrderedDict scan over self.t1.items() or self.t2.items() and filters out keys with ref_cnt > 0, keys in protected, or keys already selected in the same batch. This rescans the same rejected entries n times. Ghost-list trimming is a popitem(last=False) loop after successful selection.

## Estimated impact explanation
Large CPU offload pools and burst stores can put this scan on the scheduler prepare_store path. Faster victim selection reduces scheduler stalls, improving TPOT when many requests are storing new KV blocks while old blocks are still protected.

## Evolve rationale
Batched eviction is O(n * cache_size) when many entries are protected, referenced, or already selected. Headroom includes carrying scan cursors across victim selection inside the n-loop, maintaining a separate eviction-eligible queue keyed off ref_cnt transitions in CPUOffloadingManager.complete_load, or changing victim selection while preserving ARC partition semantics. Oracles are CachePolicy.evict's atomic exact-n-or-None contract, ref_cnt/protected-key safety, ghost-list bounds, and existing ARC tests for target_t1_size adaptation and T1/T2/B1/B2 transitions.

## Deep research proposals

### 1. Add S3-FIFO probationary filter in front of ARC to cut eviction scan cost on one-hit KV blocks
- **Finding:** `find-0008` — *FIFO Queues are All You Need for Cache Eviction*
- **Source URL:** <https://jasony.me/publication/sosp23-s3fifo.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Augment ARCCachePolicy in vllm/v1/kv_offload/cpu/policies/arc.py (evict at lines 97-156, plus touch/insert paths) with an S3-FIFO-style small probationary FIFO that sits in front of T1/T2. New blocks enter a small FIFO (sized as a small fraction of cache_capacity, e.g. 10%); only blocks that receive a second hit while in the FIFO are promoted into ARC's T1. Blocks evicted from the small FIFO without a second hit are demoted directly (optionally recorded in a ghost FIFO mirroring B1) without ever touching T1/T2. Concretely: (1) add a small FIFO queue and per-entry hit bit alongside self.t1/self.t2; (2) on insert, place the key in the small FIFO instead of T1; (3) on touch, set the hit bit if in small FIFO, otherwise apply existing T1->T2 promotion; (4) in evict, first attempt to drain non-promoted, non-protected, ref_cnt==0 victims from the small FIFO before scanning T1/T2. The atomic exact-n-or-None contract, ref_cnt/protected safety, and ghost-list bounds are preserved by performing the same two-phase select-then-commit pattern on the new FIFO. ARC's target_t1_size adaptation and B1/B2 transitions are unchanged for keys that reach T1/T2.

**Proposal rationale.**

The candidate's hot path is a per-victim rescan of T1/T2 that repeatedly skips ref-counted, protected, or already-selected entries; eligible victims are sparse precisely when the cache is full of one-hit prefill/long-context blocks held briefly by in-flight loads. S3-FIFO's central claim - a small FIFO filters most one-hit objects before they reach the main cache - directly addresses that gap: most short-lived KV blocks would be evicted from the small FIFO in O(1) FIFO order without ever polluting T1/T2 or competing with reusable prefixes. This preserves ARC's adaptive partitioning for the blocks that matter (multi-turn reusable prefixes, matching the stated multi-turn agentic workload) while reducing the n*cache_size worst case in evict, which is the explicit headroom called out in evolve_rationale and the medium-impact prepare_store scheduler stall described in estimated_impact_explanation.

---

## Agent proposals

### 1. Maintain ref-cnt-driven evictable LRU indexes for T1/T2 to make ARC.evict O(n)
- **Agent:** claude

**Detailed description.**

In vllm/v1/kv_offload/cpu/policies/arc.py (ARCCachePolicy.evict, lines 97-156), replace the per-victim full-scan of self.t1.items()/self.t2.items() with two auxiliary OrderedDicts — self.evictable_t1 and self.evictable_t2 — that mirror T1/T2 in LRU order but only contain keys whose BlockStatus.ref_cnt == 0. These evictable indexes are kept in sync at exactly the points where ref_cnt and partition membership change: (1) insert() adds to evictable_t1 only if ref_cnt == 0; (2) touch() promotes/moves the entry's mirror to the corresponding evictable index when partition or recency changes (preserving LRU ordering); (3) remove() removes from whichever evictable index holds it; (4) BlockStatus gains a thin ref_cnt API (acquire/release helpers) so that the manager — CPUOffloadingManager.complete_load and prepare_load/store call sites that currently mutate ref_cnt directly — calls into the policy on the 0->1 (remove from evictable index) and 1->0 (re-insert at the LRU tail) transitions. evict() then becomes: while we still need victims, iterate evictable_t<n> first (skipping protected and already_selected, both small batch-local sets); fall back to the other partition exactly as today when the adaptive target says so. The atomic exact-n-or-None contract is preserved by still building a candidate list before mutation, ghost-list bounds and B1/B2 transitions are unchanged, and ARC's target_t1_size adaptation is untouched. Worst case drops from O(n * cache_size) to O(n + |protected|) per evict(). Tests to keep green: existing ARC partition/ghost tests, plus new tests asserting that pinned (ref_cnt>0) and protected keys are never returned, that ref_cnt 1->0 reinserts at the LRU tail (not MRU), and that a churn workload with many pinned blocks no longer scales scan cost with cache size.

**Novelty rationale.**

The single existing deep_research_proposal (find-0008) adds an S3-FIFO probationary queue in front of T1/T2 to filter one-hit blocks by *access pattern*; it still scans T1/T2 linearly when victims must come from the main partitions, and it does not change how ineligible (ref_cnt>0 or protected) entries are handled inside that scan. This proposal is orthogonal: it attacks the eligibility-skip cost directly by maintaining a ref_cnt-indexed evictable LRU mirror, exactly the 'separate eviction-eligible queue keyed off ref_cnt transitions in CPUOffloadingManager.complete_load' direction called out in evolve_rationale but not picked up by find-0008. The two changes compose — S3-FIFO would reduce pollution; this would make whatever scans remain O(n) — but neither subsumes the other.

---

### 2. Reuse per-call scan cursors across ARC batch victim selection
- **Agent:** codex

**Detailed description.**

Change `ARCCachePolicy.evict` in `vllm/v1/kv_offload/cpu/policies/arc.py` so the existing per-victim ARC decision loop keeps local iterators over `self.t1.items()` and `self.t2.items()` for the duration of one `evict(n)` call. Preserve the current `virtual_t1_size >= int(self.target_t1_size)` branch exactly, but replace each fresh `for key, block in self.t1.items()` / `self.t2.items()` scan with helper closures such as `next_t1_candidate()` and `next_t2_candidate()` that resume from the last examined entry and return the next key whose `ref_cnt == 0` and key is not protected. Because the dictionaries are not mutated until after all `n` victims are selected, these iterators can safely skip already-selected entries by construction. If the requested `n` victims cannot be found, return `None` without mutating T1/T2/B1/B2 as today; if all are found, commit the same T1->B1 and T2->B2 moves and ghost trimming. This keeps ARC partition behavior and the exact-n-or-None contract unchanged while reducing repeated rejected-entry scans inside one batch from O(n * cache_size) toward O(cache_size + n).

**Novelty rationale.**

The deep_research proposal adds an S3-FIFO admission tier before ARC, which changes how one-hit blocks enter the cache but does not address repeated scans inside `ARCCachePolicy.evict`. Agent A's proposal adds persistent evictable LRU indexes synchronized with ref_cnt transitions across the manager and policy. This proposal is narrower and distinct: it only introduces per-call local cursors inside the existing `evict` implementation, with no admission-policy change, no auxiliary policy state, and no ref_cnt callback plumbing.

---
