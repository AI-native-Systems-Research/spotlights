# ARCCachePolicy.evict

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/policies/arc.py`](vllm/v1/kv_offload/cpu/policies/arc.py) (lines 112–170)
- **Symbol:** `ARCCachePolicy.evict`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_kv_offload-0001`

## Description
ARC batch eviction selects victims from T1/T2, moves evicted keys to B1/B2, and bounds ghost-list sizes.

## Current approach
Uses monotonic T1/T2 iterators and chooses from T1 while virtual_t1_size >= int(target_t1_size), otherwise T2. Entries with ref_cnt > 0 or protected keys are skipped, and ghost lists are trimmed only after a successful batch eviction.

## Estimated impact explanation
Primary-tier hit rate directly affects whether multi-turn agentic prompts reuse CPU KV blocks or pay a secondary-tier promotion. The ARC eviction split is one of the main controls for recency/frequency balance under shifting working sets, so it can move median TTFT and TPOT.

## Evolve rationale
The T1-vs-T2 draw rule, integer flooring of target_t1_size, skip behavior under protected/ref-counted blocks, and ghost-list trim schedule are compact ARC heuristics. Correctness oracle: CachePolicy.evict contract, existing tests in tests/v1/kv_offload, capacity bounds, protected-key exclusion, and ref_cnt == 0 for all returned victims.

## Deep research proposals

### 1. Add workflow-priority-aware skip in ARC evict victim selection
- **Finding:** `find-vllm_v1_kv_offload-0002` — *Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo*
- **Source URL:** <https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend ARCCachePolicy.evict at vllm/v1/kv_offload/cpu/policies/arc.py:112-170 so that the next_candidate scan consults a per-block retention/priority signal in addition to the existing ref_cnt == 0 and `key not in protected` checks. Concretely: (1) carry an optional priority/retention field on BlockStatus (populated during insert/touch from ReqContext, e.g. a workflow-tag or tool-call boundary hint the harness can set), and (2) in next_candidate, treat blocks whose priority exceeds a threshold as soft-protected — skip them on the first pass over the T1/T2 iterator and only fall back to them if no lower-priority victim can be produced for the requested n. Ghost-list bookkeeping (B1/B2) and the T1-vs-T2 draw rule remain unchanged; the change is purely a filter layered on top of the monotonic iterators. This lets an agent harness that already knows which blocks belong to the current tool-call turn keep those from being evicted first, matching the Dynamo pattern of `lower-priority blocks are evicted first` while preserving ARC's adaptive recency/frequency balance for the untagged majority.

**Proposal rationale.**

The candidate's evict picks victims purely from ARC's recency/frequency partitions with only ref_cnt and a hard `protected` set as gating. In a multi-turn agentic workload — the caller's stated objective — blocks belonging to an in-flight tool-call turn or a pinned system-prompt prefix have workflow value that ARC cannot see, so they get evicted at their natural LRU position and must be re-promoted from the secondary tier on the next turn, inflating TTFT. The Dynamo finding contributes exactly this missing signal: a harness-provided retention/priority that biases eviction order without discarding the underlying policy. Layering it as a soft-skip inside next_candidate is a minimal, local change that reuses the existing monotonic-iterator structure and the hard `protected` mechanism as a template, so it fits within the evict method's contract (correct n victims with ref_cnt == 0, protected excluded) while addressing the concrete gap the finding identifies.

---

### 2. Add a TinyLFU frequency-sketch admission gate around ARC eviction/insertion
- **Finding:** `find-vllm_v1_kv_offload-0005` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://paperity.org/p/377179711/tinylfu-a-highly-efficient-cache-admission-policy>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend ARCCachePolicy with an approximate frequency sketch (Count-Min Sketch plus a small doorkeeper bloom filter, aged periodically to preserve recency) that tracks CPU-tier block access frequency. Update the sketch on every `get`/`touch`/`insert` call. When `evict` (vllm/v1/kv_offload/cpu/policies/arc.py:112-170) has chosen a victim for a batch under capacity pressure, expose its estimated frequency; then in `insert` (lines 64-67) — or in the CPU manager path that would call `insert` after `evict` returns a victim — compare the incoming block's sketch frequency against the just-evicted victim's frequency, and skip admission (leave the incoming block un-cached, do not touch T1/B1/B2) when the candidate's frequency is strictly lower than the victim's. Keep ARC's T1/T2/B1/B2 bookkeeping and the adaptive `target_t1_size` unchanged; the admission gate acts only when the cache is full and a real eviction just took place. Provide a small hysteresis (e.g., allow admission on tie or on B1/B2 ghost-hit, since ARC already flags those as high-value) so ghost-list learning still functions.

**Proposal rationale.**

The candidate's current `insert` unconditionally admits every new block into T1, so a burst of one-shot prefixes (common in multi-turn agentic traffic where sibling branches produce many low-reuse blocks) can churn out established frequent entries before ARC's ghost lists get a chance to react. TinyLFU addresses exactly this gap: an approximate-frequency admission gate rejects low-value candidates when they would displace a more valuable victim, reducing eviction churn under capacity pressure and protecting repeated agent prefixes. Because TinyLFU is orthogonal to the recency/frequency split ARC learns via B1/B2, it can be layered on without altering the T1-vs-T2 draw rule, `target_t1_size` update, protected/ref_cnt skip behavior, or ghost-list trim schedule — preserving the correctness oracle in `CachePolicy.evict` and existing tests in tests/v1/kv_offload. Better CPU-tier retention of hot blocks translates directly to fewer secondary-tier promotions, which is the mechanism named in the candidate's estimated_impact for median TTFT/TPOT on multi-turn agentic workloads.

---

## Agent proposals

### 1. Re-queue skipped ref-counted/protected blocks to MRU during ARC evict scan
- **Agent:** claude

**Detailed description.**

In vllm/v1/kv_offload/cpu/policies/arc.py ARCCachePolicy.evict at lines 112-170, change the semantics of next_candidate so that when it encounters an entry with ref_cnt > 0 or key in protected, it not only skips that entry but also lazily moves it to the MRU end of its partition (T1 or T2) via OrderedDict.move_to_end. Two concrete effects: (1) the monotonic t1_iter / t2_iter no longer keep tripping over the same long-lived pinned or in-transfer blocks on every subsequent evict call — a block held with ref_cnt > 0 for a long transfer currently anchors itself at the LRU head and forces every future evict scan to walk past it before reaching real victims, so re-queueing amortizes the scan cost across the pin's lifetime; (2) it aligns LRU semantics with the observation that a block currently being read (ref_cnt > 0) or currently marked protected is by definition in *active* use and therefore behaves like a recent touch, so its LRU position was stale to begin with. Implementation is local: (a) next_candidate becomes a method on the class (or a closure with access to self) and, before advancing past a skipped entry, calls self.t1.move_to_end(key) or self.t2.move_to_end(key) as appropriate; (b) because move_to_end during iteration of an OrderedDict is not safe, materialize the scan cursor by snapshotting the iterator's next key first, then apply move_to_end after the decision (or switch to a two-pass approach: collect skip-worthy keys, requeue them, then continue). Preserve all other invariants — ARC's T1-vs-T2 draw rule keyed on virtual_t1_size, the int(target_t1_size) floor, ghost-list bookkeeping (B1/B2 additions and cache_capacity trim), and the atomic 'n candidates or None' contract. Adaptive learning is unaffected because touch/insert/get paths are untouched and B1/B2 receive only genuinely evicted keys. Add a targeted test in tests/v1/kv_offload asserting that after two successive evict calls where the first N blocks in T1 are pinned via ref_cnt > 0, the second call does not re-scan those N blocks (measurable by wrapping next_candidate or by checking that the pinned blocks have moved to the MRU end after the first call).

**Novelty rationale.**

Neither listed deep_research_proposal addresses the LRU position of skipped blocks or the amortized cost of re-scanning long-lived pinned blocks on every evict call. Proposal 1 (workflow-priority soft-skip) adds a *new* skip predicate on top of the existing ref_cnt/protected filter but leaves skipped blocks at their LRU position, so its soft-protected blocks would themselves become future scan tax. Proposal 2 (TinyLFU admission gate) operates on insert, not evict, and does not touch victim selection order or iterator behavior at all. This proposal is orthogonal to both: it changes only what next_candidate does with an entry it has already decided not to return this call, and it composes cleanly with either of the other two if they are adopted — a priority-skipped or TinyLFU-relevant block would also benefit from being requeued to MRU, so this is an enabling refinement rather than a competing design.

---

### 2. Add cross-partition fallback before ARC evict returns None
- **Agent:** codex

**Detailed description.**

Update `ARCCachePolicy.evict` in `vllm/v1/kv_offload/cpu/policies/arc.py:112-170` so victim selection first tries ARC's preferred partition, but if that partition has no eligible candidate because its iterator is exhausted or all remaining entries are `ref_cnt > 0`/`protected`, it attempts the other partition before returning `None`. Today, when `virtual_t1_size < int(target_t1_size)`, the method goes directly to T2 and returns `None` if T2 has no eligible victim, even if T1 still contains evictable blocks. The symmetric case can also matter after a preferred T1 scan fails. Preserve the current adaptive preference as the first choice, decrement `virtual_t1_size` only when selecting from T1, keep the atomic all-or-nothing mutation behavior, and continue adding actual T1/T2 victims to B1/B2 respectively. Add focused tests that set `target_t1_size` to prefer T2 while T2 is fully protected or ref-counted and T1 has an eligible block, asserting eviction succeeds from T1; include the mirror case where T1 is preferred but only T2 has eligible victims.

**Novelty rationale.**

The deep-research priority proposal adds a new soft skip predicate, but it does not address the case where ARC's preferred partition has no eligible victim while the other partition does. The TinyLFU proposal changes admission around insertion and victim value comparison, not the fallback behavior inside `evict`. Agent A's proposal requeues skipped protected/ref-counted blocks to reduce repeated scan cost, but it still preserves the existing preferred-partition scan shape; this proposal changes the failure semantics so `evict` can make progress whenever any partition has valid victims.

---
