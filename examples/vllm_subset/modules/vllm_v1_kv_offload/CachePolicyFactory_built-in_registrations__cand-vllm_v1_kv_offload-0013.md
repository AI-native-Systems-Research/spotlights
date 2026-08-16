# CachePolicyFactory built-in registrations

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/cpu/policies/factory.py`](vllm/v1/kv_offload/cpu/policies/factory.py) (lines 84–90)
- **Symbol:** `CachePolicyFactory built-in registrations`
- **Kind:** plugin_seam
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_kv_offload-0013`

## Description
Registration site for pluggable CPU cache eviction policies selected by cache-policy configuration.

## Current approach
The interface is CachePolicy in vllm/v1/kv_offload/cpu/policies/base.py; built-in implementations are LRUCachePolicy in vllm/v1/kv_offload/cpu/policies/lru.py and ARCCachePolicy in vllm/v1/kv_offload/cpu/policies/arc.py. Runtime selection uses the eviction_policy config key and optional cache_policy_module_path, resolved by CachePolicyFactory.get_cache_policy_cls.

## Estimated impact explanation
A workload-tuned primary-cache policy can materially improve hit rate for repeated agentic prefixes and reduce secondary promotions. Because callers select policies through config, this is a high-leverage extension point for TTFT and TPOT experiments.

## Evolve rationale
The CachePolicy interface is small and has strong invariants, making sibling implementations such as 2Q, LFU, TinyLFU, or learned policies measurable without changing callers. Correctness oracle: factory tests, existing LRU/ARC policy tests, and the shared get/insert/remove/touch/evict contract.

## Deep research proposals

### 1. Add a workflow-priority CachePolicy that consumes per-request retention hints
- **Finding:** `find-vllm_v1_kv_offload-0002` — *Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo*
- **Source URL:** <https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new sibling CachePolicy implementation (e.g. PriorityCachePolicy) and register it alongside LRU/ARC in vllm/v1/kv_offload/cpu/policies/factory.py (lines 84-90). The policy reuses the existing CachePolicy contract (get/insert/remove/touch/evict/clear) but derives per-block retention scores from ReqContext.kv_transfer_params (already threaded through touch(keys, req_context) in base.py). Callers pass workflow-level metadata such as a numeric priority, a 'pin' flag, or an expected reuse horizon; the policy maintains a priority queue or segmented LRU where higher-priority blocks are evicted last, and evict(n, protected) walks lowest-priority buckets first before falling back to recency. Blocks marked pinned via kv_transfer_params bypass eviction entirely (skipped just as `protected` blocks are today). Because selection is via the eviction_policy config key with optional cache_policy_module_path, the change is a pure addition at the registration site with no caller modifications, and correctness can be verified with the existing factory tests plus new tests that assert priority-ordered eviction and pin honoring.

**Proposal rationale.**

The finding argues an agent harness benefits when the cache is told which blocks matter after tool calls, which to pin, and which to drop first — not pure recency. The candidate's plugin seam is exactly the place to add such a policy: the CachePolicy ABC already takes ReqContext in touch(), and the factory pattern makes a hint-aware sibling measurable against LRU/ARC without touching callers. For the stated objective (median TTFT/TPOT on multi-turn agentic workloads) this addresses the specific gap that recency-only policies evict tool-context blocks likely to be reused in the next turn, causing avoidable reloads that dominate TTFT. The idea is concrete, transferable to the existing interface, and stays within the seam's declared scope of sibling implementations.

---

### 2. Add S3-FIFO cache policy as a sibling to LRU/ARC in the CPU cache policy factory
- **Finding:** `find-vllm_v1_kv_offload-0004` — *FIFO queues are all you need for cache eviction*
- **Source URL:** <https://s3fifo.com/blog/2023/08/01/fifo-queues-are-all-you-need-for-cache-eviction/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new S3FIFOCachePolicy implementing the CachePolicy interface in vllm/v1/kv_offload/cpu/policies/base.py, and register it in CachePolicyFactory's built-in registrations at vllm/v1/kv_offload/cpu/policies/factory.py:84-90 alongside LRUCachePolicy and ARCCachePolicy. The policy would implement S3-FIFO's three-queue structure: a small probationary FIFO (S, ~10% of capacity) for newly inserted blocks, a main FIFO (M) for blocks that receive a second hit while in S or M, and a ghost FIFO (G) tracking recently evicted keys. Track a tiny per-block frequency counter (2 bits, saturating) updated on touch; on eviction from S, promote to M if counter>0 else evict and record in G; on eviction from M, re-insert to head if counter>0 (decrementing) else evict. G is consulted on insert to decide whether a new block enters S or directly M. Wire selection via the existing eviction_policy config key (e.g., 'S3FIFO'), reusing the same get/insert/remove/touch/evict contract exercised by existing factory and policy tests. Add unit tests mirroring the LRU/ARC test structure to cover the small/main/ghost transitions and eviction ordering.

**Proposal rationale.**

The candidate is precisely the plugin seam for CPU cache eviction policies, and its evolve_rationale explicitly calls out sibling policies like 2Q/LFU/TinyLFU as measurable drop-ins. S3-FIFO is a concrete, well-documented sibling in that same family and targets exactly the gap the caller cares about: for multi-turn agentic workloads, hot shared prefix blocks must survive scans of one-hit intermediate blocks to preserve prefix cache hits and thereby reduce median TTFT (fewer prefill recomputes) and TPOT (less contention/promotion churn against the secondary tier). S3-FIFO's small probationary FIFO quickly demotes one-hit pollution while the main FIFO plus ghost queue protect blocks that have demonstrated reuse, which matches the reuse pattern of long agentic prefixes better than pure recency (LRU) and with simpler eviction metadata than ARC. Because policies are resolved via config through CachePolicyFactory.get_cache_policy_cls, adding this registration is low-risk and directly enables the TTFT/TPOT experiments the candidate description anticipates.

---

### 3. Add TinyLFU admission-gated cache policy as a built-in sibling to LRU and ARC
- **Finding:** `find-vllm_v1_kv_offload-0005` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://paperity.org/p/377179711/tinylfu-a-highly-efficient-cache-admission-policy>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new CachePolicy implementation, TinyLFUCachePolicy, in vllm/v1/kv_offload/cpu/policies/tinylfu.py, and register it as a third built-in in vllm/v1/kv_offload/cpu/policies/factory.py (lines 84-90) with a call like CachePolicyFactory.register_cache_policy("tinylfu", "vllm.v1.kv_offload.cpu.policies.tinylfu", "TinyLFUCachePolicy"). The policy subclasses CachePolicy from vllm/v1/kv_offload/cpu/policies/base.py and preserves the get/insert/remove/touch/evict/clear contract used by LRUCachePolicy and ARCCachePolicy. Internally, it maintains: (1) an approximate frequency sketch (e.g., a small count-min sketch with a periodic aging/decay step) updated on touch() to record recent access frequency per OffloadKey; (2) a main LRU (or SLRU with a small probationary segment) that structures the resident set for eviction ordering; and (3) an admission gate invoked when a new key would displace an existing victim. In evict(n, protected), candidate victims come from the LRU tail as usual; in insert(), when the cache is at capacity the newly arriving key is compared to the current victim's sketch count and admitted only if its sketch count is greater than or equal to the victim's, otherwise the incoming block is dropped and the victim retained (i.e., admission acts as a gate, mirroring the finding's rule that admission is decided "based on the recent access history"). Selection remains driven by the existing eviction_policy config key and optional cache_policy_module_path; no caller changes are required beyond the single registration line. Correctness is guarded by the existing factory tests and shared policy contract tests for LRU/ARC, extended to cover the new policy with the same get/insert/remove/touch/evict invariants.

**Proposal rationale.**

The candidate is explicitly a plugin seam: CachePolicyFactory registers sibling implementations selected by config, and its evolve_rationale calls out TinyLFU as a directly measurable sibling of LRU/ARC. The finding contributes a concrete, transferable admission-gate technique (frequency-sketch comparison between incoming and victim) that is exactly the axis on which LRU and ARC are weakest for the caller's workload: multi-turn agentic sessions have repeated hot prefixes interleaved with one-shot blocks, so gating admission by recent frequency should reduce eviction churn of high-value prefix blocks and raise CPU-tier hit rate under capacity pressure, improving median TTFT (more prefix hits avoid recompute/re-fetch) and median TPOT (less thrash in the offload path). The change is localized (one new module plus one registration line at lines 84-90) and reuses the small, well-specified CachePolicy contract, so it can be validated against the same correctness oracle the candidate names without altering callers.

---

## Agent proposals

### 1. Add SIEVE cache policy as a built-in sibling to LRU and ARC
- **Agent:** claude

**Detailed description.**

Introduce a new SIEVECachePolicy in vllm/v1/kv_offload/cpu/policies/sieve.py and register it as a third built-in at vllm/v1/kv_offload/cpu/policies/factory.py:84-90 via CachePolicyFactory.register_cache_policy("sieve", "vllm.v1.kv_offload.cpu.policies.sieve", "SIEVECachePolicy"). The implementation follows Zhang et al.'s SIEVE algorithm (NSDI 2024): maintain a single doubly-linked list of resident blocks (newest inserted at head) plus a persistent "hand" pointer that walks from tail toward head across eviction rounds, and give each BlockStatus one visited bit (packed alongside ref_cnt or held in a parallel dict keyed by OffloadKey). Contract mapping: get() returns the BlockStatus and does not move the node; touch(keys, req_context) sets visited=1 for each key already present (lazy promotion — no list reordering, so touch is O(1) per key and cheaper than LRU's move-to-head); insert(key, block) links the block at the list head with visited=0; remove(key) unlinks and if it is the current hand advances the hand to its predecessor first; evict(n, protected) walks the hand backward, and for each candidate: if the key is in protected or ref_cnt != 0 skip without changing the visited bit; else if visited==1 clear it to 0 and continue; else the block is evicted (added to the return list, hand advanced to its predecessor). If the hand reaches the head without finding n victims, wrap to tail and continue; if a full sweep completes without n evictions the operation returns None atomically (no state changes visible — accumulate victims into a temporary list and only unlink after n are found, matching the ARC/LRU atomicity guarantee documented in base.py). clear() empties the list and resets the hand to None; mark_evictable/mark_non_evictable remain no-ops as in LRU. Selection is via the existing eviction_policy config key with optional cache_policy_module_path, so no caller changes are needed. Add tests mirroring the LRU/ARC suites that assert: (a) unvisited blocks are evicted before visited ones, (b) the hand's position persists across evict() calls so a scan-heavy workload cannot repeatedly rescue the same tail block, (c) protected/pinned keys are skipped without clearing their visited bit, (d) atomic failure when n exceeds evictable population.

**Novelty rationale.**

None of the three listed deep_research_proposals cover SIEVE. Proposal #1 (PriorityCachePolicy) is a hint-driven policy that consumes ReqContext.kv_transfer_params; SIEVE is fully workload-agnostic and requires no caller-supplied hints. Proposal #2 (S3-FIFO) relies on three queues (small probationary FIFO, main FIFO, ghost FIFO) plus a 2-bit saturating frequency counter and consults a ghost queue on insert to decide S vs. M placement; SIEVE has no ghost queue, no frequency counter, no probationary segment, and no promotion between structures — it uses a single list, one visited bit per block, and a persistent scan hand that provides "quick demotion" without the metadata cost of S3-FIFO. Proposal #3 (TinyLFU) is defined by a count-min frequency sketch plus an admission gate that can reject an incoming block if its sketch count is lower than the victim's; SIEVE has no sketch, admits every insert unconditionally, and makes eviction decisions from a single visited bit rather than frequency estimation. SIEVE also occupies a distinct point on the complexity/effectiveness curve (simpler than S3-FIFO and TinyLFU, O(1) hits with no list reordering) and is a directly measurable sibling under the same CachePolicy contract, which is exactly what the candidate's evolve_rationale invites.

---

### 2. Add a size-aware GDSF cache policy for variable KV block cost
- **Agent:** codex

**Detailed description.**

Introduce a new GDSFCachePolicy in vllm/v1/kv_offload/cpu/policies/gdsf.py and register it in CachePolicyFactory's built-in registrations at vllm/v1/kv_offload/cpu/policies/factory.py:84-90 (for example, eviction_policy="gdsf"). The policy implements GreedyDual-Size-Frequency semantics under the existing CachePolicy contract: maintain per-OffloadKey metadata with frequency, an insertion/access priority, and a global inflation value L set to the priority of the most recently evicted block. On insert(), compute priority as L + frequency / cost, where cost is the resident block's CPU memory footprint when available from BlockStatus/KV block metadata, falling back to cost=1 when all blocks are uniform. touch(keys, req_context) increments frequency and recomputes priority without requiring caller hints; get() returns the BlockStatus; remove() deletes metadata; clear() resets the heap/map and L. evict(n, protected) pops the lowest-priority keys from a min-heap, skipping stale heap entries, protected keys, and non-evictable/ref-counted blocks in the same style as LRU/ARC. To preserve the atomic evict contract, first collect n valid victims into a temporary list and only unlink/update L after enough victims are found; otherwise leave policy state unchanged and return None. Add factory coverage plus policy tests mirroring existing LRU/ARC tests, with focused cases showing that a large low-reuse block is evicted before smaller frequently reused prefix blocks, protected keys are skipped, stale heap priorities are ignored, and failure to find n victims is atomic.

**Novelty rationale.**

The listed proposals add hint-driven priority/pinning, S3-FIFO, TinyLFU admission, and SIEVE. GDSF is distinct because its eviction score explicitly combines reuse frequency with object cost/size and a global aging term; it is neither caller-hint based like PriorityCachePolicy, nor queue/ghost based like S3-FIFO, nor an admission gate with a frequency sketch like TinyLFU, nor a one-bit second-chance scan like SIEVE. This targets a different lever for median TTFT/TPOT in CPU KV offload: retaining small or repeatedly reused prefix blocks when larger one-off KV blocks would consume disproportionate CPU cache capacity, while remaining a drop-in sibling at the same factory registration seam.

---
