# TieringOffloadingManager._initiate_promotion

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/tiering/manager.py`](vllm/v1/kv_offload/tiering/manager.py) (lines 410–457)
- **Symbol:** `TieringOffloadingManager._initiate_promotion`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_kv_offload-0009`

## Description
Reserves primary-tier space for a secondary-to-primary promotion and accumulates pending load submissions by tier and request.

## Current approach
Each secondary hit calls primary_tier.prepare_write([key], req_context) immediately for a single key, then appends returned keys and block IDs into a per-tier/per-request PendingPromotion batch flushed later.

## Estimated impact explanation
Promotion admission and allocation determine how quickly secondary hits become primary hits and whether the primary tier thrashes. This directly affects TTFT on cache-hit turns and TPOT when decode waits for promoted blocks.

## Evolve rationale
Per-key allocation, promotion admission, and pending-batch grouping are hot scheduling choices. Evolution can batch prepare_write, prioritize likely-needed blocks, cap promotion pressure, or defer low-value promotions. Correctness oracle: ref-count bookkeeping, no duplicate in-flight promotions for a key, submit_load fan-in tests, and total promoted keys matching reserved primary slots.

## Deep research proposals

### 1. Prioritize promotions by agentic workflow value, not just recency
- **Finding:** `find-vllm_v1_kv_offload-0002` — *Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo*
- **Source URL:** <https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In TieringOffloadingManager._initiate_promotion (vllm/v1/kv_offload/tiering/manager.py:410-457), replace the current one-key-at-a-time primary_tier.prepare_write([key], req_context) loop with a value-aware admission step that (a) batches candidate secondary-hit keys per request into a single prepare_write call so allocation and eviction decisions see the whole set at once, and (b) consults request-supplied metadata on req_context (e.g. priority/pin/prefetch hints such as an expected reuse horizon or a tool-call-boundary marker) to decide which keys to promote now, which to defer, and which to skip. Concretely: collect per-request secondary hits into a proposed batch, score each key using recency plus any hint from req_context (pinned/high-priority keys score highest; keys marked as needed-after-next-tool-call get promoted eagerly; low-value keys are dropped from the batch when primary pressure is high), call prepare_write once with the surviving ordered list, and then build the PendingPromotion entries from the returned block IDs. Keep the existing ref-count bookkeeping and the invariant that no duplicate in-flight promotion exists for a key; the admission filter runs before prepare_write so reserved primary slots still match the number of PendingPromotion entries.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out promotion admission, batching prepare_write, prioritizing likely-needed blocks, and capping promotion pressure as levers. The Dynamo finding contributes a concrete, transferable idea beyond generic LRU: let the caller (an agent harness that knows its own tool-call structure) annotate requests with prefetch/pin/evict-first hints, and have the cache manager honor those hints when admitting promotions. In a multi-turn agentic workload — the stated caller context — the harness often knows which blocks will be reused after a tool call and which are throwaway; using that signal to gate promotions directly targets TTFT on cache-hit turns (promote the blocks that actually matter next) and TPOT (avoid thrashing primary with low-value promotions that displace hot decode state). Batching prepare_write also lets the primary tier make a globally better allocation/eviction decision per request instead of one-at-a-time, which the current per-key loop cannot do.

---

### 2. Split per-request promotion batch into progressive sub-batches to reduce HOL blocking
- **Finding:** `find-vllm_v1_kv_offload-0003` — *[RFC]: Progressive KV Cache CPU Onloading*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/33526>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `TieringOffloadingManager._initiate_promotion` and `_flush_pending_promotions` (vllm/v1/kv_offload/tiering/manager.py:410-457, 459+), replace the single-`submit_load` per (tier, request) batching with a progressive-batching strategy: instead of accumulating all promoted keys/block_ids into one `PendingPromotion` and submitting one large job at `on_schedule_end`, split the accumulated batch into multiple `JobMetadata` submissions ordered by prefix position — a small first sub-batch containing the earliest (lowest block-index) keys, followed by geometrically or linearly larger sub-batches for later keys. The primary-tier allocation via `primary_tier.prepare_write([key], req_context)` at line 437 stays per-key so ref-count/in-flight semantics are preserved; only the flush step changes to slice `entry.keys`/`entry.block_ids` into ordered sub-batches, assign each its own `job_id`, and call `tier.submit_load(...)` per sub-batch. Add a `min_progressive_batch` / `growth_factor` knob (with a threshold below which we keep a single batch, so short promotions are unaffected). Preserve the correctness oracle: total submitted keys equals reserved primary slots, no duplicate in-flight promotions per key (since ref_cnt=-1 is still set at allocation time), and `submit_load` fan-in tests should be updated to accept N>=1 jobs per (tier, request).

**Proposal rationale.**

The candidate's current approach groups all promotions for a request into one `submit_load` per tier per step. Under the caller's multi-turn agentic workload with a TTFT objective, a long promoting request can monopolize a secondary-to-primary transfer channel and delay a short request that shares a prefix — exactly the head-of-line-blocking scenario the RFC describes. The finding's technique (split one transfer into multiple, small-first-then-larger) transfers directly: because the primary allocation is already per-key and deferred submission is already the flush point, we can slice the pending batch into ordered sub-batches with minimal disruption to ref-count bookkeeping. Progressive sub-batches let the early prefix land quickly (unblocking a short-request decode that only needs the front of the shared prefix) while amortizing per-job overhead on later, larger chunks — targeting median TTFT on cache-hit turns without changing store semantics.

---

### 3. Gate secondary-to-primary promotions with an S3-FIFO probationary + ghost queue
- **Finding:** `find-vllm_v1_kv_offload-0004` — *FIFO queues are all you need for cache eviction*
- **Source URL:** <https://s3fifo.com/blog/2023/08/01/fifo-queues-are-all-you-need-for-cache-eviction/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify TieringOffloadingManager._initiate_promotion in vllm/v1/kv_offload/tiering/manager.py (lines 410-457) to admit secondary-tier hits through an S3-FIFO-style two-stage admission policy instead of promoting every secondary hit directly into the primary tier. Introduce three lightweight per-manager FIFOs keyed by OffloadKey: (1) a small probationary FIFO 'S' sized to a small fraction of primary capacity, (2) the main primary region 'M' (the existing primary tier), and (3) a ghost FIFO 'G' holding recently evicted keys with no data. On a secondary hit for key k in _initiate_promotion: if k is in G (a re-reference of a recently evicted block), call primary_tier.prepare_write([k], req_context) targeting the main region as today and increment k's tiny per-key counter (2-bit) to protect it; otherwise call prepare_write into the small probationary region S and initialize the counter to 0. Keep the existing PendingPromotion batching-by-(tier, request) unchanged: entries from both admission paths append into the same tier_pending[ctx_id] batch so _flush_pending_promotions still emits one submit_load per (tier, request). When the primary tier signals capacity pressure (prepare_write returns None or a separate 'S full' signal from the tier), evict the oldest S entry: if its counter >= 1 promote it into M (which may in turn evict from M using existing policy), otherwise push its key into G and drop the block. G is a bounded FIFO of keys only; overflow discards the oldest ghost entry. The 2-bit counter is incremented on touch() calls into S/M and decremented on ghost re-references, matching the S3-FIFO reference-bit sketch. Reuse the existing prepare_write / submit_load / ref_cnt=-1 in-flight bookkeeping and the batched PendingPromotion path; the only new state is the S/G FIFOs and a per-key uint8 counter map. Correctness oracles from evolve_rationale remain enforceable: ref_cnt bookkeeping is unchanged (prepare_write still owns the slot), duplicate in-flight promotion suppression still works because prepare_write reserves the primary slot before enqueueing into PendingPromotion, submit_load fan-in per (tier, request) is preserved, and total promoted keys still equals reserved primary slots (S counts as reserved primary capacity).

**Proposal rationale.**

The candidate's _initiate_promotion currently admits every secondary-tier hit into the primary tier unconditionally, deferring only the submit_load batching. For a multi-turn agentic workload the primary tier is a small, hot region that must retain shared conversation prefixes across turns; naively promoting every secondary hit lets one-hit blocks (tool outputs, transient scratch content pulled in by a single lookup) evict hot prefixes, hurting both TTFT (prefix cache miss on next turn) and TPOT (decode stalls while re-promoted blocks reload from secondary). S3-FIFO's central mechanism — a small probationary FIFO that quickly demotes one-hit objects and a ghost FIFO that lets re-referenced objects skip probation — targets exactly this promotion-admission gap with O(1) metadata (per-block 2-bit counter, two FIFO pointers, ghost key set). It is a strict superset of the current 'admit all' policy: setting S to primary size and skipping G reproduces today's behavior. The finding provides the concrete queue-and-counter structure (blog quote: 'S3-FIFO uses three FIFO queues: a small FIFO queue (S), a main FIFO queue (M), and a ghost FIFO queue (G)'), which maps directly onto the existing prepare_write / PendingPromotion / submit_load pipeline without changing the correctness oracles listed in evolve_rationale.

---

### 4. Add a TinyLFU admission gate before promoting secondary blocks to the primary tier
- **Finding:** `find-vllm_v1_kv_offload-0005` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://paperity.org/p/377179711/tinylfu-a-highly-efficient-cache-admission-policy>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify TieringOffloadingManager._initiate_promotion at vllm/v1/kv_offload/tiering/manager.py:410-457 to consult a small Count-Min-Sketch-based frequency estimator (TinyLFU-style: a small counting sketch plus a short doorkeeper Bloom filter, both aged periodically) before calling primary_tier.prepare_write([key], req_context). On every secondary lookup and every primary hit, increment the sketch for the block's OffloadKey. In _initiate_promotion, first estimate the candidate key's recent frequency. If the primary tier is under capacity pressure (prepare_write would need to evict a resident block), peek at the primary tier's LRU/eviction victim and admit the candidate only if its sketch-estimated frequency exceeds the victim's; otherwise return False without reserving a slot and without appending to _pending_load_submissions, so the block simply stays in the secondary tier. When the primary has free slots, promote unconditionally as today. Correctness invariants from the candidate's oracle are preserved: no ref_cnt=-1 reservation is created on rejection, so no duplicate in-flight promotions can appear; PendingPromotion batches only contain keys that were successfully reserved; and the total promoted keys still equal the reserved primary slots. To keep the hot path cheap, the sketch should be a fixed-size 4-bit counter CMS sized to the working set (e.g. width proportional to primary_tier capacity), with periodic halving of all counters (aging) triggered from _flush_pending_promotions or on_schedule_end.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out capping promotion pressure and deferring low-value promotions as a lever for TTFT/TPOT, and notes that primary-tier thrashing directly hurts cache-hit turns. In the current code every secondary hit unconditionally reserves a primary slot via prepare_write, which under capacity pressure evicts an existing primary block regardless of whether the incoming block is likely to be re-used. TinyLFU's core contribution — an approximate frequency sketch used as an admission gate that only admits a candidate when its recent frequency exceeds the victim's — maps directly onto this decision point: it converts a blind evict-on-every-secondary-hit policy into a value-weighted one. This is especially well matched to the stated multi-turn agentic workload, where a small set of shared prefix blocks are accessed repeatedly (high sketch frequency, worth promoting) while long tail one-off blocks would otherwise churn the primary tier (low sketch frequency, correctly rejected). The change is localized to _initiate_promotion plus a small sketch data structure, and preserves the candidate's correctness oracle (ref-count bookkeeping, no duplicate in-flight promotions, submit_load fan-in) because rejection is a strict no-op path before any reservation or pending-batch append.

---

### 5. Batch primary-tier allocation for promotions instead of per-key prepare_write
- **Finding:** `find-vllm_v1_kv_offload-0006` — *Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving*
- **Source URL:** <https://arxiv.org/abs/2605.03375>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor TieringOffloadingManager to defer primary-tier allocation the same way it defers submit_load. In vllm/v1/kv_offload/tiering/manager.py:410-457, replace the per-key primary_tier.prepare_write([key], req_context) call inside _initiate_promotion with an accumulation step that collects (tier, req_context, key) tuples during lookup(). Then, either at the end of lookup batching or at the head of _flush_pending_promotions (lines 459-481), issue a single primary_tier.prepare_write(all_keys, req_context) per request/tier group and distribute the returned keys_to_store and block_ids into per-request PendingPromotion entries. To preserve the current in-flight semantics (ref_cnt=-1 preventing duplicate promotion attempts within the same step), track a lightweight in-memory set of "pending-allocation" keys per tier so subsequent lookups in the same step still see the key as reserved without a real primary allocation. On flush, partial-fill from prepare_write is handled by treating the returned keys_to_store subset as promoted and the rest as unavailable (matching the existing 'primary full' branch at lines 439-442). This turns N small prepare_write calls into one bulk call per (tier, request), mirroring the finding's 'bulk KV-cache object' abstraction on the allocation path in the same way _flush_pending_promotions already does for submit_load.

**Proposal rationale.**

The candidate already batches submit_load but still performs per-key prepare_write allocation in the hot lookup path. The finding's core transferable idea is that per-block tasks against a slower/coarser backing tier should be aggregated into bulk object operations to avoid fragmented work and better overlap with GPU compute. That maps cleanly to the allocation side of promotion here: today each secondary hit incurs a separate primary-tier allocation call with its own locking, ref-count bookkeeping, and eviction bookkeeping, which for a multi-turn agentic workload with many small secondary hits per turn is exactly the fragmented-tiny-work pattern the paper argues against. Batching prepare_write reduces per-hit overhead in the scheduler step and lets the primary tier make a single, potentially better global admission decision (e.g., allocate as many of the N requested slots as it can rather than N independent decisions), which directly targets TTFT on cache-hit turns called out in the caller objective. The correctness oracles listed in evolve_rationale (ref-count bookkeeping, no duplicate in-flight promotions, submit_load fan-in, promoted-count == reserved slots) are all preserved because the batched allocation is a strict refactor of the same operation and the pending-allocation set maintains the in-flight guarantee within a step.

---

### 6. Batch prepare_write allocations per step to amortize per-key primary-tier admission overhead
- **Finding:** `find-vllm_v1_kv_offload-0007` — *GPUDirect Storage Overview Guide*
- **Source URL:** <https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In TieringOffloadingManager._initiate_promotion (vllm/v1/kv_offload/tiering/manager.py:410-457), replace the per-key primary_tier.prepare_write([key], req_context) call at line 437 with a two-phase design that mirrors the batched-submission pattern already used by _flush_pending_promotions. Phase 1 (during lookup, called from _initiate_promotion): buffer secondary-hit keys into a per-(tier, request) staging list without invoking prepare_write, and mark a lightweight in-flight sentinel so subsequent lookups within the same step still see the slot as pending and do not attempt duplicate promotions. Phase 2 (in on_schedule_end, just before _flush_pending_promotions): for each (tier, request) group, call primary_tier.prepare_write(all_staged_keys, req_context) exactly once, then materialize PendingPromotion.keys/block_ids from the single returned result and submit_load as today. If the batched prepare_write returns a partial allocation (fewer keys reserved than requested because the primary tier is near-full), drop the tail keys from that request's promotion set and record them as unavailable, preserving today's fail-when-full semantics. Keep the ref_cnt=-1 in-flight bookkeeping intact so the correctness oracle (no duplicate in-flight promotions per key, submit_load fan-in, promoted keys == reserved primary slots) still holds.

**Proposal rationale.**

The GPUDirect Storage guide's core lesson - 'batching reduces the overhead by amortizing that fixed overhead across the transactions in the batch' - transfers directly to the candidate's remaining unbatched hot path. _flush_pending_promotions already batches submit_load per (tier, request), but _initiate_promotion still pays a per-key fixed cost inside primary_tier.prepare_write (line 437: prepare_write([key], ...)): each call likely takes an allocator lock, walks free lists, and constructs a CPULoadStoreSpec for one block. On multi-turn agentic workloads with wide secondary hits per step, N secondary hits cause N sequential allocator round-trips before any I/O can begin, delaying the batched submit_load and inflating TTFT on cache-hit turns. Consolidating into one prepare_write per (tier, request) per step applies the same amortization principle the finding validates for storage submissions, and fits cleanly because the surrounding code already groups by (tier, req_id) and defers work until on_schedule_end.

---

### 7. Gate secondary-to-primary promotion on predicted usefulness instead of promoting every hit
- **Finding:** `find-vllm_v1_kv_offload-0010` — *InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management*
- **Source URL:** <https://www.alphaxiv.org/abs/2406.19707>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify TieringOffloadingManager._initiate_promotion (vllm/v1/kv_offload/tiering/manager.py:410-457) so that a secondary-tier hit does not unconditionally reserve a primary-tier slot. Introduce a lightweight usefulness signal computed from information already flowing through the tiering manager — for example, per-request recency and reuse counts of the key in secondary, block position within the request (earlier prefix blocks are more likely to be re-attended in multi-turn turns), and whether the request is still in prefill vs. decode — and pass promotion candidates through a scoring/admission step before calling primary_tier.prepare_write. Concretely: (1) accumulate secondary hits for the current step into a candidate list rather than reserving primary space one-by-one; (2) at flush time (or in a small pre-flush admission step) rank candidates by the usefulness score and admit only the top-K, where K is capped by remaining primary capacity plus an adaptive promotion-pressure budget derived from recent primary evictions/hit-rate stats already exposed via TieringOffloadingMetrics; (3) skipped candidates stay as secondary-tier hits (existing async-lookup path) and are not counted as failed promotions. Keep the existing correctness oracle intact: still call prepare_write([key], req_context) exactly once per admitted key so ref_cnt=-1 in-flight bookkeeping and duplicate-promotion prevention are unchanged, still group admitted keys into per-(tier, request) PendingPromotion batches, and still guarantee total promoted keys == reserved primary slots. Expose the admission threshold and pressure cap as tunables so the heuristic can be disabled to recover current behavior.

**Proposal rationale.**

The candidate today promotes every secondary hit that fits, which under a multi-turn agentic workload can thrash the primary tier: prefix blocks from earlier turns are recalled en masse even when only a subset will actually be attended before eviction, delaying TTFT on the current turn and adding contention that lengthens TPOT while decode waits on batched submit_loads. The InfiniGen finding contributes the transferable idea that promotion/prefetch should be selective — bring back only entries predicted to be useful — rather than blindly recalling everything available. Applied to _initiate_promotion, this becomes a usefulness-scored admission step in front of prepare_write, which directly targets the two hot scheduling choices called out in the candidate's evolve_rationale (promotion admission and capping promotion pressure) while preserving the existing correctness invariants (ref-count bookkeeping, no duplicate in-flight promotions, submit_load fan-in, reserved-slot accounting). Even without InfiniGen's model-aware rehearsal (which requires attention-layer signals the block-level module does not have), a cheap block-level proxy — recency/reuse/position/phase — captures the same principle at the granularity this module operates on.

---

### 8. Speculative prefetch promotion to hide secondary-to-primary load latency
- **Finding:** `find-vllm_v1_kv_offload-0011` — *ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs*
- **Source URL:** <https://www.usenix.org/conference/osdi26/presentation/liu-guangda>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend TieringOffloadingManager._initiate_promotion at vllm/v1/kv_offload/tiering/manager.py:410-457 (plus the surrounding lookup/scheduling path) with a speculative promotion mode that runs before demand rather than only at the point of secondary-hit resolution. Concretely: (1) Add a predictor hook that, at scheduler step boundaries, proposes a bounded set of (tier, key, req_context) triples likely to be demanded soon — for the multi-turn agentic workload, seed it from the tail of each active request's already-hit secondary prefix and from sibling blocks of secondary hits observed this step (cheap heuristics that require no model signal). (2) Route those predicted triples through a new speculative variant of _initiate_promotion that calls primary_tier.prepare_write with a lower-priority admission cap (e.g., reserve at most K primary slots per step for speculation, distinct from demand promotions) and appends into the same PendingPromotion batch so _flush_pending_promotions still submits one batched load per (tier, request). (3) Preserve the correctness oracle: keep ref_cnt=-1 in-flight semantics so a subsequent demand lookup for the same key coalesces onto the in-flight speculative promotion instead of issuing a duplicate; on request finish or eviction pressure, decrement the speculative reservations first. (4) Track a hit rate on speculative promotions and back off the per-step cap when it drops, so a mispredicting workload cannot starve demand promotions of primary space.

**Proposal rationale.**

The current _initiate_promotion is purely reactive: a secondary hit at scheduling time reserves primary space and queues the load, so the load latency sits on the critical path of the very step that discovered the hit — inflating TTFT for cache-hit prefills and TPOT for decodes that wait on promoted blocks. ECHO's core idea is that predictable future KV use can be prefetched during other compute, converting exposed load latency into overlapped latency. That maps cleanly onto this candidate: prepare_write, the PendingPromotion batching, and the ref_cnt=-1 in-flight guard already provide the machinery for a promotion to be admitted, deduplicated across the step, and consumed by submit_load — what is missing is a path that initiates promotions before demand. Speculative promotion reuses all of that machinery, targets the exact gap the finding names (exposed secondary→primary load latency on the request's critical path), and stays bounded and reversible via the per-step cap and hit-rate feedback so it cannot regress workloads where prediction is unreliable.

---

### 9. Schedule promotion transfers into decode idle-bandwidth windows with batched prepare_write
- **Finding:** `find-vllm_v1_kv_offload-0012` — *Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching*
- **Source URL:** <https://ojs.aaai.org/index.php/AAAI/article/view/39224>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/kv_offload/tiering/manager.py:410-457, restructure TieringOffloadingManager._initiate_promotion so promotions are timed and batched to overlap with compute-bound decode phases rather than issued eagerly per key. Two coupled changes: (1) Batch prepare_write across a scheduler step. Instead of calling self.primary_tier.prepare_write([key], req_context) once per secondary hit, accumulate candidate keys per (tier, request) during lookup() and issue a single primary_tier.prepare_write(keys, req_context) call at on_schedule_end() (or a fixed budget mid-step), before the current _flush_pending_promotions loop. Keep the in-flight semantics by having the primary tier still mark reserved slots ref_cnt=-1 for all admitted keys during the batched allocation, and drop keys whose allocation failed so admission is bounded by remaining primary capacity. (2) Prioritize and cap admissions. Order accumulated candidates by expected imminent use (e.g., proximity to the request's next decode position or presence in the current step's prefill window), admit up to a per-step promotion budget sized to the primary tier's headroom, and defer or drop the tail. Emit a stat counter for deferred/dropped promotions and reuse the existing _pending_load_submissions structure to submit the surviving batch via tier.submit_load() so the flush path in _flush_pending_promotions() is unchanged. Correctness invariants preserved: reserved primary slots equal admitted keys, no duplicate in-flight promotion per key (batched prepare_write still returns keys_to_store), and submit_load fan-in still matches JobMetadata.keys.

**Proposal rationale.**

The candidate today allocates primary space and groups pending submissions per-key, which minimizes the promotion latency of a single hit but ignores when the transfer will actually contend with decode. The finding's central idea is scheduling KV movement into idle memory-bandwidth windows to hide it behind compute. Applied to this method, it motivates two changes the current code does not make: batching prepare_write so allocation cost and admission decisions are amortized once per step (matching the already-batched submit_load), and capping/prioritizing promotions so the primary tier does not thrash on decode-heavy steps. For the stated objective (median TPOT on multi-turn agentic workloads), this addresses the specific gap where a burst of secondary hits from a long context would otherwise flood primary allocation and delay decode, while preserving the ref-count and no-duplicate-in-flight oracles listed in evolve_rationale.

---

## Agent proposals

### 1. Coalesce cross-request duplicate promotions within a step via a shared in-flight promotion table
- **Agent:** claude

**Detailed description.**

In TieringOffloadingManager._initiate_promotion (vllm/v1/kv_offload/tiering/manager.py:410-457), add a manager-scoped in-flight promotion table keyed by OffloadKey that lets concurrent requests sharing a prefix coalesce onto a single promotion rather than each reserving and loading the same key redundantly at the (tier, request) granularity. Concretely: (1) Introduce `_promotion_inflight: dict[OffloadKey, PromotionHandle]` where PromotionHandle carries the reserved primary block_id, the originating (tier, req_context) that issued prepare_write, and a set of dependent request contexts. (2) In _initiate_promotion, before calling primary_tier.prepare_write([key], req_context), consult _promotion_inflight; if key is present, do NOT call prepare_write again — instead append this request to the handle's dependents list, increment the primary block's logical ref_cnt for the new request via the existing tier accounting, and record in this request's PendingPromotion that the target block_id is a shared reference (not a freshly reserved slot). (3) If the key is absent, call prepare_write as today, insert the returned block_id into _promotion_inflight, and proceed with the normal PendingPromotion append. (4) In _flush_pending_promotions, submit_load is issued only for the originating (tier, request); dependent requests wait on the same job completion by extending the existing job-completion callback path so all dependents are notified when the shared load finishes, then drop the handle from _promotion_inflight. (5) On request cancellation or eviction, decrement the shared ref_cnt via the primary tier's existing accounting so early-dropping one dependent does not free a block another still needs. This preserves every correctness oracle in evolve_rationale: total reserved primary slots equal distinct promoted keys (the correct invariant when multiple requests share a prefix), no duplicate in-flight promotion per key is strengthened from per-(tier, request) to global, and submit_load fan-in remains one job per originating request — dependents attach to the existing job.

**Novelty rationale.**

None of the nine listed deep_research_proposals address cross-request coalescing of promotions for the same key within a step. finding-0003 splits one request's batch progressively; findings-0006/0007/0012 batch prepare_write within a single (tier, request) group; findings-0002/0004/0005/0010 change admission policy (which keys to promote) but still promote independently per request; finding-0011 speculatively pre-promotes but does not fold concurrent demand from multiple requests onto one in-flight promotion. The novelty here is exploiting the multi-turn agentic workload's prefix sharing across concurrent requests — common when a shared system prompt or shared tool-output prefix triggers secondary hits from N requests in the same scheduling step — so N secondary hits on the same key produce one prepare_write reservation, one submit_load, and one primary block with ref_cnt=N, instead of N duplicate reservations that race and evict each other. This targets both TTFT (only the first request pays load latency; others attach for free) and TPOT (avoids primary-tier thrash from redundant reservations displacing hot decode state) with a small, localized change to _initiate_promotion plus ref_cnt bookkeeping.

---

### 2. Release reserved promotion slots when requests cancel before load completion
- **Agent:** codex

**Detailed description.**

In `TieringOffloadingManager._initiate_promotion` (`vllm/v1/kv_offload/tiering/manager.py:410-457`), make each reserved promotion cancellable by recording enough metadata in `PendingPromotion` to undo the reservation: `(key, block_id, tier, req_id/request generation)`. Add a small cleanup path used before `_flush_pending_promotions` and from request-finished/cancelled handling that removes cancelled request entries from `_pending_load_submissions`, skips their `submit_load`, and calls the primary tier's release/free path for the corresponding reserved `block_id` whose promotion has not completed yet. For promotions already submitted, mark the job as orphaned/cancelled so completion releases the reserved primary block instead of installing it for a dead request. Guard this with a request generation or context identity check so a reused request id cannot cancel a newer promotion. The correctness tests should cover: cancelling after `prepare_write` but before flush frees all reserved blocks, cancelling after submit but before completion does not install stale keys, and live requests' pending promotion counts still match reserved primary slots.

**Novelty rationale.**

The listed deep_research_proposals focus on admission policy, batching, prioritization, progressive transfer slicing, speculative prefetch, or timing transfers. Agent A focuses on cross-request duplicate coalescing. None address the lifecycle gap created by reserving primary-tier slots in `_initiate_promotion` before the later `submit_load` and before request completion is known. This proposal is about cancellation-safe cleanup of already-reserved but no-longer-needed promotion slots, which is distinct from deciding which keys to admit or how to batch/coalesce them. In multi-turn agentic workloads with tool calls, retries, and abandoned branches, avoiding stale reserved promotions can directly reduce primary-tier pressure and secondary load work, improving TTFT/TPOT for still-live requests.

---
