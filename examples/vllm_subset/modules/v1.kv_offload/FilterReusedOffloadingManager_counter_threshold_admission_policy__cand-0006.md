# FilterReusedOffloadingManager counter+threshold admission policy

[← v1.kv_offload](../v1.kv_offload.md)

- **File:** [`vllm/v1/kv_offload/reuse_manager.py`](vllm/v1/kv_offload/reuse_manager.py) (lines 44–97)
- **Symbol:** `FilterReusedOffloadingManager counter+threshold admission policy`
- **Kind:** region
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0006`

## Description
Admission-control decorator that records lookup frequency per OffloadKey in a bounded LRU tracker and forwards only keys whose count reaches store_threshold to the backing OffloadingManager.prepare_store.

## Current approach
The tracker is an exact OrderedDict counter capped by max_tracker_size. lookup() moves or inserts one key, evicting the least-recent tracker entry via popitem(last=False) when full; prepare_store() filters with counts.get(key, 0) >= store_threshold. CPUOffloadingSpec enables this decorator only when kv_connector_extra_config["store_threshold"] >= 2 and defaults max_tracker_size to 64000.

## Estimated impact explanation
In multi-turn agentic traces, many blocks are never reused while system/tool prefixes are reused heavily. Better admission increases useful offload residency, reducing prefill recomputation and TTFT on cache-warm turns, while avoiding wasted GPU<->CPU bandwidth that would otherwise hurt TPOT.

## Evolve rationale
This is the owned admission gate that decides whether a block should consume CPU capacity and transfer bandwidth. Headroom includes TinyLFU/count-min sketches for longer horizons, decay so stale popularity expires, reuse-distance-aware scoring from request patterns, and better defaults for threshold/tracker size. Oracles are the exact current contract that a key is stored iff count >= threshold and the backing manager accepts it, plus existing FilterReusedOffloadingManager tests and trace-level hit-rate comparisons against an exact reference.

## Deep research proposals

### 1. Replace exact LRU counter with S3-FIFO probationary queue for offload admission
- **Finding:** `find-0008` — *FIFO Queues are All You Need for Cache Eviction*
- **Source URL:** <https://jasony.me/publication/sosp23-s3fifo.pdf>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Re-architect FilterReusedOffloadingManager (vllm/v1/kv_offload/reuse_manager.py:44-97) so its admission tracker mirrors S3-FIFO's small probationary FIFO instead of the current OrderedDict-based exact counter. Concretely: (1) On lookup(), insert previously-unseen OffloadKeys into a small FIFO queue ("S") sized as a fraction of max_tracker_size (e.g. ~10%), without touching any main structure; if the key is already in S, increment its in-place counter (saturating at a small cap, e.g. 3); if it is in a separate "main" set/queue ("M"), bump its reuse counter there. (2) On eviction from S (FIFO order, popleft on overflow), promote keys whose S-counter has reached store_threshold-1 into M and drop the rest, giving the same "count >= store_threshold ⇒ admit" oracle while sharply reducing one-hit residency in the tracker. (3) prepare_store() returns only keys present in M (or whose S-counter already meets the threshold), keeping the exact existing contract that backing OffloadingManager.prepare_store sees a key iff its observed-count >= threshold. (4) Keep CPUOffloadingSpec wiring unchanged: only enable this filter when store_threshold >= 2, and reuse max_tracker_size as the combined |S|+|M| budget so memory is bounded as today. Existing FilterReusedOffloadingManager tests continue to pin the contract; add trace-level hit-rate comparisons against the current OrderedDict reference on multi-turn agentic traces.

**Proposal rationale.**

The candidate is literally a one-hit-wonder admission filter sitting in front of an OffloadingManager, which is exactly the role S3-FIFO assigns to its small probationary FIFO ("filters out most objects from entering the main cache, which provides a guaranteed demotion speed and high demotion precision"). The current OrderedDict tracker uses LRU recency to decide which counters to forget, so a long scan of unique blocks (common in multi-turn agentic traces) can evict the still-warm system/tool-prefix counters before they reach store_threshold, defeating the gate. A FIFO probationary queue evicts in arrival order with bounded latency, so transient one-off blocks are dropped quickly while frequently-touched prefix keys get promoted into a stable main set and reliably cross the threshold. This directly targets the candidate's evolve_rationale (admit reusable prefixes, reject one-offs) and the caller's TTFT/TPOT objective by improving useful offload residency without growing the tracker budget.

---

### 2. Replace exact LRU counter with TinyLFU admission for CPU offload stores
- **Finding:** `find-0009` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://arxiv.org/pdf/1512.00727>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/kv_offload/reuse_manager.py:44-97, replace the exact OrderedDict-based counter (self.counts capped by max_tracker_size with popitem(last=False) eviction and the counts.get(key, 0) >= store_threshold gate) with a TinyLFU-style admission filter. Concretely: (1) maintain an approximate frequency sketch (count-min sketch with a small number of rows, sized roughly to max_tracker_size, plus a doorkeeper Bloom filter to suppress one-hit wonders); (2) on lookup(key), bump the sketch counters for the key and increment a global sample counter that triggers periodic halving (the TinyLFU 'aging' / reset step) so popularity decays instead of accumulating forever; (3) in prepare_store(keys), admit a key only when its estimated frequency exceeds either a fixed store_threshold (preserving the existing contract for store_threshold>=2) or the estimated frequency of the eviction candidate the backing OffloadingManager would displace. Where the backing manager exposes (or can be extended to expose) a lookup for the would-be victim block, compare frequencies before forwarding to prepare_store; otherwise fall back to the threshold-only admission so the change is strictly additive over the current oracle. Keep CPUOffloadingSpec wiring (store_threshold>=2 enables the decorator, max_tracker_size default 64000) and reinterpret max_tracker_size as the sketch width budget. Preserve the existing FilterReusedOffloadingManager tests and add a trace-level comparison against the exact OrderedDict reference to confirm hit-rate parity or improvement.

**Proposal rationale.**

The candidate is exactly the admission gate the TinyLFU paper targets: it currently decides admission by an exact recent-window counter with LRU eviction, which (a) loses popularity signal as soon as a key falls out of the bounded OrderedDict and (b) admits any key that simply crosses a fixed count even if a more frequently reused block is about to be evicted from the CPU pool. TinyLFU's count-min-sketch + doorkeeper + periodic aging provides longer-horizon frequency estimates within the same memory budget (the evolve_rationale explicitly calls out TinyLFU/count-min sketches and decay as headroom), and the victim-vs-candidate comparison from the paper's abstract directly addresses 'wasted CPU stores' that the candidate's threshold-only policy cannot prevent. For the stated multi-turn agentic workload where system/tool prefixes are reused heavily but most blocks are one-shot, this should raise useful offload residency, improving cache-warm TTFT and avoiding GPU<->CPU bandwidth that hurts TPOT. The change is local to reuse_manager.py and preserves the existing oracle (threshold-only admission remains a valid lower bound), making it a concrete, transferable improvement rather than a topical restatement.

---

### 3. Cost-aware admission scoring for FilterReusedOffloadingManager via GreedyDual-Size H values
- **Finding:** `find-0010` — *GreedyDual-Size Algorithm*
- **Source URL:** <https://www.usenix.org/legacy/publications/library/proceedings/usits97/full_papers/cao/cao_html/node8.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the pure-frequency gate in vllm/v1/kv_offload/reuse_manager.py:44-97 (FilterReusedOffloadingManager) with a GreedyDual-Size-style admission score. Augment the bounded LRU tracker so each OffloadKey accumulates an H value rather than a raw count: on lookup(), set H[key] = L + cost(key) / size(key), where L is a global aging counter advanced to the H of the last evicted tracker entry (the standard GD-Size trick that makes recency interact with cost). On prepare_store(), forward a key only when H[key] - L >= store_threshold_h, replacing the current counts.get(key, 0) >= store_threshold check. cost(key) should approximate the marginal TTFT/TPOT saved on a hit: e.g. tokens_per_block * per_token_prefill_latency for the layer/position the key represents, optionally inflated by the block's depth in the shared prefix (deeper blocks save proportionally more recompute when reused). size(key) is the byte cost the block would occupy in CPU offload (block_size * num_layers * 2 * head_dim * dtype_bytes), so blocks for narrower KV groups (e.g. MQA/GQA) get a higher H per byte than full MHA blocks. Keep the OrderedDict + popitem(last=False) eviction structure; the only changes are (a) the value stored per key becomes a float H, (b) lookup() updates L on eviction, and (c) the threshold is compared against H - L. Defaults: choose store_threshold_h so that for a uniform unit-cost/unit-size workload the gate behaves identically to today's store_threshold=2, and continue to no-op the decorator when CPUOffloadingSpec sees store_threshold < 2. Oracles are preserved: 'stored iff score >= threshold AND backing manager accepts', existing FilterReusedOffloadingManager tests can be parameterized over (cost, size) = (1, 1) to recover current behavior, and trace-level hit-rate comparisons remain valid.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'reuse-distance-aware scoring from request patterns' and 'better defaults' as headroom over the current count-only gate; GreedyDual-Size provides a principled, well-studied way to fold per-key cost and size into a single admission score while keeping the bounded-LRU data structure the decorator already uses. In multi-turn agentic traces the impact dimensions are non-uniform: blocks at the head of long shared system/tool prefixes save much more prefill on a hit than tail blocks, and KV blocks for GQA/MQA layers consume far fewer CPU bytes than full-MHA blocks, so a frequency-only threshold systematically wastes CPU residency and PCIe bandwidth on low-value blocks while gating out high-value ones that haven't yet hit the count. GD-Size's H = L + cost/size formulation directly addresses both gaps (TTFT-saving cost in the numerator, CPU/PCIe footprint in the denominator) and the L aging term keeps stale high-cost keys from monopolizing the tracker, matching the candidate's 'decay so stale popularity expires' headroom item. The change is local to reuse_manager.py, preserves the existing prepare_store contract, and degenerates exactly to today's behavior under unit cost/size, which keeps the existing oracle and tests usable as a regression baseline.

---

### 4. Replace LRU tracker eviction in FilterReusedOffloadingManager with SIEVE queue+visited-bit
- **Finding:** `find-0011` — *SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for Web Caches*
- **Source URL:** <https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Swap the OrderedDict-based LRU eviction inside FilterReusedOffloadingManager's bounded frequency tracker (vllm/v1/kv_offload/reuse_manager.py:44-97) for a SIEVE-style structure. Concretely: replace the OrderedDict counts with (a) a FIFO queue of OffloadKey entries, (b) a per-entry visited bit alongside the existing integer count, and (c) a 'hand' pointer that walks backwards from the queue tail. On lookup(key): if the key is already tracked, simply increment its count and set its visited bit (no move_to_end / no queue reordering). If the key is new and the tracker has capacity, push it at the head with count=1 and visited=0. If full, advance the hand from its current position toward the head, clearing visited=1 entries to 0, and evict the first entry whose visited bit is 0; insert the new key at the head. prepare_store() retains its existing semantics: forward only keys whose recorded count >= store_threshold. The visited bit replaces the per-hit OrderedDict.move_to_end work, while keeping the same exact admission contract (key stored iff count >= threshold and backing manager accepts it). Existing FilterReusedOffloadingManager tests and trace-level hit-rate comparisons against the LRU reference become the oracle. The CPUOffloadingSpec gate (store_threshold >= 2, max_tracker_size default 64000) is unchanged; only the tracker's internal eviction discipline changes.

**Proposal rationale.**

The candidate's tracker is on the hot path of every block lookup and grows to ~64k entries by default; today every hit performs a move_to_end on an OrderedDict, which is the kind of per-hit promotion SIEVE was designed to eliminate. SIEVE keeps the same bounded-memory budget while replacing per-hit reordering with a single bit flip, and the paper reports better-or-equal miss ratios than LRU on web/cache traces with very high reuse skew, which matches the multi-turn agentic workload described in caller context (heavy reuse on system/tool prefixes, long tail of one-shot blocks). The visited-bit semantics also map naturally onto admission: an unvisited entry is one whose count never grew enough to matter, so evicting it preserves the high-frequency entries that actually clear store_threshold. This addresses the candidate's stated headroom around scheduler-side policy overhead and viability of larger tracker sizes, without changing the admission contract or the surface used by CPUOffloadingSpec.

---

## Agent proposals

### 1. Hyperbolic admission scoring (hits/age) for FilterReusedOffloadingManager
- **Agent:** claude

**Detailed description.**

Replace the integer counter inside FilterReusedOffloadingManager (vllm/v1/kv_offload/reuse_manager.py:44-97) with a hyperbolic-caching priority. Maintain a monotonically increasing virtual tick `now` advanced by 1 on every lookup() call. For each tracked OffloadKey store a tuple (hits: float, first_seen_tick: int) instead of the current int count, still inside the existing bounded OrderedDict so the popitem(last=False) eviction path and max_tracker_size budget are unchanged. lookup(key): if absent and capacity is full, evict the LRU tracker entry (as today); insert (1.0, now); if present, increment hits by 1.0 and move_to_end as today; advance now. prepare_store(keys): admit a key when hits / max(now - first_seen_tick, 1) >= store_rate_threshold, where store_rate_threshold is derived from the existing `store_threshold` (e.g. store_rate_threshold = (store_threshold - 1) / max_tracker_size, so that under uniform workloads a key seen `store_threshold` times in a tracker-window would clear the gate, making the change degenerate to today's behavior in the limit). Keep CPUOffloadingSpec wiring unchanged: the decorator is still gated by store_threshold>=2 and max_tracker_size still bounds the OrderedDict. Add an optional saturating cap on hits (e.g. 16.0) so a single popular block cannot dominate forever even with later quiet periods, which gives the policy graceful re-evaluation when workload phase changes. Existing FilterReusedOffloadingManager tests pin the contract; add a trace-level hit-rate comparison versus the integer-counter reference on multi-turn agentic traces.

**Novelty rationale.**

The four listed deep_research_proposals all reshape either the tracker's eviction discipline (S3-FIFO probationary FIFO in find-0008, SIEVE visited-bit clock sweep in find-0011) or the admission score's structure (TinyLFU count-min sketch + doorkeeper + batch halving in find-0009, GD-Size H = L + cost/size with cost and size terms in find-0010). None of them score admission by a per-key continuous arrival rate `hits / age` with a wall-tick virtual clock. Hyperbolic caching (Blankstein et al., USENIX ATC 2017) is a distinct family: it uses no global aging counter, no probationary queue, no frequency sketch, and no cost/size weighting; instead each key carries its own birth tick so popularity decays smoothly as a function of elapsed lookups even without any global halving step or eviction event, and the admission decision is rate-based rather than count-based. This directly addresses the candidate's `decay so stale popularity expires` headroom in a way orthogonal to TinyLFU's batch halving and GD-Size's L counter, while leaving the OrderedDict + popitem eviction path and the CPUOffloadingSpec gating untouched, so it does not duplicate any of the four prior findings.

---

### 2. Count distinct request observations, not only lookup probes
- **Agent:** codex

**Detailed description.**

Change FilterReusedOffloadingManager in vllm/v1/kv_offload/reuse_manager.py:44-97 so the tracker records whether an OffloadKey appeared in a distinct request, using both lookup() and prepare_store() as observation points. Replace the OrderedDict value from a bare int to a small entry such as (count, last_seen_req_marker), where last_seen_req_marker is derived from the ReqContext object for the request. Factor the current lookup accounting into a private _observe(key, req_context) helper that increments count only when the marker differs from the entry's last marker, then call _observe() from lookup() and call it for every key at the start of prepare_store() before applying the existing count >= store_threshold filter. This fixes a blind spot in the current policy: prefix lookup stops at the first miss, so later blocks in the same repeatedly-seen prompt may never be counted even though prepare_store() repeatedly sees them as computed blocks; similarly, repeated scheduler probes for the same request should not be mistaken for reuse by different turns. Add tests with two distinct ReqContext instances showing a two-block prompt becomes eligible after two request appearances even when only the first block was looked up, and a same-request lookup()+prepare_store() does not double-count past store_threshold.

**Novelty rationale.**

This is not another replacement eviction/admission algorithm. The S3-FIFO, TinyLFU, GD-Size, SIEVE, and Agent A hyperbolic proposals all change the data structure, priority score, or decay behavior once observations already exist. This proposal changes what counts as an observation: request-level appearances from prepare_store() plus lookup(), with per-request de-duplication. That addresses a scheduler-specific accounting gap in vLLM's offload path that none of the listed proposals cover.

---
