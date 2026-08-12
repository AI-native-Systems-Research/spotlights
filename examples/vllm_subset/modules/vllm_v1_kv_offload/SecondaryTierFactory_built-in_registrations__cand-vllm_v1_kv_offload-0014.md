# SecondaryTierFactory built-in registrations

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/tiering/factory.py`](vllm/v1/kv_offload/tiering/factory.py) (lines 103–125)
- **Symbol:** `SecondaryTierFactory built-in registrations`
- **Kind:** plugin_seam
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_kv_offload-0014`

## Description
Registration site for pluggable secondary offload tiers selected from secondary_tiers configuration entries.

## Current approach
The interface is SecondaryTierManager in vllm/v1/kv_offload/tiering/base.py; built-in implementations include ExampleSecondaryTierManager in vllm/v1/kv_offload/tiering/example/manager.py, FileSystemTierManager in vllm/v1/kv_offload/tiering/fs/manager.py, P2PSecondaryTierManager in vllm/v1/kv_offload/tiering/p2p/manager.py, and ObjectStoreSecondaryTierManager in vllm/v1/kv_offload/tiering/obj/manager.py. Runtime selection uses each secondary_tiers[].type entry and optional module_path via SecondaryTierFactory.get_tier_class.

## Estimated impact explanation
Secondary-tier latency determines how expensive non-primary hits are. A faster or locality-aware tier implementation can reduce promotion wait time and improve median TTFT for multi-turn workloads whose reused KV blocks spill beyond the CPU primary tier.

## Evolve rationale
Secondary tiers encapsulate lookup, submit_load, submit_store, polling, and completion contracts behind a stable factory. New tiers or variants can optimize storage media, network routing, batching, or locality policy while preserving JobMetadata/JobResult semantics. Correctness oracle: factory tests, tier-specific tests, and the SecondaryTierManager contract that submitted jobs eventually report exactly one result and return valid LookupResults.

## Deep research proposals

### 1. Register a Mooncake-style shared distributed KV pool as a secondary tier
- **Finding:** `find-vllm_v1_kv_offload-0001` — *Serving Agentic Workloads at Scale with vLLM x Mooncake*
- **Source URL:** <https://vllm.ai/blog/2026-05-06-mooncake-store>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new SecondaryTierManager implementation modeling a shared distributed KV cache pool (Mooncake-style) and register it in vllm/v1/kv_offload/tiering/factory.py alongside the existing 'example', 'fs', 'p2p', and 'obj' entries (lines 103-125). The new tier (e.g. type 'mooncake' or 'shared_kv_pool', imported from a new module under vllm/v1/kv_offload/tiering/) implements SecondaryTierManager.lookup / submit_load / submit_store / get_finished_jobs by talking to a cluster-wide KV store. Data movement runs on a dedicated background I/O thread (RDMA where available), so submit_* remain non-blocking as required by the SecondaryTierManager contract (base.py:114-117, 156-206). lookup() consults the shared pool so the scheduler can hit prefixes stored by *other* workers in prior turns, matching the finding's 'scheduler-side block lookup and worker-side asynchronous data movement' shape. JobMetadata/JobResult semantics are preserved: each submitted job eventually reports exactly one result, and lookup returns HIT/MISS/RETRY unchanged. Registration is a single register_tier() call appended to the built-in block at factory.py:103-125; no changes to get_tier_class or the SecondaryTierManager base are required.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out 'network routing' and 'locality policy' as levers a new tier can pull, and estimated_impact notes that better secondary-tier latency reduces promotion wait and improves median TTFT for multi-turn workloads whose reused KV blocks spill beyond the CPU primary tier — which is precisely the multi-turn agentic workload named in the caller context. The finding describes a production system (Mooncake) that reuses prefixes 'turn after turn' via a shared pool and offloads transfers to a dedicated background I/O thread, mapping cleanly onto SecondaryTierManager's async submit/poll contract and the existing plugin seam. The existing P2P tier covers only pairwise worker transfers, not a shared cluster-wide pool that survives across workers and turns, so this is additive rather than duplicative.

---

### 2. Register a TinyLFU-admission secondary tier variant behind the factory seam
- **Finding:** `find-vllm_v1_kv_offload-0005` — *TinyLFU: A Highly Efficient Cache Admission Policy*
- **Source URL:** <https://paperity.org/p/377179711/tinylfu-a-highly-efficient-cache-admission-policy>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new pluggable SecondaryTierManager variant registered via SecondaryTierFactory.register_tier (vllm/v1/kv_offload/tiering/factory.py lines 103-125) that wraps one of the existing tier backends (fs, p2p, or obj) with a TinyLFU admission gate. The wrapper implements the SecondaryTierManager contract in vllm/v1/kv_offload/tiering/base.py unchanged for lookup/submit_load/polling/completion, but intercepts submit_store: on capacity pressure, it consults an approximate frequency sketch (e.g., Count-Min Sketch with periodic aging via a doorkeeper Bloom filter) keyed by block hash, and admits the candidate only when its estimated recent frequency exceeds that of the tier's eviction victim. Under-frequency stores are dropped rather than written, and the sketch is updated on every lookup and store attempt so that repeated agent prefixes accumulate weight. Register it as a new short name (e.g. 'tinylfu_fs' or accept a base-tier config key) alongside the existing 'example', 'fs', 'p2p', 'obj' entries. JobMetadata/JobResult semantics are preserved: every submitted store still produces exactly one result (success or admission-rejected), and LookupResults remain valid because rejected blocks simply never appear on subsequent lookups.

**Proposal rationale.**

The candidate's evolve_rationale explicitly identifies 'locality policy' as a dimension new tier variants can optimize while preserving the JobMetadata/JobResult contract, and the factory seam is designed to accept exactly this kind of drop-in. The finding contributes a concrete, well-studied admission mechanism (TinyLFU's frequency-sketch gate: 'decides, based on the recent access history, whether it is worth admitting the new item') that directly addresses secondary-tier eviction churn under capacity pressure. For the stated multi-turn agentic workload, prefix KV blocks are re-accessed with skewed frequency; admitting only high-frequency candidates keeps hot prefixes resident longer, reduces wasted write bandwidth to slower media, and shortens promotion wait time on subsequent turns — directly improving median TTFT per the caller objective. The change is confined to a new tier class plus one register_tier call, so it composes with the existing seam without altering the interface or other tiers.

---

### 3. Register an SSD-object secondary tier optimized for bulk transfers and slack-aware scheduling
- **Finding:** `find-vllm_v1_kv_offload-0006` — *Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving*
- **Source URL:** <https://arxiv.org/abs/2605.03375>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new built-in secondary tier registration at vllm/v1/kv_offload/tiering/factory.py (lines 103-125) alongside the existing 'fs', 'p2p', and 'obj' entries — e.g. SecondaryTierFactory.register_tier('ssd-obj', 'vllm.v1.kv_offload.tiering.ssd_obj.manager', 'SsdObjectTierManager'). The new SecondaryTierManager implementation coalesces contiguous per-block store/load jobs into larger KV-cache 'objects' (chunked writes covering multiple blocks) instead of issuing per-block filesystem tasks, and drives loads through an asynchronous GPU-direct-style object I/O path when available (falling back to batched os.readv/pwritev). Its scheduler treats submit_load/submit_store as slack-aware: it defers non-critical stores when a load is on the critical path to first-token generation and reorders queued work by TTFT slack, while still preserving the SecondaryTierManager contract (exactly one JobResult per submitted job, valid LookupResult objects). The registration slot is the only edit at the factory; the manager, its file-object layout, and its DualQueueThreadPool-equivalent scheduler live under a new tiering/ssd_obj/ subpackage modeled on tiering/fs/.

**Proposal rationale.**

The candidate is explicitly a plugin seam whose evolve_rationale invites new tier variants that 'optimize storage media, network routing, batching, or locality policy while preserving JobMetadata/JobResult semantics'. The current 'fs' tier issues per-block filesystem I/O (see FileSystemTierManager's file naming scheme <hash_hex>.bin at vllm/v1/kv_offload/tiering/fs/manager.py:14 and its per-block batch_load_block/batch_store_block calls), which is exactly the fragmented tiny-random-I/O pattern the finding warns against for SSD-backed KV cache. The finding contributes three concrete, transferable ideas — bulk object abstractions replacing per-block tasks, asynchronous GPU-direct object I/O, and slack-aware scheduling — that map cleanly onto a new tier registration and can plausibly reduce median TTFT for the caller's multi-turn agentic workload, whose reused KV blocks spill beyond the CPU primary tier and pay the promotion cost the estimated_impact_explanation calls out.

---

### 4. Register a GPUDirect Storage (GDS) secondary tier for direct, batched storage→GPU KV loads
- **Finding:** `find-vllm_v1_kv_offload-0007` — *GPUDirect Storage Overview Guide*
- **Source URL:** <https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new built-in secondary tier registration (e.g., "gds") to SecondaryTierFactory in vllm/v1/kv_offload/tiering/factory.py (lines 103-125), alongside the existing "example", "fs", "p2p", and "obj" entries. The new tier implements the SecondaryTierManager contract from vllm/v1/kv_offload/tiering/base.py (lookup, submit_store, submit_load, get_finished_jobs, drain_jobs, take_events, shutdown) using NVIDIA GPUDirect Storage: cuFile handles per block file plus cuFileBatchIOSubmit/cuFileBatchIOGetStatus to submit and poll many block I/Os per call, with completion ordered on a dedicated CUDA stream so promotion into the primary KV view is visible to compute streams via a stream-wait rather than a host round-trip. Mirror the FileSystemTierManager's file layout and FileMapper usage so a GDS tier can coexist with or transparently replace the pure-Python "fs" tier where cuFile is available; keep JobMetadata/JobResult semantics and the "exactly one JobResult per submitted job" invariant intact. Group adjacent block IDs from a single submit_load/submit_store call into one batched cuFile submission to amortize per-submission fixed cost, and fall back to the existing "fs" tier (or refuse registration) when cuFile probing fails at construction time so behavior stays fail-closed.

**Proposal rationale.**

The candidate is explicitly the plugin seam for pluggable secondary tiers, and its estimated-impact rationale identifies secondary-tier latency as a driver of promotion wait time and median TTFT for multi-turn workloads whose KV blocks spill beyond CPU. The finding contributes a specific, transferable technique from NVIDIA's GDS documentation: batched asynchronous file I/O with CUDA-stream-ordered completion, with the cited quote ("batching reduces the overhead by amortizing that fixed overhead across the transactions in the batch.") mapping directly onto submit_load / submit_store which already receive multi-block JobMetadata batches. Today's "fs" tier stages every block through a CPU bounce buffer and a Python/pthread thread pool; a GDS tier bypasses that bounce buffer, moves storage→GPU DMA off the CPU, and lets storage completion synchronize with compute via a stream event instead of a scheduler poll. That directly shortens the promotion critical path that gates TTFT once the CPU primary tier is exceeded, which is the specific gap the caller flagged. The change lives at the factory seam as required, is additive (a new registration line), and preserves the SecondaryTierManager contract used by existing factory tests.

---

### 5. Add a bandwidth-adaptive KV-compression secondary tier (CacheGen-style)
- **Finding:** `find-vllm_v1_kv_offload-0009` — *CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving*
- **Source URL:** <https://arxiv.org/abs/2310.07240>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Introduce a new SecondaryTierManager implementation, e.g. `CacheGenTierManager` in `vllm/v1/kv_offload/tiering/cachegen/manager.py`, and register it in the built-in table at `vllm/v1/kv_offload/tiering/factory.py` (lines 103-125) alongside `example`, `fs`, `p2p`, and `obj` via `SecondaryTierFactory.register_tier("cachegen", "vllm.v1.kv_offload.tiering.cachegen.manager", "CacheGenTierManager")`. The new tier wraps an existing backing tier (fs/obj/p2p) but transforms KV blocks in `submit_store`/`submit_load` through a CacheGen-style encoder/decoder: quantize and entropy-code KV tensors before persisting, and decode on promotion. Configuration keys on the tier entry select the underlying backend, a base compression level per KV component (e.g. bits per K vs V element, per-layer sensitivity), and a policy that adapts the compression level to the measured effective bandwidth of the backend (bytes moved / job latency tracked over recent JobResults). The manager preserves the SecondaryTierManager contract exactly: `lookup` still returns valid LookupResults keyed by block hash, `submit_store`/`submit_load` still return one JobResult per submitted job through `poll_completed`, and JobMetadata carries the compressed size so callers see accurate bytes-in-flight. No changes to the factory API or to other tiers; this is purely a new built-in registration entry and its backing module.

**Proposal rationale.**

The candidate is the plugin seam where new secondary tiers are registered, and its impact note explicitly identifies bandwidth-bound promotion from CPU/remote storage as the dominant cost for multi-turn workloads whose reused KV spills beyond the primary tier. CacheGen's contribution — compressing KV tensors with a specialized encoder and adapting compression level to available bandwidth — targets exactly that bottleneck: it reduces bytes transferred on the promotion path, which in the bandwidth-bound regime translates directly into lower promotion latency and therefore lower median TTFT on multi-turn agentic workloads (the stated caller objective). Because SecondaryTierManager encapsulates lookup/submit_load/submit_store/poll_completed behind a stable contract, a compression-aware tier can be added as a peer of `fs`/`obj`/`p2p` without disturbing existing tiers, correctness oracles (factory tests, tier tests, JobMetadata/JobResult semantics) still apply, and users opt in per-deployment via `secondary_tiers[].type="cachegen"`. The finding is neither a restatement of the current approach (no existing tier compresses or adapts to bandwidth) nor merely topical — it names a concrete, transferable mechanism that fits this seam.

---

### 6. Register a usefulness-admission secondary tier variant that gates promotions by predicted reuse
- **Finding:** `find-vllm_v1_kv_offload-0010` — *InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management*
- **Source URL:** <https://www.alphaxiv.org/abs/2406.19707>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new built-in registration in vllm/v1/kv_offload/tiering/factory.py (lines 103-125) for a SecondaryTierManager variant (e.g. type="admission") that wraps an inner secondary tier and applies a lightweight promotion-admission filter before submit_load. Concretely, the new manager implements the SecondaryTierManager contract from vllm/v1/kv_offload/tiering/base.py but overrides submit_load to consult a per-key usefulness score derived from cheap block-level signals available in the scheduler (e.g. per-key hit recency/frequency accumulated via touch(), per-request kv_transfer_params hints from ReqContext, and layer/position derived from OffloadKey). Keys scored below a configurable threshold are dropped from the JobMetadata before delegation, and reported as MISS/skipped via get_finished_jobs() so JobResult semantics remain intact (exactly one result per submitted job, with successful_keys marking the admitted subset on partial admission). lookup() delegates unchanged so hits are still discoverable, but only "essential" blocks are actually promoted to CPU. The registration follows the same shape as the existing "fs", "p2p", "obj", and "example" entries at lines 103-125, with class placed under vllm/v1/kv_offload/tiering/admission/manager.py.

**Proposal rationale.**

The candidate is the registration seam for pluggable secondary tiers, and InfiniGen's transferable insight (per the finding's own technique_summary) is "promotion admission based on predicted usefulness instead of blindly recalling every available block." The candidate's evolve_rationale explicitly identifies promotion wait time as the primary lever for improving median TTFT on multi-turn agentic workloads whose reused KV spills beyond the CPU tier - which is the same workload class InfiniGen targets. InfiniGen's model-internal attention rehearsal is out of scope for this seam (secondary tiers run in the Scheduler and cannot see attention activations), but the block-level analog - selective admission using signals already available to SecondaryTierManager (touch history, ReqContext, OffloadKey) - fits cleanly within the existing contract and is a new tier variant rather than a modification of existing tiers. This addresses the gap that every current tier implementation promotes every hit block unconditionally, paying full transfer cost even for blocks unlikely to be re-referenced within the request's remaining attention window.

---

## Agent proposals

### 1. Register a hierarchical composite tier that federates existing secondary tiers with fast/slow cascade
- **Agent:** claude

**Detailed description.**

Add a new built-in registration at vllm/v1/kv_offload/tiering/factory.py:103-125 alongside 'example', 'fs', 'p2p', and 'obj' — SecondaryTierFactory.register_tier('hierarchical', 'vllm.v1.kv_offload.tiering.hierarchical.manager', 'HierarchicalTierManager'). The new SecondaryTierManager instantiates N inner tiers from a nested config (e.g. inner_tiers=[{type: 'fs', capacity_gb: 32, ...}, {type: 'obj', bucket: ..., ...}]) by calling SecondaryTierFactory.create_secondary_tier recursively, forming an ordered hierarchy from fastest to slowest medium. lookup() polls inner tiers in order and returns the first HIT (RETRY if any inner returns RETRY before all others MISS); submit_store() writes into the fastest inner tier and, on capacity eviction there, cascades the victim to the next inner tier via that tier's submit_store with is_promotion=False rather than dropping it, preserving cold-but-live blocks; submit_load() promotes from whichever inner holds the block, and may (as a write-through-on-read policy) simultaneously refill the fastest inner so the next reuse hits the fast tier. get_finished_jobs() fans in from all inner tiers, remapping their per-tier JobIds to the composite's outer JobId while preserving the exactly-one-JobResult-per-submitted-job invariant of the SecondaryTierManager contract; on_new_request, on_request_finished, touch, take_events, has_pending_work, drain_jobs, and shutdown fan out to all inners. Because it composes existing tiers via the same factory, no changes are required to base.py, JobMetadata/JobResult semantics, or other tier implementations — deployments configure e.g. secondary_tiers=[{type:'hierarchical', inner_tiers:[fs-cache, obj-store]}] and the fast tier absorbs the recency-skewed hot prefixes of a multi-turn agent while the slow tier extends effective capacity.

**Novelty rationale.**

None of the six existing deep_research_proposals construct a hierarchy across existing tiers. #1 (Mooncake shared pool) adds a single new backend that shares state cross-worker; #2 (TinyLFU admission) gates writes into one tier by estimated frequency but does not add a second slower tier for rejected/evicted blocks; #3 (SSD-object) is a single new medium with bulk coalescing and slack-aware scheduling; #4 (GDS) is a single new storage medium with cuFile batched I/O; #5 (CacheGen) is a single compression-wrapping tier; #6 (usefulness admission) filters keys inside a single load job. The composite tier is orthogonal: it reuses whichever tiers already exist (including any of the six above once implemented) and adds a fast->slow cascade with write-through-on-read and eviction-as-demotion instead of eviction-as-drop, which is the specific mechanism that turns capacity pressure in the fast tier into slow-tier hits rather than full recomputes — a lever none of the six pull. It also plugs into the exact factory seam the candidate identifies, using SecondaryTierFactory itself to build its inner tiers so it composes with, rather than replaces, prior proposals.

---
