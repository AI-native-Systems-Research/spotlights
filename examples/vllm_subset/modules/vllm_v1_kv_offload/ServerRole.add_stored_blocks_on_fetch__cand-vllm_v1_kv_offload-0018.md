# ServerRole.add_stored_blocks/on_fetch

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/tiering/p2p/session/server.py`](vllm/v1/kv_offload/tiering/p2p/session/server.py) (lines 305–390)
- **Symbol:** `ServerRole.add_stored_blocks/on_fetch`
- **Kind:** region
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_kv_offload-0018`

## Description
P2P server role matches locally stored blocks with peer fetch demand and submits transfers when supply and demand meet.

## Current approach
Maintains per-round available and demanded dicts keyed by OffloadKey. Stored blocks and FetchMsg demand are matched immediately; symmetric lookup-supplied rounds fail unmatched demand immediately, while PD rounds park demand until store supply arrives.

## Estimated impact explanation
P2P secondary hits can replace local prefill for multi-turn or disaggregated agentic traffic. Matching and batching decisions determine how quickly peer-held KV reaches the consumer, so they can directly move median TTFT and reduce TPOT stalls when decode waits for fetched blocks.

## Evolve rationale
The supply/demand matching order, per-round partitioning, immediate transfer submission, and symmetric-P2P fail-fast rule are hand-rolled routing heuristics. Correctness oracle: tests/v1/kv_offload/tiering/p2p/test_sessions.py, one StoreResult per contributing job, one TransferDoneMsg per fetch round, and preservation of key-to-block-index mapping across matched transfers.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Priority-aware first-block fast-path in add_stored_blocks/on_fetch to cut decode-blocking TPOT
- **Agent:** claude

**Detailed description.**

Extend ServerRole.add_stored_blocks and ServerRole.on_fetch (vllm/v1/kv_offload/tiering/p2p/session/server.py:305-390) so that transfer submission is ordered by fetch-round urgency rather than arrival order, and so that the very first matched block for a decode-blocking round is submitted immediately while later matches in the same batch are optionally coalesced. Concretely: (1) Add an optional `priority` hint on FetchMsg (or infer it from the peer's decode-vs-prefill state already tracked in _ServerRequestState) and stash it on _OutboundRequestState when on_fetch binds demand. (2) In add_stored_blocks, when result.local_idxs is non-empty and rnd.demand_received is True, do not unconditionally call _submit_transfer for the entire matched batch. Instead, split result.local_idxs into a `head` (the first 1-N blocks, sized to one NIC-MTU-aligned transfer) and a `tail`. Submit `head` immediately; enqueue `tail` on a per-session ready-queue keyed by priority. (3) Drain the ready-queue in priority order at the end of add_stored_blocks and on_fetch — highest-priority rounds' tails go before lower-priority rounds' heads that arrived on the same tick. (4) Preserve the existing correctness oracle: one StoreResult per contributing job, one TransferDoneMsg per fetch round (accumulate transfer callbacks so TransferDoneMsg fires only after the split submissions all complete), and preserve the key→block_index mapping — the split is purely on the local_idxs list, keys stay bound. (5) The symmetric-P2P fail-fast rule in on_fetch (lookup_supplied + demanded) is unchanged; the PD parked-demand path is unchanged; only the submission ordering and head/tail split are added. Net effect for the multi-turn agentic workload: the first block of a decode-critical fetch round leaves the server as soon as either the store or the fetch closes the match, so the consumer's decode can resume against the first block while remaining blocks stream in — directly targeting median TPOT — while lower-priority prefill-only fetches yield the NIC without blocking the queue.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete change is novel by construction. Beyond that, this proposal is specifically distinct from the obvious 'batch/coalesce stores' angle implicit in the evolve_rationale: it argues the opposite direction (split the first matched batch to shorten first-block latency) and layers a priority queue keyed by decode-vs-prefill urgency onto the existing per-round matching, rather than replacing the matching logic. It also does not change the symmetric-P2P fail-fast rule or the PD parking rule, so it composes with — rather than supplants — the hand-rolled routing heuristics the evolve_rationale flags.

---

### 2. Coalesce per-peer matched rounds into a single transfer submission burst
- **Agent:** codex

**Detailed description.**

In vllm/v1/kv_offload/tiering/p2p/session/server.py:305-390, change ServerRole.add_stored_blocks/on_fetch so that when multiple OffloadKey matches for the same peer/session become ready in the same call, the role accumulates them into a per-peer pending-transfer accumulator and invokes the lower-level transfer submission once for the whole ready set instead of one submission per matched round/key. The accumulator should preserve the existing observable contract by retaining each round's key-to-block-index mapping and completion accounting: each contributing store job still receives exactly one StoreResult, and each fetch round still emits exactly one TransferDoneMsg after all blocks for that round have completed. This is a batching-only change: keep the current per-round supply/demand matching semantics, symmetric-P2P fail-fast behavior, and PD parked-demand behavior, but defer the actual _submit_transfer calls until the end of add_stored_blocks/on_fetch so adjacent ready matches for the same peer are packed together. The expected win for multi-turn agentic traffic is lower control-plane overhead and fewer small DMA/RDMA sends when many secondary-hit blocks are discovered together, improving median TTFT for prefill continuation and reducing TPOT stalls caused by transfer setup overhead.

**Novelty rationale.**

There are no deep_research_proposals listed. This is distinct from Claude's proposal, which prioritizes decode-blocking rounds and splits a ready batch into head/tail submissions to minimize first-block latency. This proposal instead preserves whole-round readiness and focuses on coalescing multiple ready rounds or keys for the same peer into fewer transfer submissions, targeting per-transfer setup overhead rather than urgency ordering or first-block fast paths.

---
