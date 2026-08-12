# ServerRole.serve_external_requests/_process_inbound_lookup

[← vllm/v1/kv_offload](../vllm_v1_kv_offload.md)

- **File:** [`vllm/v1/kv_offload/tiering/p2p/session/server.py`](vllm/v1/kv_offload/tiering/p2p/session/server.py) (lines 449–665)
- **Symbol:** `ServerRole.serve_external_requests/_process_inbound_lookup`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_kv_offload-0019`

## Description
P2P server-side LookupMsg handling batches inbound key probes, re-polls pending keys, pins hits, and emits one aggregated LookupRespMsg.

## Current approach
Uses a per-request _serve_pending work list, de-duplicates keys within a LookupMsg using dict.fromkeys, calls parent.lookup per unique key, pins newly resolved hits through parent.create_store_job, re-polls HIT_PENDING/RETRY keys once per serve window, and forces unresolved keys to MISS after _LOOKUP_PENDING_TIMEOUT_S.

## Estimated impact explanation
Symmetric P2P consumers cannot start promotion until lookup responses arrive. Faster or better-batched inbound lookup resolution can reduce TTFT for peer-cache hits, while the effect is bounded by scheduler-step cadence and remote data-transfer time.

## Evolve rationale
The work-list scheduling, per-message de-duplication, per-key parent lookup loop, hit pinning batch, and 5-second pending timeout are explicit lookup-routing heuristics. Correctness oracle: P2P session tests, lookup response order matching the wire key order, parent.on_request_finished called once per synthetic lookup context, and no duplicate store jobs for the same resolved hit.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Emit an early head-prefix LookupRespMsg for the contiguous HIT run before deadline
- **Agent:** claude

**Detailed description.**

In vllm/v1/kv_offload/tiering/p2p/session/server.py, extend ServerRole._process_inbound_lookup / _resolve_pending_lookups to send a fast partial LookupRespMsg for the longest contiguous HIT prefix of lookup.keys as soon as it is resolved, rather than deferring the entire response until every key settles or the 5s _LOOKUP_PENDING_TIMEOUT_S fires.

Concrete shape (all changes localized to this file/region, lines 449-665):

1. Add an `early_prefix_sent: int` counter on `_ActiveLookup` (default 0) recording how many wire-order keys have already been reported to the peer.
2. In `_poll_lookup_keys` (or a small helper called right after it in both `_process_inbound_lookup` and `_resolve_pending_lookups`), after resolving new keys, walk `lookup.keys[lookup.early_prefix_sent:]` while each key is present in `lookup.resolved` as True (i.e. HIT). Let `k` be the length of this new contiguous HIT run. If `k > 0` and this is not the terminal emission (i.e. some suffix keys are still in `lookup.pending`), emit a partial `LookupRespMsg` carrying just those keys+hits and advance `early_prefix_sent`. Stop the prefix walk on the first non-HIT (either MISS or still-pending) — never speculatively send a MISS ahead of pending keys, because a later re-poll may promote the pending key to HIT and reorder wire semantics.
3. In `_finalize_lookup`, only include `lookup.keys[lookup.early_prefix_sent:]` (and the corresponding hits) in the final `LookupRespMsg`. Skip sending the final message entirely if `early_prefix_sent == len(lookup.keys)` (everything was already delivered by prefix emissions), but still call `parent.on_request_finished(lookup.ctx)` so bookkeeping is released exactly once.
4. Consumer side (symmetric receiver of `LookupRespMsg` for the same `kv_request_id`) must accept multiple LookupRespMsgs and concatenate their (keys, hits) in arrival order rather than assuming one message per LookupMsg. A `partial: bool` flag on the wire (last message has `partial=False`) keeps the change forward-compatible; older peers that don't understand `partial` still see well-formed key-aligned responses.
5. Preserve the existing pin-batch discipline: `_pin_and_register_hits` already batches `parent.create_store_job` per poll pass, so early emission does not multiply store jobs — it only changes when the wire message is flushed.

Correctness invariants preserved: wire-order key/hit alignment (concatenation of partial messages equals the current single message); exactly-one `parent.on_request_finished` per synthetic lookup ctx (still fired from `_finalize_lookup`); no duplicate store jobs for the same resolved hit (pin bookkeeping unchanged); `_finish_inbound_lookups` still safely pops parked `_ActiveLookup`s (early-emitted keys are already resolved, so a terminal-FetchMsg race just drops the residual suffix as before).

Why this helps TTFT/TPOT on a multi-turn agentic workload: agentic turns share long system-prompt and tool-schema prefixes across turns, so the peer's LookupMsg is typically head-heavy — the leading blocks correspond to reusable prefix content that is almost always already resident on the responder, while the trailing blocks (fresh user turn, tool output) are frequently in-flight (HIT_PENDING). Under the current design the consumer stalls on the whole batch, potentially up to 5s of the pending-timeout, before it can start promotion of the certain-hit prefix. Emitting the contiguous HIT prefix immediately lets the consumer kick off remote block fetches for the prefix in parallel with the responder still waiting on the pending suffix, collapsing lookup latency into fetch latency for the common case and materially reducing median TTFT on peer-cache hits without changing MISS behavior or scheduler-step cadence.

Validation hooks: existing P2P session tests that assert `parent.on_request_finished` is called once per synthetic lookup ctx and that no duplicate store jobs are created should pass unchanged; add a targeted test where a LookupMsg has a HIT prefix followed by a HIT_PENDING key, and assert (a) a partial LookupRespMsg for the prefix is observed before the deadline, (b) the final LookupRespMsg carries only the suffix, (c) the concatenation matches the wire order of the original request.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any concrete change is novel by construction. Beyond that, this proposal is specifically about carving the aggregated LookupRespMsg into an early head-prefix emission plus a suffix — a wire-protocol change to the one-message-per-LookupMsg invariant called out in the current_approach — rather than tuning the 5s timeout, changing de-duplication, batching lookups differently, or reordering the per-key parent.lookup loop. It directly targets the TTFT bottleneck for multi-turn agentic workloads (shared head prefix, freshly-in-flight tail) by decoupling head-hit signaling from tail resolution, which none of the surface-level knobs on this candidate address.

---

### 2. Throttle pending-key re-polls with a per-serve lookup budget
- **Agent:** codex

**Detailed description.**

In `vllm/v1/kv_offload/tiering/p2p/session/server.py`, change `_resolve_pending_lookups` so it does not re-call `parent.lookup` for every still-pending key on every scheduler step. Add a small per-serve budget and a cursor/queue on `_ActiveLookup` for pending keys, then poll at most that budget across parked lookups for the request, carrying unpolled keys forward until the next `serve_external_requests` call or the existing deadline. Keep the first-sighting path in `_process_inbound_lookup` unchanged so newly arrived LookupMsgs still get an immediate full pass; apply throttling only to repeated HIT_PENDING/RETRY re-polls. When the deadline expires, force all remaining pending keys to MISS exactly as today and finalize normally. This caps scheduler-step CPU spent on remote lookup bookkeeping when many peer-cache blocks are in-flight, which is common in multi-turn agentic prompts with large shared prefixes and fresh tail blocks, and should reduce TPOT jitter without changing the wire response shape or pinning semantics.

**Novelty rationale.**

There are no deep_research_proposals listed for this candidate. Agent A proposes sending partial early LookupRespMsg messages for a resolved contiguous HIT prefix, which changes response timing and wire semantics. This proposal instead preserves the one-response-per-LookupMsg contract and targets server-side scheduling cost by limiting repeated pending-key re-polls per serve window; it does not emit partial responses, add a `partial` flag, or change client-side response concatenation.

---
