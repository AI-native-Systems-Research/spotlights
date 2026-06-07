# Scheduler.schedule (WAITING-phase admission loop)

[← v1.core](../v1.core.md)

- **File:** [`vllm/v1/core/sched/scheduler.py`](vllm/v1/core/sched/scheduler.py) (lines 567–846)
- **Symbol:** `Scheduler.schedule (WAITING-phase admission loop)`
- **Kind:** loop
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0002`

## Description
Admission loop that promotes blocked WAITING requests, enforces LoRA capacity, probes local and external prefix caches, chooses prefill chunk sizes, schedules multimodal encoder inputs, and allocates KV slots before moving requests to RUNNING or back to skipped_waiting.

## Current approach
Greedy single-pass over waiting/skipped queues. The queue head is tried first, prefix-cache probing happens only after selection, allocation failure or unschedulable encoder input breaks admission, and skipped blocked/LoRA requests are requeued ahead of older skipped work.

## Estimated impact explanation
This loop controls which queued request gets first-token work. Better admission and chunking policies can move median TTFT materially for agentic turns whose prompts mostly hit prefix cache.

## Evolve rationale
TTFT for multi-turn agentic workloads is dominated by admitting cached-prefix requests quickly while preserving decode capacity. The concrete policy knobs are the queue-head selection, prefix-hit query placement, chunked-prefill clamp, break-on-fail behavior, and skipped_waiting requeue rule. Headroom includes prefix-hit-aware ordering, retry with smaller chunks, and explicit decode-prefill budget partitioning. Correctness oracles include tests/v1/core/test_scheduler.py, tests/v1/core/test_prefix_caching.py, and tests/v1/core/test_scheduler_e2e.py; PrefixCacheStats and SchedulerStats give measurable signals.

## Deep research proposals

### 1. Defer encoder-bound requests during WAITING admission to avoid encoder/prefill interference
- **Finding:** `find-0001` — *Efficiently Serving Large Multimodal Models Using EPD Disaggregation*
- **Source URL:** <https://arxiv.org/html/2501.05460v2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify the WAITING-phase admission loop in vllm/v1/core/sched/scheduler.py:567-846 so that multimodal encoder scheduling no longer gates admission with a break-on-fail. Inspired by EPD disaggregation, treat the encoder stage as a logically separate resource: when probing a WAITING request, classify its encoder inputs as (a) already-cached/encoded (multimedia-token cache hit), (b) currently encodable within remaining encoder budget, or (c) requiring fresh encoder work that would contend with prefill this step. Admit (a) and (b) as today, but for (c) push the request to skipped_waiting instead of breaking the loop, so the scheduler can continue probing other waiting requests whose encoder work is ready (especially text-only or cache-hit prompts) and fill the prefill/decode budget this step. Additionally, when ordering the queue head, prefer requests whose multimodal encoder outputs are already cached, mirroring the prefix-hit-aware ordering knob already called out in evolve_rationale. Plumb the multimodal cache-hit signal through the existing prefix-cache probe path (reuse PrefixCacheStats counters) and add SchedulerStats counters for encoder-deferred admissions. Validate using tests/v1/core/test_scheduler.py and tests/v1/core/test_scheduler_e2e.py with multimodal fixtures, and confirm no regression in tests/v1/core/test_prefix_caching.py.

**Proposal rationale.**

The candidate explicitly schedules multimodal encoder inputs inside the admission loop and currently breaks admission when an encoder input is unschedulable, which is exactly the encoder/prefill interference EPD targets. Full disaggregation is out of scope for a single loop change, but the finding's core idea — treat encoding as a separable stage and make repeated media cheap — maps directly onto two concrete policy knobs the candidate already lists: break-on-fail behavior and queue-head selection. Deferring encoder-bound requests to skipped_waiting (instead of breaking) and prioritizing media-cache-hit requests reduces media TTFT for multi-turn agentic workloads where many turns reuse the same image/audio context, without requiring new infrastructure.

---

### 2. Agent-step-aware admission ordering for cached-prefix requests
- **Finding:** `find-0003` — *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows*
- **Source URL:** <https://arxiv.org/abs/2507.07400>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In Scheduler.schedule's WAITING-phase admission loop (vllm/v1/core/sched/scheduler.py:567-846), extend the queue-head selection step to consult an optional per-request agent-step metadata field (a steps-to-execution value, supplied by the caller when running an agent workflow). When such metadata is present, before falling through to the current FIFO/skipped-queue order, prefer admitting requests whose (a) reported steps-to-execution is small (i.e. temporally imminent next activations) and (b) prefix-cache probe indicates a high local-hit ratio. Concretely: (1) thread an optional steps_to_execution attribute through Request / SchedulingRequest so it is visible to the admission loop without changing existing call sites; (2) in the WAITING loop, after computing the candidate set per LoRA/capacity gating, sort the candidates by (steps_to_execution ascending, prefix-cache hit length descending) instead of strict queue order, with a fallback to existing FIFO when metadata is absent; (3) keep the skipped_waiting requeue rule but bias requeue position by the same key so older skipped work is not unduly starved; (4) when a high-priority, high-prefix-hit candidate is selected, issue a non-blocking prefetch hint to the prefix-cache layer for its remaining uncached suffix blocks (using the existing async block load path, no new transport). All changes gated behind a feature flag so default behavior is unchanged when no agent-step metadata is supplied. Validate using tests/v1/core/test_scheduler.py and tests/v1/core/test_prefix_caching.py and measure TTFT via PrefixCacheStats and SchedulerStats on a multi-turn agentic trace.

**Proposal rationale.**

The candidate explicitly lists prefix-hit-aware ordering as headroom and targets multi-turn agentic TTFT, where the admission loop controls which queued request earns first-token work. KVFlow's transferable idea is that an Agent Step Graph yields per-request temporal-proximity values (steps-to-execution) that better predict near-future demand than LRU/FIFO. While KVFlow applies that signal to eviction and prefetch, the same signal is directly usable at admission: requests close to activation with cached prefixes are exactly the ones whose admission most reduces median TTFT for agentic turns. This addresses the gap that the current loop tries the queue head first and only probes prefix-cache after selection, which can admit a non-imminent or cache-cold request ahead of an imminent cache-hot one. The proposal stays conservative by being purely additive (opt-in metadata, fallback to current behavior) and reuses existing prefix-cache probing and async-load paths rather than introducing new mechanisms.

---

### 3. Rank WAITING admissions by resident prefix-cache benefit (RadixAttention-style cache-aware scheduling)
- **Finding:** `find-0004` — *Efficiently Programming Large Language Models using SGLang*
- **Source URL:** <https://arxiv.org/html/2312.07104v1>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In Scheduler.schedule's WAITING-phase admission loop (vllm/v1/core/sched/scheduler.py:567-846), replace the head-first greedy pass over waiting/skipped queues with a cache-aware selection that probes the local prefix cache (and, when configured, the external KV cache) for each admission candidate before choosing whom to admit, then prefers the candidate whose matched-prefix length (or matched-token fraction relative to prompt length) is largest among requests that fit current LoRA, encoder, and KV-block budgets. Concretely: (1) move the prefix-hit probe (currently performed after the queue head is selected) to a lookahead step that scores the top-K waiting/skipped entries by hit length; (2) admit in descending hit-score order, with a small bounded budget K and a tie-breaker on arrival time / existing priority to preserve fairness; (3) keep the existing LoRA capacity, multimodal encoder, and chunked-prefill clamp checks unchanged so correctness oracles in tests/v1/core/test_scheduler.py, tests/v1/core/test_prefix_caching.py, and tests/v1/core/test_scheduler_e2e.py still hold; (4) leave the skipped_waiting requeue path intact but feed it through the same scoring so older skipped work is not starved when no candidate has a hit. Telemetry should reuse PrefixCacheStats/SchedulerStats to expose hit-rate and median-TTFT deltas.

**Proposal rationale.**

The candidate's evolve_rationale explicitly lists 'prefix-hit-aware ordering' as a headroom item and notes that prefix-cache probing currently happens only after queue-head selection, so high-hit candidates behind the head pay full prefill cost. The SGLang finding contributes a concrete, transferable mechanism for this gap: RadixAttention's cache-aware scheduling ranks requests by matched-prefix length against the resident KV cache, which is exactly the signal vLLM already computes (just later in the pipeline). For the targeted multi-turn agentic workload, where successive turns share long system/tool prefixes, admitting the highest-hit request first directly reduces prefill tokens issued and therefore median TTFT, while leaving decode capacity for in-flight requests untouched. The change is local to the admission loop, reuses existing prefix-probe utilities, and is testable against the listed oracles.

---

### 4. Replace FIFO admission with k-LPM prefix-aware ordering in the WAITING loop
- **Finding:** `find-0005` — *LLM Query Scheduling with Prefix Reuse and Latency Constraints*
- **Source URL:** <https://openreview.net/forum?id=HKfZwLjSwQ>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/core/sched/scheduler.py:567-846 (Scheduler.schedule WAITING-phase admission loop), replace the current FIFO queue-head selection with a k-LPM (k-Longest Prefix Match) admission policy. Concretely: (1) before the greedy single-pass over waiting/skipped queues, probe the local and external prefix caches for the top-k waiting requests so cached-prefix length is known prior to selection (currently the probe happens only after the queue head is chosen); (2) at each admission step, pick the request whose longest-prefix-match maximizes a weighted score subject to a per-request wait-time/age bound that enforces fairness, generalizing between FCFS (k=1) and pure cache-hit prioritization (k=|queue|); (3) extend the same ordering to the skipped_waiting requeue rule so older skipped work is not starved by newer prefix-heavy arrivals; (4) expose k (and the fairness bound) as a scheduler config knob so it can be tuned per workload. Reuse PrefixCacheStats/SchedulerStats for measurement and validate with tests/v1/core/test_scheduler.py, tests/v1/core/test_prefix_caching.py, and tests/v1/core/test_scheduler_e2e.py.

**Proposal rationale.**

The candidate explicitly lists prefix-hit-aware ordering as headroom and notes that TTFT for multi-turn agentic workloads is dominated by admitting cached-prefix requests quickly. The current loop is greedy FIFO with prefix probing after selection, which leaves cache-hit information unused for ordering. k-LPM directly fills that gap: it is a formal framework that balances longest-prefix-match reuse against TTFT/TPOT fairness, matching the caller's objective of reducing median TTFT/TPOT on multi-turn agentic traffic, and it cleanly subsumes both extremes the candidate currently sits between (FCFS vs. pure cache-hit greedy).

---

### 5. Reorder WAITING admission by prompt length and decode-laxity to lower TTFT while protecting TPOT
- **Finding:** `find-0006` — *Optimal Scheduling Algorithms for LLM Inference: Theory and Practice*
- **Source URL:** <https://arxiv.org/abs/2508.01002>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify the WAITING-phase admission loop in vllm/v1/core/sched/scheduler.py:567-846 (Scheduler.schedule) to replace the strict head-of-queue pick with a SLAI-style two-signal ordering. (1) Prefill reordering by known prompt length: before the per-iteration pick, score waiting requests using their token count (and, when cheaply available, their already-known prefix-cache hit length from a lightweight pre-probe of KVCacheManager) so that short / high-cache-hit prompts can be promoted ahead of long cold prompts when they fit in the remaining token budget. Long prompts are not starved: apply a bounded look-ahead window (e.g. top-K of waiting/skipped_waiting) and a max-deferral count per request to preserve FCFS fairness. (2) Decode-laxity guard on prefill admission: track a running estimate of the next-step decode time-between-tokens (TBT) using existing SchedulerStats / RUNNING token counts, and when the projected decode budget after admitting a prefill chunk would push laxity below a threshold, clamp the chunk size (reusing the existing chunked-prefill clamp path) or skip admission this step instead of breaking the loop. This turns the current 'break on first failure' into a laxity-aware continue. The skipped_waiting requeue rule is adjusted so that requests deferred for laxity reasons go to the tail (not ahead of older skipped work), while LoRA/blocked skips keep current behavior. No changes to KV allocation correctness; the policy lives entirely in selection and chunk-size decisions.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out queue-head selection, prefix-hit-aware ordering, chunked-prefill clamp, and break-on-fail as the policy knobs, and flags median TTFT for multi-turn agentic prompts (which are typically short and prefix-cache-heavy) as the target metric. SLAI's two contributions map onto exactly those knobs: prompt-length-aware prefill reordering directly addresses the queue-head selection gap (currently strict FIFO), and real-time decode-deadline / laxity measurement directly addresses the chunk-size and break-on-fail gap (currently oblivious to decode pressure). The finding therefore contributes a concrete, transferable policy — not just a topic — that targets the same TTFT/TPOT trade-off the caller asked to optimize, and the existing SchedulerStats/PrefixCacheStats signals plus the listed correctness oracles (test_scheduler, test_prefix_caching, test_scheduler_e2e) make the change measurable and testable.

---

### 6. Adopt Sarathi-Serve stall-free chunked-prefill admission with decode-aware token budget
- **Finding:** `find-0007` — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- **Source URL:** <https://arxiv.org/abs/2403.02310>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In Scheduler.schedule's WAITING-phase admission loop (vllm/v1/core/sched/scheduler.py:567-846), replace the current greedy single-pass clamp on prefill chunk size with a Sarathi-Serve-style stall-free scheduling policy. Concretely: (1) compute a per-step token budget that first reserves capacity for all RUNNING decode tokens, then exposes the remainder as a prefill budget; (2) when admitting a WAITING request, derive its prefill chunk size as the near-equal split of the remaining prefill that fits the remaining prefill budget (rather than clamping to a fixed per-step cap), so multiple admitted prefills in the same step receive comparable shares and a single long media/tool prompt cannot consume the whole step; (3) on allocation or encoder-input failure for the chosen chunk size, retry the same request with a smaller near-equal chunk before falling through to skipped_waiting, since the headroom note explicitly calls out 'retry with smaller chunks'; (4) when the decode reservation is exhausted, stop admitting new WAITING prefills for this step instead of letting prefill admission starve in-flight decodes. Keep the existing prefix-cache probing, LoRA capacity, and skipped_waiting requeue logic intact; this proposal only changes how the prefill chunk size and the prefill-vs-decode token split are computed inside the loop. Validate using tests/v1/core/test_scheduler.py and tests/v1/core/test_scheduler_e2e.py, and observe SchedulerStats (decode steps without pause) and TPOT.

**Proposal rationale.**

The candidate's evolve_rationale lists 'chunked-prefill clamp' and 'explicit decode-prefill budget partitioning' as concrete policy knobs and 'retry with smaller chunks' as headroom. Sarathi-Serve directly targets exactly these knobs: near-equal chunking bounds the worst-case prefill cost per step, and stall-free scheduling formalizes a decode-first token budget so adding prefill admissions does not pause ongoing decodes. For the stated multi-turn agentic workload, where the caller wants both lower media TTFT and lower median TPOT, this is the standard and well-evidenced way to keep TPOT stable while still admitting new prefills - it addresses the gap that today's loop uses a static clamp and a break-on-fail rule that can either over-commit a step to one long prompt or prematurely abort admission when a slightly smaller chunk would have fit.

---

### 7. Scheduler-aware proactive prefetch of external prefix-cache KV during WAITING admission
- **Finding:** `find-0009` — *AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving*
- **Source URL:** <https://arxiv.org/html/2403.19708v2>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify the WAITING-phase admission loop in vllm/v1/core/sched/scheduler.py:567-846 so that prefix-cache probing is moved earlier and is used as a scheduling signal for asynchronous KV prefetch from lower tiers (host/external store), rather than running only after queue-head selection. Concretely: (1) at the top of each admission pass, issue a cheap batched prefix-hit query across the next N candidates in waiting/skipped_waiting (covering both local KVCacheManager and the connector-backed external cache); (2) rank/reorder the queue head by combined prefix-hit ratio so requests whose prefixes already live in any tier are admitted first, addressing the candidate's listed knob 'prefix-hit query placement' and 'queue-head selection'; (3) for high-hit candidates whose blocks are only in the external tier, kick off an async layer-wise preload (mirroring AttentionStore's overlap-with-compute scheme) using the existing KV connector hooks while the loop continues admitting local-hit requests, so the external transfer overlaps with prefill GPU work of already-admitted requests in the same step; (4) when allocation fails for a high-hit candidate because its remote blocks have not finished landing, retry with a smaller chunked-prefill clamp (the candidate's 'retry with smaller chunks' headroom) and skip-without-break instead of breaking the admission loop, and place such 'awaiting-preload' entries at the front of skipped_waiting only once their preload completes (fixing the current rule that requeues blocked requests ahead of older skipped work). Validate with tests/v1/core/test_scheduler.py, tests/v1/core/test_prefix_caching.py, tests/v1/core/test_scheduler_e2e.py, and observe PrefixCacheStats / SchedulerStats for hit-rate and TTFT deltas on multi-turn traces.

**Proposal rationale.**

The candidate's WAITING admission loop already integrates with an external prefix cache but probes it only after queue-head selection and breaks on allocation failure, so external-tier hits either stall the head or get deferred across many scheduler steps - precisely the multi-turn TTFT pathology AttentionStore targets. AttentionStore's two transferable ideas - scheduler-aware fetch/eviction decisions and asynchronous layer-wise preload that overlaps KV transfer with compute - map directly onto the candidate's named policy knobs (prefix-hit query placement, queue-head selection, break-on-fail behavior, skipped_waiting requeue rule) and onto its stated headroom (prefix-hit-aware ordering, retry with smaller chunks). This is concretely actionable inside the existing loop and the existing KV connector interface, and aligns with the caller objective of cutting median TTFT for multi-turn agentic workloads where prompts mostly hit prefix cache.

---

### 8. KVCache-centric admission scoring with remote-cache and SLO awareness
- **Finding:** `find-0011` — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*
- **Source URL:** <https://arxiv.org/abs/2407.00079>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/core/sched/scheduler.py Scheduler.schedule (lines 567-846), replace the strict FIFO queue-head admission with a KVCache-centric scoring pass over the WAITING/skipped_waiting heads. Before the existing allocation attempt, compute a per-request admission score that combines: (a) local prefix-cache hit length (already available via the local cache probe), (b) external/remote prefix-cache residency and estimated KV transfer cost surfaced through the existing KVConnector/external-cache probe path, and (c) an SLO-risk signal derived from time-in-queue and the request's TTFT budget. Admit in score order rather than queue order, with a small bounded look-ahead window (e.g., top-K of the waiting head) to keep per-step cost bounded. Move the prefix-hit probe ahead of the chunk-size clamp so the chunk budget can be biased toward the cached portion of the prompt, and add a prediction-based early-reject step that defers (back to skipped_waiting) any candidate whose estimated prefill cost given current free blocks would push another already-admitted request past its TPOT/TTFT SLO. Keep break-on-fail semantics for hard allocator/encoder failures, but on a soft failure (chunk too large) retry the same request with a smaller chunk before falling through. Reuse PrefixCacheStats for the hit-length signal and extend SchedulerStats with the new score components for offline tuning. Validate with tests/v1/core/test_scheduler.py, tests/v1/core/test_prefix_caching.py, and tests/v1/core/test_scheduler_e2e.py.

**Proposal rationale.**

The candidate's current admission loop is greedy/FIFO and only consults the local free-block count plus a post-selection prefix probe; it has no notion of remote KV residency, transfer cost, or SLO risk. Mooncake's KVCache-centric scheduler explicitly balances effective throughput against TTFT/TPOT SLOs by treating KV cache (including non-local tiers) as the primary scheduling resource and using prediction-based early rejection. That maps directly onto the candidate's documented headroom (prefix-hit-aware ordering, retry-with-smaller-chunk, decode/prefill budget partitioning) and onto the multi-turn agentic objective where most prompts hit a cache tier and median TTFT is dominated by which cached-prefix request is admitted first. The change is concrete, local to this loop, and uses signals (external cache probe, PrefixCacheStats, queue wait time) that the scheduler already computes.

---

### 9. Add explicit prefill/decode token-budget partitioning to the WAITING admission loop
- **Finding:** `find-0012` — *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*
- **Source URL:** <https://arxiv.org/abs/2401.09670>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In Scheduler.schedule (vllm/v1/core/sched/scheduler.py:567-846), introduce an explicit per-step budget split between decode tokens (already-RUNNING requests) and prefill tokens (WAITING admission), instead of the current single token_budget that prefill admission can fully consume. Concretely: before entering the WAITING-phase admission loop, reserve a decode-floor portion of the step budget based on the count and unfinished length of RUNNING requests; expose the remaining prefill-headroom to the admission loop. The chunked-prefill clamp inside the loop should size new prefill chunks against the prefill-headroom rather than the global remaining budget, and the loop should stop admitting prefill once the prefill-headroom is exhausted even when global budget remains. The decode-floor and prefill-headroom should be tunable knobs (e.g. config fields analogous to the existing chunked-prefill knobs) so they can be tuned for agentic multi-turn workloads where decode steady-state dominates TPOT. Existing oracles in tests/v1/core/test_scheduler.py, tests/v1/core/test_prefix_caching.py, and tests/v1/core/test_scheduler_e2e.py, together with PrefixCacheStats and SchedulerStats, can verify correctness and measure TTFT/TPOT impact.

**Proposal rationale.**

DistServe's core observation is that prefill work interferes with decode work and that isolating their resource allocations under separate TTFT and TPOT constraints improves goodput. The candidate's current admission loop has no explicit decode-vs-prefill budget partition: a large prefill admitted by the chunked-prefill clamp can absorb the step's token budget, delaying decode steps and inflating TPOT for the multi-turn agentic workload called out in the caller context. Disaggregating across GPUs is out of scope for a single scheduler, but the transferable idea, called out as headroom in evolve_rationale ('explicit decode-prefill budget partitioning'), is to reserve decode capacity at scheduling time so prefill admission cannot starve it. This directly targets the candidate's TPOT objective while leaving TTFT-favoring policies (prefix-hit ordering, chunk sizing) intact within the prefill-headroom.

---

### 10. Probe a host-tier encoder embedding cache during WAITING admission and prefer encoder-hit requests
- **Finding:** `find-0014` — *Embedding Cache*
- **Source URL:** <https://docs.dynamo.nvidia.com/dynamo/user-guides/multimodal/embedding-cache>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the WAITING-phase admission loop in vllm/v1/core/sched/scheduler.py:567-846 to consult a CPU-side LRU cache of vision-encoder outputs (modeled after NVIDIA Dynamo's embedding cache) at the same point where prefix-cache probing occurs. Concretely: (1) before committing to the queue head, probe both the local prefix cache and the new host-tier encoder embedding cache for each candidate's multimodal inputs; (2) compute a per-request 'admission cost' that subtracts cached encoder work (hit ⇒ skip encode entirely; miss ⇒ enqueue encode plus host→device transfer of the produced embedding) from the prefill chunk budget; (3) bias queue-head selection so requests whose multimodal inputs are fully encoder-cached (and prefix-cached) are admitted ahead of cold-encode requests, instead of strictly walking the queue head first; (4) on a miss, allow the existing encoder-input scheduling path to run, but on a hit, short-circuit `Scheduler._try_schedule_encoder_inputs` so it consumes no encoder budget and does not break admission; (5) record cache outcomes in PrefixCacheStats / SchedulerStats analogues so the policy is measurable. The cache itself is owned outside the scheduler (CPU-side, LRU, populated when the encoder runs), and the scheduler's role is restricted to a read probe plus an ordering signal—mirroring how prefix-cache hits already influence chunked-prefill decisions. Validate against tests/v1/core/test_scheduler.py and tests/v1/core/test_scheduler_e2e.py with multimodal fixtures whose media inputs repeat across turns.

**Proposal rationale.**

The candidate's evolve_rationale explicitly names prefix-hit-aware ordering and break-on-fail behavior on encoder inputs as policy knobs with headroom, and the caller objective is media TTFT on multi-turn agentic workloads—exactly the regime where the same images/audio recur across turns. Dynamo's finding contributes the transferable mechanism: a host-tier LRU keyed on media identity that lets the engine skip encoder work entirely on hit. Wiring that signal into the admission loop's queue-head selection and encoder-input scheduling closes a concrete gap (encoder cost is treated as opaque today and can break admission) and is implementable without restructuring the rest of the scheduler. The improvement is plausibly large because for repeated media, hits convert an encoder forward pass into a host→device copy, materially shortening media TTFT while leaving decode capacity untouched.

---

### 11. Group WAITING admissions by shared prefix-cache prefix to batch prefix-similar requests together
- **Finding:** `find-0015` — *ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition*
- **Source URL:** <https://aclanthology.org/2024.acl-long.623/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In Scheduler.schedule's WAITING-phase admission loop (vllm/v1/core/sched/scheduler.py:567-846), introduce a prefix-aware ordering pass that runs before the greedy queue-head selection. For each candidate in waiting (and skipped_waiting), pre-probe the local prefix cache to compute a cheap prefix-group key (e.g., longest cached block-hash prefix or root prefix-tree node) before allocation. Replace the strictly FIFO head-pop with a group-coalesced scan: once a request is admitted, prefer admitting other waiting requests that share its prefix-group key in the same step, so the resulting RUNNING batch contains prefix-similar prompts. Preserve the existing LoRA capacity check, chunked-prefill clamp, encoder-input scheduling, and KV allocation logic, and bound group preference by a fairness budget (e.g., max age in queue or max group span) to avoid starving non-grouped requests. Skipped_waiting requeue ordering should also respect group affinity rather than only blocked/LoRA-first.

**Proposal rationale.**

ChunkAttention demonstrates that throughput/locality gains from chunked KV plus a prefix tree are realized only when batches contain prefix-similar requests. The scheduler is the upstream choke point: under the current loop, prefix-cache probing happens after queue-head selection, so even when many waiting prompts share a long system prompt they may be split across steps. The candidate's evolve_rationale explicitly names 'prefix-hit-aware ordering' and 'prefix-hit query placement' as headroom; the finding contributes the concrete, transferable idea of grouping by common prefix at admission time. For the stated multi-turn agentic workload, where prompts overwhelmingly share system-prompt prefixes, coalescing prefix-similar requests at admission is plausibly impactful for both TTFT (more cache hits realized in the admitted batch) and median TPOT (denser shared-prefix batches that downstream attention can exploit). Correctness oracles (tests/v1/core/test_scheduler.py, test_prefix_caching.py, test_scheduler_e2e.py) and PrefixCacheStats/SchedulerStats provide measurable signals to validate the change.

---

### 12. Prioritize multi-turn session continuations in WAITING admission via prefix-hit-aware ordering
- **Finding:** `find-0017` — *Stateful Large Language Model Serving with Pensieve*
- **Source URL:** <https://arxiv.org/abs/2312.05516>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify the WAITING-phase admission loop in vllm/v1/core/sched/scheduler.py:567-846 so that queue-head selection is informed by prefix-cache hit ratio rather than strict FIFO. Concretely: (1) before the current per-request selection step, perform a lightweight batched prefix-cache probe (local KVCacheManager + external connector lookup) over the head of the waiting/skipped_waiting queues to estimate cached-prefix length per request; (2) reorder candidates so requests with the longest cached prefix (i.e. multi-turn agent continuations whose conversation state is still resident) are tried first, subject to a fairness cap to avoid starvation of cold requests; (3) when a continuation is admitted, prefer keeping its previous conversation-state blocks pinned in the prefix cache rather than evicting them under pressure, treating the originating session as a stateful cache owner. This requires hoisting the prefix probe (currently invoked only after a request is selected) above the selection decision and threading the resulting hit length into the ordering and chunk-size choice.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out prefix-hit-aware ordering and prefix-hit query placement as headroom items, and identifies multi-turn agentic TTFT as the dominant objective. Pensieve's central insight — that multi-turn LLM serving benefits from treating conversation history as persistent cached state and avoiding redundant prefill — maps directly onto admission ordering: a request whose prior turn's KV is still cached should be admitted before a request that requires full prefill, because its first-token work is much cheaper and its cached blocks are most valuable when reused soon (before eviction). The current loop probes the prefix cache only after committing to a queue head, so it cannot exploit hit length as a scheduling signal. Ordering by cached-prefix length is a concrete, transferable policy change that operationalizes Pensieve's stateful-session view inside the existing admission loop, and is testable via tests/v1/core/test_scheduler.py and test_prefix_caching.py with PrefixCacheStats/SchedulerStats as oracles.

---

## Agent proposals

### 1. Eviction-cost-aware WAITING admission: penalize candidates whose KV allocation displaces queued peers' cached prefixes
- **Agent:** claude

**Detailed description.**

In Scheduler.schedule's WAITING-phase admission loop (vllm/v1/core/sched/scheduler.py:567-846), add a destructive-side-effect check before committing an admission. Concretely: (1) when KV allocation for a candidate would require freeing cached blocks (i.e. the free pool is insufficient and the allocator would fall back to evicting LRU prefix-cache entries), enumerate the specific block hashes the allocator would evict; (2) cross-reference those block hashes against a cheap, scheduler-owned 'queued-prefix index' built once per admission pass — a flat map from block-hash -> list of (waiting-request, matched-prefix-length) entries derived from the same prefix probes that find-0004/find-0015/find-0017 already propose to hoist; (3) compute an eviction-cost score for the candidate equal to the sum over to-be-evicted blocks of (queued matched-prefix-length × queue-position weight × an age-decay factor for blocks belonging to RUNNING but soon-to-finish requests, estimated from observed token rate and remaining max_tokens); (4) admit the candidate only if eviction_cost <= alpha × candidate's own admission benefit (its own prefix-hit length plus its queue priority); otherwise push to skipped_waiting with a short cooldown so it can be reconsidered after the queue state changes, and try the next candidate. The queued-prefix index is rebuilt at most once per scheduler step and reuses block-hash structures already produced by KVCacheManager probes, so it adds no new probing cost beyond what proposals like find-0004 already contemplate. Validate with tests/v1/core/test_scheduler.py and tests/v1/core/test_prefix_caching.py — extend them with a fixture where two waiting continuations share a system-prompt prefix and a third 'cold' candidate's admission would evict that prefix; assert the cold candidate is deferred. Telemetry: extend SchedulerStats with counters for 'admission deferred for eviction cost' and PrefixCacheStats with 'queued-prefix protected blocks', and confirm median TTFT improves on a multi-turn agentic trace where many sessions share long system prefixes.

**Novelty rationale.**

Every listed proposal reasons about admission *benefit* to the candidate being admitted — its own prefix-hit length (find-0004, find-0005, find-0015, find-0017), its session continuation status (find-0017), its agent-step proximity (find-0003), its SLO risk (find-0006, find-0011), its encoder cache hit (find-0001, find-0014), or its remote-tier residency (find-0009, find-0011). None of them score the *destructive externality* an admission imposes on other queued requests whose cached prefixes would be evicted to make room for it. find-0017 pins continuation blocks under pressure but does not consult the waiting queue when deciding what to evict, and find-0011's SLO-risk rejection looks at whether *already-admitted* peers will miss SLOs, not whether *queued* peers' cached state will be destroyed. This proposal closes that gap: it makes admission jointly optimize benefit-to-self against eviction-cost-to-queued-peers, which is the missing mechanism for protecting the working set of a multi-turn agentic fleet where many sessions concurrently hold valuable cached prefixes.

---

### 2. Make encoder-cache mutations transactional during WAITING admission
- **Agent:** codex

**Detailed description.**

In `Scheduler.schedule`'s WAITING loop in `vllm/v1/core/sched/scheduler.py:567-846`, avoid mutating the multimodal encoder cache until the request is actually admitted. Today `_try_schedule_encoder_inputs` calls `EncoderCacheManager.check_and_update_cache` and `can_allocate` before `kv_cache_manager.allocate_slots`; both are stateful, and `can_allocate` can evict freeable media embeddings. If KV allocation then fails, or a later encoder/mamba check breaks admission, the scheduler can leak cache references or emit `free_encoder_mm_hashes` for a request that never ran, inflating media TTFT for later repeated-media turns. Refactor encoder scheduling into a side-effect-free planning phase plus commit phase: probe cache hits without touching refcounts, compute the required eviction/reservation plan, and commit those ref/refcount/eviction changes only after KV allocation succeeds and the request is popped for scheduling. On every non-commit path, discard the plan. Add a focused scheduler test that forces KV allocation failure after an encoder-cache reclaim would have been planned, then asserts the original media embedding remains cached and no freed mm hash is reported.

**Novelty rationale.**

The listed proposals change admission ordering, chunk sizing, encoder-hit preference, or remote/cache-aware scheduling; none addresses speculative side effects from the existing encoder-cache probes before KV admission is committed. Agent A's proposal scores KV prefix-cache evictions caused by a successful candidate against queued peers; this proposal targets a different cache and a different failure mode: uncommitted multimodal encoder-cache mutations caused by failed WAITING admission attempts.

---
