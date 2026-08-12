# EngineCore.step_with_batch_queue

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/core.py`](vllm/v1/engine/core.py) (lines 621–735)
- **Symbol:** `EngineCore.step_with_batch_queue`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0003`

## Description
Pipeline-parallel batch-queue step that decides when to enqueue new work, when to drain the oldest future, and when to defer sampling for structured outputs.

## Current approach
Prioritizes filling the batch queue before blocking for model outputs, appends non-deferred work immediately, pops the oldest queued future when needed, and handles deferred sampling at the end of the pop path. The local comment notes this fixed policy favors TTFT over TPOT/throughput.

## Estimated impact explanation
For pipeline-parallel serving, this policy determines bubble size and how quickly first outputs are drained, so it can move median TTFT and TPOT whenever the batch queue is enabled.

## Evolve rationale
The method owns the pipeline scheduling policy between queue fill, output drain, blocking future.result(), and deferred sampling. Candidate changes include adaptive drain-vs-fill decisions, queue-depth thresholds, and moving deferred sampling earlier or later. Correctness oracle: pipeline-parallel tests should preserve completion, ordering constraints, structured-output behavior, and token parity; vllm bench can measure TTFT/TPOT movement for pipeline_parallel_size > 1.

## Deep research proposals

### 1. Slack-guided adaptive drain-vs-fill policy in step_with_batch_queue
- **Finding:** `find-vllm_v1_engine-0001` — *Taming Request Imbalance: SLO-Aware Scheduling for Disaggregated LLM Inference*
- **Source URL:** <https://papers.cool/arxiv/2605.02329>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the fixed enqueue-before-drain policy in EngineCore.step_with_batch_queue (vllm/v1/engine/core.py:621-735), specifically the decision at lines 678-683 (`if len(batch_queue) < self.batch_queue_size and (model_executed or self.scheduler.has_requests()): return None, model_executed`), with a slack-guided adaptive rule inspired by Kairos-style slack-aware batching. Maintain a lightweight lookup table of measured per-step latency keyed by (batch size, sequence-length bucket, pipeline stage occupancy), updated online from the timings already observable when `future.result()` returns. On each call, before choosing to keep filling the queue vs. pop and drain, compute the minimum TPOT slack across the requests represented in the head-of-queue scheduler_output (deadline-to-now minus the predicted remaining step time from the table). If that minimum slack is below a threshold, drain first (pop the head of `batch_queue` and process its output) instead of enqueueing another batch, so tight-slack decode tokens are not delayed by an additional queued step. If slack is comfortable, retain the current fill-first behavior to preserve throughput and TTFT. The same slack signal can also gate the deferred-sampling branch (lines 712-733): defer only when slack allows the extra queued step, otherwise compute the grammar bitmask and sample immediately before enqueueing further work. Keep the existing correctness guarantees (queue ordering, structured-output token handling, and pipeline-parallel completion) by leaving the pop path and deferred-sampling mechanics unchanged aside from the added guard.

**Proposal rationale.**

The candidate's own comment at line 714 and its evolve_rationale explicitly call out that the current drain-vs-fill policy is fixed and favors TTFT over TPOT/throughput, and that adaptive drain-vs-fill decisions and queue-depth thresholds are within scope. The finding contributes exactly the missing ingredient: a concrete, measurable signal (per-request TPOT slack derived from an online step-time table) that turns a fixed policy into a data-driven one. Applied to the caller objective (reduce median TTFT and TPOT on a multi-turn agentic workload), draining under low slack directly protects decode-token latency for older in-flight requests, while filling under high slack keeps pipeline stages busy for new turns — addressing the pipeline-bubble/TPOT tension the candidate owns. The idea is transferable without requiring paper-specific SLOs: slack thresholds can be tuned per deployment, and the step-time table can be populated from timings already collected around `future.result()`.

---

### 2. Adopt stall-free chunked-prefill + decode-maximal batching in step_with_batch_queue
- **Finding:** `find-vllm_v1_engine-0002` — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- **Source URL:** <https://www.usenix.org/conference/osdi24/presentation/agrawal>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify EngineCore.step_with_batch_queue in vllm/v1/engine/core.py (lines 621-735) to implement Sarathi-Serve's stall-free scheduling policy for the pipeline-parallel batch queue. Instead of the current fixed policy that prioritizes filling the batch queue before blocking on the oldest future, change the enqueue/drain decision so that: (1) when composing a new batch to enqueue, the scheduler admits chunked prefill work sized to fit alongside ongoing decodes within a per-iteration token budget, rather than admitting whole prefills that stall decodes; (2) each iteration is decode-maximal — remaining token budget after in-flight decodes is filled with prefill chunks up to a near-equal chunk size, preserving TTFT for new requests without pausing ongoing decodes; (3) the drain-vs-fill branch (append new work vs. pop oldest future via future.result()) uses queue-depth and expected-chunk-cost signals so pipeline bubbles are minimized when decodes dominate, and prefill chunks are enqueued opportunistically to keep pipeline stages busy. Deferred structured-output sampling handling at the end of the pop path is preserved. The chunking policy itself lives in the scheduler; step_with_batch_queue's change is to align enqueue timing and admission with a decode-maximal, stall-free batch, and to expose the queue-depth threshold controlling when to drain vs. keep filling.

**Proposal rationale.**

The candidate's own evolve_rationale explicitly calls out adaptive drain-vs-fill decisions and queue-depth thresholds as targets, and notes the current policy trades TPOT/throughput for TTFT. Sarathi-Serve directly addresses that exact tradeoff: chunked prefills plus decode-maximal batching reduce generation stalls (improving TPOT) and shrink pipeline bubbles (relevant precisely because this method only runs when pipeline_parallel_size > 1), while still admitting new prefill work each iteration (protecting TTFT). The caller's objective is to reduce median TTFT and median TPOT under a multi-turn agentic workload, which is decode-heavy and thus benefits most from stall-free scheduling that keeps decodes flowing while interleaving small prefill chunks. The finding contributes a concrete, transferable policy — chunk size selection and decode-maximal admission — that maps onto the specific decision points this method owns, rather than restating the current approach.

---

### 3. Latency-budgeted adaptive drain policy for step_with_batch_queue
- **Finding:** `find-vllm_v1_engine-0013` — *Dynamic Request Batching*
- **Source URL:** <https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/engine/core.py (EngineCore.step_with_batch_queue, lines 621-735), replace the fixed 'fill before drain' policy with a latency-budgeted decision inspired by Ray Serve's dynamic batching knobs. Introduce two tunables analogous to batch_size_fn and batch_wait_timeout_s: (1) a cost metric that estimates the effective batch cost as a function of queued requests (e.g., total scheduled tokens across in-flight microbatches) rather than raw batch count, used to decide whether the queue is 'full enough' to justify deferring a drain; and (2) a per-step wait budget derived from a remaining-TPOT SLO that bounds how long the scheduler is willing to keep filling before it must pop the oldest queued future and drain outputs. Concretely: before the current unconditional enqueue path, compute the cost of the queued futures; if cost >= threshold OR the oldest queued future has been outstanding longer than the TPOT-derived budget, take the pop/drain branch first (blocking on future.result() and delivering tokens) instead of enqueuing another microbatch. Keep the deferred-sampling handling on the pop path unchanged, but ensure the same budget check governs whether we defer sampling to the next iteration or resolve it eagerly. Expose the threshold and budget through the existing scheduler/engine config surface so pipeline_parallel_size > 1 deployments can tune them; default to today's behavior when unset so no policy change ships silently.

**Proposal rationale.**

The candidate's own comment flags that the current fixed policy favors TTFT at the expense of TPOT/throughput, and evolve_rationale explicitly calls out adaptive drain-vs-fill decisions and queue-depth thresholds as in-scope. Ray Serve's dynamic batching guidance contributes two concrete, transferable ideas: a cost function (batch_size_fn) that measures batch size in tokens rather than requests, and a wait timeout tied to the end-to-end latency SLO. Mapping those onto the batch-queue step gives a principled knob to trade TTFT for TPOT on multi-turn agentic workloads where token-delivery latency dominates once decode is underway, directly addressing the caller's objective of reducing median TPOT without regressing TTFT. This is more than topical overlap: the finding supplies the shape of the policy (cost metric + latency budget) that the candidate currently lacks.

---

## Agent proposals

### 1. Opportunistic zero-cost drain of ready futures before enqueue in step_with_batch_queue
- **Agent:** claude

**Detailed description.**

In EngineCore.step_with_batch_queue (vllm/v1/engine/core.py:621-735), insert a non-blocking readiness peek at the head of `batch_queue` before the scheduling/enqueue block (i.e., before line 648 `if self.scheduler.has_requests():`). Concretely: if `batch_queue` is non-empty, check whether the oldest queued future (`batch_queue[-1][0]`, the one that `.pop()` on line 692 would take) is already done via `future.done()` (or `concurrent.futures.wait([future], timeout=0, return_when=FIRST_COMPLETED)`). If it is ready, take the existing pop/drain path first — call `.pop()`, run `future.result()`, `_process_aborts_queue()`, `scheduler.update_from_output(...)`, and the deferred-sampling handling at lines 715-733 — and return `engine_core_outputs, model_executed=False` without scheduling a new batch this call. Only if the head-of-queue future is not yet ready do we fall through to today's fill-first logic (schedule, execute_model non_block, appendleft, and the existing `len(batch_queue) < self.batch_queue_size` guard at lines 678-683). This exploits the fact that on decode-heavy multi-turn workloads the oldest microbatch frequently finishes on the accelerator during the Python-side overhead of the previous step, so its tokens are sitting in the future waiting to be delivered; the current code unconditionally enqueues another batch first and only drains on the next call, adding one full step of TPOT latency to those tokens for no throughput gain. The change is a single `done()` peek — O(1), no timing table, no SLO knob, no config surface — and preserves every correctness guarantee (queue ordering is FIFO, structured-output deferred sampling still runs on the pop path, pipeline-parallel completion semantics are unchanged) because the drain path taken is byte-for-byte the existing one. When nothing is ready, behavior is identical to today.

**Novelty rationale.**

The three listed deep_research_proposals all modify the drain-vs-fill *decision policy* using predictive/config signals: find-0001 adds an online step-time table plus per-request TPOT slack thresholds; find-0002 adopts Sarathi-Serve chunked-prefill + decode-maximal admission (a scheduler-side change); find-0013 adds a Ray-Serve-style token-cost function plus a TPOT-derived wait budget with new config knobs. All three answer 'should we drain even though the queue is not full?' with a predicted-cost or slack argument. This proposal answers a different question: 'is the drain already free right now?' It requires no cost model, no slack estimator, no chunk-size policy, no timing history, and no new config — just a `future.done()` peek at the head of the deque before the enqueue branch. It composes cleanly with any of the three policy proposals (they decide when to force a drain under contention; this captures the trivially-ready case that they would otherwise miss on the first pass) and captures a class of wins — tokens ready during Python overhead — that none of them target, since a slack/cost/latency-budget check will still choose to fill when the queue is not deep and budget is comfortable, even though the head future is literally done.

---

### 2. Prioritize deferred sampling completions as latency-critical queue entries
- **Agent:** codex

**Detailed description.**

In `EngineCore.step_with_batch_queue` (`vllm/v1/engine/core.py:621-735`), treat futures created by the deferred structured-output sampling path as latency-critical completion work rather than ordinary pipeline microbatches. When the pop path reaches the `if deferred_scheduler_output:` block and calls `sample_tokens(..., non_block=True)`, enqueue that sampling future with metadata marking it as a deferred-sampling completion. On the next `step_with_batch_queue` call, before scheduling more model work, drain the oldest ready deferred-sampling completion if one exists, or block on it when it is at the front of the logical completion queue. This prevents a structured-output request whose logits were already produced from waiting behind additional `execute_model` microbatches just to finalize its sampled token. Keep normal FIFO ordering for regular model execution futures, and only apply the priority to deferred sampling futures that were spawned from an already-drained scheduler output.

**Novelty rationale.**

The deep_research proposals cover adaptive drain-vs-fill policies using slack estimates, chunked prefill/decode-maximal scheduling, or latency-budgeted queue thresholds; two mention possibly gating whether deferred sampling should happen immediately, but they do not introduce a distinct priority class for already-unblocked deferred sampling completions. Agent A's proposal drains the oldest future if it is already ready before enqueueing, but it still treats every queued future uniformly. This proposal targets the specific second-stage latency of structured-output sampling after logits are available, making those completion futures drain ahead of ordinary pipeline work without adding a global cost model, SLO policy, or scheduler-side chunking change.

---
