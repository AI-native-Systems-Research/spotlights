# AsyncLLM._run_output_handler.output_handler

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/async_llm.py`](vllm/v1/engine/async_llm.py) (lines 656–707)
- **Symbol:** `AsyncLLM._run_output_handler.output_handler`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0004`

## Description
Background asyncio handler that awaits EngineCore outputs, slices them into VLLM_V1_OUTPUT_PROC_CHUNK_SIZE chunks, processes each chunk, yields between non-final chunks, dispatches stop-string aborts, updates scheduler stats, and records logger stats.

## Current approach
Fixed chunk size from env var VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, unconditional asyncio.sleep(0) between non-final chunks, synchronous abort dispatch after each chunk, and synchronous logger.record at the end of every batch.

## Estimated impact explanation
The handler is the asyncio chokepoint between EngineCore and user streams. Better chunk/yield cadence reduces TPOT jitter and tail TTFT under bursty multi-turn workloads, but the effect depends on batch size and event-loop contention.

## Evolve rationale
The chunking and yield policy at lines 667-683 is a measurable scheduling heuristic. Adaptive chunk sizing based on num_outputs and queue backlog, conditional yields, abort batching across chunks, or deferred/throttled logger.record while preserving output order are all on-table. Correctness oracle: tests/v1/engine/test_async_llm.py and OpenAI streaming tests must observe identical per-request output order, abort behavior, and final statuses; a property check can compare per-request output multiset and order before and after chunking changes.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Pipeline EngineCore output fetch with current-batch processing
- **Agent:** claude

**Detailed description.**

In vllm/v1/engine/async_llm.py:656-707, the output_handler is strictly serial: it awaits engine_core.get_output_async(), then processes all chunks, then loops back to await the next batch. While the handler is busy processing chunks, dispatching aborts, and recording logger stats (steps 2–4, lines 671-702), the EngineCore output queue can fill but the handler is not draining it — and conversely, while it is awaiting the next batch (line 660), no per-request queues are being fed. Change: introduce a bounded one-deep prefetch so that immediately after receiving outputs at line 660, the handler issues `next_outputs_fut = asyncio.ensure_future(engine_core.get_output_async())` before entering the chunk loop at line 671. After the chunk loop, scheduler-stats update, and logger.record complete (line 702), the loop body becomes `outputs = await next_outputs_fut` instead of re-issuing get_output_async. This overlaps EngineCore's IPC/queue wait with the handler's CPU work (process_outputs, abort dispatch, logger.record), hiding the dominant gap on multi-turn agentic workloads where each turn produces small num_outputs and the per-iteration handler overhead is non-trivial relative to inter-step latency. To keep correctness oracle-clean: per-request output ordering is preserved because process_outputs is still called in batch order, and aborts for batch N still flush before we begin processing batch N+1 (since we `await next_outputs_fut` only after the current batch's abort_requests_async completes). On shutdown, the pending prefetch future must be cancelled in the finally branch to avoid an orphaned task. Expected effect: lower median TPOT (reduced inter-step gap) and lower TTFT for new requests arriving while a previous batch is still being post-processed, since their first-token EngineCoreOutputs will now be picked up one handler-iteration sooner.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. The evolve_rationale enumerates four directions explicitly on-table — adaptive chunk sizing, conditional yields, abort batching across chunks, and deferred/throttled logger.record — all of which target work *within* one handler iteration. This proposal is orthogonal: it changes the *iteration boundary* by introducing a one-deep prefetch so that EngineCore queue-wait overlaps with CPU-side post-processing, an inter-iteration pipelining change rather than an intra-iteration scheduling change. None of the four on-table directions express or imply this overlap.

---

### 2. Add a batch-boundary yield after publishing outputs
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/async_llm.py:656-707`, add a final publish checkpoint for nonempty batches: after all chunks for the current `EngineCoreOutputs` have been processed, stop-string aborts have been sent, scheduler stats have been updated, and `logger.record(...)` has completed, do `await asyncio.sleep(0)` before looping back to `engine_core.get_output_async()`. Today, batches with `num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` never hit the inter-chunk yield, and `asyncio.Queue.get()` can return immediately when EngineCore already has another output queued. Under a bursty multi-turn workload, the output handler can therefore drain multiple ready EngineCore batches in a row while per-request `generate()` tasks remain only woken, not actually scheduled; this delays TTFT/TPOT delivery and increases `RequestOutputCollector` coalescing. The change is intentionally a scheduling checkpoint, not a processing reorder: per-request queues are still populated in the same order, aborts for the batch still flush before the checkpoint, and stats/logging retain their current relative order. A focused test can use a fake `engine_core.get_output_async()` that returns two already-ready batches and assert that a waiting consumer task receives the first batch’s output before the handler processes the second batch.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes one-deep prefetch to overlap waiting for batch N+1 with CPU work for batch N; it does not address event-loop starvation after outputs have already been published to per-request collectors. This proposal changes the post-publication fairness point so woken consumer tasks can run before the handler drains another ready EngineCore batch, and it composes with prefetch rather than replacing it.

---
