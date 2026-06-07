# RequestState.make_request_output

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/output_processor.py`](vllm/v1/engine/output_processor.py) (lines 269–331)
- **Symbol:** `RequestState.make_request_output`
- **Kind:** method
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0008`

## Description
Decides whether a request emits a RequestOutput on this step, builds the CompletionOutput or PoolingOutput, and merges child outputs for n>1 requests.

## Current approach
Fixed stream gating: emit if finished, if first token, or if num_output_tokens - sent_tokens_offset reaches the global stream_interval. FINAL_ONLY suppresses all intermediate outputs.

## Estimated impact explanation
Streaming cadence controls how quickly generated tokens become visible to agent clients and how much per-token Python/output overhead the frontend pays. It can improve post-first-token responsiveness and median TPOT under long generations, though the first token is already always emitted.

## Evolve rationale
The gate at lines 281-307 is a compact adaptive-policy target. It could consider request type, downstream queue depth, first-few-token latency, or long-generation batching while preserving DELTA reconstruction. Correctness oracle: tests/v1/engine/test_output_processor.py must show that concatenating all DELTA text/token chunks equals the final output, FINAL_ONLY behavior is unchanged, and finish/stop metadata is identical.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Make streaming-gate consumer-aware via RequestOutputCollector backpressure
- **Agent:** claude

**Detailed description.**

Replace the fixed `stream_interval` gate at vllm/v1/engine/output_processor.py:281-307 with a per-request adaptive gate that uses the existing `self.queue: RequestOutputCollector` as a backpressure signal. Concretely, in `RequestState.make_request_output`, after the FINAL_ONLY early-return, compute an effective interval as: 1 when `self.queue is None` or `self.queue.output is None` (the consumer has already drained the previous RequestOutput), and `self.stream_interval` (or a small grown multiple, capped) when `self.queue.output is not None` (the producer is ahead and the collector is currently aggregating/merging deltas — coalescing here is free under DELTA semantics because `RequestOutputCollector.put` already merges via `RequestOutput.add(..., aggregate=True)` at lines 67-72). The condition at lines 292-297 then becomes `effective_interval == 1 or finished or sent_tokens_offset == 0 or num_output_tokens - sent_tokens_offset >= effective_interval`. The DELTA bookkeeping at lines 300-306 stays unchanged, so token reconstruction is preserved bit-for-bit; FINAL_ONLY still returns at line 281; finish/stop and parent_req.get_outputs paths are untouched. Add a small per-request grown-interval state (e.g., `self._coalesce_interval`) that doubles up to a cap (e.g., 2*stream_interval) on consecutive 'consumer-behind' emissions and resets to `stream_interval` once the queue drains, to prevent pathological stalling under bursty consumers. For pooling outputs, the gate is bypassed today (only enters the `if self.stream_interval > 1` branch); preserve that. For n>1 / parent_req requests, leave the merge logic at lines 321-327 untouched — the gate runs per child as today. Tests in tests/v1/engine/test_output_processor.py should add: (a) consumer always drained ⇒ every token streamed individually with concatenated DELTAs equal to the final completion text/token ids; (b) consumer never drained ⇒ outputs coalesce, but final concatenation still equals the full output; (c) FINAL_ONLY still emits exactly once; (d) finish_reason/stop_reason metadata identical to baseline.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, and the candidate description only sketches general directions (request type, downstream queue depth, first-few-token latency, long-generation batching) without naming any mechanism. This proposal pins down a specific, locally-implementable mechanism — using the already-present `RequestOutputCollector.output is None` flag as a zero-cost backpressure signal — that none of the listed (empty) prior proposals cover. It is also distinct from a simple time- or first-K-token policy because it adapts per-request based on the consumer's actual drain rate, which directly targets median TPOT in multi-turn agentic workloads where the agent loop alternates between blocking and idle on the stream.

---

### 2. Honor stream_interval for CUMULATIVE outputs
- **Agent:** codex

**Detailed description.**

Update vllm/v1/engine/output_processor.py:281-307 so `sent_tokens_offset` is treated as a last-emitted token watermark for all completion output kinds, not only DELTA. Compute `num_output_tokens = self.detokenizer.num_output_tokens()` once, gate on `num_output_tokens - self.sent_tokens_offset >= self.stream_interval`, and after any emitted completion output set `self.sent_tokens_offset = num_output_tokens`. For DELTA, capture the old offset before updating and keep slicing `self.detokenizer.output_token_ids[old_offset:]`; for CUMULATIVE, emit the existing cumulative text/token_ids but advance the watermark so future non-final steps are actually suppressed until the interval or finish. Guard this token-based gate to completion outputs only so pooling final-only outputs remain unaffected. Add tests in tests/v1/engine/test_output_processor.py covering CUMULATIVE with `stream_interval > 1`: first token emits, intermediate steps below the interval do not, emitted cumulative prefixes are correct, the final output contains the full text/token_ids, and finish/stop metadata matches the DELTA/FINAL_ONLY baselines. This reduces Python RequestOutput construction, queue puts, and n>1 parent aggregation work for the default `SamplingParams.output_kind=CUMULATIVE`, where the current code effectively streams every step because `sent_tokens_offset` remains 0 outside the DELTA branch.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes adapting the effective interval from `RequestOutputCollector` backpressure and focuses on DELTA coalescing; it does not address the existing CUMULATIVE watermark gap. This proposal is a separate output-kind correctness and overhead reduction change, and it would still be required if Agent A's backpressure policy were added because `sent_tokens_offset == 0` would otherwise keep short-circuiting the gate for CUMULATIVE outputs.

---
