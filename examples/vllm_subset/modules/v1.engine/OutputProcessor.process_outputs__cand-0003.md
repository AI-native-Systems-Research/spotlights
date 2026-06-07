# OutputProcessor.process_outputs

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/output_processor.py`](vllm/v1/engine/output_processor.py) (lines 572–687)
- **Symbol:** `OutputProcessor.process_outputs`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0003`

## Description
The single full-batch EngineCoreOutput-to-RequestOutput loop. For each output it looks up request state, updates stats, detokenizes, updates logprobs, builds a RequestOutput, enqueues it, and finishes or aborts completed requests.

## Current approach
One Python loop over every EngineCoreOutput with per-item dict lookup, a stats helper call that early-returns when stats are disabled, per-item detokenizer and logprobs calls, RequestOutput construction, queue put, finish cleanup, and tracing branches in the hot path.

## Estimated impact explanation
This loop runs once per engine step and scales with batch size. CPU time here delays per-request queue puts, so optimizing it lowers frontend TPOT and helps prevent TTFT regression for new agent turns sharing the same event loop.

## Evolve rationale
The function is explicitly documented as the only full-batch EngineCoreOutput loop at lines 590-597, making it the right granularity for evolution. Headroom: no-stats/no-tracing fast paths, caching req_state fields in locals, specializing pooling vs generation paths, skipping make_request_output work when stream gating will suppress output, and grouping detokenizer/logprobs work for shared tokenizer state. Correctness oracle: tests/v1/engine/test_output_processor.py and tests/v1/engine/test_async_llm.py must emit identical RequestOutput sequences, finish statuses, abort sets, and iteration stats.

## Deep research proposals

### 1. Offload per-item detokenization in process_outputs to a tokenizer process pool
- **Finding:** `find-0002` — *[Feature]: Add tokenizer process pool to eliminate preprocessing bottleneck · Issue #25301 · vllm-project/vllm*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/25301>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/engine/output_processor.py at OutputProcessor.process_outputs (lines 572-687), the per-EngineCoreOutput loop currently invokes the detokenizer (and logprobs text decoding) inline before constructing each RequestOutput and enqueuing it. Apply the finding by routing the detokenization step through a multiprocessing tokenizer/decoder pool: (1) pre-pass the batch to gather (req_id, new_token_ids, skip_special_tokens, spaces_between_special_tokens, finish_reason) tuples for outputs whose req_state is not stream-gated/aborted, (2) submit them to a shared decoder pool (sized per tokenizer) for parallel detokenize_incrementally / decode_logprob_token calls, and (3) iterate the original loop using the pre-decoded text/offsets, keeping RequestOutput construction, queue.put_nowait, finish-handling, and tracing on the main thread. The pool can be the same one proposed in #25301 for input preprocessing, reused for output decoding so the engine step's hot path no longer blocks on serialized CPU tokenizer work. Preserve correctness by keeping the per-request IncrementalDetokenizer state authoritative on the main thread (workers receive a copy of the prefix state and return the new text plus updated offsets, which the main loop commits before the next step), and keep no-detokenize / pooling-only paths on the in-process fast path so they avoid IPC overhead.

**Proposal rationale.**

The candidate's documented headroom explicitly calls out 'grouping detokenizer/logprobs work for shared tokenizer state' as a target, and the function is the only full-batch EngineCoreOutput loop, making it the natural integration point for the finding's process-pool idea. The finding identifies tokenizer encode/decode as a CPU bottleneck that serializes work behind the GPU; in process_outputs the per-item detokenizer call sits directly between EngineCoreOutput arrival and queue.put_nowait, so removing it from the critical path is the specific lever that lowers per-request enqueue latency, which the caller context flags as the dominant TPOT contributor for the multi-turn agentic workload. Unlike a generic 'use the pool everywhere' restatement, this targets the one loop where detokenization currently blocks RequestOutput delivery, so the finding contributes a concrete, transferable mechanism rather than just topical adjacency.

---

### 2. Offload detokenization and stop-check work from process_outputs to a non-GIL worker pool
- **Finding:** `find-0003` — *SMG: The Case for Disaggregating CPU from GPU in LLM Serving*
- **Source URL:** <https://vuink.com/post/clgbepu-d-dbet/blog/lightseek-smg>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor OutputProcessor.process_outputs (vllm/v1/engine/output_processor.py:572-687) so the per-EngineCoreOutput Python loop no longer performs CPU-bound detokenization, stop-string checks, and logprobs materialization inline. Instead, the loop becomes a thin dispatcher: it looks up req_state, updates lightweight counters, and submits a work item (token ids, req_state handle, stop config) to a dedicated detokenizer worker pool that runs off the engine event loop (e.g., a thread pool released-from-GIL via the fast tokenizer's Rust core, or a separate process/gRPC gateway as the finding describes). The worker produces the finalized RequestOutput (or stream delta) and pushes it onto the per-request output queue directly, while the hot loop continues to the next EngineCoreOutput. Finish/abort bookkeeping that must remain ordered with engine state (request_states pop, iteration stats finalize) stays on the main path; only the detok/stop/logprobs CPU work moves. Concretely: introduce a DetokenizerDispatcher with a bounded queue, change make_request_output and the detokenizer.update calls in the candidate range to be invoked from the worker rather than inline, and gate stream queue.put on the worker's completion. Preserve the existing oracles in tests/v1/engine/test_output_processor.py and tests/v1/engine/test_async_llm.py by ensuring per-request ordering of RequestOutput emissions and identical finish/abort semantics.

**Proposal rationale.**

The candidate is explicitly the single full-batch EngineCoreOutput loop and its hottest per-item costs are detokenizer.update, stop-string checks, and RequestOutput construction — exactly the CPU-bound, GIL-bound work the finding identifies as the bottleneck ("tokenization and detokenization had become bottlenecks" / "Hitting the GIL Wall at Scale"). The caller objective (reduce median TTFT/TPOT under multi-turn agentic load) maps directly onto the finding's claim: when many concurrent requests share one event loop, time spent detokenizing inside process_outputs delays every other request's queue put. Disaggregating that work to a non-GIL worker (thread pool over the Rust fast tokenizer, or a separate process as in the finding's gateway design) lets the main loop iterate and dispatch finish/abort decisions faster, lowering per-step latency without changing engine↔frontend semantics. This is a transferable, concrete idea distinct from the candidate's current inline approach, not merely topically adjacent.

---

## Agent proposals

### 1. Pre-partition outputs by activity class and skip RequestOutput build for stream-suppressed/aborted items
- **Agent:** claude

**Detailed description.**

In vllm/v1/engine/output_processor.py at OutputProcessor.process_outputs (lines 572-687), replace the single monolithic Python loop with a two-stage structure that eliminates work for outputs whose RequestOutput will never be consumed. Stage 1 is a tight first pass over engine_core_outputs that does only the bookkeeping that must run for every output regardless of activity (lookup req_state via dict.get, mark abort_set membership, accumulate iteration-stats raw counters into local arrays, and partition the outputs into four pre-sized lists: (a) aborted/cancelled — skip detokenize, logprobs, make_request_output, and queue.put entirely; (b) stream-suppressed (req_state.queue is None or already finished) — same skip; (c) pooling-only — bypass detokenizer/logprobs branches; (d) active generation — full path). Stage 2 iterates only the active and pooling buckets with specialized inner code: pre-bind req_state.detokenizer, req_state.logprobs_processor, req_state.queue, and req_state.stats into loop-local variables, drop tracing branches behind a single hoisted `if tracer_enabled` guarding a separate specialized variant of the body so the common no-tracing case has zero per-item tracing overhead, and likewise hoist the no-stats fast path. After Stage 2, finalize iteration stats from the accumulated arrays in a single bulk update (replacing the per-item helper invocation), and run finish_request / abort cleanup in a third pass over the discardable/finished buckets. Preserve correctness oracles in tests/v1/engine/test_output_processor.py and tests/v1/engine/test_async_llm.py: per-request emission order is unchanged because each request still appears at most once in engine_core_outputs per step, and the aborted/finished sets are processed in the same iteration order. The win is twofold — wasted detokenizer.update / make_request_output calls for stream-gated or aborted outputs are eliminated entirely (not merely offloaded), and the active-path body avoids per-item branching on tracing/stats/pooling flags that today defeat branch prediction inside the hot loop.

**Novelty rationale.**

Both existing deep_research_proposals (find-0002, find-0003) attack the same bottleneck — inline detokenization — by **moving** that CPU work to a process pool or non-GIL worker pool. They retain a single uniform loop and still build a RequestOutput for every output. This proposal is orthogonal: it does not add any worker, IPC, or thread pool, and it composes with either offload approach. Its lever is **work elimination** (skipping detokenize / make_request_output / queue.put for outputs whose consumer is gone or suppressed) plus **specialization** of the hot path via hoisted tracing/stats/pooling guards and pre-bound req_state locals — neither of which appears in the existing proposals. The evolve_rationale explicitly lists 'no-stats/no-tracing fast paths', 'caching req_state fields in locals', 'specializing pooling vs generation paths', and 'skipping make_request_output work when stream gating will suppress output' as targets, and none of those are addressed by the listed deep_research_proposals.

---

### 2. Merge queued DELTA outputs in place instead of building throwaway RequestOutputs
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/output_processor.py` inside `OutputProcessor.process_outputs`, add a collector-aware fast path for the AsyncLLM DELTA case where `req_state.queue` already holds a pending `RequestOutput`. Today the loop calls `req_state.make_request_output(...)` for every emitted token, then `RequestOutputCollector.put()` immediately merges that newly allocated object into the existing pending output and discards it. Introduce a narrow helper such as `RequestOutputCollector.try_merge_delta_completion(...)` plus a `RequestState` helper that materializes only the new `CompletionOutput` after detokenizer/logprobs update. Gate it to safe cases first: `queue.aggregate`, `queue.output` is a non-exception `RequestOutput`, `pooling_output is None`, `req_state.output_kind == RequestOutputKind.DELTA`, and no parent request fan-in that needs `ParentRequest.get_outputs()`. The fast path should update the pending output's `finished`, `kv_transfer_params`, text, token ids, logprobs, cumulative logprob, finish_reason, and stop_reason in place, then skip full `RequestOutput` construction and the generic `RequestOutput.add()` list scan. Fall back to the existing `make_request_output` path for first output, FINAL_ONLY/CUMULATIVE, pooling, parent sampling, stream-interval suppression, and uncommon edge cases. Validate with `tests/v1/engine/test_output_processor.py::test_request_output_collector` plus a process-output test that emits multiple DELTA chunks before `get()` and asserts identical merged text/token/logprob/final status.

**Novelty rationale.**

The deep research proposals move detokenization and stop/logprob work off the hot loop; they still materialize RequestOutput objects for emitted chunks. Agent A removes work for aborted, suppressed, pooling, and no-stats/no-tracing paths, but it does not address the AsyncLLM collector behavior where an output is genuinely emitted yet immediately merged into an already pending DELTA output. This proposal targets allocation and merge overhead after detokenization by avoiding transient RequestOutput construction only when collector aggregation makes it redundant, so it is orthogonal to both offloading and pre-partition/suppression specialization.

---
