# OutputProcessor.process_outputs

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/output_processor.py`](vllm/v1/engine/output_processor.py) (lines 589–711)
- **Symbol:** `OutputProcessor.process_outputs`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_engine-0005`

## Description
Main per-batch EngineCoreOutput loop that updates stats, detokenizes, processes logprobs, builds RequestOutputs, enqueues them, and frees finished requests.

## Current approach
Runs a sequential Python loop over engine_core_outputs. Each iteration performs request-state dict lookup, stats updates, detokenizer.update, logprobs processing, RequestOutput construction, queue insertion, finish handling, and optional tracing.

## Estimated impact explanation
This loop touches every emitted token before the caller can observe it; constant-factor reductions compound directly into lower frontend CPU cost and median TPOT.

## Evolve rationale
The docstring explicitly marks this as the only full-batch Python loop in V1 output processing. Optimization units include hoisting hot lookups, specializing pooling vs generation and streaming vs non-streaming branches, batching finish/abort handling, and reducing allocation churn. Correctness oracle: tests/v1/engine/test_output_processor.py covers stream_interval, stop handling, logprobs, pooling, streaming input, and finished-request cleanup.

## Deep research proposals

### 1. Batch DecodeStream.step over multi-token engine outputs in fast detokenizer path
- **Finding:** `find-vllm_v1_engine-0011` — *Decoders*
- **Source URL:** <https://huggingface.co/docs/tokenizers/main/en/api/decoders>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In OutputProcessor.process_outputs (vllm/v1/engine/output_processor.py:589-711), the per-iteration call req_state.detokenizer.update(new_token_ids, finish_reason == FinishReason.STOP) at line 656 fans out to a Python for-loop over new_token_ids in BaseIncrementalDetokenizer.update (vllm/v1/engine/detokenizer.py:118), which calls self.decode_next(new_token_id) -> stream.step(tokenizer, next_token_id) once per token (detokenizer.py:211, :226). When the engine emits multi-token outputs in a single step (speculative decoding, chunked prefill emission), this pays N Python-call round-trips into the tokenizers Rust extension for what the HuggingFace tokenizers DecodeStream API already exposes as a single list-of-ids step. Change: in FastIncrementalDetokenizer, add a batched decode_next_batch(new_token_ids) that invokes stream.step(tokenizer, new_token_ids) (list overload documented at https://huggingface.co/docs/tokenizers/main/en/api/decoders) once per engine output and returns the concatenated decoded chunk, preserving UTF-8 boundary buffering inside DecodeStream. Override BaseIncrementalDetokenizer.update in the fast subclass (or take a batched fast path when len(new_token_ids) > 1 and no per-token-sensitive features are active) so that: (a) with stop_terminated and not include_stop_str_in_output, the final stop token is still peeled off before the batched step; (b) self.output_text is extended in one operation, and stop_check_offset is set once at entry (skip the batched path when self.min_tokens is non-zero and num_output_tokens() is near min_tokens, falling back to per-token so the existing min_tokens gating of stop_check_offset is preserved bit-for-bit); (c) spaces_between_special_tokens handling — currently interleaved with decode_next — is applied by post-scanning new_token_ids against self.added_token_ids after the batched decode, so last_special state matches the per-token result; (d) the invalid-prefix / OverflowError recovery in _protected_step is retained by wrapping the batched step in the same try/except and, on error, falling back to the per-token loop for that engine output. token_ids.extend(new_token_ids) becomes a single list extend rather than N appends. No visible streamed-text change: DecodeStream's own UTF-8 buffering guarantees byte-identical output.

**Proposal rationale.**

The candidate's evolve_rationale explicitly asks for reductions in the per-iteration Python cost of the process_outputs loop, and detokenization is one of the heaviest per-iteration calls (line 656). The current fast path already uses tokenizers' DecodeStream, but only in single-id mode, so every extra token in a multi-token engine output pays a full Python->Rust call and a Python string append. The finding contributes a specific, transferable capability of the same library already in use (list-of-ids step) that collapses those calls into one for the multi-token case that matters most for the stated workload (multi-turn agentic — heavy speculative-decoding acceptance patterns) and directly targets median TPOT by shrinking the constant factor on every emitted token batch. It does not restate the current approach: the current approach is single-id step in a Python loop; the proposal is batched step per engine output. The correctness oracle (tests/v1/engine/test_output_processor.py) plus the existing detokenizer tests already cover stop handling, logprobs, and streaming, which is exactly the surface this change must not perturb.

---

## Agent proposals

### 1. Partition engine_core_outputs by branch and hoist bound-method locals in process_outputs
- **Agent:** claude

**Detailed description.**

In OutputProcessor.process_outputs (vllm/v1/engine/output_processor.py:589-711), replace the single monomorphic-in-name-but-polymorphic-in-behavior for-loop with a two-pass structure that first classifies each EngineCoreOutput and then runs specialized tight loops per class. Concretely: (1) In one pass, walk engine_core_outputs and produce four small lists of (req_state, engine_core_output) pairs: pooling_queue, pooling_list, gen_queue, gen_list — using a single dict.get per output on self.request_states (already the case today) plus one attribute read of req_state.queue and one of req_state.pooling_output-is-None equivalent (use engine_core_output.pooling_output is None, which is already read at line 632). Skip outputs whose req_state is None here so the specialized loops need no None-check. (2) Run four separate specialized loops. Each loop begins by binding the hot attributes/methods to locals so the CPython bytecode uses LOAD_FAST instead of LOAD_ATTR chains: e.g. for the generation+queue variant, hoist `_update_stats = self._update_stats_from_output`, `_finish = self._finish_request`, `_do_tracing = self.do_tracing if self.tracing_enabled else None`, and inside the loop bind `det_update = req_state.detokenizer.update`, `lp_update = req_state.logprobs_processor.update_from_output`, `make_ro = req_state.make_request_output`, `q_put = req_state.queue.put` once per iteration. In the pooling variants, skip the detokenizer/logprobs block entirely rather than gating on `if pooling_output is None:` per iteration — the branch is resolved by which list the item is in. In the LLMEngine (no-queue) variants, append directly to a preallocated `request_outputs` list via a hoisted `ro_append = request_outputs.append` rather than the current `if req_state.queue is not None: ... else: request_outputs.append(...)` check per iteration. (3) Also hoist the streaming_input finish-handling: because `req_state.streaming_input` is a per-request property that does not change within a single process_outputs call for a given req_state, in the classification pass tag pairs where streaming_input is True and route them to the streaming-input finish path only when finish_reason is set, rather than re-testing `req_state.streaming_input` twice per iteration (once at line 676 for the finished=False override and once at line 688 for input_chunk_queue handling). (4) Aggregate reqs_to_abort by appending to a single hoisted `abort_append = reqs_to_abort.append`. Preserve the docstring's contract that this is the only full-batch Python loop in V1 output processing by keeping the classification pass and the four specialized loops all inside process_outputs; the total number of Python iterations over the batch is 1 (classify) + N (specialized, but each output visited by exactly one). Correctness is preserved because the branching in the current loop is data-dependent only on properties that are stable within one process_outputs call (pooling vs generation is fixed by request setup; queue vs no-queue is fixed by AsyncLLM vs LLMEngine choice; tracing_enabled is process-global; streaming_input flips only at request finish and is handled explicitly). Expected wins: removes 4-6 attribute-chain LOAD_ATTRs per output on the CPython path (req_state.detokenizer.update, req_state.logprobs_processor.update_from_output, req_state.make_request_output, req_state.queue.put, self._update_stats_from_output, self._finish_request), removes 2 per-iteration `is None` branch tests (pooling_output, queue) and 1 per-iteration `if self.tracing_enabled` test, and reduces bytecode dispatch cost proportionally to batch size. Correctness oracle is unchanged: tests/v1/engine/test_output_processor.py already covers stream_interval, stop handling, logprobs, pooling, streaming input, and finished-request cleanup — the specialization must produce byte-identical behavior on each.

**Novelty rationale.**

The existing deep_research_proposal (find-vllm_v1_engine-0011) targets step 2 of the loop only — batching DecodeStream.step inside the fast detokenizer to collapse Python->Rust round-trips for multi-token engine outputs. It does not restructure process_outputs itself and it does not touch attribute-chain lookup cost, per-iteration branch tests (pooling_output is None, queue is not None, tracing_enabled, streaming_input), or the LOAD_ATTR/LOAD_METHOD overhead that dominates a Python-level hot loop of this shape. This proposal is complementary and non-overlapping: it addresses the two evolve_rationale bullets that the existing proposal leaves untouched — 'hoisting hot lookups' and 'specializing pooling vs generation and streaming vs non-streaming branches' — via loop partitioning plus local-name binding, and it composes cleanly with the batched detokenizer step (the fast-path det_update local would just call the batched entry point if that other proposal lands).

---

### 2. Coalesce queued DELTA outputs before allocating RequestOutput
- **Agent:** codex

**Detailed description.**

In `OutputProcessor.process_outputs` (`vllm/v1/engine/output_processor.py:589-711`), add an AsyncLLM-only fast path for generation requests whose `req_state.queue` is a `RequestOutputCollector` in aggregating DELTA mode and already holds a pending `RequestOutput`. Today the loop always calls `req_state.make_request_output(...)`, which allocates a fresh `CompletionOutput`, output list, and `RequestOutput`, then `RequestOutputCollector.put()` immediately merges that object into the already-pending queued output when the consumer has not drained it yet. Change the queue handling so `process_outputs` first tries `req_state.queue.try_append_delta(...)` after detokenization/logprobs update and before `make_request_output`. The helper should be conservative: only handle non-pooling generation, `RequestOutputKind.DELTA`, `req_state.parent_req is None`, collector `aggregate` is true, collector `output` is an unfinished/pending `RequestOutput`, and the stream-interval/final-only emission rules say an output is due. It can then compute the delta fields using the same stateful primitives used by `RequestState._new_completion_output` (`detokenizer.get_next_output_text(finished, delta=True)`, current token slice, logprobs tail, cumulative logprob, finish/stop metadata) and mutate the pending `CompletionOutput` in place, mirroring `RequestOutput.add(..., aggregate=True)`, including `kv_transfer_params`, `ec_transfer_params`, `finished`, and final routed-experts concatenation. If any condition is not met, fall back to the existing `make_request_output` path. Add/extend tests in `tests/v1/engine/test_output_processor.py` with an AsyncLLM-style queued DELTA request where multiple `process_outputs` calls happen before `collector.get_nowait()`, asserting byte-identical text/token/logprob aggregation and finished cleanup.

**Novelty rationale.**

The deep-research proposal optimizes detokenization inside `detokenizer.update` by batching tokenizer DecodeStream calls for multi-token engine outputs; it does not address the downstream allocation of `RequestOutput` objects or collector aggregation. Agent A's proposal specializes branches and hoists lookups in `process_outputs`, but it still calls `make_request_output` for every emitted queued delta and then lets `RequestOutputCollector.put()` merge the freshly allocated object. This proposal targets a different source of per-token frontend CPU and allocation churn: bypassing throwaway `RequestOutput` construction when the AsyncLLM queue is already aggregating an unread DELTA output, which is common when generation outruns the consumer in multi-turn agentic workloads.

---
