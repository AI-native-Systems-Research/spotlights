# LogprobsProcessor._update_sample_logprobs / _update_prompt_logprobs / _verify_tokens / _correct_decoded_token

[← v1.engine](../v1.engine.md)

- **File:** [`vllm/v1/engine/logprobs.py`](vllm/v1/engine/logprobs.py) (lines 69–352)
- **Symbol:** `LogprobsProcessor._update_sample_logprobs / _update_prompt_logprobs / _verify_tokens / _correct_decoded_token`
- **Kind:** region
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0010`

## Description
Materializes sample and prompt logprobs into Python output containers, detokenizes logprob token ids, applies UTF-8 replacement-character repair using sequential context, and updates cumulative logprob state.

## Current approach
Per-position Python loops convert numpy/torch rows to Python lists, detokenize top-k token ids, recover recent sampled context per call, scan decoded strings for the replacement character, and may decode up to four context lengths per affected token.

## Estimated impact explanation
When clients request sample or prompt logprobs, this CPU path can dominate frontend work for each output token or prefill chunk. Optimizing it reduces TPOT and prefill-to-first-output latency for logprobs-enabled agentic workloads; impact is bounded when logprobs are disabled.

## Evolve rationale
The concrete constructs are the loops at lines 86-119 and 158-187, the .tolist() conversions at lines 89-91 and 153-155, and the UTF-8 repair helpers at lines 249-346. Headroom: cache context tails, skip _verify_tokens entirely when no decoded token ends with \ufffd, reduce tolist churn, build FlatLogprobs directly, or move the repair loop to a vectorized/native path. Correctness oracle: tests/v1/engine/test_output_processor.py::test_logprobs_processor and related stop/logprob tests must match token ids, ranks, decoded_token strings, cumulative_logprob, and prompt_logprobs exactly.

## Deep research proposals

### 1. Offload LogprobsProcessor detokenization off the engine GIL-bound thread
- **Finding:** `find-0003` — *SMG: The Case for Disaggregating CPU from GPU in LLM Serving*
- **Source URL:** <https://vuink.com/post/clgbepu-d-dbet/blog/lightseek-smg>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Apply the finding's CPU/GPU-disaggregation pattern to vllm/v1/engine/logprobs.py:69-352. The per-position Python loops in _update_sample_logprobs (86-119) and _update_prompt_logprobs (158-187), the .tolist() row conversions (89-91, 153-155), and the UTF-8 repair path in _verify_tokens / _correct_decoded_token (249-346) are exactly the kind of CPU-bound, GIL-contending detokenization work the blog identifies as bottlenecking the engine. Concrete change: split LogprobsProcessor.update_from_output into (a) a fast in-engine step that snapshots the raw numpy/torch logprob rows plus the appended sampled token ids and pushes them to a queue, and (b) a worker (a dedicated detokenizer process, an existing detokenizer thread pool, or a future Rust/gRPC gateway path) that performs convert_ids_to_tokens, builds the Logprob/FlatLogprobs containers, runs replacement-character repair, and updates cumulative_logprob before the result is attached to the RequestOutput. Keep the existing _verify_tokens semantics so test_output_processor.py::test_logprobs_processor still matches token ids, ranks, decoded_token, cumulative_logprob, and prompt_logprobs exactly; the disaggregation is purely a thread/process boundary, not a semantic change.

**Proposal rationale.**

The finding directly names tokenization/detokenization as the GIL-wall bottleneck under concurrent agentic traffic, which is the same workload the candidate's caller context targets (multi-turn agentic, median TTFT/TPOT). LogprobsProcessor today runs entirely on the engine's Python thread for every output token (and every prompt position when prompt_logprobs is set), so when logprobs are enabled it competes for the GIL with scheduling, sampling postprocessing, and stop checks. Moving its detokenize+repair work onto a non-engine executor is the candidate-specific instantiation of the finding's recommendation and addresses the headroom called out in evolve_rationale (loop churn, repeated context decodes, FlatLogprobs construction) by simply removing it from the critical path rather than micro-optimizing it in place.

---

## Agent proposals

### 1. Add batch-level UTF-8 fast path and bulk FlatLogprobs assembly to LogprobsProcessor
- **Agent:** claude

**Detailed description.**

In vllm/v1/engine/logprobs.py:69-352, change LogprobsProcessor so the per-position _verify_tokens / _get_sampled_context_ids machinery only runs when actually needed, and so FlatLogprobs is filled via bulk array extension instead of per-position append. Concrete edits: (1) In _update_sample_logprobs (86-119) and _update_prompt_logprobs (158-187), after the single batched convert_ids_list_to_tokens call, do one scan `any('�' in t for t in decoded_tokens_list)` (sample) or `'�' in joined` over the flat all_decoded_tokens (prompt). If the scan is negative — overwhelmingly common for BPE/SentencePiece tokenizers on non-CJK text — skip _verify_tokens and _get_sampled_context_ids entirely for every position in the batch and pass the raw decoded slice straight through; only fall into the existing per-position correction path when the scan flags a replacement char, and even then only invoke _correct_decoded_token for the affected indices (already done) without re-running _get_sampled_context_ids per position. (2) When self.logprobs / self.prompt_logprobs is a FlatLogprobs (created by sampling_params.flat_logprobs), bypass the per-position append_logprobs_for_next_position call and instead extend FlatLogprobs.token_ids / .logprobs / .decoded_tokens / .ranks / .start_indices / .end_indices with the already-Pythonized rows in one shot per batch — sizes are known up front (num_prompt_tokens * num_logprobs for prompt; len(token_ids_lst) * (num_logprobs+1) for sample), so this becomes a few list.extend / numpy concatenation calls rather than O(num_tokens) Python-level dict-or-append churn. (3) Cache the (max_context=4) tail returned by _get_sampled_context_ids once per _update_prompt_logprobs invocation — its value cannot change inside the inner loop because new prompt_logprobs entries are appended after the loop body via append_logprobs_for_next_position, but only the *prior* sampled context matters for byte-fallback repair and that prior tail is fixed across the batch — eliminating the per-position O(1) but allocation-heavy list rebuild at line 171. Correctness: tests/v1/engine/test_output_processor.py::test_logprobs_processor and stop/logprob tests must still match token ids, ranks, decoded_token strings, cumulative_logprob, and prompt_logprobs exactly; the fast path is a pure no-op when no � is present, and the bulk FlatLogprobs path produces structurally identical containers because append_logprobs_for_next_position's only effect on FlatLogprobs is the same extend operations performed in a different order.

**Novelty rationale.**

The listed deep_research_proposal (find-0003) only proposes moving the existing LogprobsProcessor work onto a separate detokenizer thread/process — a thread-boundary change with no reduction in CPU work. This proposal is orthogonal and complementary: it reduces the work itself by (a) short-circuiting _verify_tokens/_get_sampled_context_ids for the common no-� batch via a single upfront scan, (b) replacing the per-position append_logprobs_for_next_position dispatch with bulk FlatLogprobs array extends, and (c) hoisting the context-tail computation out of the prompt-logprobs inner loop. None of these changes are about where the work runs (the existing proposal's focus); they shrink the per-token Python overhead even if the work stays on the engine thread, and they compound with offloading rather than overlapping it.

---

### 2. Deduplicate sampled-token/top-k logprob rows before detokenization
- **Agent:** codex

**Detailed description.**

In `vllm/v1/engine/logprobs.py`, add a small row-normalization helper used by both `_update_sample_logprobs` and `_update_prompt_logprobs` before `convert_ids_list_to_tokens`. Each row has the sampled or prompt token in column 0 followed by top-k token ids; when that token is also present in the top-k set, the current path decodes it twice and either relies on dict overwrite semantics or stores duplicate primitive entries in `FlatLogprobs`. Build explicit per-row entries `(token_id, logprob, rank)` using the same rank sequence as `append_logprobs_for_next_position`, but collapse duplicate token ids with the current dict-visible semantics: later top-k entries win over the column-0 sampled/prompt entry, while `cumulative_logprob` still uses the original `logprobs[0]`. Then detokenize only the unique token ids and append with an explicit-ranks path for both list-backed and `FlatLogprobs` containers. Add focused coverage for sample and prompt rows where the sampled/prompt token appears in top-k, with `flat_logprobs` true and false, asserting the public dict view is unchanged and the flat backing arrays no longer carry duplicate ids for that position.

**Novelty rationale.**

The deep-research proposal moves the existing detokenization and container-building work off the engine thread, but does not remove redundant work inside a logprob row. Agent A proposes UTF-8 fast paths, bulk `FlatLogprobs` assembly, and context-tail caching, but explicitly preserves structurally identical FlatLogprobs entries and still processes duplicate sampled/top-k token ids. This proposal targets a different source of overhead: eliminating duplicate per-row ids before token decoding and storage while preserving the existing dict-visible output semantics.

---
