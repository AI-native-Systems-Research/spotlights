# vllm/v1/engine

[← All modules](../index.md)

## Module
- **Path:** `vllm/v1/engine`
- **Description:** Outer engine API and per-process lifecycle: async/sync entrypoints, EngineCore loop, and input/output/detokenizer staging.
- **Depends on:** _(none)_
- **Main files:**
  - `vllm/v1/engine/async_llm.py` — AsyncLLM front-end
  - `vllm/v1/engine/core.py` — EngineCore process loop (scheduler + executor)
  - `vllm/v1/engine/llm_engine.py` — Sync LLMEngine wrapper
  - `vllm/v1/engine/output_processor.py` — Detokenize and package engine outputs
- **Run status:** SUCCEEDED
- **Findings:** 15
- **Issues:** 0

## Candidates

| Candidate | Impact | Deep research proposals |
|---|---|---:|
| [`DPLBAsyncMPClient.get_core_engine_for_request`](vllm_v1_engine/DPLBAsyncMPClient.get_core_engine_for_request__cand-vllm_v1_engine-0001.md) | high | 5 |
| [`DPEngineCoreProc._should_throttle_prefills`](vllm_v1_engine/DPEngineCoreProc._should_throttle_prefills__cand-vllm_v1_engine-0010.md) | medium | 4 |
| [`EngineCore.step_with_batch_queue`](vllm_v1_engine/EngineCore.step_with_batch_queue__cand-vllm_v1_engine-0003.md) | medium | 3 |
| [`BaseIncrementalDetokenizer.update`](vllm_v1_engine/BaseIncrementalDetokenizer.update__cand-vllm_v1_engine-0006.md) | high | 2 |
| [`DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task`](vllm_v1_engine/DPAsyncMPClient._ensure_stats_update_task.run_engine_stats_update_task__cand-vllm_v1_engine-0011.md) | medium | 2 |
| [`EngineCoreProc.process_output_sockets`](vllm_v1_engine/EngineCoreProc.process_output_sockets__cand-vllm_v1_engine-0018.md) | medium | 2 |
| [`DPEngineCoreProc._has_global_unfinished_reqs`](vllm_v1_engine/DPEngineCoreProc._has_global_unfinished_reqs__cand-vllm_v1_engine-0002.md) | medium | 1 |
| [`OutputProcessor.process_outputs`](vllm_v1_engine/OutputProcessor.process_outputs__cand-vllm_v1_engine-0005.md) | high | 1 |
| [`check_stop_strings`](vllm_v1_engine/check_stop_strings__cand-vllm_v1_engine-0007.md) | medium | 1 |
| [`DPCoordinator.run polling/publish loop`](vllm_v1_engine/DPCoordinator.run_polling_publish_loop__cand-vllm_v1_engine-0014.md) | medium | 1 |
| [`EngineCore.step`](vllm_v1_engine/EngineCore.step__cand-vllm_v1_engine-0015.md) | medium | 1 |
| [`AsyncLLM._run_output_handler.output_handler`](vllm_v1_engine/AsyncLLM._run_output_handler.output_handler__cand-vllm_v1_engine-0004.md) | high | 0 |
| [`LogprobsProcessor._update_prompt_logprobs`](vllm_v1_engine/LogprobsProcessor._update_prompt_logprobs__cand-vllm_v1_engine-0008.md) | medium | 0 |
| [`EngineCoreProc._process_engine_step`](vllm_v1_engine/EngineCoreProc._process_engine_step__cand-vllm_v1_engine-0009.md) | medium | 0 |
| [`RequestState.make_request_output`](vllm_v1_engine/RequestState.make_request_output__cand-vllm_v1_engine-0012.md) | medium | 0 |
| [`LogprobsProcessor._update_sample_logprobs`](vllm_v1_engine/LogprobsProcessor._update_sample_logprobs__cand-vllm_v1_engine-0013.md) | medium | 0 |
| [`InputProcessor.process_inputs`](vllm_v1_engine/InputProcessor.process_inputs__cand-vllm_v1_engine-0016.md) | medium | 0 |
| [`RequestOutputCollector.put/get_nowait/get`](vllm_v1_engine/RequestOutputCollector.put_get_nowait_get__cand-vllm_v1_engine-0017.md) | medium | 0 |

## Findings (full list)

1. **Taming Request Imbalance: SLO-Aware Scheduling for Disaggregated LLM Inference**
   - Source type: paper
   - URL: <https://papers.cool/arxiv/2605.02329>
   - Technique: Adopt slack-guided adaptive decode batching: profile/update a lookup table for step time by batch size and sequence length, compute per-request TPOT slack, then selectively run requests whose batch packing improves throughput without violating the tightest slack. The same source also contributes urgency-based prefill selection, useful for reducing median TTFT in long-tail multi-turn workloads where short turns can otherwise sit behind long prefills.
   - Evidence: "Kairos employs urgency-based priority scheduling" and "introduces slack-guided adaptive batching"; lines 6-9 and Algorithm 3 excerpt in the alphaXiv/papers.cool text.
2. **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**
   - Source type: paper
   - URL: <https://www.usenix.org/conference/osdi24/presentation/agrawal>
   - Technique: Use stall-free chunked prefill plus decode-maximal batching: split long prefills into near-equal chunks and fill the rest of each iteration with decode work. This directly targets TPOT stalls and pipeline bubbles while preserving TTFT for new requests.
   - Evidence: "Sarathi-Serve introduces chunked-prefills" and "creates stall-free schedules that adds new requests in a batch without pausing ongoing decodes"; USENIX abstract.
3. **Supported load balancers**
   - Source type: docs
   - URL: <https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/load_balancers.html>
   - Technique: Replace full-rank scans in the DP admission hot path with Power-of-Two Choices for large DP deployments, while retaining full scan for tiny or low-rate cases. Envoy’s weighted least-request variant also gives a concrete active-request bias formula that could replace fixed hand-tuned load slopes.
   - Evidence: "An O(1) algorithm which selects N random available hosts ... and picks the host which has the fewest active requests"; Weighted least request section, lines 43-52.
4. **[Feature] RFC: Paper Reproduction of LMetric Multiplication Scheduling in SGLang Gateway**
   - Source type: issue
   - URL: <https://github.com/sgl-project/sglang/issues/29573>
   - Technique: Use a single multiplicative routing score combining estimated new prefill work after prefix reuse with current worker load. This could reduce TTFT for repeated multi-turn prefixes without a brittle threshold switch between cache locality and load balance.
   - Evidence: "score = new_prefill_work * (worker.load() + 1)" and "The worker with the smallest score is selected"; Design / Policy Formula, lines 225-237.
5. **Prefix-aware routing**
   - Source type: docs
   - URL: <https://docs.ray.io/en/latest/serve/llm/user-guides/prefix-aware-routing.html>
   - Technique: Maintain an approximate prefix tree at the router and use a three-tier decision: prefer prefix affinity only while queue lengths are balanced, fall back to low cache utilization for weak matches, and use P2C under imbalance. This is directly aligned with multi-turn agentic traffic where shared system prompts and session history can make cache-affine DP routing lower TTFT.
   - Evidence: "routes requests with similar prefixes to the same replicas" and "balances cache locality with load distribution"; How it works section.
6. **Scheduler**
   - Source type: docs
   - URL: <https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/03-scheduler.md>
   - Technique: Adopt shape-aware DP rank selection and prefix-aware scheduling fallback limits: balance input/prefill and output/decode token load as separate dimensions, and disable expensive prefix-priority computation when queue length makes scheduler overhead itself the bottleneck. This targets both DP routing quality and Python scheduling overhead.
   - Evidence: "shape_aware | Balances input/prefill and output/decode token load separately" and "LPM automatically falls back to FCFS when waiting_queue exceeds 128 requests"; Scheduling Policies, lines 377-393.
7. **FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference**
   - Source type: paper
   - URL: <https://bingyangwu.github.io/publication/fastgen/>
   - Technique: Use iteration-level preemption with skip-join Multi-Level Feedback Queues, assigning initial priority from input length and allowing preemption at output-token boundaries. This suggests a concrete way to reduce head-of-line blocking in interactive agentic workloads without needing exact output lengths.
   - Evidence: "enable preemption at the granularity of each output token" and "skip-join Multi-Level Feedback Queue scheduler"; abstract lines 12-14.
8. **NanoFlow: Towards Optimal Large Language Model Serving Throughput**
   - Source type: paper
   - URL: <https://huggingface.co/papers/2408.12757>
   - Technique: Split work into operation-level nano-batches and use an operation pipeline with execution-unit scheduling so CPU scheduling, network/KV movement, and GPU operations overlap more tightly. The top-level scheduling idea can inform pipeline batch-queue decisions and deferred output handling to reduce TPOT overhead under high concurrency.
   - Evidence: "splits requests into nano-batches at the granularity of operations" and "uses an operation-level pipeline with execution unit scheduling"; abstract lines 90-92.
9. **Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs**
   - Source type: paper
   - URL: <https://arxiv.gg/abs/2506.03296>
   - Technique: Use profiling-informed scheduling that predicts CPU and GPU subtask times, dispatching work to maximize overlap while limiting scheduling overhead. This is transferable to event-driven wakeups and adaptive backoff around KV-transfer readiness and delayed frees, replacing fixed sleeps with observed-progress signals.
   - Evidence: "dynamically dispatches compute across heterogeneous resources by predicting execution times"; abstract lines 8-10.
10. **Efficient string matching**
   - Source type: paper
   - URL: <https://cir.nii.ac.jp/crid/1364233268843300096>
   - Technique: Replace repeated per-stop-string suffix searches with an incremental Aho-Corasick automaton per request, retaining state across decoded chunks. This gives earliest-completion multi-stop matching in a single pass over new text and avoids work growing with the number of stop strings.
   - Evidence: "constructing a finite state pattern matching machine" and processing "the text string in a single pass"; abstract lines 80-81.
11. **Decoders**
   - Source type: docs
   - URL: <https://huggingface.co/docs/tokenizers/main/en/api/decoders>
   - Technique: Use DecodeStream’s ability to accept a list of token IDs to batch incremental detokenization for speculative or multi-token steps, while preserving UTF-8 buffering semantics. This can reduce per-token Python calls in output processing without changing visible streamed text.
   - Evidence: "id (`int` or `List[int]`) — The next token ID, or a list of token IDs"; DecodeStream step API.
12. **ZeroMQ | Socket API**
   - Source type: docs
   - URL: <https://zeromq.org/socket-api/?language=go&library=zmq4>
   - Technique: Apply explicit high-water-mark driven backpressure and socket-type-aware send policy to engine output/stat channels. This supports bounded queues, fewer unbounded message bursts, and safer coalescing of small stats messages without changing ordering semantics.
   - Evidence: "The high water mark is a hard limit on the maximum number of outstanding messages"; High-Water-Mark section, lines 144-148.
13. **Dynamic Request Batching**
   - Source type: docs
   - URL: <https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html>
   - Technique: Use latency-budgeted dynamic batching knobs for output and admission queues: cap batch size by a custom cost metric such as total tokens, and use a wait timeout derived from the remaining latency SLO. This is applicable to chunking EngineCore outputs and abort batches so throughput gains do not hide token delivery latency.
   - Evidence: "batch_size_fn optional function to compute the effective batch size" and "Set batch_wait_timeout_s considering the end-to-end latency SLO"; Dynamic Request Batching configuration and tuning sections.
14. **Direct streaming**
   - Source type: docs
   - URL: <https://docs.ray.io/en/master/serve/llm/user-guides/direct-streaming.html>
   - Technique: Remove unnecessary proxy/ingress hops from streaming token delivery and bound any body-aware routing copy with truncation. The transferable idea is to keep the per-token response path as direct as possible and make router-body inspection explicitly budgeted against TTFT.
   - Evidence: "Removing the ingress proxy hop cuts per-token overhead" and "buffering and re-emitting large bodies adds time to first token"; When to use direct streaming and Body-aware routers sections.
15. **GitHub - Netflix/concurrency-limits**
   - Source type: codebase
   - URL: <https://github.com/Netflix/concurrency-limits>
   - Technique: Use short-window and long-window EWMA divergence to adapt concurrency/admission limits from observed latency rather than fixed throttle intervals. This can guide DP prefill throttling and stats smoothing when rank imbalance or KV pressure changes over time.
   - Evidence: "tracks the measure of divergence between two exponential averages" and uses it "to identify a queueing trend"; Gradient2 section, lines 199-201.

## Issues

_No issues._
