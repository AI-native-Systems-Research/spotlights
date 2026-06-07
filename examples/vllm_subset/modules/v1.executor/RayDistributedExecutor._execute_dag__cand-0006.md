# RayDistributedExecutor._execute_dag

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/ray_executor.py`](vllm/v1/executor/ray_executor.py) (lines 451–485)
- **Symbol:** `RayDistributedExecutor._execute_dag`
- **Kind:** method
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0006`

## Description
Driver-side Ray compiled-DAG execution path. It lazily compiles the DAG, submits scheduler and grammar outputs, then either gets a single worker result, wraps refs in FutureWrapper, or gathers and aggregates all worker outputs for KV connector mode.

## Current approach
Each call executes forward_dag with a freshly built (scheduler_output, grammar_output) tuple, branches on connector and non_block state, allocates FutureWrapper for async paths, and in blocking connector mode ray.gets all refs, loops over outputs to detach Ray zero-copy buffers, then calls kv_output_aggregator.aggregate().

## Estimated impact explanation
This is the Ray driver data path for every decode step. Reducing Python-side submission, wrapper, and aggregation overhead should lower median TPOT for Ray deployments and reduce TTFT when the first prefill triggers lazy DAG compilation.

## Evolve rationale
The concrete hot constructs are forward_dag.execute((...)) at line 461, the branch ladder at lines 463-485, and the detach-plus-aggregate loop at lines 479-482. Evolution can specialize no-connector async mode, reduce per-step tuple/wrapper allocation, and fuse detach with aggregation for connector mode. Correctness oracle: Ray distributed tests and end-to-end generation parity with the multiprocess executor on identical prompts, including connector and no-connector configurations.

## Deep research proposals

### 1. Use msgspec structs + Ray custom serializer on the compiled-DAG hot path
- **Finding:** `find-0009` — *msgspec*
- **Source URL:** <https://jcristharif.com/msgspec/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

On the driver-side _execute_dag path in vllm/v1/executor/ray_executor.py:451-485, every decode step calls forward_dag.execute((scheduler_output, grammar_output)) (line 461) and then ray.get(refs) plus detach_zero_copy_from_model_runner_output on each ModelRunnerOutput (lines 467-482). Today SchedulerOutput (vllm/v1/core/sched/output.py:181) is a plain dataclass and ModelRunnerOutput is similarly non-msgspec, so Ray falls back to cloudpickle/pickle to serialize the tuple sent across the compiled DAG and the outputs returned by workers. Convert these hot-path message types (SchedulerOutput, GrammarOutput, ModelRunnerOutput, plus the small NewRequestData / CachedRequestData / SpecDecodeMetadata leaf types they own) into msgspec.Struct subclasses with explicit fields (mirroring the pattern already used in vllm/v1/engine/__init__.py, vllm/v1/serial_utils.py, and vllm/v1/engine/core_client.py), and register a Ray custom serializer (ray.util.register_serializer) for these types that uses msgspec.msgpack.Encoder/Decoder. Inside _execute_dag, keep the current control flow (the lazy-compile, the no-connector single-worker fast path, the FutureWrapper non-block branch, and the connector aggregate branch) but the underlying transport will now encode/decode via msgspec instead of cloudpickle. Tensors and other zero-copy buffers should remain Ray ObjectRef / shared-memory payloads via msgspec custom hooks (enc_hook/dec_hook) that pass numpy / torch tensors through unchanged, so detach_zero_copy_from_model_runner_output continues to work as-is. Validate with the existing Ray distributed tests under tests/ and end-to-end generation parity (greedy + sampled) versus the multiprocess executor on identical prompts, in both no-connector and KV-connector configurations.

**Proposal rationale.**

_execute_dag is invoked once per decode step and submits/receives the largest structured payloads on the Ray driver path: a fresh (scheduler_output, grammar_output) tuple in, and one ModelRunnerOutput per worker out. Profiling of similar hot paths consistently shows pickle/cloudpickle dominating Python-side CPU when the structured payload has many small fields (per-request lists, token id arrays, block tables, scheduled spec tokens). The finding's claim of 10-80x faster encode/decode with msgspec directly attacks this overhead. This is a stronger fit than generic 'use a faster serializer' because vLLM v1 already uses msgspec elsewhere (engine core client/coordinator, serial_utils), so the schemas, custom enc/dec hooks for tensors, and integration patterns are already established and just need to be extended to the scheduler/model-runner message types that traverse the compiled DAG. It addresses a concrete gap the candidate's evolve_rationale calls out (per-step tuple/wrapper allocation and connector-mode aggregation cost) by shrinking the per-step CPU spent in serialization, which is on the critical path for both first-token (lazy DAG compile + first execute) and steady-state TPOT in Ray deployments.

---

## Agent proposals

### 1. Specialize _execute_dag at DAG-compile time and fuse detach into aggregate
- **Agent:** claude

**Detailed description.**

In vllm/v1/executor/ray_executor.py:451-485, replace the per-step branch ladder (lines 463-485) with a configuration-specialized callable bound once when forward_dag is built. At the end of _compiled_ray_dag (and at the lazy-compile site at line 459), inspect the fixed-for-the-engine state — self.has_connector, whether PP is enabled (which determines if non_block is ever True), and whether kv_output_aggregator is present — and assign self._execute_step to one of four pre-defined specialized closures: (a) no-connector blocking that calls refs[0].get() + detach inline and returns the ModelRunnerOutput; (b) no-connector non-blocking that wraps refs[0] in FutureWrapper; (c) connector blocking that calls a new fused method on KVOutputAggregator (e.g., aggregate_with_detach(refs)) which performs ray.get, the per-output detach_zero_copy_from_model_runner_output, and the existing aggregate reduction in a single Python pass over the outputs list — eliminating the separate `for output in outputs: detach(...)` loop at lines 480-481 and the second iteration inside aggregate; (d) connector non-blocking that returns FutureWrapper(refs, kv_output_aggregator). The execute_model entry point (line 449) then calls self._execute_step(scheduler_output, grammar_output) directly, skipping every per-step `if self.has_connector / if not non_block` Python check. Also hoist the `(scheduler_output, grammar_output)` tuple construction by passing the two args positionally into forward_dag.execute via a thin helper that calls execute(scheduler_output, grammar_output) if the compiled-DAG signature accepts varargs, otherwise reuses a single 2-element list buffer rather than allocating a fresh tuple per decode step. The lazy-compile branch at line 458 stays, but the steady-state path becomes one attribute load + one indirect call + one fused output pass. Validate with the existing Ray distributed tests and end-to-end greedy/sampled generation parity vs the multiprocess executor in both no-connector and KV-connector configurations, and confirm with py-spy that the four-arm `if` ladder no longer appears in the driver hot path.

**Novelty rationale.**

The single existing deep_research_proposal (find-0009) is exclusively about converting SchedulerOutput / ModelRunnerOutput to msgspec.Struct and registering a Ray custom serializer — it explicitly preserves the current control flow ('keep the current control flow ... the connector aggregate branch'). It does not touch the per-step branch ladder, does not eliminate the per-step tuple allocation, and does not fuse detach_zero_copy_from_model_runner_output with kv_output_aggregator.aggregate — all three of which are called out in the candidate's evolve_rationale as separate optimization targets. This proposal attacks Python dispatch and iteration overhead orthogonally to serialization cost, and the two changes compose (faster bytes via msgspec + fewer Python ops via specialization).

---

### 2. Drop discarded no-connector worker outputs before they cross Ray
- **Agent:** codex

**Detailed description.**

In `vllm/v1/executor/ray_executor.py`, `_execute_dag` only consumes `refs[0]` when `self.has_connector` is false, but `_compiled_ray_dag` still exposes every last-stage TP worker through `MultiOutputNode(outputs)`. Add a no-connector Ray DAG mode that still schedules every last-stage worker, but has non-output TP ranks return a lightweight sentinel after executing the model/sample step instead of a full `ModelRunnerOutput`. Concretely, pass or derive a `return_driver_output` flag in the Ray compiled-DAG worker path (`RayWorkerWrapper.execute_model_ray`) so final PP rank TP0 preserves the current output, connector mode preserves all outputs for `KVOutputAggregator`, and final PP non-output ranks in no-connector mode return `None` after participating in all collectives. `_execute_dag` can keep using `refs[0]` as today; the difference is that Ray no longer has to serialize, publish, and retain discarded full outputs for `refs[1:]` on every decode step. Validate with Ray distributed TP>1/PP>1 tests, no-connector generation parity, connector parity to ensure all-rank aggregation is unchanged, and a driver profile confirming smaller Ray compiled-DAG result-channel traffic in TP>1 no-connector runs.

**Novelty rationale.**

The deep-research proposal speeds serialization of the same hot payload types; it does not reduce how many worker outputs are produced or sent. Agent A specializes the driver branch ladder, tuple construction, and connector detach/aggregate loop, but still assumes the DAG returns the same `refs` and the driver discards non-output refs in no-connector mode. This proposal changes the Ray DAG/worker return contract so unused no-connector outputs are never materialized as full driver-facing payloads, which is orthogonal to both serializer choice and driver-side control-flow specialization.

---
