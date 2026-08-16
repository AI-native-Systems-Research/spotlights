# AsyncOutputFuture.result/UniProcExecutor.collective_rpc

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/uniproc_executor.py`](vllm/v1/executor/uniproc_executor.py) (lines 26–106)
- **Symbol:** `AsyncOutputFuture.result/UniProcExecutor.collective_rpc`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_executor-0014`

## Description
Single-process executor RPC and async-output materialization path used by UniProcExecutor execute_model, sample_tokens, and draft-token calls.

## Current approach
collective_rpc normalizes kwargs, calls run_method on the driver_worker immediately, calls AsyncModelRunnerOutput.get_output inline for blocking calls, and wraps results in a list unless single_value is requested. For non_block calls, run_method still executes immediately; AsyncModelRunnerOutput is wrapped in AsyncOutputFuture, whose result() later calls get_output and stores either the single output or a one-item list.

## Estimated impact explanation
The path is paid for every token step on UniProcExecutor. It is medium because single-process deployments avoid interprocess/Ray overhead, but removing get_output blocking and Python wrapper churn can still lower median TPOT for short multi-turn decode workloads.

## Evolve rationale
The concrete constructs are run_method at lines 92 and 98 and AsyncModelRunnerOutput.get_output at lines 38 and 94. This is the per-step executor path for single-process serving. Headroom includes avoiding avoidable list wrapping and Future allocation on single-value hot paths, staging async output materialization so result() does less blocking host/device synchronization, and sharing the same no-copy output contract across blocking and non-blocking calls. Correctness oracle: UniProcExecutor.execute_model, sample_tokens, and take_draft_token_ids return the same values as today, exceptions are surfaced through Future.result or direct calls identically, and existing uniproc async scheduling tests preserve result ordering.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Fast-path collective_rpc: cache execute_model/sample_tokens bound methods and elide list wrapping via a dedicated async_call path
- **Agent:** claude

**Detailed description.**

Rework vllm/v1/executor/uniproc_executor.py:79-106 (and its two hot callers execute_model at lines 108-121 and sample_tokens at lines 123-131) to remove three per-step overheads that recur every decode token:

1. Bypass run_method dispatch on the hot loop. In _init_executor, resolve and cache bound method references for the small set of methods actually called on every step ('execute_model', 'sample_tokens', 'take_draft_token_ids') as attributes on self (e.g. self._m_execute_model = getattr(self.driver_worker, 'execute_model')). Add a private fast helper _hot_call(bound, args, kwargs) that just invokes bound(*args, **kwargs). run_method in vllm/v1/serial_utils performs string lookup / attribute resolution and Callable branching that is wasted work when the method name is a compile-time constant. execute_model and sample_tokens should call the cached bound method directly; collective_rpc remains for the generic path (unit tests, take_draft_token_ids, and arbitrary rpc invocations).

2. Eliminate the '[result]' list allocation on the blocking single_value=False branch by having collective_rpc return the raw result when single_value is True and only building the one-element list once, lazily, when a caller actually requires list form. Every current in-tree caller of execute_model/sample_tokens passes single_value=True, so the list branch is only exercised by generic rpc; keep it but make the single_value=True path allocation-free (no list, no tuple, no Future wrapping when the result is not an AsyncModelRunnerOutput).

3. Collapse AsyncOutputFuture for single_value non_block calls. When run_method returns an AsyncModelRunnerOutput and single_value is True (the only shape produced by execute_model/sample_tokens today), return a lighter object that implements only .done()/.result()/.exception()/.add_done_callback with a single boolean and a cached slot, instead of subclassing concurrent.futures.Future (which pays for an RLock, waiter list, and state machine on every construction). Use __slots__ = ('_async_output','_result','_exc','_done'). Materialize get_output() lazily on first .result() as today, but skip the (result if self.single_value else [result]) branch and the set_result path that acquires the Future's internal lock. Keep the general Future path for single_value=False and for non-AsyncModelRunnerOutput non_block results so semantics of collective_rpc for arbitrary callers are preserved.

Additionally, in execute_model at lines 118-120, the current 'if non_block and output.done(): output.result()' early-raise pattern calls .done() on the lightweight future; ensure the new slotted future implements .done() cheaply (single bool check) so this remains O(1).

Correctness oracle stays the ones stated on the candidate: UniProcExecutor.execute_model, sample_tokens, take_draft_token_ids return the same values, exceptions surface identically through .result(), and async scheduling ordering is preserved. Existing tests in tests/v1 (uniproc executor async scheduling) exercise both blocking and non_block paths.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any concrete proposal here is novel by construction. Beyond that, this proposal targets three specific micro-costs — run_method string dispatch on a compile-time-constant method name, the '[result]' one-element list allocation on the blocking path, and the heavyweight concurrent.futures.Future subclass on the non_block single-value hot path — and proposes caching bound driver_worker methods on the executor plus a __slots__-based lightweight future that avoids Future's RLock/waiter machinery. These are distinct from the higher-level 'stage async output materialization' and 'share no-copy output contract' directions hinted at in evolve_rationale, and are individually measurable in TPOT microbenchmarks for single-process multi-turn decode.

---

### 2. Release AsyncModelRunnerOutput after future materialization
- **Agent:** codex

**Detailed description.**

In vllm/v1/executor/uniproc_executor.py:26-42, change AsyncOutputFuture.result() so that once async_output.get_output() has either produced the ModelRunnerOutput or raised, the Future no longer holds a strong reference to the AsyncModelRunnerOutput wrapper. For example, copy self.async_output into a local variable, set self.async_output = None in a finally block after get_output(), and then cache the result or exception via set_result/set_exception as today. Adjust the attribute type accordingly. This preserves the one-shot get_output contract and the public Future behavior, but lets the CUDA event wrapper, temporary CPU arrays, and any retained device tensor references inside AsyncOutput/AsyncGPUModelRunnerOutput be released as soon as result() materializes, even if the engine keeps completed futures around briefly across multi-turn scheduling.

**Novelty rationale.**

There are no deep_research_proposals listed. Agent A focuses on dispatch overhead, list allocation, and replacing concurrent.futures.Future with a lightweight slotted future. This proposal targets a different cost: object lifetime and retained tensor/copy-buffer references after AsyncOutputFuture.result() has already consumed the one-shot AsyncModelRunnerOutput. It can be implemented independently of Agent A's fast path and does not duplicate its allocation or dispatch changes.

---
