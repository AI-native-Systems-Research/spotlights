# RayDistributedExecutor.collective_rpc

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/ray_executor.py`](vllm/v1/executor/ray_executor.py) (lines 470–495)
- **Symbol:** `RayDistributedExecutor.collective_rpc`
- **Kind:** method
- **Estimated impact:** low
- **Id:** `cand-vllm_v1_executor-0009`

## Description
Broadcast an RPC to all Ray worker actors: cloudpickle-serializes callable methods per call, fans out execute_method.remote across every worker, then either wraps refs in FutureWrapper or ray.gets synchronously on all ranks.

## Current approach
For non-string methods, cloudpickle.dumps is invoked on every call. A list comprehension issues worker.execute_method.remote for every worker in self.workers, then ray.get blocks on the full ref list when non_block is False because the collective_rpc contract returns one result per rank.

## Estimated impact explanation
The path can improve Ray startup, warmup, and occasional control-call latency, which may move TTFT around engine setup or reconfiguration. It is low for median TPOT because steady-state Ray execute_model/sample_tokens use the compiled DAG rather than this RPC path.

## Evolve rationale
The concrete constructs are the cloudpickle.dumps call and the full-worker list comprehension plus ray.get. This path is used for initialization, warmup, and runtime control calls such as LoRA, cache, profiling, and reconfiguration RPCs. Headroom includes memoizing cloudpickle payloads for repeated callables, reducing per-call list/ref allocation, and using cheaper string-method fast paths while preserving the list-returning collective contract. Correctness oracle: broadcast semantics are preserved, non_block=False returns a len(world_size) list in rank order, and existing Ray collective_rpc/control-plane tests observe identical exception propagation.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Cache cloudpickle payloads for repeated callable methods in collective_rpc
- **Agent:** claude

**Detailed description.**

In RayDistributedExecutor.collective_rpc at vllm/v1/executor/ray_executor.py:470-495, replace the unconditional `cloudpickle.dumps(method)` on every non-string invocation with a small bounded cache keyed by the callable's identity (e.g., `id(method)` guarded by a weakref to keep the callable alive, or `(method.__module__, method.__qualname__, id(method))`). Concretely: introduce an instance-level `self._cloudpickle_cache: dict[int, bytes]` populated lazily inside the method; on entry, if `method` is not a str, look up the id and reuse the cached bytes, otherwise cloudpickle.dumps once and store. This directly targets the `sent_method = ... cloudpickle.dumps(method)` line. Many control-plane callables (LoRA add/remove, cache reset, profiling toggles, reconfiguration hooks, warmup helpers) are the same bound method or module-level function repeatedly, so the pickled payload is invariant across calls and re-serialization is pure overhead — cloudpickle.dumps of even small closures typically costs tens to hundreds of microseconds and is invoked once per collective_rpc. Because the serialized bytes are only sent to Ray workers (which unpickle them independently), caching bytes is safe: no shared mutable state is exposed. Preserve correctness by (1) only caching when `method` is not a str, (2) using a `WeakValueDictionary`-based reverse map or a size cap (e.g., 128 entries with FIFO eviction) so garbage-collected callables don't leak ids that later get reused, and (3) leaving `del method` in place after `sent_method` is assigned. The collective contract (list of len(world_size) in rank order, exception propagation, non_block FutureWrapper wrapping) is untouched because only the derivation of `sent_method` changes. Expected effect: shaves serialization cost off every non-string collective_rpc — meaningful for multi-turn agentic workloads that trigger repeated LoRA swaps, KV cache resets, or per-turn reconfiguration RPCs during a session, improving control-plane latency around TTFT for turns that require adapter/cache manipulation.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals, so any concrete change is novel by construction. This proposal specifically targets the cloudpickle.dumps hot spot called out in evolve_rationale with a concrete memoization scheme (id-keyed bounded cache with eviction and safety for id reuse), rather than the alternative headroom items mentioned (list/ref allocation reduction or string-method fast paths), and it defines the correctness envelope needed to keep broadcast semantics identical.

---

### 2. Put large shared RPC arguments once before worker fanout
- **Agent:** codex

**Detailed description.**

In `RayDistributedExecutor.collective_rpc` at `vllm/v1/executor/ray_executor.py:470-495`, add a size-gated path that stores large immutable broadcast arguments in Ray's object store once before issuing the per-worker `execute_method.remote` calls. Concretely, after normalizing `kwargs`, estimate serialized size for each positional arg and kwarg value, and for values above a conservative threshold replace the value with `ray.put(value)` before the list comprehension. Ray resolves `ObjectRef` arguments before invoking the actor method, so `execute_method` still receives the same Python objects and the collective return contract is unchanged. This targets startup and reconfiguration calls such as `init_worker`, cache setup, and model or adapter control RPCs where the same config-like payload is currently embedded in every actor task submission. Keep the optimization disabled for small scalars and strings because `ray.put` overhead would dominate there, and add a focused Ray executor test with a large shared argument to verify each rank receives the original value and synchronous/non-blocking result ordering remains identical.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A's proposal only memoizes the serialized callable payload for non-string methods; this proposal addresses a different source of fanout overhead: repeated serialization and task-argument embedding of large shared `args`/`kwargs` values across all workers. It preserves the existing method serialization behavior and is orthogonal to callable cloudpickle caching.

---
