# WorkerProc.enqueue_output/handle_output/async_output_busy_loop

[← vllm/v1/executor](../vllm_v1_executor.md)

- **File:** [`vllm/v1/executor/multiproc_executor.py`](vllm/v1/executor/multiproc_executor.py) (lines 961–1006)
- **Symbol:** `WorkerProc.enqueue_output/handle_output/async_output_busy_loop`
- **Kind:** region
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_executor-0004`

## Description
Worker output materialization and response enqueue path, including the async-output thread that drains queued AsyncModelRunnerOutput objects.

## Current approach
enqueue_output converts AsyncModelRunnerOutput by calling get_output inline, maps exceptions to FAILURE tuples and other outputs to SUCCESS tuples, then enqueues on worker_response_mq. handle_output either calls enqueue_output immediately or puts the object on async_output_queue. async_output_busy_loop sets the worker device once and then handles one queued output at a time with blocking queue.Queue.get().

## Estimated impact explanation
This is the multiprocessing worker's main host/device synchronization point for model outputs. Reducing blocking get_output time or batching copies directly shortens token-produced to token-visible latency, moving median TPOT in async multi-turn workloads.

## Evolve rationale
The key code constructs are AsyncModelRunnerOutput.get_output at line 968 and the one-at-a-time async_output_queue drain at lines 1004-1006. get_output is the host/device synchronization and copy boundary for async model output. Headroom includes overlapping output materialization with the next worker RPC, batching accumulated async outputs, using pinned staging buffers, or making the response enqueue path consume already-staged CPU data. Correctness oracle: each produced output or exception is emitted exactly once, response status semantics stay unchanged, and per-rank FIFO order matches the synchronous path in existing async scheduling tests.

## Deep research proposals

### 1. Stage async model outputs into pinned host buffers on a non-default CUDA stream to overlap D2H copies with the next worker step
- **Finding:** `find-vllm_v1_executor-0003` — *CUDA C++ Best Practices Guide*
- **Source URL:** <https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-best-practices-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Rework the async output materialization path in vllm/v1/executor/multiproc_executor.py (WorkerProc.enqueue_output / handle_output / async_output_busy_loop, lines 961-1006) so that AsyncModelRunnerOutput.get_output no longer performs a blocking, on-demand D2H synchronization when the response is dequeued.

Concrete changes:
1. In async_output_busy_loop (lines 990-1006), before entering the drain loop, acquire (or lazily create) a dedicated non-default CUDA stream owned by the worker for output staging, in addition to the existing current_platform.set_device call. This mirrors the CUDA C++ Best Practices Guide guidance in section 9.1.2 that overlap requires 'different, non-default streams' plus 'pinned host memory'.
2. Extend the AsyncModelRunnerOutput contract (or add a thin wrapper used at line 966-968) so that instead of a single blocking get_output(), it exposes: (a) start_copy(stream) which enqueues D2H copies of logprobs, routed-expert metadata, sampled-token tensors, and any other device buffers into pre-allocated pinned host tensors on the staging stream, and (b) finalize() which waits on the copy event and returns the concrete ModelRunnerOutput built from the already-staged pinned buffers.
3. Change enqueue_output at lines 966-971 so it does not call get_output() inline. Instead, when the async_output_busy_loop drains an AsyncModelRunnerOutput at line 1005, it first calls start_copy(staging_stream) to launch pinned D2H copies, then only calls finalize() (with the event wait) when the message queue consumer would otherwise stall. In steady state the previous step's copies complete while the next scheduler/worker RPC is already running, so finalize()'s wait becomes a no-op instead of a full sync.
4. Preserve the current exception mapping (lines 969-974): any exception raised by start_copy or finalize is caught and turned into a (FAILURE, str(e)) tuple, and any non-exception output still becomes a (SUCCESS, output) tuple written to worker_response_mq exactly once, in FIFO order per rank.
5. Allocate the pinned host staging buffers once at worker init (sized to the maximum batch shape used by the scheduler) and reuse them across steps to avoid per-step host allocation, which the guide notes is expensive; fall back to the current synchronous path if pinned allocation fails or if the platform does not support non-default streams.

Correctness oracle from the candidate is preserved: each async output or exception is emitted exactly once, response status semantics are unchanged, and per-rank FIFO ordering matches the synchronous path exercised by existing async scheduling tests.

**Proposal rationale.**

The candidate identifies AsyncModelRunnerOutput.get_output at line 968 as the worker's host/device synchronization and copy boundary and explicitly names 'overlapping output materialization with the next worker RPC' and 'using pinned staging buffers' as headroom. The finding is the canonical NVIDIA guidance for exactly that pattern: async H<->D transfers require pinned host memory and a non-default stream, and it is the mechanism by which D2H copies for logprobs, routed experts, and token metadata (the exact payloads named in the candidate) can be overlapped with subsequent GPU work. In a multi-turn agentic workload, per-step median TPOT is bounded by how quickly each token becomes visible to the scheduler; today that latency includes a full get_output() sync inside the async drain thread, which currently handles one queued output at a time with a blocking queue.Queue.get(). Restructuring the drain to launch pinned-buffer D2H copies on a dedicated stream and only wait at the message-queue enqueue point converts a serial sync into overlapped work, directly targeting the estimated_impact rationale (shorten token-produced to token-visible latency). The finding therefore contributes a concrete, transferable mechanism, not just topical adjacency, and its constraints (pinned buffers, non-default stream) map cleanly onto the identified extension points without changing the response-status contract or FIFO ordering that the candidate's correctness oracle requires.

---

## Agent proposals

### 1. Batch-drain async_output_queue and launch get_output() with lookahead so per-rank sync cost amortizes across queued outputs
- **Agent:** claude

**Detailed description.**

Rework WorkerProc.async_output_busy_loop and enqueue_output in vllm/v1/executor/multiproc_executor.py (lines 961-1006) so the drain thread coalesces all currently-queued AsyncModelRunnerOutput objects per loop iteration instead of handling one at a time with a blocking queue.Queue.get(). Concrete changes:

1. Replace the `while True: output = self.async_output_queue.get(); self.enqueue_output(output)` loop (lines 1004-1006) with a batch-drain pattern: block on `async_output_queue.get()` for the first item, then non-blockingly pull every additional item currently in the queue via `async_output_queue.get_nowait()` in a tight loop (catching `queue.Empty`) to form an ordered batch `[o_0, ..., o_k]`.

2. Introduce a two-phase resolution over the batch that preserves per-rank FIFO order at the response_mq:
   (a) Launch phase — iterate the batch in order and, for every AsyncModelRunnerOutput, invoke a non-blocking `kick_off()` hook (added on AsyncModelRunnerOutput, defaulting to a no-op that returns self so today's subclasses keep working). Subclasses whose get_output() enqueues D2H copies or futures can use kick_off() to start that work eagerly, so the next batch item's copies are launched before the current one's sync completes. Non-AsyncModelRunnerOutput items and exceptions are left untouched in the batch list.
   (b) Finalize phase — iterate the batch in the same original order and, for each AsyncModelRunnerOutput, call the existing `get_output()` (which now sees copies already in-flight from step (a)), map exceptions to (FAILURE, str(e)) exactly as lines 969-974 do today, wrap non-exceptions as (SUCCESS, output), and call `worker_response_mq.enqueue(result)` immediately for that entry before moving on. Enqueuing per-item during finalization (rather than after collecting all results) keeps the message queue producer-side latency for early items unchanged while still amortizing device sync cost.

3. Keep enqueue_output's inline branch (lines 966-978) unchanged for the synchronous (non-async-scheduling) path used by handle_output at line 988, and keep handle_output's dispatch (lines 985-988) unchanged; only the drain loop and an optional kick_off() hook are new.

4. Cap the batch size at a small constant (e.g., 8 or a config knob defaulting to the scheduler's max concurrent async steps in flight) so that finalize latency for the first item in a burst is bounded — if the queue holds more than the cap, pending items simply wait for the next loop iteration, matching current single-item behavior in the worst case.

5. Correctness oracle from the candidate is preserved by construction: each output/exception is dequeued exactly once, mapped to exactly one SUCCESS/FAILURE tuple via the same code path as today, and enqueued to worker_response_mq in the exact order in which it entered async_output_queue (which is the order handle_output produced it in on the main worker thread).

Rationale for TPOT: in multi-turn agentic workloads the scheduler often produces bursts of AsyncModelRunnerOutput objects (e.g., trailing steps of a reasoning chain, or several ranks landing near-simultaneously into the same worker's queue). Today each such burst pays k serial get_output() syncs; with kick_off() lookahead the copies for outputs 1..k-1 execute concurrently with output 0's sync, so the drain wall-clock collapses toward one sync's worth for the whole batch.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_executor-0003) restructures AsyncModelRunnerOutput internals to use pinned host buffers on a dedicated non-default CUDA stream, splitting get_output into start_copy(stream)/finalize() so a single output's D2H sync overlaps with the *next scheduler RPC*. It does not change the drain loop's one-at-a-time scheduling: at lines 1004-1006 it still processes queued outputs sequentially with a blocking queue.Queue.get() and only wins when there is subsequent worker-side GPU work to overlap against. This proposal is orthogonal and complementary: it changes the drain-loop's *batch-dequeue strategy* and adds a `kick_off()` lookahead hook so that multiple already-queued outputs' materialization is pipelined against each other within the drain thread itself, independent of whether get_output uses a dedicated stream or the default one. The two proposals compose (batching amplifies the stream-overlap win by allowing multiple in-flight D2H copies), but neither implies the other — batching provides a distinct mechanism (queue-level pipelining) that helps even when there is no next RPC to hide the sync behind, e.g., at the tail of an agentic turn where the scheduler is waiting on exactly these outputs before deciding the next action.

---

### 2. Pipeline async output materialization separately from response-mq serialization
- **Agent:** codex

**Detailed description.**

In vllm/v1/executor/multiproc_executor.py, split the async-output path around WorkerProc.enqueue_output/async_output_busy_loop into two ordered stages. Keep async_output_busy_loop responsible for resolving each AsyncModelRunnerOutput by calling get_output() and mapping exceptions to the same (FAILURE, str(e)) tuple or successful outputs to (SUCCESS, output), but enqueue that already-materialized tuple into a small bounded local response_send_queue instead of calling worker_response_mq.enqueue() inline. Add a dedicated response-sender thread for the async path that drains response_send_queue in FIFO order and performs worker_response_mq.enqueue(result). The queue should be bounded to preserve backpressure, and the sender should use the same shutdown/error handling conventions as the existing worker threads. This lets the materialization thread begin get_output() for the next async result while the previous result is being serialized or blocked on the multiprocessing response queue. Correctness remains local: every input still produces exactly one SUCCESS/FAILURE tuple, tuple semantics are unchanged, and FIFO is preserved because both local queues are single-producer/single-consumer ordered stages.

**Novelty rationale.**

The deep_research proposal focuses on changing get_output() itself by staging D2H copies with pinned host buffers and a non-default CUDA stream. Agent A focuses on batch-draining async_output_queue and launching output materialization with lookahead. This proposal targets a separate serial section after materialization: worker_response_mq.enqueue(), including IPC serialization and possible queue backpressure. It does not require pinned buffers, CUDA streams, batch dequeueing, or new kick_off/start_copy hooks; it pipelines the existing get_output() boundary against response transport work that currently sits in the same one-at-a-time drain loop.

---
