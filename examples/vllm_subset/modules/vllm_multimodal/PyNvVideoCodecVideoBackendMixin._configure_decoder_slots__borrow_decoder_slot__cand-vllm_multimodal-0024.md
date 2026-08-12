# PyNvVideoCodecVideoBackendMixin._configure_decoder_slots/_borrow_decoder_slot

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/video.py`](vllm/multimodal/video.py) (lines 689–745)
- **Symbol:** `PyNvVideoCodecVideoBackendMixin._configure_decoder_slots/_borrow_decoder_slot`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0024`

## Description
Configures and borrows retained PyNvVideoCodec decoder slots guarded by a process-wide condition variable.

## Current approach
The first _configure_decoder_slots call fixes _max_decoder_slots. _borrow_decoder_slot pops an idle slot from a LIFO list, creates a new slot while _active_decoder_slots is below the max, or waits on _decoder_slot_cond until a release appends a slot and notify wakes one waiter.

## Estimated impact explanation
Under concurrent NVDEC video requests, decoder-slot admission determines whether media preprocessing queues or overlaps. Better scheduling reduces TTFT tail and can improve median TPOT by avoiding frontend GPU contention spikes.

## Evolve rationale
The concrete synchronization and scheduling constructs are _max_decoder_slots first-caller state, the LIFO _decoder_slots.pop(), the while True wait loop, and _decoder_slot_cond.notify(). FIFO or size-aware wait queues, warm slot preallocation, or per-device slot pools can reduce decoder queueing and avoid unfair reuse while preserving the max-slot invariant. Correctness oracle: tests/multimodal/test_video.py and tests/multimodal/test_gpu_ipc_memory.py; concurrent decodes must respect configured slot limits and return the same frames.

## Deep research proposals

### 1. Adopt PyNvVideoCodec ThreadedDecoder and decoder-reuse primitives for slot admission
- **Finding:** `find-vllm_multimodal-0004` — *PyNvVideoCodec API Programming Guide*
- **Source URL:** <https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor PyNvVideoCodecVideoBackendMixin._configure_decoder_slots and _borrow_decoder_slot (vllm/multimodal/video.py:689-745) to use PyNvVideoCodec's native ThreadedDecoder and Decoder Reuse APIs as the underlying admission and scheduling mechanism, rather than the hand-rolled condition-variable + LIFO pool. Concretely: (1) at first-call configuration time, preallocate warm decoder instances up to _max_decoder_slots using PyNvVideoCodec's decoder reuse/reconfiguration API so that _borrow_decoder_slot never pays first-decode setup cost on the hot path; (2) replace the LIFO cls._decoder_slots.pop() with a FIFO or ThreadedDecoder-backed submission queue that dispatches decode work to worker threads owning retained decoders, using batch frame retrieval where the caller requests multiple frames; (3) keep the process-wide max-slot invariant by capping the number of ThreadedDecoder workers at _max_decoder_slots and gating admission on a fair (FIFO) wait queue instead of _decoder_slot_cond.notify()'s arbitrary wake ordering. Preserve the existing public contract of _borrow_decoder_slot as a context manager so callers in the file are unchanged, and keep the correctness oracle (tests/multimodal/test_video.py and tests/multimodal/test_gpu_ipc_memory.py) passing by verifying identical frame outputs and slot-count invariants under concurrent decode.

**Proposal rationale.**

The finding points at two PyNvVideoCodec API sections that directly correspond to the two gaps called out in the candidate's evolve_rationale: 'High-Throughput Pipelines Using ThreadedDecoder' targets the concurrent scheduling / unfair LIFO reuse concern, and 'Decoder Reuse' targets the warm-slot preallocation and fixed setup cost concern. The candidate currently reinvents these behaviors with a manual condition-variable pool and LIFO stack; adopting the vendor primitives as first-class policies is a concrete transferable idea that plausibly reduces decoder-slot queueing tail and frontend GPU contention in the multi-turn agentic workload, improving median TTFT while preserving the max-slot invariant the tests verify.

---

## Agent proposals

### 1. Priority- and deadline-aware decoder-slot admission with cancellation propagation
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/video.py:689-745, replace the arbitrary-order condition-variable admission in PyNvVideoCodecVideoBackendMixin with a priority-and-deadline aware admission structure while preserving the max-slot invariant and the _borrow_decoder_slot context-manager contract.

Concretely: (1) extend the classmethod signature to _borrow_decoder_slot(cls, *, priority: int = 0, deadline_s: float | None = None, cancel_event: threading.Event | None = None) and thread these values through the two call sites in this file (_read_source_metadata and the frame-decode helper that borrows a slot) so the request-serving path can pass in the request's remaining TTFT budget and its cancellation token. (2) Replace cls._decoder_slots: list plus cls._decoder_slot_cond with a small custom scheduler object holding an idle-slot deque and a heapq of waiters keyed by (priority, enqueue_seq); on release, wake exactly the highest-priority waiter (FIFO within priority) instead of _decoder_slot_cond.notify(). (3) In the wait loop, use cond.wait(timeout=remaining_deadline) and, on wakeup, re-check cancel_event.is_set() and time.monotonic() >= deadline; if either fires while still queued, dequeue the waiter and raise a typed DecoderAdmissionTimeout / DecoderAdmissionCanceled so the caller can skip work and return a synthetic empty/degraded frame set rather than blocking the request. (4) Preserve _max_decoder_slots first-caller semantics and the 'append on release + notify' invariant; make the release path drain the highest-priority waiter's future and pass the slot directly to it, avoiding the current thundering-herd wake pattern where notify() may hand the slot to a lower-priority waiter that then re-enqueues.

Correctness oracle stays tests/multimodal/test_video.py and tests/multimodal/test_gpu_ipc_memory.py; add a targeted test that verifies (a) with all slots busy, a high-priority waiter enqueued after a low-priority waiter obtains the next released slot first, and (b) a waiter whose cancel_event is set while queued does not consume a slot and never invokes _create_decoder_slot.

**Novelty rationale.**

The existing deep_research_proposal (find-vllm_multimodal-0004) proposes swapping the hand-rolled pool for PyNvVideoCodec's ThreadedDecoder + Decoder Reuse, warm preallocation, and a fair FIFO wait queue. It is deliberately fairness-only and workload-agnostic: every waiter is treated equally and there is no notion of per-request deadline or cancellation. This proposal is orthogonal and complementary: it changes the admission policy itself to be priority- and deadline-aware and adds cooperative cancellation propagation so a critical-path multi-turn request can jump the queue and an abandoned request can free its slot without doing decode work. Neither priority scheduling nor deadline/cancellation-aware admission is mentioned in the DR proposal, and both directly target median TTFT tail behavior under the stated multi-turn agentic workload rather than average throughput.

---

### 2. Shard decoder-slot pools by CUDA device
- **Agent:** codex

**Detailed description.**

In `vllm/multimodal/video.py:689-745`, change `PyNvVideoCodecVideoBackendMixin` so `_configure_decoder_slots` and `_borrow_decoder_slot` manage decoder admission per target CUDA device instead of through one process-wide `_max_decoder_slots`, `_active_decoder_slots`, `_decoder_slots`, and `_decoder_slot_cond`. Concretely, key the retained-slot state by the device actually used to create the PyNvVideoCodec decoder slot, with a small per-device pool object containing its own max, active count, idle-slot deque/list, and condition variable. `_configure_decoder_slots` should initialize or validate the pool for that device, while `_borrow_decoder_slot` should only block when that device's pool is saturated, not when unrelated devices are busy. Release returns the slot to the same device pool it came from. Add a concurrency test in `tests/multimodal/test_video.py` that monkeypatches slot creation to record device keys and verifies saturation on one device does not prevent borrowing on another device, while still enforcing the configured per-device slot limit.

**Novelty rationale.**

The deep-research proposal focuses on replacing the hand-rolled pool with PyNvVideoCodec ThreadedDecoder/Decoder Reuse, warm preallocation, and FIFO/threaded scheduling under the existing global admission shape. Agent A focuses on priority, deadline, and cancellation semantics for queued waiters. This proposal changes the resource partitioning boundary itself: it prevents a busy NVDEC workload on one CUDA device from consuming the entire process-wide decoder-slot budget and blocking independent work on another device. That per-device isolation is mentioned only as a possible evolution hint in the candidate rationale, but it is not covered by either emitted proposal's concrete implementation plan.

---
