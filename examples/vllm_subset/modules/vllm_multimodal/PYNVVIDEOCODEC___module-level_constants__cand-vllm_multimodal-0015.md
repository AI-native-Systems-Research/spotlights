# PYNVVIDEOCODEC_* module-level constants

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/video.py`](vllm/multimodal/video.py) (lines 217–224)
- **Symbol:** `PYNVVIDEOCODEC_* module-level constants`
- **Kind:** config_block
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0015`

## Description
Defines PyNvVideoCodec resource-sizing defaults: backend name, per-decoder GPU memory reservation, decoder cache size, default HW decoder slots, and CUDA-context reservation.

## Current approach
Uses hard-coded constants: 128 MiB per decoder, decoder cache size 2, default HW decoders 2, and a 1.8 GiB CUDA-context reservation. These feed decoder-slot construction and GPU memory reservation logic.

## Estimated impact explanation
For NVDEC deployments, decoder slot count and reserved memory affect whether concurrent video requests queue or overlap. Better tuning can reduce TTFT tail and improve TPOT under video bursts.

## Evolve rationale
The optimization unit is the policy-defining constants PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES, PYNVVIDEOCODEC_DECODER_CACHE_SIZE, PYNVVIDEOCODEC_DEFAULT_HW_DECODERS, and PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES. Sweeping them trades KV-cache headroom against concurrent decode throughput. Correctness oracle: tests/multimodal/test_gpu_ipc_memory.py and tests/multimodal/test_video.py; reservations must leave positive KV cache and decoded frames must remain correct.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Replace static PyNvVideoCodec reservation with floor/ceiling + idle-teardown decoder pool
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/video.py:217-224, change the PyNvVideoCodec sizing constants from a single static value into a dynamic reservation policy that returns idle GPU memory to the KV-cache pool:

1. Split PYNVVIDEOCODEC_DEFAULT_HW_DECODERS into two constants: PYNVVIDEOCODEC_MIN_HW_DECODERS (default 0) and PYNVVIDEOCODEC_MAX_HW_DECODERS (current default 2). Reserve only `MIN * PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES + PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES` up front in the memory-reservation path that consumes these constants; the remaining `(MAX - MIN) * 128 MiB` stays available to the KV allocator until actually needed.

2. Extend _configure_decoder_slots / _borrow_decoder_slot (video.py:688-745) to grow the pool lazily. Rather than the current behavior where cls._max_decoder_slots is fixed at config time and the pool never shrinks, treat `_max_decoder_slots` as the ceiling and `_min_decoder_slots` as the floor. On borrow, create a new slot only if _active_decoder_slots < ceiling AND (a) queue is contended or (b) below floor. Track a per-slot last_release_time.

3. Add PYNVVIDEOCODEC_SLOT_IDLE_SECS (default e.g. 30s). A lightweight watchdog (or an opportunistic check on borrow) tears down decoder slots that have been idle for that long and are above the floor: call SimpleDecoder destruction, drop the CUDA stream, decrement _active_decoder_slots, and — critically — return the 128 MiB surface-pool reservation to the caching allocator so the KV manager can reuse it.

4. Add PYNVVIDEOCODEC_DECODER_CACHE_SIZE_IDLE (default 1) used when only one source is being decoded; keep the current value 2 when contention is observed. This shrinks the switching-buffer footprint when it isn't paying rent.

Keep the existing validate_pynvvideocodec_hw_decoders API by treating the argument as the ceiling; add a companion validate_pynvvideocodec_min_hw_decoders. Preserve backward compatibility by defaulting MIN=0 (no eager reservation) and MAX unchanged.

Verification: existing tests/multimodal/test_gpu_ipc_memory.py must still show positive KV cache post-reservation; tests/multimodal/test_video.py must still decode correctly across bursty and single-source patterns. Add a targeted test that (a) checks steady-state RSS/GPU-reservation after decoding one video and going idle drops back to the floor, and (b) a burst of concurrent video requests still scales up to the ceiling without regression.

Expected impact under the caller's workload (multi-turn agentic, minimizing median TTFT/TPOT): most turns are text-only, so the fleet-wide steady state releases up to `MAX * 128 MiB` (256 MiB with defaults) back to KV cache, reducing prefix eviction and increasing per-step batch runway — directly lowering median TTFT and TPOT during the majority of turns that do not include video, while preserving today's throughput on bursts.

**Novelty rationale.**

No deep_research_proposals exist for this candidate, so overlap is by definition zero. Beyond that, the proposal is structurally distinct from a constant-value sweep: it reshapes the reservation curve over time (lazy grow + idle teardown + adaptive cache size) rather than picking better fixed values for PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES / _CACHE_SIZE / _DEFAULT_HW_DECODERS / _CUDA_CONTEXT_BYTES. This is the specific lever that matters for a multi-turn agentic workload where video is bursty rather than steady, which a static-constant tuning cannot capture.

---

### 2. Make PyNvVideoCodec memory reservation tunable per deployment
- **Agent:** codex

**Detailed description.**

Keep the current constants in vllm/multimodal/video.py:217-224 as safe defaults, but add validated media_io_kwargs overrides for the two memory-sizing assumptions: per-decoder surface reservation and per-process CUDA-context reservation. Concretely, introduce helper validation near validate_pynvvideocodec_hw_decoders, e.g. accepting `pynvvideocodec_decoder_gpu_memory_mb` and `pynvvideocodec_cuda_context_memory_mb` as positive integer MiB values, and have vllm/multimodal/gpu_ipc_memory.py use those effective values when computing `per_server_decoder_bytes` instead of always using `PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES` and `PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES`. Preserve backward compatibility by defaulting to 128 MiB and 1.8 GiB. Add tests beside tests/multimodal/test_gpu_ipc_memory.py that verify the override changes the reserved KV-cache deduction, scales with `hw_decoders` and `api_process_count`, and rejects invalid values. This lets operators who have measured a smaller PyNvVideoCodec footprint on their GPU, codec mix, and PyNvVideoCodec version reclaim otherwise stranded KV-cache capacity, improving median TTFT/TPOT for mostly text multi-turn agent workloads without changing decoder-pool behavior.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes a dynamic lazy-grow and idle-teardown decoder pool with floor/ceiling slot counts and adaptive cache size. This proposal is different: it keeps the pool lifecycle and slot count semantics unchanged and instead makes the hard-coded memory constants configurable so deployments can right-size the reservation based on measured hardware/library behavior.

---
