# reserve_mm_ipc_gpu_memory

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/gpu_ipc_memory.py`](vllm/multimodal/gpu_ipc_memory.py) (lines 155–257)
- **Symbol:** `reserve_mm_ipc_gpu_memory`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0025`

## Description
Subtracts frontend multimodal GPU decode reservations from available KV-cache memory based on raw-frame budget, API process count, video backend, and PyNvVideoCodec decoder constants.

## Current approach
Computes raw_frame_reserved_bytes from mm_ipc_gpu_memory_gb, detects PyNvVideoCodec use from media_io_kwargs or VLLM_VIDEO_LOADER_BACKEND, validates hw_decoders, adds num_api_servers times a fixed per-server decoder reservation when use_gpu_video_backend is true, and raises if the remaining KV cache is non-positive.

## Estimated impact explanation
This directly controls KV-cache capacity in GPU video deployments. Less pessimistic but safe reservation can improve TPOT and batching capacity, while adequate reservation prevents TTFT-damaging OOM retries.

## Evolve rationale
The concrete policy constructs are raw_frame_reserved_bytes = int(mm_config.mm_ipc_gpu_memory_gb * GiB_bytes), uses_pynvvideocodec backend detection, hw_decoders defaulting, and reserved_bytes = raw_frame_reserved_bytes + decoder_reserved_bytes. Workload-aware or backend-specific reservation can trade decode concurrency against KV-cache headroom without violating the safety invariant. Correctness oracle: tests/multimodal/test_gpu_ipc_memory.py and config tests; reservations must reject impossible configurations and keep remaining KV-cache memory positive.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Adapt mm_ipc_gpu_memory reservation to observed peak decode residency in agentic multi-turn workloads
- **Agent:** claude

**Detailed description.**

Change reserve_mm_ipc_gpu_memory (vllm/multimodal/gpu_ipc_memory.py:155-257) from a static ceiling of raw_frame_reserved_bytes = int(mm_config.mm_ipc_gpu_memory_gb * GiB_bytes) plus a fixed per-server decoder reservation, into a two-tier adaptive reservation with a startup safe floor and a steady-state right-sizing pass driven by the pool's observed high-water mark. Concretely: (1) On first call, keep the current worst-case value as an initial safety ceiling so the KV cache is sized conservatively at engine bring-up (preserving the existing 'remaining <= 0' invariant and the tests/multimodal/test_gpu_ipc_memory.py oracle unchanged). (2) Extend MultiModalGPUMemoryPool (vllm/multimodal/gpu_ipc_memory.py:60-110) to track a peak_bytes_in_use counter alongside _available/_outstanding, updated inside _acquire/_release under the existing lock (no new synchronization). (3) Add a lightweight periodic reconciler on the frontend (invoked every N requests or T seconds from the same pool) that, once the workload has run past a warmup window, computes an adaptive_target = max(startup_floor_fraction * raw_frame_reserved_bytes, peak_bytes_in_use * headroom_factor) capped by the original raw_frame_reserved_bytes; the delta is *returned to the KV-cache manager only via a documented, opt-in callback* (behind mm_config.mm_ipc_gpu_memory_adaptive, default False) so existing deployments are byte-for-byte unchanged. (4) Split the current single log_once into an initial-reservation log plus a one-time steady-state log reporting observed peak vs. reserved, so operators can tune mm_ipc_gpu_memory_gb without trial-and-error. For agentic multi-turn workloads where a fixed fraction of turns actually carry a video/image payload, this recovers the over-provisioned raw-frame slack as KV-cache blocks, directly raising achievable batch size and thus lowering median TPOT, while the startup floor keeps first-turn TTFT safe from OOM retries.

**Novelty rationale.**

The candidate lists no deep_research_proposals, so nothing overlaps by construction. Beyond that, the proposal is specifically about turning the current one-shot worst-case reservation into a measurement-driven adaptive reservation using peak_bytes_in_use inside MultiModalGPUMemoryPool, with an explicit opt-in flag and a preserved safety floor — a mechanism that is not implied by the candidate's description (which only enumerates the static policy constructs raw_frame_reserved_bytes, uses_pynvvideocodec detection, hw_decoders defaulting, and reserved_bytes summation).

---

### 2. Make PyNvVideoCodec reservation conditional on actual local API processes
- **Agent:** codex

**Detailed description.**

In `reserve_mm_ipc_gpu_memory` (`vllm/multimodal/gpu_ipc_memory.py:155-257`), replace the blanket `num_api_servers * decoder_reserved_bytes_per_server` charge with a helper that derives the number of GPU-video-decoding frontend processes that will actually exist on the current node. Today the function multiplies by `parallel_config.data_parallel_size`, but in common deployments only a subset of DP/API ranks are colocated with this engine process, and some API processes may not own a PyNvVideoCodec decoder at all. Add a small, testable resolver that prefers an explicit config/env override such as `VLLM_MM_GPU_VIDEO_DECODER_PROCESSES` when set, otherwise falls back to the current `data_parallel_size` behavior for compatibility. Validate that the resolved count is non-negative and no greater than the configured API process count, then compute decoder reservation from that resolved local count. This keeps the existing raw-frame budget and positive-KV-cache invariant intact, but avoids reserving decoder memory for remote or non-decoding API servers in multi-turn agentic serving topologies where TTFT/TPOT suffer from unnecessarily reduced KV-cache blocks.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. This does not overlap with Claude's adaptive high-water-mark proposal: it is not measurement-driven, does not change `MultiModalGPUMemoryPool`, and does not return memory to the KV-cache manager after startup. It targets a different source of over-reservation: the static multiplier for PyNvVideoCodec decoder memory, making it topology-aware while preserving the existing one-shot reservation model.

---
