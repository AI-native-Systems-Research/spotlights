# PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/video.py`](vllm/multimodal/video.py) (lines 842–875)
- **Symbol:** `PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0026`

## Description
Stages raw video bytes for PyNvVideoCodec, reads metadata, computes sampled frame indices, gates raw decoded frame memory, decodes to pinned host memory, and returns frames plus valid indices.

## Current approach
Creates a temporary .mp4 file with tempfile.mkstemp, writes the whole input bytes to it, uses that path for metadata and decode, computes raw_frame_bytes from sampled frames, optionally acquires the global GPU IPC pool, and deletes the temp file in finally.

## Estimated impact explanation
For NVDEC video, byte staging happens before any frame reaches the processor. Removing whole-file temp I/O reduces video TTFT, especially for long clips and multi-turn workloads that repeatedly submit video bytes.

## Evolve rationale
The concrete hot constructs are tempfile.mkstemp, temp_file.write(data), _read_source_metadata(temp_path, nvc), and _decode_to_pinned_host(temp_path, frame_idx, nvc). Avoiding per-request filesystem staging, using a reusable memory-backed path, or coalescing metadata/decode decoder use can reduce ingress latency while preserving the CPU NHWC ndarray contract. Correctness oracle: tests/multimodal/test_video.py and tests/multimodal/test_gpu_ipc_memory.py; metadata, sampled indices, valid indices, shape, dtype, and pool accounting must remain correct.

## Deep research proposals

### 1. Replace per-request temp-file staging with PyNvVideoCodec memory demuxing
- **Finding:** `find-vllm_multimodal-0004` — *PyNvVideoCodec API Programming Guide*
- **Source URL:** <https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec` (vllm/multimodal/video.py:842-875), stop materializing every incoming video into a `.mp4` on disk via `tempfile.mkstemp` + `temp_file.write(data)` and stop passing a filesystem path into `_read_source_metadata` and `_decode_to_pinned_host`. Instead, use the memory-demuxing entry points described in the PyNvVideoCodec API Programming Guide: construct an in-memory demuxer/source directly from the `data: bytes` buffer (e.g., `nvc.CreateDemuxer` with a memory source / BufferInterface, or the equivalent bytes-source API documented in the guide) and pass that demuxer object to the decoder acquired from `decoder_slot.get_decoder(...)` instead of `file_path`. Update `_read_source_metadata(source, nvc)` and `_decode_to_pinned_host(source, frame_idx, nvc)` to accept either a path or a memory demuxer, and rely on the guide's Decoder Reuse guidance so the cached decoder slot can be reconfigured for the new memory source rather than rebuilt. The `try/finally` that unlinks the temp path is removed; the raw-frame-bytes computation, GPU IPC pool acquisition, sampled-index computation, pinned-host copy, and returned `(frames, source, frame_idx, valid_frame_indices)` contract are unchanged, so `tests/multimodal/test_video.py` and `tests/multimodal/test_gpu_ipc_memory.py` (shape, dtype, valid indices, pool accounting) continue to be the correctness oracle.

**Proposal rationale.**

The finding points at the vendor's own guide, which promotes memory demuxing, decoder reuse, and batch frame retrieval as first-class policies — exactly the constructs the current implementation avoids by staging to `/tmp` and re-opening the container twice (metadata + decode) through a filesystem path. For a multi-turn agentic workload, every video turn currently pays fixed disk-write + open + parse cost before any frame reaches the processor, which is on the TTFT critical path. Memory demuxing removes the whole-file write and the temp-path lifecycle, decoder reuse (already partially in place via `_borrow_decoder_slot`) can be leveraged more aggressively across the metadata+decode pair when both consume the same in-memory source, and neither change alters the NHWC pinned-host CPU ndarray contract that downstream tests pin down. The gap it closes is specifically the per-request filesystem ingress the candidate flags as the hot construct.

---

## Agent proposals

### 1. Fuse metadata + decode into a single decoder-slot borrow, with content-hash metadata cache for multi-turn reuse
- **Agent:** claude

**Detailed description.**

In `PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec` (vllm/multimodal/video.py:842-875), collapse the two independent decoder-slot borrows (currently one inside `_read_source_metadata` at line 857 and a second inside `_decode_to_pinned_host` at line 866/869) into a single fused inner path. Concretely: acquire `_borrow_decoder_slot()` once, enter `_torch_stream_context(decoder_slot.stream)` once, call `decoder_slot.get_decoder(source, nvc, device_index=cls._DEVICE_INDEX)` once, and use that same decoder object to (a) read `get_stream_metadata()` + `len(decoder)` and build the `PyNvVideoCodecSourceMetadata`, (b) run `cls.compute_frames_index_to_sample(source=..., target=target, **kwargs)` on the CPU while still holding the slot briefly (or release+re-borrow deliberately if slots are scarce — measure), (c) acquire the GPU IPC pool via `get_mm_gpu_ipc_pool()` under `pool.acquire(raw_frame_bytes)` if applicable, and (d) call `decoder.get_batch_frames_by_index(frame_idx)` followed by the existing pinned-host copy and stream sync. Keep `_read_source_metadata` and `_decode_to_pinned_host` as-is for other callers, but route the hot path through a new fused helper `_read_metadata_and_decode(source, target, nvc, **kwargs)` that returns `(frames, gpu_source, frame_idx)`. Additionally, since the caller context is a multi-turn agentic workload where the same video bytes are frequently re-submitted turn after turn, add a small process-wide LRU cache (e.g. `functools.lru_cache`-backed or a bounded dict guarded by `_decoder_slot_cond`) keyed on a cheap content fingerprint `(len(data), blake2b(data[:8192] + data[-8192:]).digest())` mapping to `PyNvVideoCodecSourceMetadata`. On a fingerprint hit, skip the metadata portion entirely and go straight to the decode inside a single slot borrow. All existing invariants — NHWC pinned-host CPU ndarray contract, `raw_frame_bytes` computation, pool acquisition semantics, and returned `(frames, source, frame_idx, valid_frame_indices)` tuple — are preserved, so `tests/multimodal/test_video.py` and `tests/multimodal/test_gpu_ipc_memory.py` remain the correctness oracle.

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_multimodal-0004) targets the *source type* — replacing the on-disk `.mp4` temp file with an in-memory demuxer via `nvc.CreateDemuxer` — and only mentions decoder reuse in passing as guidance from the vendor docs. It does not restructure the two-borrow pattern in `decode_frames_pynvvideocodec` and does not propose caching metadata across repeat submissions. This proposal is orthogonal: it changes the decoder-slot *lifecycle* (two `_borrow_decoder_slot()` acquires and two `get_decoder` cache lookups per request → one of each), and adds a content-fingerprint LRU that eliminates the metadata-read pass entirely for repeat videos in multi-turn agentic sessions. The two ideas compose — even after migrating to memory demuxing, a request would still pay two slot acquires and two decoder builds without the fused borrow, and the metadata cache still saves work on the second and later turns.

---

### 2. Stream PyNvVideoCodec frames into pinned host memory without GPU batch stacking
- **Agent:** codex

**Detailed description.**

In `PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host`, which is the decode helper called by `decode_frames_pynvvideocodec` for lines 842-875, replace the current `torch_frames = [...]` + `torch.stack(torch_frames)` + batch `_pynvvc_frames_to_nhwc(...)` path with a streaming copy into the final pinned CPU tensor. After `decoder.get_batch_frames_by_index(frame_idx)`, convert the first DLPack frame to a tensor, infer whether per-frame layout is HWC or CHW, allocate the pinned host batch with final NHWC shape, then iterate over decoded frames and copy each normalized NHWC frame into `host_frames[i]` with `non_blocking=True`, synchronizing once at the end as today. Keep the existing warning when fewer frames are returned, preserve the empty-result behavior, and keep the returned NumPy array shape/dtype contract. This removes an extra full-batch GPU allocation and device-to-device copy caused by `torch.stack`, and prevents peak GPU memory from briefly including both the decoder-owned frame outputs and a second contiguous stacked batch before the CPU copy. Add/adjust tests around mocked PyNvVideoCodec HWC and CHW frames to verify output shape, dtype, valid frame truncation, and that the empty path remains unchanged.

**Novelty rationale.**

The deep research proposal changes the input source from a temporary filesystem path to PyNvVideoCodec memory demuxing, while Agent A changes decoder-slot lifetime and caches metadata across repeated video bytes. This proposal targets a different hot construct inside the decode helper after frames have already been decoded: avoiding `torch.stack` and the batch contiguous GPU copy before pinned-host transfer. It composes with either source-type or decoder-lifecycle change and is still useful if temp-file staging and duplicate metadata reads are eliminated.

---
