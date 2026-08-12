# PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/video.py`](vllm/multimodal/video.py) (lines 795–839)
- **Symbol:** `PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0017`

## Description
Decodes selected frames with PyNvVideoCodec, wraps DLPack outputs as tensors, converts to NHWC, copies to pinned CPU memory, synchronizes, and returns a numpy array.

## Current approach
Builds torch_frames with a Python list comprehension, stacks them into device_frames, converts layout, allocates a fresh pinned CPU tensor with torch.empty, copies non-blocking, then calls stream.synchronize before exposing host_frames.numpy().

## Estimated impact explanation
On NVDEC paths, decode plus D2H transfer is the dominant video ingress cost. Reducing pinned allocation and synchronization lowers bytes-to-frames latency and TTFT.

## Evolve rationale
The concrete constructs are torch.stack(torch_frames), torch.empty(..., pin_memory=True), host_frames.copy_(..., non_blocking=True), and stream.synchronize(). A pinned host buffer pool and stream-pipelined D2H copies can reduce allocation and overlap transfer with subsequent decode while preserving the CPU NHWC ndarray contract. Correctness oracle: tests/multimodal/test_video.py; dtype, shape, valid frame count, and frame values must remain correct.

## Deep research proposals

### 1. Adopt PyNvVideoCodec ThreadedDecoder and memory demuxing in _decode_to_pinned_host
- **Finding:** `find-vllm_multimodal-0004` — *PyNvVideoCodec API Programming Guide*
- **Source URL:** <https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Rework vllm/multimodal/video.py:795-839 (PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host) to use PyNvVideoCodec's ThreadedDecoder plus memory demuxing as first-class policies, per the API Programming Guide's 'High-Throughput Pipelines Using ThreadedDecoder' and 'Decoder Reuse' sections. Concrete changes, scoped to this method and the surrounding decoder-slot machinery it already borrows via cls._borrow_decoder_slot(): (1) Route source bytes through PyNvVideoCodec's in-memory demuxer instead of a file_path argument, eliminating the file-staging step ahead of decode. (2) Have PyNvVideoCodecDecoderSlot vend a ThreadedDecoder (still cached/reconfigured across requests, matching the existing decoder-reuse pattern in get_decoder / reconfigure_decoder) so that get_batch_frames_by_index(frame_idx) produces DLPack frames while the previous batch's stack + D2H copy is still in flight on cls._torch_stream_context(stream). (3) Pipeline the D2H stage: overlap torch.stack(torch_frames) -> _pynvvc_frames_to_nhwc -> host_frames.copy_(device_frames, non_blocking=True) with the next ThreadedDecoder batch, keeping a single stream.synchronize() only at the boundary where host_frames.numpy() is exposed to the caller. Preserve the current NHWC ndarray return contract, dtype, shape, and valid frame count validated by tests/multimodal/test_video.py; keep the empty-frame_idx fast path and the 'unexpected shape' guard on device_frames.ndim.

**Proposal rationale.**

The finding names four PyNvVideoCodec techniques; the candidate already applies two (decoder caching/reconfiguration in PyNvVideoCodecDecoderSlot.get_decoder, and batch retrieval via get_batch_frames_by_index). ThreadedDecoder and memory demuxing are exactly the untapped, concrete constructs from the guide that map onto this method's remaining bottlenecks: serial decode/copy with a single stream.synchronize() before host_frames.numpy(), and file-path staging upstream of decode. On multi-turn agentic workloads where NVDEC decode + D2H is the dominant video ingress cost, pipelining decode with D2H transfer via ThreadedDecoder lowers bytes-to-frames latency (and TTFT), while memory demuxing removes a fixed setup cost per request. The correctness oracle (tests/multimodal/test_video.py) is unchanged: frame count, dtype, shape, and values remain the observable contract.

---

### 2. Pool pinned host buffers and stream-pipeline D2H in _decode_to_pinned_host
- **Finding:** `find-vllm_multimodal-0005` — *A guide on good usage of non_blocking and pin_memory() in PyTorch*
- **Source URL:** <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/multimodal/video.py:795-839 (PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host), replace the per-call torch.empty(..., pin_memory=True) allocation and the immediate stream.synchronize() with a size-keyed pool of reusable pinned host tensors and a stream-aware D2H handoff. Concretely: (1) keep the existing NHWC device_frames construction on the decoder's CUDA stream; (2) acquire a pinned host tensor from a class-level cache keyed by (shape, dtype), falling back to torch.empty(..., pin_memory=True) on miss and caching on release; (3) issue host_frames.copy_(device_frames, non_blocking=True) on the same non-default stream used for decode so the copy overlaps with the next decode; (4) record a CUDA event on that stream instead of calling stream.synchronize() unconditionally, and only synchronize (event.synchronize() or stream.synchronize()) immediately before .numpy() is materialized for the caller, since numpy() requires a host-consistent view; (5) return the numpy view and defer buffer release until the caller drops the array (either via a small wrapper that returns the pooled tensor on finalization, or by copying into a fresh numpy array from the pinned buffer and returning the pinned tensor to the pool inside this function). Bound the pool by count and/or bytes to avoid unbounded growth, and continue to del decoded_frames/torch_frames/device_frames as today. Preserve the existing shape/dtype/frame-count invariants so tests/multimodal/test_video.py still passes.

**Proposal rationale.**

The finding is the canonical PyTorch guidance on pinned memory and non_blocking transfers, and it directly targets the two costs visible in this candidate: per-request pinned allocation (torch.empty(..., pin_memory=True) on every decode) and an eager stream.synchronize() that blocks the decoder stream before returning. The guide's prescribed remedies — reuse pinned buffers across their real lifecycle, run copies on a non-default stream, and synchronize only at the point of CPU consumption — map 1:1 onto the four constructs called out in evolve_rationale (torch.stack, torch.empty pinned, copy_ non_blocking, stream.synchronize). Because NVDEC decode + D2H is the dominant video-ingress cost on this path, shrinking allocation and permitting decode/copy overlap plausibly lowers bytes-to-frames latency and thus median TTFT under the multi-turn agentic workload named in the caller context, without changing the CPU NHWC ndarray contract that tests/multimodal/test_video.py validates.

---

## Agent proposals

### 1. Eliminate on-device stack+contiguous by copying DLPack frames directly into a pre-allocated pinned NHWC host buffer
- **Agent:** claude

**Detailed description.**

In vllm/multimodal/video.py:795-839 (PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host), replace the two-step 'materialize an on-device NHWC batch, then D2H the batch' pattern with N direct D2H copies into a single pre-allocated pinned NHWC host buffer, removing the intermediate stacked device tensor entirely. Concretely: (1) after decoder.get_batch_frames_by_index(frame_idx) and the empty/mismatch guards, probe decoded_frames[0] via torch.from_dlpack to determine (H, W, C) and channel-axis order using the same NHWC-vs-NCHW detection currently in _pynvvc_frames_to_nhwc (frames.shape[-1] == 3 vs frames.shape[-3] == 3); (2) allocate exactly one host_frames = torch.empty((N, H, W, 3), dtype=uint8, device='cpu', pin_memory=True); (3) inside the existing cls._torch_stream_context(stream) block, iterate frames and issue host_frames[i].copy_(torch.from_dlpack(frame) if HWC else torch.from_dlpack(frame).permute(1, 2, 0), non_blocking=True) — the permutation becomes a strided device read into a dense pinned host write during the same DMA, rather than a separate on-device permute+.contiguous() copy; (4) keep the single stream.synchronize() before host_frames.numpy(); (5) preserve the empty-frame_idx fast path and add a shape guard on the probed frame (must be 3D with a channel axis of size 3) that mirrors today's 'unexpected shape' ValueError. Delete the torch.stack(torch_frames) call, drop the _pynvvc_frames_to_nhwc invocation on the batched tensor, and delete the intermediate device_frames variable and its del. tests/multimodal/test_video.py continues to enforce dtype, shape (N, H, W, 3), valid frame count, and frame values as the observable contract.

**Novelty rationale.**

Both existing deep_research_proposals explicitly preserve the on-device batched NHWC tensor: proposal 1 keeps 'torch.stack(torch_frames) -> _pynvvc_frames_to_nhwc -> host_frames.copy_(device_frames, ...)' as the D2H stage it pipelines, and proposal 2 states 'keep the existing NHWC device_frames construction on the decoder's CUDA stream'. Neither addresses the intermediate N*H*W*3-byte device allocation from torch.stack nor the additional device-to-device copy that .contiguous() triggers inside _pynvvc_frames_to_nhwc whenever PyNvVideoCodec returns NCHW. This proposal removes both by writing each DLPack frame directly into its offset slice of a single pinned NHWC host buffer, folding the layout permutation into the D2H DMA. It is orthogonal to and composable with proposal 2's pinned-buffer pool (host-side reuse) and proposal 1's ThreadedDecoder pipelining (which still applies per batch).

---

### 2. Batch DLPack conversion with torch.utils.dlpack.from_dlpack to avoid transient Python tensor retention
- **Agent:** codex

**Detailed description.**

In vllm/multimodal/video.py:795-839 (PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host), replace the current list-comprehension lifetime pattern around DLPack frames with a narrower, streaming conversion that feeds stack/copy without retaining both PyNvVideoCodec frame wrappers and Torch tensor wrappers longer than needed. Concretely, after decoder.get_batch_frames_by_index(frame_idx), validate the returned count, then build the stacked tensor from a generator or short-lived tuple created inside the CUDA stream context using torch.utils.dlpack.from_dlpack(frame) and immediately release decoded_frames once stack completes. Keep the existing _pynvvc_frames_to_nhwc, pinned host allocation, non_blocking copy, and synchronize semantics unchanged. Add a small regression test or assertion path in tests/multimodal/test_video.py if feasible to cover mismatched frame counts and confirm dtype/shape/value parity. This is primarily a memory-pressure and lifetime hygiene change for large frame batches: it reduces the window where decoded frame owners and Torch DLPack views are simultaneously live, which can reduce peak GPU memory pressure and allocator stalls on multi-turn video ingress.

**Novelty rationale.**

The deep-research proposals target PyNvVideoCodec ThreadedDecoder/memory demuxing and pinned-host buffer pooling with stream-pipelined D2H. Agent A targets eliminating torch.stack and copying each DLPack frame directly into host NHWC. This proposal does not change the transfer strategy, introduce pooling, or remove the batched device tensor; it narrows DLPack/Torch object lifetimes around the existing stack path to reduce transient ownership and peak memory pressure. That lifetime-focused change is orthogonal to the already proposed decode pipelining, pinned memory reuse, and direct per-frame D2H copy ideas.

---
