# fast_plan_decode

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flashinfer.py`](vllm/v1/attention/backends/flashinfer.py) (lines 2296–2386)
- **Symbol:** `fast_plan_decode`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0006`

## Description
Provides a CUDA-graph-aware fast path for FlashInfer BatchDecodeWithPagedKVCacheWrapper planning.

## Current approach
Uses a first-call boolean to warm the wrapper with plan(), then delegates subsequent CUDA-graph calls to flashinfer.decode.fast_decode_plan with supplied indptr and last_page_len CPU buffers. It does not short-circuit identical metadata or coalesce repeated host metadata preparation at the vLLM layer.

## Estimated impact explanation
Steady-state FlashInfer CUDA-graph decode calls this every token. Skipping unchanged plans or reducing host metadata traffic cuts per-step overhead and improves median TPOT.

## Evolve rationale
The owned policy is the first-call/subsequent-call routing and the decision to call fast_decode_plan every CUDA-graph decode step. Correctness oracle is identical decode wrapper outputs versus the current fast_plan_decode across random paged-KV metadata and existing FlashInfer decode tests.

## Deep research proposals

### 1. Cache and reuse fast_decode_plan auxiliary structures across identical decode steps
- **Finding:** `find-vllm_v1_attention-0004` — *flashinfer.cascade*
- **Source URL:** <https://docs.flashinfer.ai/api/cascade.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `fast_plan_decode` (vllm/v1/attention/backends/flashinfer.py:2296-2386), add a lightweight metadata fingerprint check on the decode wrapper before invoking `fast_decode_plan`. Compute a stable key from the shape/plan inputs that determine the auxiliary structures — `indptr_cpu` contents (or its data_ptr + len + a cheap hash/tail bytes), `last_page_len_cpu` contents, `indices.data_ptr()` + length, `num_qo_heads`, `num_kv_heads`, `head_dim`, `page_size`, `page_encoding_mode`, `window_left`, `logits_soft_cap`, `q_data_type`, `kv_data_type`, `o_data_type`, `fixed_split_size`, `disable_split_kv`. Store the last key on the wrapper (e.g. `self._vllm_last_plan_key`) alongside the already-set `vllm_first_call` sentinel. On subsequent calls, if the key matches, return early and skip the `fast_decode_plan` invocation entirely, since the FlashInfer-owned auxiliary buffers on the wrapper are still valid and the CUDA-graph-captured decode kernel already references the persistent CUDA-graph buffers. On mismatch, run `fast_decode_plan` as today and update the cached key. Keep the first-call `self.plan(...)` warm-up path unchanged so the `_cached_module` is still populated. The correctness oracle from the candidate — identical decode wrapper outputs vs. the current `fast_plan_decode` across random paged-KV metadata and existing FlashInfer decode tests — is unchanged because auxiliary reuse is only taken when inputs are byte-identical.

**Proposal rationale.**

The candidate's evolve_rationale explicitly targets 'short-circuit identical metadata or coalesce repeated host metadata preparation at the vLLM layer,' which is exactly the gap this finding addresses. FlashInfer's cascade documentation states that 'auxiliary data structures can be reused across multiple batch decode attention calls,' establishing that FlashInfer's own design contract permits reuse of these plan artifacts across calls when inputs are unchanged. In a multi-turn agentic workload, many CUDA-graph decode steps within a request see stable paged-KV metadata across consecutive tokens for the same batch shape (indptr grows only when a request finishes, last_page_len changes only at page rollovers), so an exact-match short-circuit will hit frequently on the steady-state hot path. Skipping the host-side `fast_decode_plan` on cache hits removes per-step H2D copies of `indptr`/`last_page_len` and Python-level plan orchestration, directly reducing median TPOT without changing correctness or the first-call warm-up behavior the candidate already owns.

---

### 2. Coalesce indptr and last_page_len into a single pinned H2D copy per fast_plan_decode call
- **Finding:** `find-vllm_v1_attention-0006` — *CUDA C++ Best Practices Guide*
- **Source URL:** <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In `vllm/v1/attention/backends/flashinfer.py` `fast_plan_decode` (lines 2296-2386), replace the two independent CPU-side tensors (`indptr_cpu`, `last_page_len_cpu`) that `fast_decode_plan` implicitly copies to the device with a single contiguous pinned-memory host staging buffer that packs both arrays back-to-back. At the vLLM metadata-preparation layer that feeds this function on every CUDA-graph decode step, allocate the staging buffer once (pinned, persistent across steps, sized for `max_num_seqs + 1 + max_num_seqs`), fill the indptr and last_page_len slices in place, and issue one non-blocking `copy_` into a matching persistent device buffer whose two views are then passed to `fast_decode_plan`. This changes the owned policy from "call fast_decode_plan with two separately-allocated CPU tensors every step" to "pack the small host metadata into one pinned staging tensor and issue a single async H2D transfer per step," while keeping the first-call warmup path unchanged. Correctness is verified against the current fast_plan_decode by comparing decode-wrapper outputs on random paged-KV metadata and the existing FlashInfer decode tests.

**Proposal rationale.**

The CUDA C++ Best Practices Guide explicitly recommends batching many small transfers into one larger transfer and using pinned memory for asynchronous copies. `fast_plan_decode` runs on every CUDA-graph decode token in the FlashInfer path and currently drives two separate small H2D copies (indptr and last_page_len) through `fast_decode_plan`, plus any per-step reallocation of those small CPU tensors upstream. The candidate's evolve_rationale calls out exactly this gap: "coalesce repeated host metadata preparation at the vLLM layer" and it does not short-circuit or batch the host buffers. Coalescing the two arrays into one pinned staging buffer directly applies the finding's mechanism to reduce per-step host overhead, which is on the critical path for median TPOT in the multi-turn agentic workload described in the caller context.

---

### 3. Short-circuit fast_plan_decode on unchanged batch descriptor for CUDA-graph decode replays
- **Finding:** `find-vllm_v1_attention-0009` — *CUDA Graphs*
- **Source URL:** <https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In vllm/v1/attention/backends/flashinfer.py::fast_plan_decode (lines 2296-2386), after the first-call warm-up path (lines 2335-2361), introduce a lightweight batch-descriptor cache on the decode wrapper that guards the fast_decode_plan call at lines 2365-2386. Compute a descriptor from the arguments that actually change plan state across CUDA-graph decode steps: the shape/contents of indptr_cpu and last_page_len_cpu (e.g., their byte-content digest or an (nbytes, data_ptr, sum, tail-value) tuple that is cheap on CPU tensors), plus the scalar fields (num_qo_heads, num_kv_heads, head_dim, page_size, pos_encoding_mode, window_left, logits_soft_cap, dtypes, sm_scale, rope_scale, rope_theta, fixed_split_size, disable_split_kv). Store the last descriptor on `self` (e.g., `self.vllm_last_plan_desc`) alongside the existing `vllm_first_call` flag. When the new descriptor equals the cached one and the wrapper is CUDA-graph enabled, return without invoking fast_decode_plan — the previously planned state on the wrapper is still valid for replay. On mismatch (or when any input is unavailable for hashing), fall back to the current fast_decode_plan call and update the cached descriptor. Keep the indices tensor out of the descriptor (its device buffer is captured by the graph and is not re-copied by fast_decode_plan for cudagraph paths), so the check stays cheap and CPU-only.

**Proposal rationale.**

The finding highlights vLLM's CUDA-graph dispatcher pattern of using stable batch descriptors to route uniform-decode batches through a fast, plan-preserving path while mixed batches fall back safely. fast_plan_decode is exactly the per-step decode planner on that CUDA-graph-compatible route, and today it unconditionally re-runs fast_decode_plan every decode step (candidate's stated gap: no short-circuit on identical metadata, no coalescing of host metadata preparation). In a multi-turn agentic decode workload, consecutive steps within a captured CUDA-graph shape frequently share identical (num_qo_heads, num_kv_heads, page_size, indptr, last_page_len) descriptors between token boundaries where only paged indices contents mutate — precisely the regime where a descriptor equality check removes redundant host-side plan work and the associated H2D copies of indptr/last_page_len, directly cutting per-token overhead and improving median TPOT. The change is local to the owned policy (first-call/subsequent-call routing) and preserves the correctness oracle: whenever the descriptor differs the code still calls fast_decode_plan exactly as before, so decode wrapper outputs remain identical to the current implementation on any metadata that actually changed.

---

## Agent proposals

### 1. Overlap fast_plan_decode on a side CUDA stream with the previous decode step's kernels
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/backends/flashinfer.py `fast_plan_decode` (lines 2296-2386), change the CUDA-graph-enabled subsequent-call path (the `fast_decode_plan(self, ...)` invocation at lines 2365-2386) from a synchronous same-stream call into an overlapped side-stream call gated by a per-wrapper CUDA event. Concretely: lazily attach two attributes to the decode wrapper on first use — `self._vllm_plan_stream: torch.cuda.Stream` (a dedicated, non-blocking side stream created once per wrapper) and `self._vllm_plan_ready_event: torch.cuda.Event` (with `enable_timing=False`, `blocking=False`). Keep lines 2335-2361 (first-call `self.plan(...)` warm-up) exactly as today, still on the current compute stream, so `_cached_module` and any FlashInfer-owned device buffers are populated deterministically. On subsequent CUDA-graph calls, instead of calling `fast_decode_plan(self, ...)` on the current stream, do: (a) `with torch.cuda.stream(self._vllm_plan_stream):` invoke `fast_decode_plan(self, indptr=indptr_cpu, indices=indices, last_page_len=last_page_len_cpu, ..., non_blocking=True, ...)` — this issues the H2D copies of `indptr_cpu`/`last_page_len_cpu` and any planning kernels on the side stream; (b) `self._vllm_plan_ready_event.record(self._vllm_plan_stream)`; (c) `torch.cuda.current_stream().wait_event(self._vllm_plan_ready_event)` before returning. Because the caller in `flashinfer.py:1497-1518` invokes `fast_plan_decode` from the metadata-preparation stage that runs *ahead* of the captured decode graph replay, the side-stream `fast_decode_plan` for step N runs concurrently with any still-in-flight kernels from step N-1 on the compute stream, hiding its host-visible latency in the common steady-state case where metadata changes every token (`last_page_len[i] += 1`) and cache-based short-circuits cannot fire. The compute-stream `wait_event` guarantees the captured graph never replays before the plan artifacts are visible, so the correctness oracle stated in the candidate — identical decode wrapper outputs versus current `fast_plan_decode` across random paged-KV metadata and existing FlashInfer decode tests — is preserved bit-for-bit. To make host-side arguments safe for the deferred issue, ensure `indptr_cpu` and `last_page_len_cpu` are backed by pinned memory (they already come from the persistent `self.paged_kv_indptr.cpu` / `self.paged_kv_last_page_len.cpu` staging buffers, which the metadata builder can allocate pinned) so the non-blocking H2D issued on the side stream does not stall on a synchronous CPU copy. Verification plan: run existing FlashInfer decode tests under `tests/` for correctness; add a targeted micro-benchmark under `benchmarks/kernels/` that measures per-step CUDA-graph decode overhead before/after across batch sizes to confirm TPOT wins on multi-turn agentic decode.

**Novelty rationale.**

The three listed deep_research_proposals all target the case where step-to-step decode metadata is *identical* (findings 4 and 9 short-circuit `fast_decode_plan` on a matching fingerprint/descriptor) or reduce the size of a single H2D transfer (finding 6 coalesces `indptr` and `last_page_len` into one pinned copy). None of them help in the steady-state common case — which dominates a multi-turn agentic decode workload — where `last_page_len[i]` increments by 1 every token for each active sequence, so the descriptor legitimately changes and `fast_decode_plan` must actually execute. This proposal is orthogonal and additive: it does not attempt to skip the call, it hides its latency by issuing it on a dedicated side CUDA stream that overlaps with the previous step's compute on the main stream, gated by a lightweight recorded event. It composes cleanly with all three listed proposals (a cache hit simply skips the side-stream dispatch; a coalesced pinned buffer becomes the payload of the overlapped copy), and it addresses a different root cause — serialization between planning and compute — that the listed proposals leave on the table.

---

### 2. Add a page-rollover-only fast path for last_page_len-only updates
- **Agent:** codex

**Detailed description.**

In `vllm/v1/attention/backends/flashinfer.py::fast_plan_decode` (lines 2296-2386), split the subsequent CUDA-graph path into two cases based on whether the paged-KV layout actually changed versus only token positions advanced within already-allocated pages. Keep the current first-call `self.plan(...)` warm-up unchanged. On later calls, store the previous `indptr_cpu`, `last_page_len_cpu`, and scalar planning parameters on the wrapper. If `indptr_cpu` and all scalar plan parameters are unchanged and every `last_page_len_cpu` entry advanced within the same page without wrapping or changing the effective page count, avoid the full `flashinfer.decode.fast_decode_plan(...)` call and instead update only the wrapper state/buffer that FlashInfer uses for per-sequence last-page lengths, using the existing persistent CPU staging tensor and a direct copy into the cached last-page-length device buffer if that buffer is exposed by the wrapper. If FlashInfer does not expose that buffer stably, add a small helper in this backend that locates it once after the first plan and fails closed to the existing full `fast_decode_plan` path when unavailable. Any change to `indptr_cpu`, page rollover (`last_page_len` crossing the page-size boundary), dtype/head/page scalar parameters, split-kv policy, or buffer identity falls back to the current full plan call and refreshes the cached descriptor. This targets the common decode-step case where the page table and split schedule remain valid but only last-page lengths increment, so vLLM can avoid rebuilding FlashInfer auxiliary planning structures while still keeping the single mutable length vector current for the replayed decode graph.

**Novelty rationale.**

The existing deep-research proposals either skip `fast_decode_plan` only when all relevant metadata is byte-identical, or coalesce the host metadata copies into one pinned transfer. This proposal handles a different non-identical steady-state: `last_page_len_cpu` changes every token but the paged-KV structure and planning schedule remain the same until a page rollover or batch-shape change. Agent A's proposal overlaps planning work on a side CUDA stream but still executes the full `fast_decode_plan` whenever metadata changes; this proposal reduces the work done for the common last-page-length-only update by refreshing only the mutable length buffer and reusing the existing auxiliary plan.

---
