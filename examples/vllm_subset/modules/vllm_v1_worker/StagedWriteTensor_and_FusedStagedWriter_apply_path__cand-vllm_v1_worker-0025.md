# StagedWriteTensor and FusedStagedWriter apply path

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/buffer_utils.py`](vllm/v1/worker/gpu/buffer_utils.py) (lines 128–310)
- **Symbol:** `StagedWriteTensor and FusedStagedWriter apply path`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0025`

## Description
Staged GPU-buffer write machinery used by the new runner to apply request and block-table mutations.

## Current approach
Accumulates staged indices, starts, contents, and cumulative lengths in Python lists; copies several metadata lists through UVA buffers; transfers contents separately; then launches _apply_write_kernel with fixed BLOCK_SIZE=1024.

## Estimated impact explanation
This cost scales with request churn and KV groups; better packed staging or persistent device-side append buffers improves TTFT for admitted/resumed requests and reduces TPOT during churn-heavy agent traffic.

## Evolve rationale
The Python staging lists, UvaBufferPool copies, async_tensor_h2d(contents), and _apply_write_kernel launch are a self-contained write path. tests/v1/worker/test_gpu_block_table.py validates fused multi-group writes and single-group behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Stage writes directly into persistent pinned/UVA ring buffers to eliminate list rebuilding and the synchronous contents H2D
- **Agent:** claude

**Detailed description.**

In `vllm/v1/worker/gpu/buffer_utils.py` at `StagedWriteTensor` (lines 114-207) and `FusedStagedWriter.apply` (lines 210-271), replace the four Python staging lists (`_staged_write_indices`, `_staged_write_starts`, `_staged_write_contents`, `_staged_write_cu_lens`) with a set of persistent pinned/UVA-backed ring buffers owned by each `StagedWriteTensor` plus small counters (`n_writes`, `contents_used`).

Concretely: (1) On construction, allocate `UvaBuffer` (or a `UvaBufferPool` slot) for `indices`, `starts`, `cu_lens` sized `num_rows` (int32) and a `contents` UVA buffer sized to the tensor's maximum possible content footprint (num_rows * row_width for row-shaped tensors; a per-instance cap otherwise). Keep a numpy view of each for fast slice assignment. (2) In `stage_write`, write `index`, `start`, and the incoming `x` directly into these numpy-backed pinned regions at positions `n_writes` and `contents_used` using `np.asarray(x, dtype=...)` + slice assignment (which uses vectorized memcpy in C), then bump the counters and record `cu_lens[n_writes] = contents_used`. `stage_write_elem` becomes a single scalar store. This removes the per-call Python `list.append`/`list.extend` overhead — currently the dominant CPU cost when many small requests are admitted or resumed each step. (3) In `apply_write`, issue a single async H2D of `contents[:contents_used]` from pinned memory to a device tensor (reusing a per-instance device scratch of size `contents_capacity` if available) instead of `async_tensor_h2d(list, ...)`, which today walks a Python list into a fresh unpinned CPU tensor and then pins-copies it. The `indices`/`starts`/`cu_lens` UVA views are already device-visible with no explicit H2D. (4) In `FusedStagedWriter.apply`, drop the per-group Python `extend`/comprehension re-materialization entirely: launch `_apply_write_kernel` with `MULTI_GROUP=True` using each tensor's own already-staged UVA `indices/starts/cu_lens` plus a small per-group `(group_id, write_offset, content_base)` table computed from `t.n_writes` and `t.contents_used`. Pass a device pointer table of per-group `contents` bases (like `output_ptrs`) and let the kernel use `group_id` to index both `output_ptr` and a new `contents_ptrs` argument, or concatenate the small `cu_lens` arrays into one UVA scratch via a single pinned memcpy loop rather than a Python-level rebuild. (5) Additionally, launch one Triton program per contiguous chunk of up to BLOCK_SIZE elements instead of one program per write when writes are short: precompute a program→(write_id, chunk_id) mapping in the same pinned pass so `stage_write_elem` calls (many 1-element writes for block_table updates) each cost O(1) work rather than a BLOCK_SIZE=1024 masked kernel program.

This targets `tests/v1/worker/test_gpu_block_table.py` (fused multi-group and single-group write paths) for correctness. Verification: run that test suite, plus a microbench that stages N=10k small `stage_write_elem` calls (typical for block-table churn during agentic decode) and measures apply_write wall time vs. current implementation, and end-to-end `vllm bench serve` on a multi-turn agentic trace measuring median TTFT/TPOT.

**Novelty rationale.**

The candidate lists no existing deep_research_proposals (the field is empty), so any concrete proposal is novel by construction. Beyond that, this specifically attacks two costs the candidate description flags but does not itself address: the Python staging lists (replaced with pinned/UVA ring buffers written via numpy slice assignment, not just a bigger buffer or different kernel tuning) and the separate `async_tensor_h2d(contents)` H2D (fused into the pre-staged pinned copy). It also introduces a program-per-chunk kernel launch scheme rather than program-per-write, which is orthogonal to simply changing BLOCK_SIZE. These are concrete algorithmic changes to the write path, not just kernel autotune or a bigger buffer.

---

### 2. Coalesce contiguous staged writes per row before applying
- **Agent:** codex

**Detailed description.**

Add a lightweight coalescing path inside `StagedWriteTensor.stage_write` and `stage_write_elem` in `vllm/v1/worker/gpu/buffer_utils.py`: when the newly staged mutation targets the same `index` as the most recent staged write and its `start` is exactly the previous write's end offset, append the new contents to the existing staged content span and update only the last cumulative length instead of appending a new write record. For `stage_write_elem`, treat the element as a length-1 write and coalesce when it extends the prior contiguous span. This keeps current last-write-wins behavior because it only merges non-overlapping adjacent ranges, but reduces `n` passed to `_apply_write_kernel`, shrinks the `indices`/`starts`/`cu_lens` UVA copies, and lowers per-write Triton program overhead. This should help churn-heavy steps where block tables or per-request state receive several adjacent appends for the same request before `apply_write`. Add focused coverage to `tests/v1/worker/test_gpu_block_table.py` or the nearest buffer-utils test to stage multiple adjacent writes to the same row, verify the final GPU row contents match the uncoalesced semantics, and verify a non-contiguous or different-row write remains a separate record.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes replacing Python lists with persistent pinned/UVA ring buffers, avoiding the contents H2D rebuild, changing the fused path to use per-group pointer tables, and chunking kernel programs. This proposal is different: it preserves the current staging/list and kernel architecture, but changes the staging semantics to merge adjacent same-row writes so fewer logical writes reach the existing apply path. It is a smaller, orthogonal reduction in metadata traffic and kernel program count, not a persistent-buffer or H2D-transfer redesign.

---
