# WorkspaceManager._ensure_workspace_size

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/workspace.py`](vllm/v1/worker/workspace.py) (lines 119–191)
- **Symbol:** `WorkspaceManager._ensure_workspace_size`
- **Kind:** region
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_worker-0021`

## Description
Lazy per-ubatch GPU scratch workspace allocation and resize policy.

## Current approach
When a larger request arrives, drops the old workspace, calls torch.accelerator.empty_cache(), and allocates exactly required_bytes for the requesting ubatch.

## Estimated impact explanation
This is a tail-latency and occasional TTFT lever rather than steady-state TPOT; exponential high-water growth can avoid repeated allocator stalls for long tool-response prefills.

## Evolve rationale
The empty_cache call and exact-size growth policy are the optimization constructs. Workspace lock/growth assertions and workspace tests validate allocation safety and no-growth-after-lock behavior.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Geometric high-water growth with lazy empty_cache to eliminate re-allocation stalls on the hot path
- **Agent:** claude

**Detailed description.**

In `WorkspaceManager._ensure_workspace_size` (vllm/v1/worker/workspace.py:119-191), replace the exact-size growth policy with a geometric high-water strategy tuned for the pre-lock warmup window and any subsequent unlock/relock episodes (e.g. elastic EP scaling):

1. Round `required_bytes` up to a `max(next_pow2(required_bytes), current_size * growth_factor)` where `growth_factor` defaults to 2.0 (configurable via a new `VLLM_WORKSPACE_GROWTH_FACTOR` env var), and also round up to a coarse alignment (e.g. 2 MiB) so successive small deltas do not each trigger a reallocation. Allocate `torch.empty` at this rounded size rather than `required_bytes`.
2. Skip the `torch.accelerator.empty_cache()` call on the common warm-up growth path. `empty_cache` forces a device synchronize and returns cached blocks to the driver, which is a well-known stall on CUDA/HIP. Instead, do a two-step reallocation: (a) drop the reference to the old tensor first (`self._current_workspaces[ubatch_id] = None; del current_workspace`) then (b) directly call `torch.empty` at the new geometric size. The PyTorch caching allocator will reuse the just-freed block when the new size fits or split a larger cached segment; only fall back to `empty_cache()` if the fresh allocation raises `torch.cuda.OutOfMemoryError`, retrying once after the cache flush. This preserves OOM safety while removing the synchronous stall from the hot warmup path.
3. Track a `_high_water_bytes` per ubatch slot and, when `lock()` is called, log both the observed high water and the geometric-rounded allocated size so operators can tune `VLLM_WORKSPACE_GROWTH_FACTOR` if the slack is excessive. No behavior change post-lock: locked growth still asserts.
4. Optionally, when `num_ubatches > 1`, proactively resize sibling ubatch slots to the new high-water on the next `get_simultaneous` call for that slot (already the existing lazy behavior) — but pre-round their target size to the same high-water so the second ubatch does not incur an independent geometric ramp.

Expected effect for the caller context (multi-turn agentic workloads): tool-response prefills produce highly variable ubatch sizes across turns, so the current exact-fit policy hits `_ensure_workspace_size`'s reallocation branch many times per session before the workspace saturates. Each hit incurs a `cudaFree`-inducing `empty_cache()` plus a fresh `cudaMalloc`, both of which serialize the CUDA stream and land directly on TTFT for the affected request. Geometric growth converges to the session-wide high water in O(log N) reallocations; dropping `empty_cache()` on the common path removes the per-growth device-side sync. Peak reserved memory grows by at most `growth_factor` × the true high water — bounded and observable.

**Novelty rationale.**

The candidate has zero listed deep_research_proposals, so any concrete change is novel by construction. Substantively, this proposal targets two specific mechanisms (geometric growth policy and elimination of the synchronous `empty_cache()` on the hot growth path with an OOM-fallback retry) that are not otherwise being investigated on this symbol, and ties them to the multi-turn agentic TTFT objective in the caller context.

---

### 2. Consolidate ubatch scratch buffers into one growable workspace slab
- **Agent:** codex

**Detailed description.**

Change `WorkspaceManager._ensure_workspace_size` in `vllm/v1/worker/workspace.py:119-191` so the manager owns a single contiguous `torch.uint8` slab sized for all ubatch slots, with each ubatch receiving a non-overlapping slice of that slab. Track per-ubatch required sizes and offsets; when any ubatch needs more space before lock, recompute the total slab size as the sum of aligned per-ubatch sizes, allocate one replacement slab, and refresh `_current_workspaces` as views into that slab. Keep the existing DBO safety rule by only resizing while unlocked, and preserve the locked-growth assertion. This reduces the number of independent CUDA allocator blocks from `num_ubatches` to one, avoids fragmentation between DBO slots, and makes the post-warmup steady state cheaper for multi-turn workloads that repeatedly alternate ubatches during prefill/decode.

**Novelty rationale.**

There are no deep_research_proposals listed for this candidate. Agent A proposed geometric per-ubatch growth, lazy `empty_cache`, high-water tracking, and optional sibling high-water alignment. This proposal is different: it changes the allocation topology from multiple independent per-ubatch tensors to a single partitioned slab, targeting allocator fragmentation and multi-ubatch steady-state behavior rather than the exact growth factor or `empty_cache` policy.

---
