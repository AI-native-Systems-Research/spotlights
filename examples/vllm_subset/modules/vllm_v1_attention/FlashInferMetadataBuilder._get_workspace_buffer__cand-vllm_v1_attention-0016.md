# FlashInferMetadataBuilder._get_workspace_buffer

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flashinfer.py`](vllm/v1/attention/backends/flashinfer.py) (lines 940–963)
- **Symbol:** `FlashInferMetadataBuilder._get_workspace_buffer`
- **Kind:** method
- **Estimated impact:** low
- **Id:** `cand-vllm_v1_attention-0016`

## Description
Lazily allocates the FlashInfer workspace buffer using an environment default and a head-footprint estimate.

## Current approach
Allocates once at max(env buffer size, max_num_batched_tokens * num_qo_heads * head_dim * 16), never shrinks, and does not adapt to observed batch shapes after allocation.

## Estimated impact explanation
Workspace sizing mainly affects memory footprint and edge-case long prefills rather than steady-state median latency. It can still reduce TTFT outliers by avoiding FlashInfer workspace failures or fallback behavior.

## Evolve rationale
Concrete policy is the buffer_size estimate and one-shot allocation. Correctness oracle is existing FlashInfer runtime size checks and output equality; under-provisioning should fail tests while over-provisioning only changes memory use.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Share workspace buffer across FlashInfer wrappers and reuse a per-device singleton to cut allocation and TTFT jitter
- **Agent:** claude

**Detailed description.**

In `FlashInferMetadataBuilder._get_workspace_buffer` (`vllm/v1/attention/backends/flashinfer.py:940-963`), replace the per-builder eager `torch.zeros` allocation with a per-device, per-dtype cached singleton owned by the FlashInfer backend module (e.g., a module-level `dict[(device, size_bucket)] -> torch.Tensor` guarded by a lock). Concretely: (1) compute the sizing estimate as today, but round `buffer_size` up to a small set of power-of-two buckets (e.g., 256 MiB, 512 MiB, 1 GiB) so multiple builders/wrappers (prefill, non-causal prefill, decode, DCP) with slightly different `max_num_batched_tokens`/`num_qo_heads` share the same physical tensor instead of each allocating their own. (2) Store the tensor with `torch.empty` rather than `torch.zeros` — FlashInfer treats it as scratch and does not require zero-initialization, so the current `zeros` performs an unnecessary GPU-wide memset on the multi-hundred-MB buffer at first prefill, directly adding to cold-start TTFT of the very first request in a multi-turn agentic session. (3) If a later builder requests a larger bucket, grow the singleton in place (allocate the larger tensor, drop the smaller) and rebind all live wrappers via `set_workspace_buffer`, so subsequent iterations don't keep two copies resident. (4) Keep the `VLLM_BATCH_INVARIANT` and env-override paths behaving exactly as today (fixed size, no bucketing) to preserve determinism guarantees. The correctness oracle is unchanged (FlashInfer's runtime size checks + output equality); the payoff is one large scratch allocation per device instead of N (prefill wrapper + non-causal prefill wrapper + decode wrappers + any speculative/DCP wrapper), plus removal of the `zeros` memset from the first-request critical path — both of which cut median TTFT under the agentic workload where new sessions repeatedly trigger cold prefill wrappers.

**Novelty rationale.**

There are no existing deep_research_proposals listed on this candidate, and the `evolve_rationale` framing only contemplates tweaking the `buffer_size` estimate and one-shot allocation policy. This proposal is structurally different: it targets (a) cross-wrapper sharing via a module-level bucketed singleton so prefill/non-causal-prefill/decode/DCP wrappers stop each holding their own hundreds-of-MB scratch, and (b) switching `torch.zeros` to `torch.empty` to eliminate an unnecessary large memset on the first-prefill critical path. Neither the sharing nor the zeros→empty change is implied by 'resize the estimate' — they attack allocation count and cold-start memset cost rather than the size formula.

---

### 2. Grow the FlashInfer workspace on demand after runtime size failures
- **Agent:** codex

**Detailed description.**

Change `FlashInferMetadataBuilder._get_workspace_buffer` in `vllm/v1/attention/backends/flashinfer.py:940-963` from a strictly one-shot allocation into an adaptive high-watermark policy that can retry with a larger scratch buffer when FlashInfer reports the workspace is too small for an observed batch shape. Keep the existing initial estimate and environment minimum, but record the allocated size and expose a helper such as `_ensure_workspace_buffer(min_size_bytes)` that reallocates only when an actual planning/run path needs more workspace. The retry path should parse or map FlashInfer's insufficient-workspace error to a required size when available, otherwise grow geometrically (for example 2x, capped by available device memory policy) and retry the wrapper planning once. This targets long or unusual prefills in multi-turn agentic sessions: normal sessions keep today's footprint, while rare larger batches avoid hard failures or fallback/reinitialization costs that can inflate TTFT. Add a focused test by monkeypatching the FlashInfer wrapper/planning call to first raise a deterministic insufficient-workspace exception and then succeed after `_get_workspace_buffer` grows, asserting the same wrapper receives the larger buffer and normal output behavior is unchanged.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes sharing a bucketed per-device singleton across wrappers and replacing `torch.zeros` with `torch.empty`, with optional singleton growth when later builders request a larger bucket. This proposal is different: it keeps ownership local to the existing builder/wrapper path and focuses on adaptive growth driven by observed FlashInfer insufficient-workspace failures at runtime, preserving the smaller initial allocation until a real batch shape proves it inadequate. It addresses under-provisioning and retry behavior rather than allocation deduplication, bucketing, or memset removal.

---
