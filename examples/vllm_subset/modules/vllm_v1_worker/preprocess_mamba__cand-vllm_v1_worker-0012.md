# preprocess_mamba

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/mamba_utils.py`](vllm/v1/worker/mamba_utils.py) (lines 1038–1142)
- **Symbol:** `preprocess_mamba`
- **Kind:** function
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0012`

## Description
Legacy per-step Mamba align-mode preprocessing that computes previous/current state block indices and stages copy metadata before forward.

## Current approach
Contains an explicit optimization TODO and loops over input_batch.req_ids, doing dict lookups, block arithmetic, per-request CPU buffer writes, and either fused-buffer staging or Python metadata collection.

## Estimated impact explanation
Hybrid Mamba/SSM models pay this on the per-step path; vectorizing scheduler-derived arrays reduces O(num_reqs x layer-groups) Python work and improves TPOT for long agent loops.

## Evolve rationale
The TODO at line 1058 and the per-request loop are concrete optimization targets. Fused and unfused Mamba parity tests in tests/v1/worker/test_mamba_utils.py validate the same state-copy semantics.

## Deep research proposals

### 1. Vectorize preprocess_mamba by making mamba_state_idx a persistent GPU-resident tensor indexed by MRV2-style stable request rows
- **Finding:** `find-vllm_v1_worker-0003` — *Model Runner V2 Design Document*
- **Source URL:** <https://docs.vllm.ai/en/v0.17.0/design/model_runner_v2/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor preprocess_mamba (vllm/v1/worker/mamba_utils.py:1038-1142) so the per-request Python loop over input_batch.req_ids is replaced by vectorized array operations, following the Model Runner V2 pattern of decoupling persistent per-request state from per-step input tensors. Concretely: (1) Replace the mamba_state_idx: dict[str, int] with a persistent int32 tensor sized to input_batch's stable-row capacity, kept alongside the other MRV2 persistent tensors (mirrored CPU/GPU as needed), so lookup and update become row-indexed gathers/scatters rather than dict.get/dict[]= per req. (2) Derive num_scheduled_tokens and num_computed_tokens as 1D arrays over the active-row slice (they already exist in the input batch / scheduler output as arrays), then compute num_blocks = cdiv(num_computed + num_scheduled, block_size) + num_speculative_blocks and curr_state_idx = num_blocks - 1 - num_speculative_blocks as elementwise numpy/torch ops. (3) Compute prev_state_idx as a gather from the persistent state-idx tensor at the current active rows, with a masked fallback (num_computed_tokens - 1) // block_size for rows the cleanup step just invalidated. (4) Build src_col, token_bias, and state_idx buffers by masked vector assignment (mask = (prev != -1) & (prev != curr)) using input_batch.num_accepted_tokens_cpu as an array, and write the reset num_accepted_tokens_cpu[mask] = 1 in one shot. (5) Feed the resulting arrays directly into fused.ctx.run_fused_precopy without staging per-request writes into np buffers inside the loop; for the unfused path, do collect_mamba_copy_meta over the masked indices rather than one call per req. This removes the O(num_reqs × mamba_group_ids) Python overhead flagged by the TODO at line 1058 while preserving the fused/unfused parity covered by tests/v1/worker/test_mamba_utils.py.

**Proposal rationale.**

The finding's core MRV2 principle — "decouple persistent state tensors from per-step input tensors" and gather per-step inputs from mostly GPU-resident state — targets exactly the gap in this candidate: mamba_state_idx is a Python dict keyed by req_id string, and every field the loop touches (num_scheduled_tokens, num_accepted_tokens, block_table) is looked up per request even though the input batch already stores them as arrays over stable rows. Applying the MRV2 shape here converts the loop into vectorized numpy/torch ops on the active-row slice, directly attacking the per-step Python cost the candidate calls out as the optimization target and shrinking TPOT for multi-turn agentic workloads that pay this preprocessing every decode step.

---

### 2. Fold Mamba pre-copy staging into a fused selective_state_update with separate src/dst indices
- **Finding:** `find-vllm_v1_worker-0011` — *flashinfer.mamba.selective_state_update*
- **Source URL:** <https://docs.flashinfer.ai/generated/flashinfer.mamba.selective_state_update.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor `preprocess_mamba` in vllm/v1/worker/mamba_utils.py (lines 1038-1142) so that, when the runtime backs Mamba updates with flashinfer's `selective_state_update`, we no longer stage a separate previous->current state block copy at all. Concretely: (1) keep the scheduler-derived bookkeeping that computes `prev_state_idx` and `curr_state_idx` per request, but stop populating `MambaCopyBuffers` / running `do_mamba_copy_block` / `run_fused_precopy` on the pre-forward path; (2) instead, build two int32 device arrays sized `num_reqs` — `state_batch_indices` (== prev_state_idx, with -1/new-request slots pointing at a scratch/zero slot) and `dst_state_batch_indices` (== curr_state_idx) — plus a per-request `accept_token_bias` array; (3) plumb these directly into the flashinfer `selective_state_update` invocation inside the Mamba layer forward so the kernel reads from the previous block and writes the updated state to the current block in one pass. The unfused/`copy_bufs`-based branch (lines 1116-1127, 1142) and the fused-precopy branch (lines 1066-1082, 1130-1140) both collapse into filling these two index tensors. Vectorize the per-request loop over `input_batch.req_ids` by computing `curr_state_idx` and `prev_state_idx` from already-GPU-resident tensors on `input_batch` (num_computed_tokens, num_scheduled_tokens, num_accepted_tokens) so the Python loop at lines 1083-1128 shrinks to array ops. Preserve current parity by falling back to today's staged-copy path when the active kernel does not accept `dst_state_batch_indices`, and validate against `tests/v1/worker/test_mamba_utils.py` fused/unfused parity tests.

**Proposal rationale.**

The candidate explicitly flags the per-step preprocessing loop as an optimization target (TODO at line 1058) and the copy-spec staging (fused and unfused branches) is the largest chunk of per-request Python work on the hybrid-model hot path. The finding shows that flashinfer's `selective_state_update` accepts a `dst_state_batch_indices` parameter that decouples read and write state slots inside the update kernel itself — exactly the primitive needed to eliminate the separate pre-copy stage that this function exists to orchestrate. Adopting it addresses two constraints at once: it removes a full GPU pass (the pre-copy) and it lets the remaining scheduler-side work reduce to two small index tensors that can be built from already-CPU-resident scheduler arrays, cutting O(num_reqs x layer-groups) Python dict/attr work per step. For a multi-turn agentic workload with many short decode steps, that Python overhead is a real component of TPOT, and removing an extra device kernel/copy per step also trims TTFT on the first decode after prefill.

---

## Agent proposals

### 1. Cache last-step curr_state_idx per stable row and short-circuit preprocess_mamba on unchanged steady-state decode steps
- **Agent:** claude

**Detailed description.**

Exploit temporal coherence across decode steps in preprocess_mamba (vllm/v1/worker/mamba_utils.py:1038-1142) so that steady-state single-token decode — the dominant regime for multi-turn agentic TPOT — pays near-zero per-step cost. Concretely: (1) Add a persistent int32 `last_curr_state_idx` buffer sized to input_batch's stable-row capacity (mirrored CPU/GPU via CpuGpuBuffer, kept alongside the other persistent per-row tensors and updated by `cleanup_mamba_state_idx` when rows are freed/resumed) that stores the previous step's `curr_state_idx` per stable batch row. (2) At the top of preprocess_mamba, build a small boolean `needs_recompute` mask over the active-row slice: True iff the row is a new/resumed slot after the cleanup call (detectable via `scheduled_new_reqs` / a sentinel value in `last_curr_state_idx`), OR `input_batch.num_accepted_tokens_cpu[i] > 1` (draft acceptance changed the block boundary), OR `(num_computed_tokens[i] + num_scheduled_tokens[i]) // block_size != num_computed_tokens[i] // block_size` (a block boundary is crossed this step) — all computed as vector ops on numpy arrays already resident on `input_batch`. (3) For `~needs_recompute` rows, set `prev_state_idx[i] = curr_state_idx[i] = last_curr_state_idx[i]`; because `prev == curr`, the existing mask at line 1110 makes these rows contribute nothing to `src_col`, `token_bias`, `collect_mamba_copy_meta`, or the `num_accepted_tokens_cpu` reset — no CpuGpuBuffer writes, no metadata staging. (4) Run the current per-row work only over `needs_recompute` (typically a handful of rows), then scatter the updated `curr_state_idx` back into `last_curr_state_idx`. (5) Add an empty-set fast-path: if `needs_recompute.sum() == 0` on the fused path, skip the three `fused.*.copy_to_gpu(num_reqs)` H2D transfers and the `fused.ctx.run_fused_precopy(...)` launch entirely — with `src_col` all -1 the kernel would be a no-op, but a launch and three H2D copies still cost latency every decode step. On the unfused path, skip `do_mamba_copy_block(copy_bufs)` when `copy_bufs.offset == 0`. (6) Preserve fused/unfused parity (validated by tests/v1/worker/test_mamba_utils.py) and add a targeted test that mocks `ctx.run_fused_precopy` and asserts call_count stays flat across a synthetic 3-step decode with no block crossings.

**Novelty rationale.**

The two existing DR proposals both attack per-request Python cost by making the current per-row computation cheaper — DR#0003 vectorizes the loop against persistent MRV2-style tensors, and DR#0011 folds staging into `selective_state_update` via `dst_state_batch_indices`. Both still touch every active row on every step and still H2D-copy `num_reqs`-sized index tensors on every step. This proposal is orthogonal and composable with either: it exploits temporal coherence between decode steps by caching the previous step's `curr_state_idx` per stable row and reducing the working set from `num_reqs` to `|changed_rows|` — which is typically 0 in steady-state agentic decode because no block boundary is crossed and no draft acceptance occurred. Critically, it adds an empty-mask fast-return that eliminates the fused-precopy kernel launch and its three H2D transfers entirely on those steps, a knob neither DR touches (DR#0011 removes one kernel but still builds/uploads the two index tensors each step; DR#0003 still gathers/scatters over all active rows). The temporal-caching angle and the launch-elision fast-path are the novel contribution.

---

### 2. Let fused precopy reset accepted-token GPU state in-kernel
- **Agent:** codex

**Detailed description.**

For the fused align path in `preprocess_mamba` (`vllm/v1/worker/mamba_utils.py:1066-1140`), extend `MambaSpecDecodeGPUContext.run_fused_precopy` and `precopy_mamba_align_fused_kernel` so the kernel takes `num_accepted_tokens_gpu` directly, computes `accept_token_bias = num_accepted_tokens_gpu[req] - 1` for rows with `src_col != -1`, and stores `1` back to `num_accepted_tokens_gpu[req]` after performing the previous->current state copy. Then change the `gpu_model_runner.py` align call-site immediately after `preprocess_mamba` to skip the unconditional `self.num_accepted_tokens.np[:num_reqs] = input_batch.num_accepted_tokens_cpu[:num_reqs]` plus `copy_to_gpu(num_reqs)` when the fused precopy path is active, because the device buffer has already been corrected for the Mamba forward. Keep the existing CPU `input_batch.num_accepted_tokens_cpu[i] = 1` update for host-side bookkeeping and keep the current H2D resync for the unfused path. Validate with an align fused/unfused parity test that asserts copied rows both use the original accepted-token bias for the copy and observe `num_accepted_tokens_gpu == 1` before forward.

**Novelty rationale.**

The deep-research proposals focus on replacing the per-request loop with persistent/vectorized request-row state, or removing the separate precopy by folding source/destination indices into `selective_state_update`. Agent A focuses on caching unchanged rows and skipping no-op launches/transfers. None of them specifically removes the extra full `num_accepted_tokens` CPU-to-GPU resynchronization that the runner performs after `preprocess_mamba` mutates `num_accepted_tokens_cpu`. This proposal is narrower and orthogonal: keep the fused precopy design, but make that existing kernel also update the already-live GPU accepted-token buffer so copied rows do not require a second host-to-device pass before forward.

---
