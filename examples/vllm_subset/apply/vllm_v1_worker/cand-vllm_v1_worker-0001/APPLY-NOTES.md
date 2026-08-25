# Apply notes — cand-vllm_v1_worker-0001

- **Candidate:** `cand-vllm_v1_worker-0001`
- **Module:** `vllm/v1/worker`
- **Repo:** `/Users/iklamer/ai-native-systems/VSCodeProjects/vllm`
- **Base commit:** `83ad767eed3be3ee7f2df63be693bfaca5c7c922`
- **Objective:** reduce the median TTFT and median TPOT (Time Per Output Token) (direction: minimize)

## In-scope files

- vllm/v1/worker/gpu_model_runner.py:2001-2323 — GPUModelRunner._prepare_inputs

## Files changed

What the diff actually touched, derived from the patch itself — not from the
declared scope above. Compare it against "In-scope files": any disagreement
is exactly what a reviewer needs to see before applying this patch.

| File | Change | Lines | Scope |
| --- | --- | --- | --- |
| `vllm/v1/worker/gpu_model_runner.py` | modified | +77/-42 | in scope |

**Totals:** 1 file changed, +77/-42.


## The proposal

Main legacy runner per-step CPU-to-GPU input preparation for request indices, positions, token indices, query_start_loc, optimistic sequence lengths, discard masks, slot mapping, and spec-decode counts.

**Current approach:** Uses several NumPy/Torch CPU passes, torch.index_select, repeated CpuGpuBuffer.copy_to_gpu calls, a per-request prompt-embeds loop, a req_ids list walk for num_tokens, and an event synchronize before staging num_accepted_tokens.

**Why it was worth changing:** The method runs every non-empty legacy runner step on the TTFT/TPOT path. Coalescing H2D staging, removing Python request walks, and replacing the accepted-token sync contract can preserve positions, query_start_loc, slot_mapping, and request-state semantics covered by tests/v1/worker/test_gpu_model_runner.py::test_update_states_* and tests/v1/attention/test_attention_backends.py.

## Research findings used

- [RFC]: Multi-Step Scheduling (issue) — https://github.com/vllm-project/vllm/issues/6854
  Adopt a worker-local multi-step decode loop that keeps sampled tokens on GPU, advances next-step input metadata with CUDA kernels, and delays scheduler/output synchronization until lookahead slots are exhausted. This directly targets median TPOT by amortizing per-token Python input prep, output materialization, and scheduler overhead across multiple decode iterations in agentic multi-turn traffic.
- [Performance]: Fully Async Spec-Decoding | Make `seq_lens_cpu` in CommonAttentionMetadata optional (issue) — https://github.com/vllm-project/vllm/issues/29134
  Make CPU sequence-length metadata optional and push spec-decode metadata consumers toward device-resident sequence lengths and upper bounds. This would remove host/device syncs that block overlapping next-step input preparation with current forward execution, especially when verifying multiple drafted tokens.
- Model Runner V2 Design Document (docs) — https://docs.vllm.ai/en/v0.17.0/design/model_runner_v2/
  Decouple persistent request state from per-step input tensors, assign active requests stable rows, and gather per-step inputs from mostly GPU-resident state. This reduces request-churn bookkeeping and avoids tensor-wide reordering on the critical TTFT/TPOT path.
- vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction (blog) — https://vllm-project.github.io/2024/09/05/perf-update.html
  Overlap output processing with the next model execution step instead of synchronously converting GPU tensors to Python lists and checking stop criteria after every token. The worker output path could further delay or batch materialization to reduce TPOT bubbles from token IDs, logprobs, NaN counts, and fault flags.

## Recorded oracles

These are carried forward **verbatim** from the candidate. They are the
verification recipe for a machine that can run them.

**Correctness:**
- `pytest tests/v1/worker/test_gpu_model_runner.py`
- `pytest tests/v1/attention/test_attention_backends.py`

**Performance:** TTFT, TPOT

> **Nothing in this directory was verified.** No test was run, no benchmark was
> measured, no build was attempted. The patch was produced in a fresh detached
> worktree with no virtualenv and no compiled extensions, on a machine that may
> lack the hardware the performance oracle needs. Treat this as a proposal
> faithfully implemented — not as a measured win.

## What changed and why

# _prepare_inputs pure-decode CPU fast path

## What changed

Added a small `pure_decode` branch at the top of
`GPUModelRunner._prepare_inputs`
(`vllm/v1/worker/gpu_model_runner.py`, formerly lines 2018–2073).
When every scheduled request has exactly one token (i.e.
`total_num_scheduled_tokens == num_reqs`, the common multi-turn agentic
decode case) and no request uses inline prompt embeddings, the CPU input
preparation is short-circuited:

1. `req_indices` becomes `self.arange_np[:num_reqs]` (a view) instead of
   the `np.repeat(arange, num_scheduled_tokens)` call.
2. `cu_num_tokens` becomes the view `self.arange_np[1 : num_reqs + 1]`
   instead of `_get_cumsum_and_arange(num_scheduled_tokens, ...)` which
   otherwise runs a cumsum, a `np.repeat`, and a `np.subtract` into
   `self.query_pos.np`.
3. `self.query_pos.np[:num_reqs]` is filled with zeros directly, so the
   downstream `self.query_pos.copy_to_gpu(total_num_scheduled_tokens)`
   still copies the correct values.
4. `positions_np` is aliased to
   `self.input_batch.num_computed_tokens_cpu[:num_reqs]` — the "compute
   pos = num_computed + query_pos" step degenerates to just
   `num_computed`.
5. The `token_ids_cpu_tensor.flatten() + torch.index_select` gather —
   which builds a (max_num_reqs * max_model_len) flattened view every
   step — is replaced by a direct 2D fancy-index gather
   `token_ids_cpu[arange, positions]` sized `num_reqs`. The equivalent
   change is applied to `is_token_ids` when
   `enable_prompt_embeds` is on.

The `else` branch is unchanged and is byte-for-byte the same as the
prior implementation, so all non-decode-only batches (chunked prefill,
mixed prefill/decode, speculative decode with drafts, prompt-embed
batches) keep the exact existing code path.

Everything downstream of the branch (`query_start_loc`,
`optimistic_seq_lens_cpu`, `_compute_prev_positions`, `num_tokens`
discard mask, `num_accepted_tokens` sync, `req_indices.copy_to_gpu`,
`num_computed_tokens` update, GPU `positions` / `seq_lens`
computation, `_prepare_input_ids`, mrope/xdrope copies, spec-decode
metadata, LoRA hot-swap, return values) is untouched.

## Why this should reduce TTFT / TPOT

For the workload named in the task (multi-turn agentic decode), the
overwhelmingly common step shape is "N requests, each with one
scheduled token" (bonus tokens per accepted decode step, no draft or
partial prefill). The CPU wall time inside `_prepare_inputs` is a real
fraction of that step's TTFT/TPOT because the method runs every
non-empty legacy runner step and the model itself is fast per step at
low batch sizes.

Removed per-step CPU work when the fast path is taken:

- `np.repeat(arange_np[:num_reqs], num_scheduled_tokens)` — allocates
  and writes a fresh `num_reqs`-element int64 array.
- `_get_cumsum_and_arange`: `np.cumsum` + `np.repeat` +
  `np.subtract(..., out=...)` — three passes over
  `num_scheduled_tokens`-sized arrays.
- `num_computed_tokens_cpu[req_indices] + query_pos.np[:total]` —
  numpy fancy-index gather + addition into a new array.
- `positions_np + req_indices * max_model_len` — another new
  intermediate array.
- `torch.from_numpy(token_indices)` + `torch.index_select` on the
  flattened `token_ids_cpu_tensor` (whose flat size is
  `max_num_reqs * max_model_len`, which for the default max_model_len
  is enormous even though the gather only pulls `num_reqs` values).

The replacement performs one contiguous 1D copy of length `num_reqs`
plus a tiny bookkeeping fill. There are no additional H2D transfers —
the GPU work below the branch is identical either way (same
`query_pos`, `req_indices`, `num_scheduled_tokens` H2Ds of size
`num_reqs`, same `positions`/`seq_lens`/`slot_mapping` construction).

## Proposals this draws on

Directly implements the `[claude]` proposal:
"Add a pure-decode fast path in `_prepare_inputs` that skips
`np.repeat`, cumsum, and `index_select` when every request has
`num_scheduled_tokens == 1`". That proposal explicitly argues this is
the only in-scope change of the five listed that eliminates the CPU
side of the per-step work (proposals #1–#4 all attack sync/staging or
propose cross-file state refactors and metadata contract changes,
which would require edits outside the allowed file range).

Not adopted (and why):

- Multi-step decoding, async spec metadata, MRV2 persistent state,
  and one-step-deferred `num_accepted_tokens` sync all require changes
  to files outside the allowed scope (attention metadata contracts,
  scheduler/output pipeline, request-state ownership, worker/runner
  glue). None are implementable as a faithful, self-contained edit to
  `_prepare_inputs`.

## What a reviewer should check by hand

1. **Correctness of the fast-path gate.** `pure_decode = (total_num_scheduled_tokens == num_reqs) and not req_prompt_embeds`.
   Since every scheduled request has `num_scheduled_tokens[i] >= 0` and
   the sum equals the count, each `num_scheduled_tokens[i]` must be
   exactly 1. Confirm this holds — in particular that `num_reqs` here
   is the count of active input-batch rows, not something that could
   include padding with zero scheduled tokens. (Reading
   `self.input_batch.num_reqs` at 2015 and the way callers construct
   `num_scheduled_tokens` should confirm this: the caller builds
   `num_scheduled_tokens` from the same active-row set.)
2. **`cu_num_tokens` dtype.** In the slow path it comes from
   `np.cumsum(int32)`; on 64-bit numpy this is typically int64. In the
   fast path it is the int64 view `arange_np[1:num_reqs+1]`. Downstream
   uses (`query_start_loc.np[1:num_reqs+1] = cu_num_tokens`, an int32
   destination that will safely downcast; `.fill(cu_num_tokens[-1])`;
   and passing to `_calc_spec_decode_metadata`, which cannot be reached
   in pure-decode because spec decode requires at least one request
   with `num_scheduled_tokens[i] >= 2`) are dtype-flexible.
3. **`query_pos.np[:num_reqs] = 0`** is required so the later
   `self.query_pos.copy_to_gpu(total_num_scheduled_tokens)` uploads
   zeros for the `num_reqs` positions we care about. Confirm no other
   consumer of `query_pos.gpu[num_reqs:]` matters in this call — the
   `positions[:total_num_scheduled_tokens]` computation only reads
   `query_pos.gpu[:total_num_scheduled_tokens]` and
   `total_num_scheduled_tokens == num_reqs` in the fast path.
4. **`token_ids_cpu` gather semantics.** `token_ids_cpu` is an int32
   2D numpy view of `token_ids_cpu_tensor` (shape
   `(max_num_reqs, max_model_len)`). Fancy indexing
   `token_ids_cpu[arange, positions]` returns a 1D int32 array of
   length `num_reqs`; assigning it into `self.input_ids.np[:num_reqs]`
   (an int32 view of the pinned CPU buffer) writes the same values the
   old flatten + `index_select` did. Similarly for `is_token_ids`.
5. **No behavioral change on non-pure-decode.** The `else` branch is
   the previous code verbatim; there are no shared side effects
   between branches beyond writing to buffers that both branches also
   write to (`query_pos.np`, `input_ids.cpu`, `is_token_ids.cpu`).
6. **Recorded oracles.** `pytest tests/v1/worker/test_gpu_model_runner.py`
   and `pytest tests/v1/attention/test_attention_backends.py` should
   both pass; the change does not alter attention metadata, request
   states, or output shapes.

## Applying and verifying this patch

Run this from the directory containing this file — the same directory
`apply.patch` sits in:

```bash
git -C /Users/iklamer/ai-native-systems/VSCodeProjects/vllm checkout 83ad767eed3be3ee7f2df63be693bfaca5c7c922
git -C /Users/iklamer/ai-native-systems/VSCodeProjects/vllm apply --check "$PWD/apply.patch" && git -C /Users/iklamer/ai-native-systems/VSCodeProjects/vllm apply "$PWD/apply.patch"

# the recorded correctness oracles — run them all on a machine that can:
pytest tests/v1/worker/test_gpu_model_runner.py
pytest tests/v1/attention/test_attention_backends.py
```

If the patch does not apply cleanly, `git -C /Users/iklamer/ai-native-systems/VSCodeProjects/vllm apply -3 "$PWD/apply.patch"`
falls back to a three-way merge. Without git, `patch -p1 < apply.patch` works
from the repo root.
