# detach_zero_copy_from_model_runner_output

[← v1.executor](../v1.executor.md)

- **File:** [`vllm/v1/executor/ray_utils.py`](vllm/v1/executor/ray_utils.py) (lines 195–234)
- **Symbol:** `detach_zero_copy_from_model_runner_output`
- **Kind:** function
- **Estimated impact:** low
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0007`

## Description
Ray zero-copy detachment helper for ModelRunnerOutput.logprobs. It copies read-only NumPy arrays out of Ray shared-memory channel buffers so later channel reads do not block on retained references.

## Current approach
The function returns immediately when output.logprobs is None. Otherwise it allocates an inner _copy_if_readonly helper, checks token_ids/logprobs/ranks independently, copies each read-only ndarray, and reconstructs output.logprobs only if at least one field changed.

## Estimated impact explanation
This only matters for Ray responses with logprobs, and array copies dominate when read-only buffers are present. It still sits inside the per-token Ray output path, so reducing avoidable Python allocation can marginally improve median TPOT for logprobs-heavy workloads.

## Evolve rationale
The concrete optimization unit is the per-output logprobs walk at lines 217-233. Evolution can avoid the per-call nested helper allocation, combine ndarray/writeability checks, skip reconstruction for known writable logprob containers, or use reusable writable buffers where ownership is clear. Correctness oracle: Ray logprobs tests and sampler/logprob tests must preserve dtype, shape, values, mutability safety, and the invariant that Ray SHM read-only arrays are not retained across iterations.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Gate the detach call at the executor with a config-derived flag, so the per-output Ray hot path skips the function entirely when SHM zero-copy cannot produce read-only NumPy views
- **Agent:** claude

**Detailed description.**

At the call sites in vllm/v1/executor/ray_executor.py:468 and :481, the executor unconditionally invokes detach_zero_copy_from_model_runner_output for every ModelRunnerOutput coming back from the compiled Ray DAG. Whether Ray's SHM channels actually produce read-only NumPy views for ModelRunnerOutput.logprobs is a property of the executor's configuration (Ray version, compiled DAG enabled, channel transport / SHM enabled, output_rank routing, presence of a connector) and does not change per iteration. Compute this once during RayDistributedExecutor initialization into a boolean self._needs_zero_copy_detach (e.g., during _compiled_ray_dag construction, by inspecting the channel implementation chosen for the output ref or by a one-shot probe of the first returned output's logprobs.token_ids.flags.writeable when output.logprobs is not None) and guard both call sites with `if self._needs_zero_copy_detach: detach_zero_copy_from_model_runner_output(output)`. In deployments where the DAG does not return SHM-backed read-only arrays (e.g., when output is materialized through pickle/serialization, when compiled DAG is disabled, or when SHM is not the channel transport), this eliminates the entire function call — Python dispatch, the `output.logprobs is None` branch, the tuple unpack at vllm/v1/executor/ray_utils.py:217, and (critically for the candidate's evolution unit) the per-call definition of the inner `_copy_if_readonly` closure at lines 219-222. For multi-turn agentic workloads where logprobs may not even be requested, the saved cost is dominated by the function-call frame and per-iteration closure allocation, not by NumPy copies. When the flag is true the existing function body is preserved verbatim, so correctness is unchanged; the Ray logprobs / sampler tests cover both branches by exercising configurations where the SHM path is and isn't engaged. Optionally, perform a self-correction on the first observed output: if the flag was set False but a subsequent output exposes a read-only ndarray, fall back to True permanently and copy that output, so a misjudged probe never causes a Ray channel stall.

**Novelty rationale.**

The candidate has zero existing deep_research_proposals. The candidate's own evolve_rationale targets micro-optimizations *inside* the function body (avoiding the nested helper allocation, combining ndarray/writeability checks, reusable writable buffers, skipping reconstruction). This proposal is orthogonal: it leaves the function body untouched and instead eliminates the call site entirely in configurations that cannot produce SHM-backed read-only views, keyed on a single boolean computed once at executor init. None of the rationale bullets address call-site gating or executor-level dispatch state, so the savings (function-frame + closure allocation per output) compose with — rather than overlap — any in-body evolution that may later optimize the slow path.

---

### 2. Copy retained logprob slices lazily instead of detaching the whole Ray batch
- **Agent:** codex

**Detailed description.**

Change `vllm/v1/executor/ray_utils.py:217-233` so read-only Ray-backed `LogprobsLists` arrays are not eagerly copied in full. When `detach_zero_copy_from_model_runner_output` sees any read-only ndarray, wrap `output.logprobs` in a private `LogprobsLists` subclass or adapter that preserves the tuple fields but overrides `slice_request()`: it slices first, then copies only the read-only sliced `logprob_token_ids`, `logprobs`, and `sampled_token_ranks`, returning a plain detached `LogprobsLists`. The scheduler only retains `logprobs.slice_request(...)` for requests that actually need returned sample logprobs, so mixed multi-turn batches avoid copying rows for requests with no logprob output while still ensuring any retained arrays no longer point at Ray SHM. Keep the current eager copy as a fallback if a direct full-container consumer is introduced, and add tests that retained slices are writable/detached, writable arrays remain untouched, and Ray SHM read-only arrays are not held across iterations.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes executor-level call-site gating to skip the detach function when zero-copy SHM cannot occur; this proposal applies when the detach path is needed and changes copy granularity from full-batch eager copies to retained-slice lazy copies. It also does not duplicate the candidate rationale's in-body micro-optimizations such as removing the nested helper, combining checks, skipping reconstruction, or reusing writable buffers.

---
