# GPUModelRunner.prepare_inputs

[← vllm/v1/worker](../vllm_v1_worker.md)

- **File:** [`vllm/v1/worker/gpu/model_runner.py`](vllm/v1/worker/gpu/model_runner.py) (lines 971–1149)
- **Symbol:** `GPUModelRunner.prepare_inputs`
- **Kind:** method
- **Estimated impact:** high
- **Id:** `cand-vllm_v1_worker-0023`

## Description
New GPU runner per-step input preparation for request ordering, idx mappings, scheduled-token arrays, query_start_loc, prefill inputs, positions, seq_lens, logits indices, and InputBatch assembly.

## Current approach
Sorts scheduler request IDs in Python, builds multiple NumPy arrays from dict iterators, performs separate H2D copies for idx_mapping and query_start_loc/cu_num_logits, then launches separate kernels for prefill token copy, positions/seq_lens, and sampled/draft token combination.

## Estimated impact explanation
This is the new runner's central per-step CPU/GPU prep path; coalescing scheduler arrays and reducing H2D/kernel count directly lowers median TPOT and TTFT in multi-turn agent decode.

## Evolve rationale
The concrete constructs are sort_batch_req_ids, np.fromiter array construction, async_copy_to_gpu calls, and the prepare_* kernel sequence. tests/v1/worker/test_gpu_input_batch.py, tests/v1/worker/test_gpu_model_runner.py, and attention backend tests validate the InputBatch contract.

## Deep research proposals

### 1. Eliminate host-blocking seq_lens_cpu dependency in prepare_inputs to enable async spec-decode overlap
- **Finding:** `find-vllm_v1_worker-0002` — *[Performance]: Fully Async Spec-Decoding | Make `seq_lens_cpu` in CommonAttentionMetadata optional*
- **Source URL:** <https://github.com/vllm-project/vllm/issues/29134>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In GPUModelRunner.prepare_inputs (vllm/v1/worker/gpu/model_runner.py:971-1149), refactor the per-step input preparation so that constructing CommonAttentionMetadata and its consumers no longer requires a materialized seq_lens_cpu tensor. Concretely: (1) treat seq_lens_cpu as an optional field on CommonAttentionMetadata and stop unconditionally producing it during the np.fromiter/async_copy_to_gpu sequence used to build seq_lens; (2) keep seq_lens resident on device and derive per-request upper bounds / query_start_loc offsets from device-side scans or precomputed slot tables rather than host-side scheduler dict iterations that force a sync; (3) route the prepare_* kernel chain (prefill token copy, positions/seq_lens, sampled/draft token combination) so that spec-decode verification of multiple drafted tokens reads device seq_lens directly, allowing prepare_inputs for step N+1 to be enqueued while step N's forward is still executing. Preserve a CPU fallback only for attention backends that still require it, guarded behind an explicit request from the backend, and update tests/v1/worker/test_gpu_input_batch.py and test_gpu_model_runner.py assertions on the InputBatch contract accordingly.

**Proposal rationale.**

The candidate's current approach performs multiple separate H2D copies (idx_mapping, query_start_loc/cu_num_logits) and constructs seq_lens via host-side np.fromiter over scheduler dicts, which forces the host to know sequence lengths before it can build attention metadata. This synchronization point is exactly what the linked issue identifies as the barrier to fully async spec decoding: as long as CommonAttentionMetadata requires seq_lens_cpu, the CPU cannot begin preparing step N+1's inputs while the GPU is still verifying drafted tokens for step N. Making seq_lens_cpu optional and letting metadata consumers read device seq_lens removes that barrier at the exact location (prepare_inputs) where the sync is introduced, directly lowering median TPOT in the multi-turn agentic decode workload where spec-decode verification of several drafted tokens compounds the cost of every avoidable host/device sync. The finding contributes a concrete, transferable structural change (optional CPU metadata + device-resident upper bounds) rather than merely restating that prepare_inputs is slow.

---

### 2. Coalesce per-step H2D transfers and move CPU idx-gathers onto GPU in prepare_inputs
- **Finding:** `find-vllm_v1_worker-0003` — *Model Runner V2 Design Document*
- **Source URL:** <https://docs.vllm.ai/en/v0.17.0/design/model_runner_v2/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor GPUModelRunner.prepare_inputs (vllm/v1/worker/gpu/model_runner.py:971-1149) to more fully realize the MRV2 principle that per-step inputs should be gathered from GPU-resident persistent state with minimal CPU/H2D overhead:

1. Coalesce H2D copies. Today the method issues at least three separate async_copy_to_gpu calls per step: idx_mapping (line 994), cu_num_logits (line 1024, when draft tokens exist), and query_start_loc (line 1040). Pack idx_mapping_np, cu_num_logits_np, and query_start_loc_np into a single pinned CPU staging buffer (a slice of a preallocated input_buffers scratch tensor sized max_num_reqs*3+1) and issue one H2D copy per step, then take views into the destination. This preserves the existing InputBatch contract while reducing PCIe transactions on the critical TPOT path.

2. Move CPU-side idx-gathers to GPU. Lines 1043-1046, 1099, and 1111 currently perform NumPy fancy-indexing (self.req_states.prefill_len.np[idx_mapping_np], num_computed_prefill_tokens_np[idx_mapping_np], num_computed_tokens_np_np[idx_mapping_np], max_seq_len[idx_mapping_np]) against CPU mirrors of persistent state. Per the finding's guidance that 'large state tensors are mostly stored on GPU memory, so gather runs in parallel on the GPU with low overhead,' fuse these gathers into a single Triton/torch kernel that reads idx_mapping (already on GPU) and gathers the required scalars from the GPU-resident req_states tensors, producing an is_prefilling mask and seq_lens_cpu_upper_bound on GPU. Only D2H the small scalars (or the is_prefilling mask reduction) that downstream Python control flow strictly needs — e.g., the np.any(is_prefilling_np) branch on line 1049 can become a single-byte D2H flag rather than a full CPU array.

3. Fuse the prep kernel sequence. The current three kernel launches (prepare_prefill_inputs, prepare_pos_seq_lens, combine_sampled_and_draft_tokens) all read idx_mapping and query_start_loc and write into input_buffers. Where their write footprints do not overlap, launch them as a single fused kernel (or at minimum a single stream with no interleaved host sync) to cut launch overhead and enable persistent-kernel scheduling.

Existing tests (tests/v1/worker/test_gpu_input_batch.py, tests/v1/worker/test_gpu_model_runner.py, attention backend tests) validate the InputBatch contract and should continue to pass unchanged; add a microbenchmark under benchmarks/ that measures per-step prepare_inputs wall time for representative multi-turn agent decode batches.

**Proposal rationale.**

The candidate already reflects the top-level MRV2 design (persistent req_states vs per-step input_buffers, stable idx_mapping rows, GPU-resident state), so the finding's *concept* is not a gap. What the finding contributes over the current implementation is its specific emphasis that gathers should run on the GPU because state is GPU-resident, and that per-step bookkeeping should avoid tensor-wide CPU work. Applied to this method, that principle identifies three concrete inefficiencies still present on the critical TTFT/TPOT path: (a) three separate H2D copies per step where one coalesced transfer would suffice, (b) four CPU-side fancy-index gathers (.np[idx_mapping_np]) that duplicate work already implicitly available on GPU via idx_mapping, and (c) three sequential kernel launches that share inputs. For the caller's multi-turn agentic decode workload — where prepare_inputs runs on every step at small-to-moderate num_reqs and its latency is directly additive to TPOT — reducing per-step CPU work, PCIe transactions, and kernel launches is a plausible, high-impact win. The change is bounded to a single method with well-tested public contract (InputBatch), so it is transferable without touching the scheduler, attention backends, or persistent-state layout.

---

## Agent proposals

### 1. Precompute per-step prepare_inputs metadata in a scheduler-side worker thread overlapped with model forward
- **Agent:** claude

**Detailed description.**

In GPUModelRunner.prepare_inputs (vllm/v1/worker/gpu/model_runner.py:971-1149), split the method into a pure-Python 'plan' phase and a 'materialize' phase, and run the plan phase for step N+1 concurrently with the model forward of step N using a dedicated CPU worker thread (or asyncio task) bound to the scheduler output pipeline.

Concretely: (1) Factor out everything that only depends on the SchedulerOutput dict and CPU-mirror state — sort_batch_req_ids over scheduled_req_ids, construction of idx_mapping_np, num_scheduled_tokens_np, cu_num_logits_np, query_start_loc_np, the .np[idx_mapping_np] gathers against req_states mirrors (prefill_len, num_computed_prefill_tokens, num_computed_tokens, max_seq_len), and the is_prefilling mask — into a `_plan_step(scheduler_output) -> PreparedPlan` free function that produces a plain dataclass of NumPy arrays plus Python scalars. (2) Keep the H2D copies and prepare_* kernel launches in a thin `_materialize_plan(plan) -> InputBatch` phase that runs on the model-execution thread. (3) Add a single-slot producer/consumer: right after the scheduler emits SchedulerOutput for step N+1 (while the runner is still awaiting step N's forward/sampler), submit `_plan_step` to a `concurrent.futures.ThreadPoolExecutor(max_workers=1)` owned by the runner; when prepare_inputs is called for step N+1, it just `future.result()`s the already-computed plan and proceeds to materialize. (4) Fall back to inline planning when the future is not ready (first step, recompilation, profile runs) so correctness is unchanged.

This overlaps the ~pure-Python CPU cost of sort_batch_req_ids, dict iteration into np.fromiter, and the fancy-index gathers with GPU-side model forward time, which on multi-turn agentic decode (small-to-moderate num_reqs, many short forward passes) is where prepare_inputs' Python overhead is a visible fraction of TPOT. It is a structural change to when the work runs, orthogonal to how the tensors are laid out or transferred. Update tests/v1/worker/test_gpu_model_runner.py to exercise both the inline-fallback and overlapped paths, and add an assertion that _plan_step is a pure function of its inputs (no reads from self.input_buffers or GPU tensors).

**Novelty rationale.**

The two existing deep_research_proposals both attack the *content* of prepare_inputs' work on the critical path: proposal 1 removes the seq_lens_cpu sync so the CPU can start step N+1 sooner, and proposal 2 coalesces H2D copies, moves gathers to GPU, and fuses the prep kernel sequence — both stay entirely on the model-execution thread and reduce the amount of work done there. This proposal is orthogonal: it does not change what tensors are built, where they live, or how many H2D transfers occur; it changes *when the CPU-side planning runs* by pipelining it into a scheduler-adjacent worker thread that overlaps step N+1's Python plan with step N's GPU forward. Neither existing proposal introduces threading, a plan/materialize split, or a scheduler-side producer for prepared metadata, and the two approaches compose (a smaller, faster plan phase from proposal 2 makes this overlap cheaper; removing the seq_lens_cpu sync from proposal 1 makes the plan phase truly independent of device state).

---

### 2. Emit decode-first ordered scheduler arrays to remove per-step sorting in prepare_inputs
- **Agent:** codex

**Detailed description.**

Extend the SchedulerOutput contract consumed by GPUModelRunner.prepare_inputs (vllm/v1/worker/gpu/model_runner.py:971-1149) so the scheduler provides the batch order and scheduled-token counts in the exact decode -> short_extend -> prefill order that sort_batch_req_ids currently reconstructs. Concretely, while vllm/v1/core/sched/scheduler.py is already deciding each request's num_new_tokens, populate two parallel fields such as ordered_scheduled_req_ids: list[str] and ordered_num_scheduled_tokens: np.ndarray/list[int], appending decode-length requests to a leading bucket and non-decode requests to buckets ordered by num_new_tokens. Then prepare_inputs can replace sort_batch_req_ids(...), map(num_tokens_per_req.__getitem__, req_ids), and the np.fromiter construction of num_scheduled_tokens with direct views/copies from SchedulerOutput. Keep the existing num_scheduled_tokens dict for compatibility during rollout, assert in debug/tests that the new ordered fields match sort_batch_req_ids, and update tests/v1/worker/test_gpu_model_runner.py or tests/v1/worker/test_gpu_input_batch.py to verify the decode-leading invariant because split_decodes_and_prefills relies on it.

**Novelty rationale.**

The existing deep_research proposals optimize host/device synchronization, H2D transfer coalescing, CPU idx gathers, and kernel fusion inside prepare_inputs; they do not change the scheduler-to-worker contract or remove the O(num_reqs log num_reqs) sort and dict lookups at the source. Agent A overlaps the same Python planning work with model forward by moving it into a worker thread, but still performs sort_batch_req_ids and np.fromiter in that plan phase. This proposal eliminates that work by having the scheduler emit the already-known ordered batch metadata once, making it orthogonal to both GPU-side coalescing and thread-based overlap.

---
