# FlashAttentionImpl._forward_with_dcp

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/backends/flash_attn.py`](vllm/v1/attention/backends/flash_attn.py) (lines 1177–1351)
- **Symbol:** `FlashAttentionImpl._forward_with_dcp`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0003`

## Description
Runs FlashAttention with Decode Context Parallelism by gathering query tensors, computing context attention, combining DCP states, computing new-token attention, and merging attention states.

## Current approach
Executes a fixed sequence of contiguous(), get_dcp_group().all_gather, workspace reservation, context flash_attn_varlen_func, dcp_combine, new-token flash_attn_varlen_func, and merge_attn_states. The FA2/FA4 context split is static and the local new-token kernel is not overlapped with the collective/context path.

## Estimated impact explanation
For DCP-enabled serving this path runs in every attention layer and can move TTFT and TPOT through better overlap or split selection. Impact is medium because the workload hint is agentic but not necessarily DCP-enabled.

## Evolve rationale
The optimization unit is communication/compute scheduling around all_gather, dcp_combine, workspace reuse, the two flash_attn calls, and should_split_fa2_dcp_context_attention. Correctness oracle is exact output equality against current DCP and non-DCP baselines, with split_dcp_context_queries and DCP tests covering shape logic.

## Deep research proposals

### 1. Adaptive Flash-Decoding split policy for the DCP context attention call
- **Finding:** `find-vllm_v1_attention-0002` — *Flash-Decoding for Long-Context Inference*
- **Source URL:** <https://princeton-nlp.github.io/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In FlashAttentionImpl._forward_with_dcp (vllm/v1/attention/backends/flash_attn.py:1177-1351), replace the current uniform use of attn_metadata.max_num_splits with an adaptive Flash-Decoding-style split policy on the DCP context flash_attn_varlen_func call (both the split_dcp_context branch via run_split_fa2_dcp_context_attention and the else-branch flash_attn_varlen_func at ~lines 1290-1312). Choose num_splits per invocation as a function of (a) the effective decode/prefill query token count (num_decode_tokens + num_context_prefill_tokens), (b) max_dcp_context_kv_len / mean dcp_context_kv_lens, and (c) SM occupancy (available SMs on the device / num_heads). Concretely: when num_decode_tokens dominates and batch*heads is small while max_dcp_context_kv_len is large, raise num_splits so KV-length parallelism saturates SMs; when queries are long (prefill-heavy) and heads*batch already saturate SMs, keep num_splits low to avoid the extra partial-LSE reduction. Apply this only to the context (long-KV) call; leave the new-token flash_attn_varlen_func at lines 1322-1342 using its existing splits since its cu_seqlens_k equals cu_seqlens_q and is short. Reuse existing merge_attn_states/dcp_combine for correctness; the change is confined to the num_splits argument selection and a small helper (e.g. _pick_dcp_context_num_splits) alongside should_split_fa2_dcp_context_attention. Correctness oracle remains exact output equality against the current DCP baseline (the FA kernel already reduces partial LSEs internally when num_splits > 1).

**Proposal rationale.**

The finding's core transferable idea is parallelizing decode attention over the KV sequence-length dimension and merging partial log-sum-exp states — exactly the regime the DCP context call operates in (long gathered-KV, small per-rank query token count in decode-heavy multi-turn agentic workloads). The candidate already exposes num_splits and merges LSEs, but the split count is a static max_num_splits precomputed per metadata rather than tuned to the specific ratio of query tokens vs. context KV length seen at this call site. Making the split adaptive addresses TPOT for small-batch long-context decode — the exact gap Flash-Decoding targets — while leaving the FA2/FA4 split-context path, workspace layout, all_gather, dcp_combine, and merge_attn_states unchanged, which keeps the correctness surface small.

---

### 2. Split new-token attention into overlap-friendly attention-state chunks and merge n-ary with DCP context state
- **Finding:** `find-vllm_v1_attention-0003` — *Attention States and Recursive Attention*
- **Source URL:** <https://docs.flashinfer.ai/tutorials/recursive_attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Refactor FlashAttentionImpl._forward_with_dcp (vllm/v1/attention/backends/flash_attn.py:1177-1351) to treat every partial attention result as a first-class (output, log-sum-exp) attention state and use an n-ary merge instead of the current fixed 2-way merge_attn_states at lines 1345-1351. Concretely: (1) after the get_dcp_group().all_gather(query) at line 1225, launch the context flash_attn_varlen_func (lines 1290-1312 or the split FA2 path at 1262-1288) and, on a separate CUDA stream, immediately begin the new-token flash_attn_varlen_func at lines 1322-1342, so that the local new-token kernel overlaps the collective/context path rather than being serialized after dcp_combine (lines 1314-1319). (2) Allow the new-token and context computations to each emit multiple (out, lse) chunks (for example, split the new-token attention by request or by KV-chunk when max_seqlen_q or num_decodes is large, or produce per-rank partial context states before dcp_combine reduces them) and merge them with a single n-ary attention-state combiner, exploiting the associativity noted in the FlashInfer recursive-attention tutorial ('the merge operator can be generalized to any number of attention state inputs'). (3) Fuse the transpose+contiguous of context_lse (line 1320) and the final merge into that n-ary combiner so LSE-layout gymnastics are done once. Correctness is preserved because every step still composes exact attention states; the equality oracle against the current DCP and non-DCP baselines and the split_dcp_context_queries / DCP tests continue to apply.

**Proposal rationale.**

The current implementation already uses attention states between the context and new-token paths but is locked into a strict serial schedule: all_gather -> context attn -> dcp_combine -> new-token attn -> 2-way merge_attn_states. The evolve_rationale explicitly names communication/compute scheduling around all_gather, dcp_combine, and the two flash_attn calls as the optimization unit, and flags that 'the local new-token kernel is not overlapped with the collective/context path.' The finding's associative n-ary merge property is exactly the invariant that licenses breaking that serial dependency: because any number of (out, lse) states can be combined losslessly, the new-token kernel can be issued on a separate stream in parallel with the collective/context branch and reconciled at the end without changing outputs. This directly attacks TTFT/TPOT on the DCP path (agentic multi-turn workload) by hiding all_gather + dcp_combine latency behind local new-token compute and by allowing finer-grained chunking of the new-token or context computations when they are the critical path. This is a concrete transferable idea beyond the current 2-way merge, not merely a restatement of it.

---

### 3. Occupancy-aware split policy for DCP context/new-token attention
- **Finding:** `find-vllm_v1_attention-0005` — *Multi-Head, Multi-Query, and Group-Query Attention*
- **Source URL:** <https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Replace the static FA2/FA4 split decision in FlashAttentionImpl._forward_with_dcp (vllm/v1/attention/backends/flash_attn.py:1177-1351), currently gated by should_split_fa2_dcp_context_attention with fixed constants, with an occupancy-aware policy that decides whether to split the DCP context attention and how to schedule the two flash_attn_varlen_func calls. The policy would take batch size (num_decode_tokens after all_gather), num_heads / num_kv_heads (GQA ratio), device SM count, and the context-vs-new-token token counts, and only enable the split path when a single-block-per-head kernel would leave SMs idle — mirroring TRT-LLM's multi-block generation heuristic. When occupancy is already high (large batch, many KV heads, small context), fall back to a single unsplit context call to avoid the extra dcp_combine/merge_attn_states overhead. The heuristic thresholds would be encoded once at engine init using torch.cuda.get_device_properties(...).multi_processor_count and the layer's head configuration, so per-step cost is a cheap comparison.

**Proposal rationale.**

The candidate's evolve_rationale explicitly calls out should_split_fa2_dcp_context_attention and the static FA2/FA4 context split as an optimization unit, and notes the local new-token kernel is not overlapped with the collective/context path. The TRT-LLM finding describes exactly this class of decision — enabling a multi-block kernel path only when low occupancy makes one-block-per-head inefficient, using batch size, head count, and SM count as inputs. Porting that occupancy-aware framing to the DCP split decision addresses a concrete gap (fixed constants that cannot adapt to the agentic multi-turn workload's variable batch/context shapes) and can improve TTFT/TPOT by avoiding split overhead when it does not pay for itself and enabling it when idle SMs would otherwise be wasted. The correctness oracle (exact output equality vs current DCP/non-DCP baselines) is unchanged because both branches already exist and are covered by split_dcp_context_queries and DCP tests.

---

### 4. Replace DCP all-gather with Ulysses-style all-to-all head redistribution in _forward_with_dcp
- **Finding:** `find-vllm_v1_attention-0008` — *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*
- **Source URL:** <https://raw.githubusercontent.com/deepspeedai/DeepSpeed/master/blogs/deepspeed-ulysses/README.md>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In FlashAttentionImpl._forward_with_dcp (vllm/v1/attention/backends/flash_attn.py:1177-1351), replace the current sequence-sharded all_gather of query tensors followed by a context flash_attn_varlen_func + dcp_combine pipeline with a DeepSpeed-Ulysses-style redistribution when num_heads is divisible by the DCP world size. Concretely: before the context attention block, issue an all-to-all that converts the sequence-sharded (Q, K, V) into head-sharded tensors so each DCP rank owns a full-sequence view of num_heads / dcp_world_size heads; call a single flash_attn_varlen_func on the full sequence for that head slice (removing the need for dcp_combine and the two-phase context/new-token split logic driven by should_split_fa2_dcp_context_attention); then all-to-all the output back to sequence-sharded layout before merge_attn_states. Keep the existing new-token flash_attn_varlen_func path as the fallback when heads do not divide evenly, and preserve the workspace reservation and merge_attn_states call so the correctness oracle (exact output equality against current DCP and non-DCP baselines, split_dcp_context_queries tests) still holds. Gate the new path behind a config flag that also checks head divisibility and the presence of an efficient all-to-all in get_dcp_group().

**Proposal rationale.**

The candidate's current gather-heavy DCP path communicates O(seq_len * num_heads * head_dim) bytes per rank via all_gather, then relies on dcp_combine to merge partial attention states — a pattern whose communication volume grows with sequence length and dominates long-context TPOT. DeepSpeed-Ulysses's finding shows that swapping sequence-sharding for head-sharding via all-to-all lets each rank run full-sequence attention on fewer heads with communication volume that scales as O(seq_len * num_heads * head_dim / world_size) per rank, and it eliminates the need for a partial-state combine because each rank computes exact attention for its head slice. For the multi-turn agentic workload (long, growing KV context) targeted by the caller objective, that directly attacks the TPOT bottleneck the evolve_rationale identifies (communication/compute scheduling around all_gather + dcp_combine) and also removes the FA2/FA4 static-split decision, simplifying the schedule the candidate flags as not overlapped. The technique is transferable because DCP already partitions along the sequence dimension and the attention kernel in use (flash_attn_varlen_func) supports the head-parallel layout Ulysses requires.

---

## Agent proposals

### 1. Token-chunked pipelining of the DCP head-dim all_gather with the context flash_attn call
- **Agent:** claude

**Detailed description.**

In FlashAttentionImpl._forward_with_dcp (vllm/v1/attention/backends/flash_attn.py:1177-1351), replace the monolithic `query_across_dcp = get_dcp_group().all_gather(query, dim=1)` at line 1225 followed by a single context `flash_attn_varlen_func` (lines 1290-1312, or the split_dcp_context path at 1262-1288) with a token-chunk pipeline over the query token dimension. Concretely: (1) partition the local `query` tensor along dim=0 (token dim) into K chunks aligned to request boundaries in `cu_seqlens_q` so causal / seqused_k semantics stay intact per chunk; (2) issue chunk-wise async collectives via a ring/staged all_gather (either NCCL send/recv pairs on a communication CUDA stream, or `dist.all_gather_into_tensor` sub-invocations for slices of the head axis) so ring stage k+1 for the head-gather is in flight while the context `flash_attn_varlen_func` kernel executes on chunk k on the compute stream; (3) collect per-chunk (context_attn_out, context_lse) partial states into the existing `dcp_context_out_workspace` region using the offsets already implied by `cu_seqlens_q`, then perform a single `dcp_combine` on the concatenated tensor (equivalent to the current call at lines 1314-1319 because dcp_combine reduces along the DCP head-shard axis, not the token axis); (4) leave the new-token `flash_attn_varlen_func` (lines 1322-1342) and the final `merge_attn_states` (lines 1345-1351) unchanged. Chunk count K is chosen from `max_query_len`, `num_reqs`, and dcp_world_size so each chunk's collective step roughly matches a chunk kernel's runtime. Gate the path behind a config flag with a fallback to today's monolithic all_gather when K==1 or when chunking would break causal masking within a request (e.g. very long prefill requests whose tokens cannot be split without a mid-request causal boundary). Correctness follows because the head-dim gather is independent per token chunk and the context kernel is per-request/per-token; the (out, lse) states from each chunk are exactly what today's single-shot call would emit for that token slice, so `dcp_combine` and `merge_attn_states` operate on identical data.

**Novelty rationale.**

None of the four listed deep_research_proposals restructures the all_gather itself into a token-chunked producer/consumer pipeline with the context kernel. Finding 0003 overlaps the *new-token* kernel with the collective/context branch but keeps the all_gather monolithic and still serializes the context flash_attn call strictly after all_gather completes; this proposal instead makes the collective and the context kernel co-execute across ring stages via token-axis chunking, which is orthogonal and composes with it. Finding 0008 (Ulysses) replaces the head-dim all_gather with an all-to-all redistribution and requires head divisibility by dcp_world_size; the present proposal preserves the existing head-gather topology and improves its scheduling, so it applies when Ulysses cannot (indivisible head counts, no efficient all-to-all in the DCP group, or as a fallback path) and does not remove `dcp_combine`. Findings 0002 and 0005 only alter num_splits selection / FA2-vs-FA4 dispatch on an unchanged serial pipeline and do not chunk the collective or overlap it with the kernel. The concrete artefacts introduced here — chunk sizing keyed to `cu_seqlens_q`, staged NCCL calls on a comm stream, per-chunk writes into `dcp_context_out_workspace`, and a single downstream `dcp_combine` — are absent from all four.

---

### 2. Compact zero-context requests out of the DCP context attention call
- **Agent:** codex

**Detailed description.**

In `FlashAttentionImpl._forward_with_dcp` (`vllm/v1/attention/backends/flash_attn.py:1177-1351`), add a fast compaction path for mixed batches where `attn_metadata.max_dcp_context_kv_len > 0` but some requests have `dcp_context_kv_lens == 0`. Today the context `flash_attn_varlen_func` is invoked over all query tokens once any request has remote DCP context, so decode/prefill tokens for requests with no DCP context still participate in the gathered-query context path before their state is merged with local new-token attention. Build a filtered request list from `dcp_context_kv_lens > 0`, compact the corresponding token ranges from `query_across_dcp`, construct filtered `cu_seqlens_q`, `dcp_context_kv_lens`, and `block_table`, and run the existing split or unsplit context attention only on that compacted batch. Scatter the resulting `(context_attn_out, context_lse)` back into full token order, initializing skipped tokens as the neutral attention state (`out=0`, `lse=-inf`) before the existing `dcp_combine` and `merge_attn_states`. Keep the current full-batch path when all requests have nonzero context, and add tests with mixed zero/nonzero DCP context lengths to verify exact equality with the current implementation.

**Novelty rationale.**

The listed proposals optimize split selection, occupancy policy, all-to-all redistribution, overlapping the new-token kernel, or token-chunk pipelining of the all_gather/context kernel. None removes unnecessary context-attention work for requests whose DCP context length is zero inside an otherwise DCP-enabled batch. This proposal preserves the existing collective topology, split policy, `dcp_combine`, and final merge, but reduces the effective query batch for the long-context call by exploiting a per-request sparsity case common in mixed multi-turn serving batches.

---
