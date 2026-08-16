# resolve_seq_and_query_len / find_seq_idx

[← vllm/v1/attention](../vllm_v1_attention.md)

- **File:** [`vllm/v1/attention/ops/triton_attention_helpers.py`](vllm/v1/attention/ops/triton_attention_helpers.py) (lines 44–106)
- **Symbol:** `resolve_seq_and_query_len / find_seq_idx`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_attention-0022`

## Description
Maps each unified-attention program id to its sequence index and local query-block index inside Triton kernels.

## Current approach
Every program performs a binary search over query_start_len_ptr via find_seq_idx, then reloads prefix entries and sequence length to derive q_block_local_idx and per-sequence lengths.

## Estimated impact explanation
This executes for every Triton attention program and again in reduce_segments. Removing repeated prefix searches can reduce kernel overhead and improve median TPOT for decode-heavy agentic batches.

## Evolve rationale
Concrete construct is the per-program binary search loop in find_seq_idx. Alternatives include precomputed q-block-to-sequence metadata, exact launch grids, or specialized decode paths. Correctness oracle is equality of resolved sequence/block indices for randomized query_start_len prefixes plus full Triton attention output equality.

## Deep research proposals

### 1. Batch-descriptor dispatch to a decode-specialized attention kernel that bypasses find_seq_idx
- **Finding:** `find-vllm_v1_attention-0009` — *CUDA Graphs*
- **Source URL:** <https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Extend the unified Triton attention launcher so that uniform-decode batches (each sequence contributes exactly one query token / one BLOCK_Q q-block) dispatch to a specialized kernel variant that omits the binary search in resolve_seq_and_query_len / find_seq_idx (vllm/v1/attention/ops/triton_attention_helpers.py:44-106). In this variant, q_block_global_idx == seq_idx by construction, so seq_idx, q_block_local_idx=0, cur_start=tl.load(query_start_len_ptr+pid), cur_batch_query_len=1, and seq_len=tl.load(seq_lens_ptr+pid) are recovered directly from the program id with a single small pair of loads and no while-loop. Gate the dispatch on the CUDA-graph batch descriptor already used elsewhere in v1 (uniform-decode vs mixed/prefill) so the descriptor's stability under CUDA graph capture is preserved: uniform-decode descriptors take the fast, search-free kernel path, while mixed/prefill descriptors keep the current generic kernel that uses find_seq_idx. Apply the same specialization to the reduce_segments launcher, where use_q_block_mode=False search over the raw cumulative-length prefix also collapses to identity in the uniform-decode case. The mixed-batch path is unchanged; only the launcher's plan/select step and a new decode-specialized JIT entry are added.

**Proposal rationale.**

The finding explicitly recommends tightening fast-plan and metadata decisions around stable batch descriptors so CUDA-graph-compatible decode routes avoid redundant planning while mixed batches safely fall back. The candidate's evolve_rationale enumerates 'specialized decode paths' and 'exact launch grids' as the intended alternatives to the per-program binary search — a direct match. Uniform-decode is both the workload where find_seq_idx is pure overhead (the map is the identity) and the workload where CUDA graphs already stabilize the batch shape, so a descriptor-driven dispatch is a natural, low-risk way to eliminate the search and its associated prefix reloads on the exact multi-turn agentic decode batches the caller wants to speed up (median TPOT). The mixed-batch fallback keeps correctness for prefill and heterogeneous batches, matching the finding's 'safe fallback' guidance and preserving the correctness oracle (equality of resolved indices and full attention output) on all non-decode-only descriptors.

---

## Agent proposals

### 1. Precompute per-q-block metadata table in AttentionMetadata; kernel does a single indexed load
- **Agent:** claude

**Detailed description.**

In vllm/v1/attention/ops/triton_attention_helpers.py:44-106, replace the per-program binary search plus four `tl.load(query_start_len_ptr + …)` reloads inside `resolve_seq_and_query_len` / `find_seq_idx` with a single indexed load from a precomputed metadata tensor. Concretely, add a small SoA of int32 tensors of length `total_num_q_blocks` (computed today as `q.shape[0] // BLOCK_Q + num_seqs` in the launcher of vllm/v1/attention/ops/triton_unified_attention.py:966) holding, per q-block program id: `seq_idx`, `q_block_local_idx`, `cur_batch_in_all_start_index`, `cur_batch_query_len`, and `seq_len`. Build this table once per forward inside the Triton attention metadata builder (vllm/v1/attention/backends/triton_attn.py) — either on the host from `cu_seqlens_q`/`seqused_k` and copied to device, or via a tiny one-shot Triton kernel that runs a cumulative-length expansion — so the cost is amortized across every layer, kv-head, and 3D segment that reuse the same query layout for that step. Change `resolve_seq_and_query_len` to accept a `q_block_meta_ptr` (or five separate pointers) and simply do `seq_idx = tl.load(seq_idx_ptr + pid)`, etc., dropping the while-loop, the `query_start_len_ptr` reloads, and the `//BLOCK_Q + seq_idx` arithmetic. For `reduce_segments` (vllm/v1/attention/ops/triton_unified_attention.py:701-716 and the diffkv/int4 mirrors), add a second small table of length `q.shape[0]` mapping `query_token_idx → seq_idx` and load `seq_len` from a companion tensor. The launcher gates on whether the metadata was populated, so backends that do not construct it fall back to the current search — preserving correctness across the six call sites (triton_unified_attention.py, triton_unified_attention_diffkv.py, int4_per_token_head.py, chunked_prefill_paged_decode.py, and both `reduce_segments` variants).

**Novelty rationale.**

The listed deep_research_proposal (find-vllm_v1_attention-0009) is scoped strictly to uniform-decode batches: it adds a decode-specialized kernel variant gated by a CUDA-graph batch descriptor where the mapping `q_block_global_idx == seq_idx` collapses to identity, and falls back to the current binary-search kernel on every mixed or prefill batch. This proposal is orthogonal — it targets the general case (mixed prefill/decode, chunked prefill, non-graph-captured batches, and any BLOCK_Q that yields q_block_local_idx > 0), which the descriptor-dispatch approach explicitly leaves on the slow path. It also amortizes the cost across all layers of a forward pass by building the metadata once in the AttentionMetadata builder rather than re-deriving it in every kernel program, and it applies uniformly to `reduce_segments` and the diffkv/int4 variants without needing a new specialized JIT entry per variant. The evolve_rationale explicitly names 'precomputed q-block-to-sequence metadata' alongside 'specialized decode paths / exact launch grids' as distinct alternatives; the existing proposal chose the latter, this one chooses the former.

---

### 2. Add a seq-major Triton launch path for low-waste mixed batches
- **Agent:** codex

**Detailed description.**

Add an alternate launcher path for `resolve_seq_and_query_len` users in `vllm/v1/attention/ops/triton_unified_attention.py` that changes the program grid from flattened `q_block_global_idx` to `(q_block_local_idx, seq_idx, ...)` when the batch descriptor shows bounded per-sequence query-block counts. In that kernel variant, `seq_idx = tl.program_id(seq_dim)` and `q_block_local_idx = tl.program_id(block_dim)`, so `resolve_seq_and_query_len` in `vllm/v1/attention/ops/triton_attention_helpers.py:44-106` can skip `find_seq_idx` entirely and derive `cur_batch_in_all_start_index`, `cur_batch_query_len`, and `seq_len` from direct `query_start_len_ptr + seq_idx` / `seq_lens_ptr + seq_idx` loads. To avoid wasting too many programs on ragged prefill, gate this path with a simple planning heuristic such as `num_seqs * max_q_blocks_per_seq <= total_num_q_blocks * 1.10` and keep the current flattened binary-search path otherwise. This targets common agentic turns where most active sequences have one or a few q-blocks but the batch is not strictly uniform decode, preserving the generic fallback for highly ragged prefill.

**Novelty rationale.**

The deep-research proposal only covers the strict uniform-decode identity case where `q_block_global_idx == seq_idx`; this proposal handles non-uniform but low-raggedness mixed batches by changing the launch geometry so both `seq_idx` and local q-block index are native program ids. Agent A's proposal precomputes per-q-block lookup tables in metadata and then keeps a flattened launch; this proposal adds no per-q-block metadata tensor and instead trades a bounded amount of masked extra programs for removing the binary search through grid structure. It is therefore a distinct implementation path from both descriptor-dispatched decode specialization and precomputed q-block metadata.

---
