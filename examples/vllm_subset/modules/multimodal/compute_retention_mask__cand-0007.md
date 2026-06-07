# compute_retention_mask

[← multimodal](../multimodal.md)

- **File:** [`vllm/multimodal/evs.py`](vllm/multimodal/evs.py) (lines 38–92)
- **Symbol:** `compute_retention_mask`
- **Kind:** function
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0007`

## Description
Builds the EVS boolean retention mask by ranking per-token temporal dissimilarity between adjacent video-frame embeddings.

## Current approach
The function computes cosine similarity, prepends a sentinel first-frame dissimilarity tensor with torch.cat, flattens the result, performs a stable full torch.argsort over all tokens, then marks the first retain_num_tokens indices in a bool mask.

## Estimated impact explanation
When EVS is enabled, this runs once per video before the LLM consumes visual tokens. Replacing a full sort and extra allocation with selection reduces video preprocessing latency and helps downstream TPOT by preserving the pruning contract with less front-end overhead.

## Evolve rationale
The full stable argsort at line 81 is an algorithmic replacement target. Headroom includes a top-k/selection implementation that preserves current stable tie-breaking, unconditional first-frame retention without materializing a 255-sentinel cat, and scatter into a preallocated bool tensor. Oracle: model EVS tests and Qwen/Nano-Nemotron callers must see identical mask shape, first-frame retention, retained-token count from compute_retained_tokens_count, and deterministic tie behavior for equal dissimilarities.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Add q/T fast paths and reduce top-k input to non-first-frame tokens
- **Agent:** claude

**Detailed description.**

Restructure compute_retention_mask in vllm/multimodal/evs.py:38-92 with two layered optimizations beyond a simple argsort→topk swap.

1) Early-exit fast paths before any similarity work:
   - If T == 1, return torch.ones(tokens_per_frame, dtype=torch.bool, device=video_embeds.device). With one frame there is no inter-frame dissimilarity and compute_retained_tokens_count's min_num_tokens floor guarantees the whole frame is retained.
   - Compute retain_num_tokens up front (already cheap, integer math). If retain_num_tokens >= T * tokens_per_frame (e.g. q very small or rounding pushes us to the floor), return torch.ones(...) immediately. This skips the cosine_similarity, the cat, the sort, and the scatter for short videos and low-q multi-turn turns where pruning is a no-op.

2) Eliminate the 255-sentinel cat AND shrink the selection input. Instead of allocating dissimilarity over T frames and then selecting from T*tokens_per_frame elements:
   - Compute dissimilarity only for frames 1..T-1 (shape (T-1, h, w)) and flatten to (T-1)*tokens_per_frame.
   - Allocate retention_mask = torch.zeros(T * tokens_per_frame, dtype=torch.bool, device=...).
   - Pre-set retention_mask[:tokens_per_frame] = True (first-frame retention is invariant; no need to encode it as a +inf sentinel routed through a sort).
   - Run torch.topk(dissimilarity_flat, k=retain_num_tokens - tokens_per_frame, sorted=False) on the (T-1)*tokens_per_frame buffer, then scatter True at offset tokens_per_frame + topk.indices into retention_mask.
   - This reduces the selection-input size by tokens_per_frame elements and removes the torch.ones_like allocation+cat, plus the redundant reshape→view round-trip currently at lines 89-91.

Tie-determinism: torch.topk does not guarantee stable tie-breaking like the current stable argsort. To preserve the oracle (deterministic tie behavior for equal dissimilarities required by EVS tests and Qwen/Nano-Nemotron callers), break ties by a fixed-position penalty: dissimilarity_for_topk = dissimilarity_flat - eps * arange(N) / N with eps small enough not to cross adjacent dissimilarity values (e.g. eps = (max-min) * 1e-6, or use a lexicographic int64 key). Document the tie-breaking equivalence in a one-liner. If the equivalence cannot be made bit-exact, fall back to torch.sort(..., stable=True) on only the (T-1)*tokens_per_frame buffer—still strictly smaller than today's full sort.

Validation: existing tests in tests/multimodal covering EVS plus the Qwen/Nano-Nemotron paths must produce identical masks (shape, first-frame all-True, total True count == compute_retained_tokens_count, identical tie behavior). Benchmark with a representative T=16, H=W=24, spatial_merge_size=2 video at q=0.5 and q=0.0 to confirm TTFT reduction in the agentic multi-turn case where short videos and low-q turns are common.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate. Beyond the evolve_rationale's general hint of 'top-k + scatter + skip the sentinel cat', this proposal adds three specific items the rationale does not articulate: (a) two algorithmic short-circuits (T==1 and retain==total) that bypass cosine_similarity, the cat, and the selection entirely—directly relevant to the multi-turn agentic workload where many turns reuse the same video at low effective pruning; (b) shrinking the top-k input by tokens_per_frame elements by excluding the first frame from the selection set rather than just from the cat; (c) an explicit deterministic-tie-breaking strategy (arange penalty or lexicographic key) that lets torch.topk stand in for stable argsort while preserving the oracle, plus removal of the redundant reshape→view round-trip on lines 89-91.

---

### 2. Select the smaller side of the EVS cutoff
- **Agent:** codex

**Detailed description.**

Update `compute_retention_mask` in `vllm/multimodal/evs.py` so the selection step chooses whichever side of the cutoff is smaller. After deriving `retain_num_tokens`, compute `keep_after_first = retain_num_tokens - tokens_per_frame` and `drop_after_first = (T - 1) * tokens_per_frame - keep_after_first`. For low pruning rates where `drop_after_first < keep_after_first`, initialize the mask as all-true, then select the `drop_after_first` lowest-dissimilarity non-first-frame tokens and set only those positions to false. For higher pruning rates, keep the normal highest-dissimilarity retention path. Preserve the current stable descending `argsort` contract at cutoff ties: if dropping from the bottom, equal-dissimilarity boundary tokens should drop later flat indices first, so implement the bottom branch with a threshold plus reverse-index tie completion, or an equivalent lexicographic key `(dissimilarity ascending, flat_index descending)`. Add focused parity tests comparing old and new masks for small/large `q`, especially all-equal dissimilarity tensors where the boundary cuts through a tie group.

**Novelty rationale.**

There are no deep_research_proposals for this candidate. Agent A proposes fast paths, removing the first-frame sentinel from the top-k input, and deterministic top-k tie handling, but it always frames selection as choosing retained non-first-frame tokens. This proposal is different: for near-no-op pruning it selects the smaller complement set of pruned tokens and flips those bits, reducing selection work from roughly `(1 - q)` of non-first tokens to `q` of them while preserving the same mask semantics.

---
