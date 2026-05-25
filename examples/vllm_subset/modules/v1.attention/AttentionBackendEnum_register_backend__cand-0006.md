# AttentionBackendEnum/register_backend

[← v1.attention](../v1.attention.md)

- **File:** [`vllm/v1/attention/backends/registry.py`](vllm/v1/attention/backends/registry.py) (lines 34–263)
- **Symbol:** `AttentionBackendEnum/register_backend`
- **Kind:** plugin_seam
- **Estimated impact:** medium
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0006`

## Description
Attention backend registration seam: default enum entries, CUSTOM placeholder, override dictionaries, and register_backend factory for runtime backend overrides.

## Current approach
Interface: vllm/v1/attention/backend.py:AttentionBackend. Reference implementations include vllm/v1/attention/backends/flash_attn.py:FlashAttentionBackend and vllm/v1/attention/backends/triton_attn.py:TritonAttentionBackend. Runtime selector: vllm/v1/attention/selector.py:_cached_get_attn_backend reads vllm_config.attention_config.backend, typically set by --attention-backend or attention_config.backend, and resolves through current_platform.get_attn_backend_cls plus AttentionBackendEnum.get_class/register_backend overrides.

## Estimated impact explanation
The registry itself does not reduce latency, but it is the supported route for adding a specialized backend for shared-prefix or decode-heavy agentic workloads. The measurable signal would be TTFT/TPOT from the backend selected through this seam.

## Evolve rationale
This is the actual backend registration and override site, not just an interface declaration. It is the supported insertion point for a workload-specialized in-repo attention backend. Correctness oracle: tests/test_attention_backend_registry.py validates registration overrides, while tests/v1/attention/test_attention_backends.py and tests/v1/attention/test_attention_backends_selection.py validate selection and output equivalence for supported backends.

## Deep research proposals

### 1. Register a POD-Attention fused prefill-decode backend via the registry seam
- **Finding:** `find-0001` — *POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference*
- **Source URL:** <https://arxiv.org/abs/2410.18038>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new entry to AttentionBackendEnum in vllm/v1/attention/backends/registry.py (lines 34-263) for a POD-style backend (e.g. AttentionBackendEnum.POD_ATTN) and wire it through register_backend so the runtime selector at vllm/v1/attention/selector.py:_cached_get_attn_backend can resolve it when --attention-backend=POD_ATTN (or attention_config.backend) is set. The new backend implements vllm/v1/attention/backend.py:AttentionBackend by dispatching a single fused kernel that co-schedules prefill and decode tiles from the same hybrid batch on one SM, instead of routing prefill tokens to a flash-attn-style kernel and decode tokens to a separate decode kernel. Concretely: (1) introduce vllm/v1/attention/backends/pod_attn.py modeled after FlashAttentionBackend/TritonAttentionBackend with a metadata builder that preserves the existing prefill/decode token partitioning the scheduler already produces, (2) register it as a default enum entry in registry.py (no need for the CUSTOM override path) so it becomes a first-class selectable backend, (3) extend tests/test_attention_backend_registry.py to cover the new enum entry and tests/v1/attention/test_attention_backends.py / test_attention_backends_selection.py to validate output equivalence against FlashAttention on hybrid batches. The registry candidate itself is unchanged in shape - this proposal uses it as the supported insertion point exactly as evolve_rationale describes.

**Proposal rationale.**

The candidate is explicitly the registration/override seam for adding a workload-specialized backend, and the caller context (multi-turn agentic, optimize median TTFT and TPOT) is the precise regime POD-Attention targets: continuous-batching steps where prefill chunks and decode tokens coexist and where independent kernels under-utilize SMs and trade TTFT against TPOT. POD-Attention's contribution - 'prefill and decode operations happen concurrently on the same multiprocessor' - is a concrete, transferable kernel-design idea that maps onto a new AttentionBackend implementation rather than a tweak to existing FlashAttention/Triton backends, and the registry is exactly the place to expose it without disturbing other backends or the selector contract. Existing oracles (registry test plus the two attention-backend test files) make correctness verification tractable, and the impact is bounded to the new backend path while still letting TTFT/TPOT be measured end-to-end through the documented selection flow.

---

### 2. Register a Hydragen-style shared-prefix attention backend via the registry seam
- **Finding:** `find-0002` — *Hydragen: High-Throughput LLM Inference with Shared Prefixes*
- **Source URL:** <https://arxiv.org/abs/2402.05099>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Use the registration seam in vllm/v1/attention/backends/registry.py (AttentionBackendEnum + register_backend, lines 34-263) to add a new in-repo backend that implements Hydragen-style decomposed attention: it splits each request's attention computation into (a) a shared-prefix pass over a KV span common to a group of sequences and (b) per-sequence unique-suffix passes, then combines the two with a log-sum-exp merge. Concretely: (1) add a new AttentionBackendEnum entry (e.g. HYDRAGEN) alongside the existing defaults, (2) implement the backend as a sibling of vllm/v1/attention/backends/flash_attn.py:FlashAttentionBackend conforming to vllm/v1/attention/backend.py:AttentionBackend, reusing FlashAttention kernels for both the shared-prefix GEMM-like pass (large query batch, single KV) and the suffix pass, (3) detect/track shared-prefix groups at the scheduler/attention-metadata layer (multi-turn agentic conversations with identical system prompts and tool schemas are the target case) and pass a shared_prefix_len plus group membership through the per-layer metadata, (4) register the backend so it can be selected end-to-end via --attention-backend hydragen, which flows through vllm/v1/attention/selector.py:_cached_get_attn_backend and AttentionBackendEnum.get_class. Correctness oracle is tests/v1/attention/test_attention_backends.py (output equivalence vs. FlashAttention on identical KV) and tests/test_attention_backend_registry.py (registration override). The registry itself is unchanged in behavior; only a new enum entry and class are added.

**Proposal rationale.**

The candidate explicitly identifies itself as the supported insertion point for a workload-specialized in-repo backend, and notes that the measurable signal is TTFT/TPOT through the backend selected via this seam. The caller context targets median TTFT and median TPOT for a multi-turn agentic workload, which is exactly the regime Hydragen addresses: repeated system prompts, tool schemas, and branching continuations cause large fractions of KV to be redundantly re-read on every decode step and re-attended during prefill of follow-up turns. Hydragen's decomposition into shared-prefix + unique-suffix attention (with LSE merge) reduces the bandwidth-bound cost of decode over long shared contexts and the prefill cost of repeated system/tool preambles, which directly attacks both TPOT (decode bandwidth) and TTFT (prefix prefill reuse). The registry seam lets this be added without touching the AttentionBackend interface or the selector, and existing FlashAttention kernels can be reused for the two sub-passes, so the change is concrete and transferable to this specific candidate rather than a generic restatement of its current approach.

---

### 3. Register a modality-aware permutation sparse prefill backend via AttentionBackendEnum
- **Finding:** `find-0003` — *MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention*
- **Source URL:** <https://proceedings.mlr.press/v267/li25aq.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Use the registry seam at vllm/v1/attention/backends/registry.py:34-263 (AttentionBackendEnum and register_backend) to add an in-repo attention backend that implements MMInference-style modality-aware permutation sparse attention for the prefill path. Concretely: (1) add a new enum entry (e.g., MM_SPARSE_PREFILL) alongside the existing defaults so it can be selected via --attention-backend or attention_config.backend resolved by vllm/v1/attention/selector.py:_cached_get_attn_backend; (2) implement the backend class against vllm/v1/attention/backend.py:AttentionBackend, mirroring the structure of vllm/v1/attention/backends/flash_attn.py and triton_attn.py, but replacing the dense prefill kernel with a permutation step that reorders tokens to expose the Grid-pattern locality of contiguous image/video token spans, applies head-specific offline-chosen sparse masks across modality boundaries, and inverse-permutes outputs; (3) keep decode unchanged by delegating to the existing FlashAttention path, so only prefill is specialized; (4) expose the per-head sparse pattern table and modality-boundary metadata as backend init args supplied from the multimodal input pipeline. Validation: extend tests/test_attention_backend_registry.py to assert the new enum/override is registrable, and reuse tests/v1/attention/test_attention_backends.py and tests/v1/attention/test_attention_backends_selection.py to verify selection plumbing and output equivalence to a dense reference within tolerance for text-only prompts (sparse pattern collapses to dense), with a separate multimodal correctness check for mixed image/text prompts.

**Proposal rationale.**

The candidate is explicitly the supported insertion point for adding a workload-specialized in-repo backend, and the caller objective is reducing media TTFT for a multi-turn agentic workload that plausibly includes long multimodal prompts. MMInference targets exactly the prefill phase for long-context VLMs and offers a concrete, weight-preserving recipe (token permutation to expose Grid sparsity, head-specific patterns, modality-boundary handling) that can be packaged behind the AttentionBackend interface without touching the selector contract. This addresses a real gap: none of the currently registered default backends are media-aware, so there is no way to exploit modality-locality for prefill speedups today. The registry seam, rather than a model-side change, is the right level because the technique is a kernel/scheduling specialization swappable per deployment, and the existing override tests confirm the seam supports this exact pattern.

---

### 4. Register a FlashInfer-derived agentic backend variant exposing block-sparse KV and load-balanced scheduling as policy
- **Finding:** `find-0004` — *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*
- **Source URL:** <https://openreview.net/forum?id=RXPofAsL8F>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new in-repo attention backend (e.g., AttentionBackendEnum.FLASHINFER_AGENTIC or a CUSTOM-registered entry) wired through vllm/v1/attention/backends/registry.py at lines 34-263, alongside the existing default enum entries and the register_backend factory. The backend class would live next to vllm/v1/attention/backends/flashinfer.py and subclass AttentionBackend (vllm/v1/attention/backend.py), but expose two pieces of FlashInfer behavior as first-class policy rather than fixed wrapper calls: (1) a composable/block-sparse KV layout selector that picks per-request KV format based on shared-prefix structure typical of multi-turn agent traces, and (2) a load-balanced scheduling step in the metadata builder that rebalances per-request work across SMs for dynamic agent batches while preserving the CUDA Graph capture path used by _cached_get_attn_backend in vllm/v1/attention/selector.py. Registration would go through the existing register_backend seam (so users opt in via --attention-backend or attention_config.backend), with overrides validated by tests/test_attention_backend_registry.py and output equivalence checked via tests/v1/attention/test_attention_backends.py and tests/v1/attention/test_attention_backends_selection.py. No change to AttentionBackend interface is required; the new entry only adds a row to the registry's default enum and a class registration.

**Proposal rationale.**

The candidate is explicitly the supported insertion point for a workload-specialized backend, and its evolve_rationale calls out shared-prefix/decode-heavy agentic workloads as the target. The finding contributes a concrete, transferable mechanism — block-sparse/composable KV formats plus load-balanced scheduling, with the explicit constraint that CUDA Graph compatibility is preserved ("load-balanced scheduling algorithm adjusts to dynamism of user requests while maintaining compatibility with CUDAGraph"). That is directly aligned with the caller objective of reducing median TTFT (shared-prefix reuse via composable KV) and median TPOT (load-balanced decode scheduling without losing CUDA Graphs) on multi-turn agentic batches. vLLM already integrates FlashInfer as a wrapper backend, so the new idea here is promoting these two behaviors from fixed wrapper calls to backend policy, which is exactly what the registry seam is designed to enable without disturbing other backends.

---

### 5. Register a Flash-Decoding-style long-context decode backend via the attention registry seam
- **Finding:** `find-0005` — *Flash-Decoding for long-context inference – PyTorch Blog*
- **Source URL:** <https://pytorch.org/blog/flash-decoding/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Use the registration seam in vllm/v1/attention/backends/registry.py (AttentionBackendEnum/register_backend, lines 34-263) to add an in-repo attention backend specialized for long-context, small-effective-batch decode. The backend should implement vllm/v1/attention/backend.py:AttentionBackend and adopt the Flash-Decoding parallelization strategy from the finding: split each query's KV range into chunks, run attention on each chunk in parallel to expose work along the KV-sequence-length dimension when batch and head parallelism leave the GPU under-occupied, and then merge partial outputs by combining their per-chunk softmax log-sum-exp state into the final output. Wire the new backend through the existing CUSTOM placeholder / override dictionaries so it can be selected via vllm_config.attention_config.backend (i.e. --attention-backend) and resolved by vllm/v1/attention/selector.py:_cached_get_attn_backend without changing the selector. Reference implementations to model the structure on are vllm/v1/attention/backends/flash_attn.py:FlashAttentionBackend and vllm/v1/attention/backends/triton_attn.py:TritonAttentionBackend. Validate via tests/test_attention_backend_registry.py for registration/override correctness and tests/v1/attention/test_attention_backends.py plus tests/v1/attention/test_attention_backends_selection.py for selection and numerical equivalence to an existing backend on supported shapes; gate enablement on long-context decode regimes where the KV-length split actually increases occupancy.

**Proposal rationale.**

The candidate explicitly identifies itself as the supported insertion point for a workload-specialized attention backend, and the caller's objective is to reduce median TPOT for a multi-turn agentic workload. Flash-Decoding directly targets the decode-time pathology of that workload: long accumulated KV histories and small effective batch leave standard split-K decode kernels under-occupying the GPU because parallelism comes only from batch and heads. The finding contributes a concrete, transferable mechanism (additional parallelism along the KV sequence-length dimension with log-sum-exp partial-output merging) that maps cleanly onto a new AttentionBackend class registered through this seam, leaving the selector and override surface unchanged and exercised by existing registry/selection tests.

---

### 6. Register a QUEST query-aware sparse decode backend via the attention backend seam
- **Finding:** `find-0006` — *QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference*
- **Source URL:** <https://proceedings.mlr.press/v235/tang24l.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new in-repo attention backend (e.g., QuestAttentionBackend implementing vllm/v1/attention/backend.py:AttentionBackend) and expose it through the registration seam in vllm/v1/attention/backends/registry.py:34-263. Concretely: (1) add a QUEST entry to AttentionBackendEnum (or wire it via the CUSTOM placeholder + register_backend factory) so it is selectable through vllm_config.attention_config.backend / --attention-backend and resolved by vllm/v1/attention/selector.py:_cached_get_attn_backend; (2) implement the backend's decode path so that, for each paged KV block, it maintains per-page key min/max metadata and a cheap query-aware page scoring step that selects the top-K pages before loading full K/V tiles for the inner attention kernel; (3) keep the prefill path equivalent to the dense reference (e.g., FlashAttentionBackend) so prefill correctness/perf is unchanged, and gate sparsity by sequence length / K threshold so short contexts fall back to dense. Reuse the existing registration override tests (tests/test_attention_backend_registry.py) to verify the new enum/override path, and reuse tests/v1/attention/test_attention_backends.py and tests/v1/attention/test_attention_backends_selection.py patterns to validate output equivalence (within tolerance) against a dense backend on long-context inputs. No changes to AttentionBackend interface should be required; the registry seam is the intended insertion point.

**Proposal rationale.**

The candidate is explicitly framed as the supported route for adding a workload-specialized backend, and the caller objective targets median TPOT on multi-turn agentic workloads, which typically accumulate long KV histories across turns. QUEST's contribution is a concrete, transferable decode-time mechanism (per-page key min/max metadata + query-aware top-K page selection) that reduces KV memory traffic during paged attention without changing model weights or the AttentionBackend contract. That maps cleanly onto registering a new backend behind the existing enum/override seam and selector path, so the finding contributes a specific algorithmic idea (not just a topical restatement) that plausibly improves TPOT for the stated workload while leaving prefill (TTFT) unaffected when gated to long contexts.

---

### 7. Register a MInference dynamic-sparse prefill backend via the attention registry seam
- **Finding:** `find-0007` — *MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention*
- **Source URL:** <https://arxiv.org/abs/2407.02490>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new in-repo attention backend (e.g., MInferenceAttentionBackend) implementing vllm/v1/attention/backend.py:AttentionBackend, and wire it through the registration seam at vllm/v1/attention/backends/registry.py:34-263. Concretely: (1) add a new AttentionBackendEnum entry (or use the existing CUSTOM placeholder plus register_backend) so the backend is selectable via vllm_config.attention_config.backend / --attention-backend and resolved by vllm/v1/attention/selector.py:_cached_get_attn_backend; (2) implement the backend to apply MInference's head-specific sparse prefill: load an offline-determined per-head pattern table (e.g., A-shape / Vertical-Slash / Block-Sparse) and at prefill time build the online sparse index per head before dispatching to a sparse attention kernel, falling back to a dense path for heads/lengths where dense is faster; (3) keep the decode path on the existing dense kernel so only long-prefill traffic uses sparse routing, preserving correctness on short prompts. The override dictionaries and register_backend factory in registry.py are the supported insertion point, and tests/test_attention_backend_registry.py plus tests/v1/attention/test_attention_backends.py / test_attention_backends_selection.py act as the correctness oracle for registration and output equivalence.

**Proposal rationale.**

The candidate is explicitly the supported seam for adding a workload-specialized backend, and the caller's workload is multi-turn agentic with long tool-history prompts where prefill dominates TTFT. MInference contributes a concrete, transferable mechanism (offline per-head pattern selection plus online sparse-index construction) that targets exactly this regime, and it slots cleanly behind the AttentionBackend interface without changing the selector contract. This addresses the candidate's stated impact gap: the registry itself does not reduce latency, but routing long-prefill requests to a sparse-prefill backend through this seam is the path by which TTFT improvements become measurable, while leaving short-prompt and decode behavior on existing backends to protect TPOT and correctness.

---

### 8. Register a FlashDecoding++-style decode backend via the attention registry seam
- **Finding:** `find-0008` — *FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics*
- **Source URL:** <https://proceedings.mlsys.org/paper_files/paper/2024/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new in-repo attention backend (e.g. vllm/v1/attention/backends/flash_decoding_pp.py) implementing the three FlashDecoding++ techniques on the decode path: (1) a unified-max asynchronous partial-softmax merge that lets split-KV partials be combined without a global synchronization on per-split max values, (2) double-buffered 'flat GEMM' handling for the skinny M=1..few decode matmuls so the small-N tail is overlapped with the next split's load, and (3) a small heuristic that picks among split-K, split-KV, and non-split dataflows based on (batch, num_heads, kv_len, head_dim) at backend-selection time. Wire it into vllm/v1/attention/backends/registry.py:34-263 by adding a new AttentionBackendEnum entry (or by using the existing CUSTOM placeholder + register_backend factory) so vllm/v1/attention/selector.py:_cached_get_attn_backend can select it via --attention-backend / attention_config.backend, exactly as TritonAttentionBackend and FlashAttentionBackend are selected today. Validate with tests/test_attention_backend_registry.py for the registration override and tests/v1/attention/test_attention_backends.py + tests/v1/attention/test_attention_backends_selection.py for output equivalence against an existing backend on the same prompts.

**Proposal rationale.**

The candidate is explicitly the supported insertion point for workload-specialized in-repo backends, and the caller's workload (multi-turn agentic, TTFT/TPOT-bound) is dominated by decode steps where FlashDecoding++'s contributions land: the unified-max async softmax merge removes a synchronization that today sits on split-KV decode reductions, and the flat-GEMM double buffering targets exactly the skinny-M shapes that decode produces and that current decode kernels under-utilize. Because the change is additive — a new backend registered through register_backend / AttentionBackendEnum — it inherits the registry's correctness oracle (registration + selection + numerical-equivalence tests) without perturbing existing backends, which is the conservative way to land a kernel-level idea at this seam.

---

### 9. Register an XQA-style MQA/GQA decode backend through the attention registry seam
- **Finding:** `find-0009` — *New XQA-kernel provides 2.4x more Llama-70B throughput within the same latency budget*
- **Source URL:** <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new entry to AttentionBackendEnum in vllm/v1/attention/backends/registry.py (lines 34-263) for an XQA-style decode-specialized backend, and wire it through register_backend so it can be selected via --attention-backend or attention_config.backend. The backend class itself would live alongside existing reference implementations (e.g., vllm/v1/attention/backends/xqa_attn.py, mirroring flash_attn.py/triton_attn.py), implementing the AttentionBackend interface from vllm/v1/attention/backend.py. At selection time (vllm/v1/attention/selector.py:_cached_get_attn_backend, via current_platform.get_attn_backend_cls), the backend would be eligible only when the model uses MQA/GQA (num_kv_heads < num_query_heads), the hardware exposes the required tensor-core path, and the KV cache layout matches what the XQA-style kernel expects; otherwise selection falls back to the existing default. The registry change is small and localized: a new enum member plus, optionally, a CUSTOM-style override entry for opt-in experimentation. Correctness can be validated against tests/test_attention_backend_registry.py (registration/override) and tests/v1/attention/test_attention_backends.py and test_attention_backends_selection.py (selection and output equivalence vs. existing backends).

**Proposal rationale.**

The candidate is explicitly the supported insertion point for workload-specialized backends, and the finding describes a decode-phase kernel family (XQA) targeting precisely the shape (MQA/GQA) and phase (generation/TPOT) that the caller's multi-turn agentic workload is bottlenecked on. The change is transferable rather than topical: it uses the registry's existing extension contract (enum entry + register_backend + AttentionBackend implementation) and is gated by model/hardware/KV-layout checks at the selector, so it does not perturb the default path. The medium impact estimate aligns with the finding's claim of substantially better decode throughput within the same latency budget for Llama-class GQA models, which is the dominant decode regime for agentic workloads.

---

### 10. Register a FlexAttention+FA4 backend via the registry seam for agentic mask variants
- **Finding:** `find-0010` — *FlexAttention + FlashAttention-4: Fast and Flexible – PyTorch*
- **Source URL:** <https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new entry through vllm/v1/attention/backends/registry.py (AttentionBackendEnum/register_backend, lines 34-263) that exposes a FlexAttention-based backend whose kernels are JIT-instantiated by FlashAttention-4 for workload-specific score_mod/mask_mod variants relevant to multi-turn agentic traffic (e.g., document/turn prefix masks that recognize shared system+history prefixes, sliding-window for long context tails, optional soft-capping). Concretely: (1) introduce a new backend class implementing vllm/v1/attention/backend.py:AttentionBackend (sibling of vllm/v1/attention/backends/flash_attn.py:FlashAttentionBackend and triton_attn.py:TritonAttentionBackend) that wraps torch.nn.attention.flex_attention with the FA4 backend selected at compile time and parameterizes the mask_mod/score_mod from request metadata; (2) register it through the existing seam either as a new AttentionBackendEnum member or via register_backend(...) so it can be selected by --attention-backend / attention_config.backend and resolved by vllm/v1/attention/selector.py:_cached_get_attn_backend without bypassing current_platform.get_attn_backend_cls; (3) keep the override dictionaries / CUSTOM placeholder semantics intact so the new backend is opt-in and falls back to the platform default. Validation reuses the existing oracles: tests/test_attention_backend_registry.py for registration/override behavior, and tests/v1/attention/test_attention_backends.py + test_attention_backends_selection.py for selection and output equivalence against a reference backend on the supported mask variants.

**Proposal rationale.**

The candidate is explicitly the supported insertion point for a workload-specialized in-repo attention backend, and the finding describes exactly the kind of backend that fits that seam: FlexAttention with an FA4 backend that JIT-instantiates fast kernels for custom score/mask combinations (sliding window, soft-capping, document/prefix masks) that today either force a fallback to slower generic attention or require bespoke kernels. Multi-turn agentic workloads frequently exhibit such mask shapes (shared system/history prefixes across turns, long-context tails), so plugging a FlexAttention+FA4 backend through register_backend gives a concrete, transferable path to keep these variants on a fast kernel — directly targeting media TTFT (prefix-heavy prefill) and median TPOT (decode under custom masks) — without modifying the selector or interface. The finding contributes a specific implementation route (FA4 JIT instantiation behind FlexAttention), not just a restatement of the registry's existing capability.

---

### 11. Register a KIVI-style asymmetric 2-bit KV-quantized attention backend via the registry seam
- **Finding:** `find-0011` — *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*
- **Source URL:** <https://proceedings.mlr.press/v235/liu24bz.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new entry to AttentionBackendEnum in vllm/v1/attention/backends/registry.py (lines 34-263) for a workload-specialized backend (e.g., AttentionBackendEnum.KIVI_KV2 or registered dynamically via register_backend) that implements the AttentionBackend interface from vllm/v1/attention/backend.py with KIVI-style KV-cache quantization: per-channel quantization for the key cache and per-token quantization for the value cache, at low bitwidth (2-bit asymmetric, with optional residual full-precision window for the most recent tokens as in the paper). The new backend reuses an existing reference implementation (e.g., flash_attn.py or triton_attn.py) for the compute path, but stores K/V in a packed low-bit layout with the appropriate scale/zero-point tensors and dequantizes on the fly during attention. Selection flows through the existing path: users opt in via --attention-backend (resolved by vllm/v1/attention/selector.py:_cached_get_attn_backend through current_platform.get_attn_backend_cls and AttentionBackendEnum.get_class), so no new plumbing is required outside the registry. Correctness is gated by tests/test_attention_backend_registry.py for registration and by tests/v1/attention/test_attention_backends.py / test_attention_backends_selection.py for output equivalence (within a quantization tolerance) against a reference backend.

**Proposal rationale.**

The candidate is explicitly the supported insertion point for a workload-specialized in-repo attention backend, and the caller objective targets median TPOT for a multi-turn agentic workload — a regime that is KV-bandwidth-bound and benefits directly from low-bit KV. KIVI provides a concrete, tuning-free recipe (per-channel keys, per-token values, 2-bit) whose asymmetry across K vs V is the non-obvious idea that makes 2-bit work, and it maps cleanly onto an AttentionBackend implementation registered through register_backend without changing the selector or interface. This addresses the gap that none of the default registry entries currently provide a low-bit KV path tuned for decode-heavy long-context agent traces.

---

### 12. Register a KVQuant-style low-bit KV-cache attention backend via the registry seam
- **Finding:** `find-0012` — *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*
- **Source URL:** <https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Use the registration seam at vllm/v1/attention/backends/registry.py:34-263 (AttentionBackendEnum / register_backend) to plug in a new in-repo attention backend that implements KVQuant's three techniques: (1) pre-RoPE key quantization, applied during the cache write path so keys are quantized before rotary embedding is fused into them; (2) non-uniform per-channel/per-token KV datatypes with channel-wise scales for keys and token-wise scales for values; (3) dense-sparse outlier handling, where a small fraction of high-magnitude entries are stored densely (e.g., FP16) while the bulk of the cache uses the low-bit non-uniform format. The backend would mirror the structure of vllm/v1/attention/backends/flash_attn.py:FlashAttentionBackend and vllm/v1/attention/backends/triton_attn.py:TritonAttentionBackend, exposing a fused cache-update + dequantize-on-the-fly attention kernel so the quantized layout is invisible to the rest of v1. It would be made selectable through current_platform.get_attn_backend_cls and the AttentionBackendEnum.get_class/register_backend override path used by vllm/v1/attention/selector.py:_cached_get_attn_backend, so users opt in via attention_config.backend (e.g., --attention-backend KVQUANT) without touching call sites. Correctness would be gated by extending tests/test_attention_backend_registry.py for the new registration entry and tests/v1/attention/test_attention_backends.py / test_attention_backends_selection.py for output equivalence (within a quantization-error tolerance) against an FP16 reference backend.

**Proposal rationale.**

The candidate is explicitly described in evolve_rationale as the supported insertion point for a workload-specialized in-repo attention backend, and the finding contributes concrete, transferable kernel-level ideas (pre-RoPE key quantization, non-uniform KV datatypes, dense-sparse outliers) that are described as attention-backend-relevant because they require fused cache-update/dequantization kernels. For the caller's multi-turn agentic workload, KV cache size dominates long-context decode bandwidth, so a backend that shrinks per-token KV bytes via these techniques directly targets the median TPOT objective while preserving TTFT through fused dequantize-in-attention. Routing this capability through register_backend keeps the change isolated to a new backend module plus its registry entry, matches the existing override mechanism, and is validated by the registry's existing correctness oracles (test_attention_backend_registry.py, test_attention_backends.py, test_attention_backends_selection.py).

---

### 13. Register a cascade/shared-prefix attention backend via the registry seam for multi-turn agentic prefixes
- **Finding:** `find-0013` — *Cascade and Recursive Attention | flashinfer-ai/flashinfer | DeepWiki*
- **Source URL:** <https://deepwiki.com/flashinfer-ai/flashinfer/2.4-cascade-and-recursive-attention>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Add a new in-repo attention backend that implements cascade/recursive attention as described by FlashInfer: compute attention over disjoint KV index subsets independently and merge their partial outputs via associative log-sum-exp (LSE) state merging, preserving exactness. Wire it into the seam at vllm/v1/attention/backends/registry.py (lines 34-263) by (a) adding a new AttentionBackendEnum entry (e.g. CASCADE) alongside the existing defaults, and (b) implementing the backend in a sibling module (e.g. vllm/v1/attention/backends/cascade_attn.py) that conforms to vllm/v1/attention/backend.py:AttentionBackend, internally dispatching to an existing exact backend (FlashAttentionBackend / TritonAttentionBackend) for each KV subset and then merging per-subset (output, LSE) pairs. The backend should detect a shared prefix span (system prompt + prior turns common across the batch) versus per-request suffix tokens, run attention twice (once over the shared-prefix KV subset, once over the per-request suffix KV subset), and combine results with the LSE-merge primitive. Selection flows through the existing path: users set attention_config.backend=CASCADE (or via --attention-backend), and vllm/v1/attention/selector.py:_cached_get_attn_backend resolves it through AttentionBackendEnum.get_class. Equivalence is validated by reusing tests/v1/attention/test_attention_backends.py output-equivalence harness; the registry override mechanics are already exercised by tests/test_attention_backend_registry.py. Impact target: lower media TTFT and median TPOT on multi-turn agentic workloads where many requests share a long prefix, by reducing redundant prefix-KV reads/compute while staying numerically exact via LSE merging.

**Proposal rationale.**

The candidate is explicitly described as the supported insertion point for a workload-specialized backend, and the caller's workload (multi-turn agentic) is exactly the regime where a shared-prefix cascade pays off: the system prompt and prior-turn context recur across requests, so splitting attention into (shared-prefix subset, per-request suffix subset) and merging via LSE removes redundant prefix work without changing outputs. The finding contributes the concrete, transferable mechanism missing from the current default backends - associative LSE merging of partial attention states over disjoint KV subsets - which is what makes a hierarchical/shared-KV decomposition exact rather than approximate. This addresses the gap that today's registry defaults (FLASH_ATTN, TRITON_ATTN, etc.) treat each request's KV as monolithic and have no exposed seam for prefix-sharing decomposition; routing it through register_backend / AttentionBackendEnum keeps the change localized to the documented plugin seam and keeps selection/override behavior covered by the existing registry and selection tests.

---

## Agent proposals

### 1. Register a DuoAttention head-specialized KV-retention backend via the registry seam
- **Agent:** claude

**Detailed description.**

Add a new entry to AttentionBackendEnum in vllm/v1/attention/backends/registry.py (lines 34-263) for a DuoAttention-style backend (e.g., AttentionBackendEnum.DUO_ATTN) and wire it through register_backend so vllm/v1/attention/selector.py:_cached_get_attn_backend resolves it via --attention-backend / attention_config.backend. The backend implements vllm/v1/attention/backend.py:AttentionBackend in a sibling module (vllm/v1/attention/backends/duo_attn.py) that splits attention heads into two classes per layer using an offline-identified mask: (a) a small set of "retrieval heads" that retain the full paged KV cache and run an existing exact kernel (delegating to FlashAttentionBackend or TritonAttentionBackend per head), and (b) the majority "streaming heads" that only retain a sliding window plus a fixed number of leading attention-sink tokens, with KV beyond the window evicted from the paged allocator on append. Concretely: (1) consume a per-layer/per-head retention policy table (loaded from a model-side artifact or, as a fallback, a length-thresholded heuristic) as backend init metadata, (2) extend the attention-metadata builder to track per-head KV index ranges (full vs. window+sink) without changing the AttentionBackend interface, (3) at decode time dispatch streaming-head batches to a window-attention path and retrieval-head batches to the dense path, fused under one backend call so the model code is unchanged, (4) keep prefill on the existing dense kernel for both head classes so prefill correctness is unaffected and the saving is realized only on accumulated multi-turn KV. Validation reuses tests/test_attention_backend_registry.py for the registration override and tests/v1/attention/test_attention_backends.py / test_attention_backends_selection.py for selection plumbing and output equivalence on short contexts (where window+sink covers the full history and the backend collapses to dense), with an additional approximate-equivalence check on long contexts where retrieval heads remain exact and only streaming heads diverge within tolerance.

**Novelty rationale.**

None of the 13 listed deep_research_proposals propose per-head differentiation of KV retention policy. The existing proposals cover prefix sharing across requests/batch (Hydragen find-0002, Cascade find-0013), full-cache sparse decode selection (QUEST find-0006), low-bit KV quantization that still keeps every token (KIVI find-0011, KVQuant find-0012), sparse prefill patterns (MMInference find-0003, MInference find-0007), decode-kernel layout/scheduling (Flash-Decoding find-0005, FlashDecoding++ find-0008, XQA find-0009, FlashInfer find-0004), fused prefill-decode (POD find-0001), and FlexAttention mask variants (find-0010). DuoAttention's contribution is orthogonal and specifically distinct: an offline-identified head taxonomy (retrieval vs. streaming) that lets the majority of heads physically discard most of the multi-turn KV (window + attention sinks) while a small minority of heads stay exact. This directly attacks median TPOT on growing agentic histories by reducing both KV memory traffic and capacity per layer in a way that quantization (which reads every byte at lower precision) and query-aware page selection (which still stores every page) do not.

---

### 2. Invalidate selector caches when backends are registered or cleared
- **Agent:** codex

**Detailed description.**

Harden the runtime override contract in vllm/v1/attention/backends/registry.py:register_backend by adding registry epoch counters for attention and mamba backends, incrementing them on decorator registration, direct class_path registration, and clear_override(). Expose a small getter from registry.py, then include the relevant epoch as an extra argument in vllm/v1/attention/selector.py:_cached_get_attn_backend and _cached_get_mamba_attn_backend so functools.cache cannot return a class resolved before an override was installed. Add a focused regression test that resolves a backend once, calls register_backend on the same enum member with a different mock class path, and verifies the next selector call with the same selector config returns the new class without requiring a process restart.

**Novelty rationale.**

The listed deep_research_proposals and Claude's proposal all add new specialized attention algorithms behind the registry seam: fused prefill/decode, shared-prefix attention, sparse prefill/decode, FlashDecoding variants, XQA, FlexAttention/FA4, KV quantization, cascade attention, and DuoAttention retention. None address the registry/selector correctness issue that register_backend can mutate _ATTN_OVERRIDES while selector.py's cached resolution may still return the previously resolved backend for the same configuration. This proposal is therefore orthogonal: it makes the plugin seam reliable for all of those future backend experiments rather than proposing another backend algorithm.

---
