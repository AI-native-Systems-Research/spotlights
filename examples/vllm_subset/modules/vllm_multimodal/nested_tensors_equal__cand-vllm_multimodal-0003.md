# nested_tensors_equal

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/inputs.py`](vllm/multimodal/inputs.py) (lines 232–286)
- **Symbol:** `nested_tensors_equal`
- **Kind:** function
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0003`

## Description
Recursively compares NestedTensors structures used by batching and shared-field compatibility checks.

## Current approach
Builds a check_dtype lambda on every call, has symmetric duplicated branches for tensor/list/tuple cases, and reaches torch.equal for tensor leaves before explicit dtype/device/shape or object-identity short-circuits.

## Estimated impact explanation
Shared multimodal fields are compared during batching on every prefill. Avoiding elementwise tensor comparisons, especially CUDA comparisons that can synchronize, reduces TTFT for batched or broadcast multimodal inputs.

## Evolve rationale
The optimization unit is nested_tensors_equal, called from _can_batch_mm_items for adjacent multimodal items. Cheap early exits for dtype, shape, device, object identity, and identical storage metadata can avoid torch.equal in common shared-field cases; recursion can also be de-duplicated. Correctness oracle: tests/multimodal/test_inputs.py and tests/multimodal/test_utils.py; preserved cases must return the same boolean result.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Short-circuit nested_tensors_equal via storage-view identity and per-call memoization
- **Agent:** claude

**Detailed description.**

Rewrite `nested_tensors_equal` in vllm/multimodal/inputs.py (lines 232-286) so that the expensive `torch.equal` call on tensor leaves is only reached when structurally required. Concretely:

1. Drop the per-call `check_dtype_func` lambda in favor of a plain `if check_dtype and a.dtype != b.dtype: return False` guard, so the closure isn't rebuilt on every recursion frame.
2. Collapse the symmetric `isinstance(a, torch.Tensor)` / `isinstance(b, torch.Tensor)` and `list` / `tuple` branches into single-branch checks that assert both sides have the same type upfront (`type(a) is type(b)` for the container cases, `isinstance(a, torch.Tensor) == isinstance(b, torch.Tensor)` for leaves) — the current code duplicates each branch to handle the case where only one side is the container/tensor, but if the types differ they're unequal regardless.
3. For the tensor-leaf case, insert cheap fast paths **before** `torch.equal`:
   - `a is b` → True (very common for shared multimodal fields like `is_embed` masks that come from a single upstream tensor and are held by reference across batched items).
   - `a.device != b.device` → False (avoids CUDA cross-device sync that `torch.equal` would otherwise trigger).
   - `a.shape != b.shape` → False.
   - If `a.device.type == b.device.type` and `a.data_ptr() == b.data_ptr()` and `a.storage_offset() == b.storage_offset()` and `a.stride() == b.stride()` and `a.dtype == b.dtype`, return True — same view over same storage is provably equal without a device-side comparison. This is the key win for `_can_batch_mm_items`, where adjacent multimodal items typically share the exact underlying tensor for hashed/broadcast fields.
   - Only then fall through to `torch.equal(a, b)`.
4. Add an optional `_seen: dict[tuple[int, int], bool] | None = None` recursion parameter (defaulting to a fresh dict at the top-level call) keyed by `(id(a), id(b))` to memoize container-level comparisons. The same shared MM field is often reached along multiple paths inside a single `_can_batch_mm_items` walk; memoizing on Python object identity is O(1) per revisit and cannot produce false positives because equal Python objects with identical `id()` are the same object.

Correctness is preserved because every early-exit condition (identity, shape mismatch, device mismatch, same storage view with same dtype/stride/offset) is a strict logical implication of `torch.equal`'s definition. Behavior on the scalar tail (`return a == b`) and container recursion is unchanged. Cover with `tests/multimodal/test_inputs.py` / `tests/multimodal/test_utils.py` as noted in the candidate's oracle.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so overlap is not against prior proposals but against the candidate's own `evolve_rationale`. The rationale gestures at generic "cheap early exits for dtype, shape, device, object identity, and identical storage metadata," but this proposal specifies the exact fast-path predicate (data_ptr + storage_offset + stride + dtype forming a same-view proof that skips `torch.equal` even for non-identical Python objects that view the same storage — the common case for shared MM fields propagated by reference) and adds an orthogonal mechanism the rationale does not mention at all: an `id()`-keyed memoization dict threaded through the recursion so repeated visits to shared subtrees during a single `_can_batch_mm_items` sweep collapse to a dict lookup. Together these target the CUDA-sync cost specifically, which is the dominant TTFT contributor for batched multimodal prefill in multi-turn agentic workloads.

---

### 2. Replace recursive container traversal with an iterative stack walk
- **Agent:** codex

**Detailed description.**

Refactor `nested_tensors_equal` in `vllm/multimodal/inputs.py` so container comparison is driven by a local stack of `(left, right)` pairs instead of recursive calls wrapped in `all(... for zip(...))`. The tensor/scalar comparison logic can remain in a small local leaf block, but list/tuple handling should push children onto the stack after checking exact container type and length. This removes one Python function call and one generator frame per nested edge, which matters when `_can_batch_mm_items` repeatedly compares multimodal kwargs made of many small tensor leaves or nested per-frame/per-image lists. Preserve the public signature and exact boolean semantics, including `check_dtype`; add a focused test in the existing multimodal input tests for a multi-level list/tuple structure with an early mismatch to ensure the iterative walk still short-circuits without visiting later leaves.

**Novelty rationale.**

There are no deep_research proposals for this candidate. Claude's proposal optimizes tensor-leaf equality, symmetric branch duplication, and per-call memoization while still describing recursion threaded through `_seen`. This proposal targets a different cost center: the Python traversal overhead from recursive calls and generator-based `all()` over nested containers. It is complementary to leaf fast paths and does not depend on storage identity, object identity, dtype/device/shape ordering, or memoization.

---
