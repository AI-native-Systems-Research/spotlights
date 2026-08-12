# InputProcessor.process_inputs

[← vllm/v1/engine](../vllm_v1_engine.md)

- **File:** [`vllm/v1/engine/input_processor.py`](vllm/v1/engine/input_processor.py) (lines 244–389)
- **Symbol:** `InputProcessor.process_inputs`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_v1_engine-0016`

## Description
Per-request input-to-EngineCoreRequest converter that validates request parameters, preprocesses raw prompts, clones sampling or pooling params, normalizes multimodal features, and stamps admission metadata.

## Current approach
Runs validation on every call, preprocesses raw prompts synchronously through InputPreprocessor when needed, clones params, applies generation-config and tokenizer updates, sorts multimodal placeholders with argsort_mm_positions, builds MultiModalFeatureSpec objects, and returns a fresh EngineCoreRequest.

## Estimated impact explanation
The work is per request rather than per token, so it mainly moves median TTFT; multi-turn agentic traffic creates many short requests where repeated validation, cloning, and preprocessing overhead is visible.

## Evolve rationale
This method sits on the TTFT path for every new turn before the request can enter EngineCore. Optimization units include specializing already-rendered EngineInput, avoiding repeated SamplingParams validation/clone work for streaming inputs, reducing multimodal flatten/sort allocation, and caching tokenizer-derived updates when params are reused. Correctness oracle: tests/v1/engine/test_preprocess_error_handling.py, tests/v1/engine/test_async_llm.py, and engine request construction tests should preserve validation failures, arrival_time, data_parallel_rank bounds, multimodal feature ordering, and params semantics.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Fast-path pre-rendered EngineInput to skip validation, cloning, and tokenizer updates for reused SamplingParams
- **Agent:** claude

**Detailed description.**

In InputProcessor.process_inputs (vllm/v1/engine/input_processor.py:244-389), the hot path for multi-turn agentic traffic already receives an already-rendered EngineInput (the `isinstance(prompt, dict) and 'type' in prompt` branch at line 272). Even on this fast path, every request still pays for: (1) full `_validate_params` + `_validate_lora` (line 259-260) which walks all SamplingParams fields; (2) an unconditional `SamplingParams.clone()` (line 318); (3) `update_from_generation_config` + `update_from_tokenizer` (lines 326-331) that touch tokenizer-derived state like EOS/stop tokens; and (4) a fresh `argsort_mm_positions` + MultiModalFeatureSpec list build even when the caller already sorted placeholders.

Concrete change:

1. Add an identity-based cache on the InputProcessor keyed by `id(params)` with a weakref (or a monotonic 'version' stamp on SamplingParams) that stores a tuple `(validated_marker, cloned_and_updated_params)`. When a subsequent call arrives with the same `params` object (common in agent loops that mutate max_tokens between turns but re-use the same SamplingParams instance, and in streaming loops that pass the same object across chunks), skip `_validate_params`, skip `clone()`, skip `update_from_generation_config`, and skip `update_from_tokenizer`. Only re-derive `max_tokens` from the current prompt length (line 320-324) since that is prompt-dependent.

2. On the pre-rendered EngineInput branch, allow the renderer to attach a boolean flag `_mm_already_sorted: bool` (and pass through the already-flattened list) so we can bypass `argsort_mm_positions` and the per-item feature-spec build loop (lines 355-371) when the renderer has produced pre-sorted output. Fall back to the current path if the flag is absent.

3. For the `_validate_lora` call, cache the result keyed by `(id(lora_request), lora_request.lora_int_id)` — this is trivially safe because LoRARequest is effectively immutable in normal use.

Invalidation: the params cache entry is dropped when the SamplingParams' bump-counter changes (add a lightweight `_vllm_version` int mutated on `clone`/set), or when the tokenizer/generation-config identity changes. Correctness is preserved because we still emit a defensive clone whenever the cached object is about to escape into EngineCoreRequest for a request that mutates it (i.e., when `max_tokens` was unset). The path collapses process_inputs on the agentic hot path to just the max_tokens patch + EngineCoreRequest construction.

Oracle: preserve tests/v1/engine/test_preprocess_error_handling.py (validation still runs on first sighting of a params object and on any lora change), tests/v1/engine/test_async_llm.py (arrival_time, dp rank, MM ordering unchanged), and add a micro-benchmark asserting >2x throughput of process_inputs for repeated SamplingParams.

**Novelty rationale.**

There are no existing deep_research_proposals on this candidate, so any concrete optimization is novel by construction. Specifically, the change combines three ideas — identity/version-keyed params-clone-and-validate memoization, a renderer-hinted 'already sorted MM' fast-path, and LoRA-validation memoization — that target exactly the per-request overhead cited in evolve_rationale (repeated validation/clone, MM flatten/sort, tokenizer-derived updates) without altering the observable request semantics or arrival_time/dp-rank checks.

---

### 2. Defer multimodal feature materialization until EngineCore needs it
- **Agent:** codex

**Detailed description.**

Change `InputProcessor.process_inputs` in `vllm/v1/engine/input_processor.py:244-389` so the multimodal branch stops eagerly allocating a full `mm_features` list of `MultiModalFeatureSpec` objects on admission. Today, after preprocessing, it sorts `mm_inputs`/`mm_positions` and immediately constructs one spec per feature before building `EngineCoreRequest`. Introduce a lightweight lazy carrier on `EngineCoreRequest`, such as a tuple of `(sorted_mm_inputs, sorted_mm_positions)` or a small `LazyMultiModalFeatureSpecs` sequence, and materialize `MultiModalFeatureSpec` only at the first consumer boundary that actually requires the concrete specs. For text-only requests this changes nothing; for multimodal requests it keeps ordering semantics identical by still applying `argsort_mm_positions` once in `process_inputs`, but avoids per-request object churn on the TTFT path when EngineCore admission/scheduling only needs to pass the data through. Add a focused request-construction test that compares the realized specs against the current eager list ordering and values for multiple out-of-order placeholders, plus a no-multimodal regression asserting the field remains empty/cheap.

**Novelty rationale.**

There are no deep_research proposals for this candidate. This is distinct from Agent A's proposal because it does not cache validation, cloning, tokenizer updates, LoRA validation, or add an already-sorted renderer hint. It targets a different cost center: delaying `MultiModalFeatureSpec` object construction itself while preserving the existing sort and request semantics.

---
