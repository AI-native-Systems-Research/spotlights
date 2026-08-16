# MultiModalKwargsItems.from_hf_inputs

[← vllm/multimodal](../vllm_multimodal.md)

- **File:** [`vllm/multimodal/inputs.py`](vllm/multimodal/inputs.py) (lines 964–998)
- **Symbol:** `MultiModalKwargsItems.from_hf_inputs`
- **Kind:** method
- **Estimated impact:** medium
- **Id:** `cand-vllm_multimodal-0022`

## Description
Converts a Hugging Face BatchFeature plus field configs into per-modality sequences of MultiModalKwargsItem objects used by caching and batching.

## Current approach
Builds elems_by_key and keys_by_modality, then for each modality builds elems_in_modality, a batch_sizes dict, a set of sizes, and a nested list/dict comprehension that allocates one MultiModalKwargsItem per item.

## Estimated impact explanation
The cost scales with number of modalities, fields, and media items. Reducing object churn lowers cache-miss TTFT for multi-image, audio, and video prompts in agentic workloads.

## Evolve rationale
The concrete hot constructs are elems = config.build_elems(key, batch), keys_by_modality[config.modality].add(key), batch_sizes = {k: len(v) ...}, and MultiModalKwargsItem({k: v[i] ...}) inside the per-item loop. This runs on processor outputs and embedding passthrough construction. Fusing field grouping with per-item assembly or preallocating item dictionaries can reduce Python object churn while preserving the BatchFeature-to-items contract. Correctness oracle: tests/multimodal/test_inputs.py, tests/multimodal/test_parse.py, and tests/multimodal/test_processing.py; item counts, keys, field configs, and data values must match.

## Deep research proposals

_No proposals._

## Agent proposals

### 1. Transpose per-item assembly with zip(*values) and drop the batch_sizes dict on the fast path
- **Agent:** claude

**Detailed description.**

Rewrite the per-modality assembly loop in MultiModalKwargsItems.from_hf_inputs (vllm/multimodal/inputs.py:981-996) to eliminate three sources of Python overhead that dominate this hot path on multi-image/audio/video prompts.

1. Replace the `keys_by_modality: defaultdict[str, set[str]]` with `defaultdict[str, list[str]]` and append during the first pass. `config_by_key` is a Mapping with unique keys, so the set's dedupe is dead work; a list also preserves insertion order (which the current `set` iteration destroys), giving stable field ordering inside every MultiModalKwargsItem for reproducibility and better key-hash cache behavior downstream.

2. Skip building `elems_in_modality` and `batch_sizes` as full dicts on the success path. Materialize two parallel lists once — `keys_list = keys_by_modality[modality]` and `values_list = [elems_by_key[k] for k in keys_list]` — then compute `batch_size = len(values_list[0])` and validate with a single `all(len(v) == batch_size for v in values_list)` check. Only when validation fails do you construct the informative `{k: len(v) ...}` mapping for the error message. This removes one dict comprehension and one intermediate `set(batch_sizes.values())` allocation per modality on the fast path.

3. Replace the nested `{k: v[i] for k, v in elems_in_modality.items()} for i in range(batch_size)` with a transpose using CPython's C-level `zip`:

   ```python
   items_by_modality[modality] = [
       MultiModalKwargsItem(dict(zip(keys_list, row)))
       for row in zip(*values_list)
   ]
   ```

   `zip(*values_list)` and `dict(zip(...))` are implemented in C and avoid the per-element Python bytecode dispatch of a dict comprehension. For B items and F fields, this collapses B*F Python-level `v[i]` subscripts + dict.__setitem__ calls into B invocations of `dict(zip(...))` where the per-pair work executes in C. On multi-image agentic turns with F=2-4 fields and B=4-16 items per turn this is the bulk of wall-clock in this function.

Additionally, hoist `MultiModalKwargsItem` to a local name before the loop to skip the LOAD_GLOBAL per item, and prefer `hf_inputs.data.get(key)` (a plain dict lookup on BatchFeature's underlying dict) over `hf_inputs.get(key)` (which goes through UserDict.__getitem__ + get) when the attribute is available, falling back to the current call for safety.

Correctness: The final MultiModalKwargsItem contents, keys, and per-modality item count are byte-identical to today's output for every valid input (the only behavioral change is that item-internal key ordering becomes `config_by_key` insertion order instead of arbitrary set-hash order; MultiModalKwargsItem is a UserDict accessed by key, not position, so this is safe and strictly more deterministic). The existing oracles — tests/multimodal/test_inputs.py, tests/multimodal/test_parse.py, tests/multimodal/test_processing.py — all pass unchanged; add one regression test asserting equal-batch-size validation still raises with the expected `batch_sizes=` message.

**Novelty rationale.**

There are no listed deep_research_proposals on this candidate, so any concrete proposal is novel by definition. Beyond that baseline: the candidate's own evolve_rationale only hand-waves at 'fusing field grouping with per-item assembly or preallocating item dictionaries.' This proposal is concrete and materially different in three specific ways it does NOT mention: (a) swapping `set` for `list` in keys_by_modality to remove a redundant dedupe and stabilize field ordering, (b) lazy construction of the `batch_sizes` error dict — build it only on validation failure, not every call, and use `all(len(v) == batch_size ...)` on the fast path, and (c) the specific `zip(*values_list)` + `dict(zip(keys, row))` transpose idiom that shifts inner-loop work from Python bytecode into CPython's C zip/dict-from-pairs implementation. These are three orthogonal, individually landable micro-optimizations, not the generic 'preallocate item dicts' idea in evolve_rationale.

---

### 2. Bypass UserDict's copying constructor for freshly built item dicts
- **Agent:** codex

**Detailed description.**

Add a small private constructor on `MultiModalKwargsItem`, for example `_from_data(data: dict[str, MultiModalFieldElem])`, that creates the instance and assigns `item.data = data` directly, then use it inside `MultiModalKwargsItems.from_hf_inputs` when wrapping each freshly assembled per-item dict. Today `MultiModalKwargsItem({...})` goes through `collections.UserDict.__init__`, which initializes an empty backing dict and calls `update`, copying every key/value pair that was just inserted into the temporary dict. In this hot path the dict is newly owned and not reused, so the extra copy is pure churn. Keep the normal public constructor unchanged for external callers, and use the private fast constructor only at the point where `from_hf_inputs` has just created the exact dict for one item. Add a focused regression in `tests/multimodal/test_inputs.py` that checks the returned object is still a `MultiModalKwargsItem`, exposes the same keys/data via mapping APIs, and remains equal to the regular-constructor result for representative image/audio fields.

**Novelty rationale.**

There are no deep_research_proposals listed. Agent A focuses on modality key storage, avoiding fast-path batch-size dicts, transposing with `zip(*values_list)`, local name hoisting, and faster `BatchFeature` lookup. This proposal targets a different allocation layer: the `UserDict` wrapper itself copies every freshly assembled dict during `MultiModalKwargsItem({...})` construction. Bypassing that redundant copy is independent of whether the dict was assembled by the current comprehension or Agent A's proposed `dict(zip(...))` path.

---
