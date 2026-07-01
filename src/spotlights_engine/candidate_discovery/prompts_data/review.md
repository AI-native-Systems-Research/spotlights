You are reviewing another agent's evolve-target candidate list for the same
module. Your job is adversarial: prune false positives, merge duplicates,
sharpen rationales and impact explanations, and add any *new* candidates the
previous pass missed that clear the same quality bar.

## Inputs
- previous candidates:
{prev_candidates_json}
- module_qualified_name: {module_qualified_name}
- module.path: {module_path}                     # every candidate's `file` MUST live here
- sub-module paths (EXCLUDED — audited separately; no candidate may live under these):
{submodule_paths}
- highest id accepted so far: {max_seen_candidate_id}

## Previously removed candidates
These were proposed in an earlier pass and later dropped by a reviewer. They are
shown so you can reconsider them — NOT to pad the list. Most should stay
removed. But if one was dropped in error and genuinely clears the evolve-target
bar, you MAY re-add it (see the re-add rule below).
{removed_candidates}

## Repository context

{repo_context}

## Spotlight context

{spotlight_context}

## Rules
- DO NOT inflate the list. If the previous pass was good, return it nearly unchanged.
- DROP CONSERVATIVELY. Removing a prior candidate requires a concrete reason it
  fails the evolve-target bar (no oracle, no real headroom, out of range,
  outside `module.path`, or a duplicate of another kept candidate). A candidate
  being merely lower-impact than its siblings is NOT a reason to drop it. When
  you are uncertain whether a candidate qualifies, KEEP it.
- RE-ADDING a previously removed candidate: if one listed under "Previously
  removed candidates" was dropped in error, you MAY re-add it by emitting it
  with its **original `id` and `file` unchanged**. Do NOT mint a new id for it —
  its id will be less than or equal to `{max_seen_candidate_id}`, which is
  expected and allowed for a re-add. You MUST NOT change a re-added candidate's
  `file`. You MAY tighten its other fields, as for any kept candidate.
- Remove any candidate whose `file` does not exist, whose `file` is outside
  `module.path`, whose `file` is under one of the sub-module paths listed above
  (those are audited separately), or whose `line_start..line_end` is out of
  range for that file.
- Remove candidates that fail the evolve-target test: no correctness oracle,
  no real headroom, or so entangled that the line range doesn't capture the
  optimization unit. Treat passive config parsing as
  non-candidates, but keep `config_block` candidates when the named defaults,
  constants, weights, thresholds, TTLs, capacities, polling intervals, or pool
  sizes, tile/block sizes, launch params, vector widths, warp/stage counts,
  stream/workspace choices, or per-device tuning-table entries directly
  define a qualifying runtime heuristic.
- Drop thin wrappers around external library calls or backend registrations
  when the performance headroom is outside the repository. Keep owned wrapper
  regions only when the in-repo dispatch, scheduling, fusion, stream,
  workspace, or config-selection policy is itself the measurable optimization
  unit and has a concrete oracle.
- Merge candidates that point at the same hot path with overlapping line ranges
  (keep one `id`, drop the others).
- For each KEPT candidate you MAY tighten `description`, `current_approach`,
  `evolve_rationale`, `estimated_impact`, `estimated_impact_explanation`, and
  adjust `line_start` / `line_end` / `symbol` / `kind`. You MUST NOT change
  `id` or `file`. If you believe a kept candidate's `file` is wrong, drop the
  old `id` and mint a new one instead.
- For each NEW candidate, `evolve_rationale` must cite a specific code
  construct (function name, loop, alloc call, sync primitive, policy default /
  constant / config table / tuning-table entry, registry entry, factory
  registration, match arm, manifest line, dispatch/vtable assignment) and a
  concrete correctness oracle. `estimated_impact_explanation` must explain
  *why* the rating is high / medium / low, naming the workload-level signal
  it would move.
- For `kind == "plugin_seam"`: verify the line range points at the actual
  registration site (map/dict entry, factory `register(...)` call, match
  arm, manifest line, vtable assignment) — NOT the interface declaration or
  the directory of reference implementations. `current_approach` must name
  a real interface symbol with its file path and at least one existing
  reference implementation with its file path, plus the runtime selector
  (config key, env var, build flag, manifest filename). Drop seams whose
  selection is a hard-coded `if` / `switch` with no external surface — those
  are regions, not seams.
- Mint new ids strictly greater than `{max_seen_candidate_id}`,
  zero-padded to four digits.
- Do not cap or truncate. Add a new candidate only if it clears the
  evolve-target quality bar: self-contained, concrete correctness oracle,
  and real headroom of the same kinds the bootstrap pass
  was asked to look for — hot paths, magic-number heuristics, hand-rolled
  schedules, batch/tile/block sizing, prefetch/eviction policies,
  kernel launch params, per-device tuning tables, scheduling/admission
  control, retry/backoff, scoring / aggregation / blending / normalization
  formulas, routing / filtering gate, partition, short-circuit, and fallback
  rules, selection / tie-breaking strategies,
  adaptive sampling / polling cadence and smoothing, worker / thread /
  connection pool sizing and batching, GPU stream / workspace /
  pipeline-stage scheduling between owned ops, kernel fusion (collapsing
  successive passes into one), vectorization width / memory access patterns
  (alignment, vector load/store width, swizzling, async copy / TMA / DMA),
  host↔device (CPU↔GPU) synchronization elimination and transfer-prep
  reduction (`.item()` / `synchronize()` / blocking `cudaMemcpy` removal,
  H2D/D2H copy coalescing, pinned tensor packing, sync-free kernel contracts),
  algorithm
  replacement at fixed contract (full sort → quickselect / partial-sort /
  top-k-specific, naive scan → suffix automaton / rolling hash, exact
  softmax → online / blockwise), native-compilation transitions for hot
  CPU paths (pure Python / interpreted → Numba / Cython / torch.compile /
  native extension), batching/vectorization of per-token operations (hash
  construction, mask building, scatter/gather inner loops), and
  policy-defining defaults/constants/config tables for any of the above
  behaviors. If nothing genuinely qualifies, add nothing.

## Output (REQUIRED — strict JSON, no markdown fence)
Same schema as the previous pass:
{
  "module_qualified_name": "{module_qualified_name}",
  "candidates": [ ... ]
}

If you genuinely have no changes, return the previous list unchanged. Do NOT pad.
Do NOT explain outside the JSON object.
