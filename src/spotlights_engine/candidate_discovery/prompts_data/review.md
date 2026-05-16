You are reviewing another agent's evolve-target candidate list for the same
module. Your job is adversarial: prune false positives, merge duplicates,
sharpen rationales / metrics, and add any *new* candidates the previous pass
missed that clear the same quality bar.

## Inputs
- previous candidates:
{prev_candidates_json}
- module_qualified_name: {module_qualified_name}
- module.path: {module_path}                     # every candidate's `file` MUST live here
- highest id accepted so far: {max_seen_candidate_id}

## Rules
- DO NOT inflate the list. If the previous pass was good, return it nearly unchanged.
- Remove any candidate whose `file` does not exist, whose `file` is outside
  `module.path`, or whose `line_start..line_end` is out of range for that file.
- Remove candidates that fail the evolve-target test: no measurable metric,
  no correctness oracle, no real headroom, or so entangled that the line
  range doesn't capture the optimization unit.
- Merge candidates that point at the same hot path with overlapping line ranges
  (keep one `id`, drop the others).
- For each KEPT candidate you MAY tighten `description`, `current_approach`,
  `evolve_rationale`, `metrics`, `estimated_impact`, and adjust `line_start` /
  `line_end` / `symbol` / `kind`. You MUST NOT change `id` or `file`. If you
  believe a kept candidate's `file` is wrong, drop the old `id` and mint a new
  one instead.
- For each NEW candidate, `evolve_rationale` must cite a specific code
  construct (function name, loop, alloc call, sync primitive, registry entry,
  factory registration, match arm, manifest line, dispatch/vtable assignment)
  and a concrete correctness oracle; `metrics` must name at least one
  quantifiable measurement.
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
  evolve-target quality bar (self-contained, measurable metric, real headroom,
  concrete correctness oracle). If nothing genuinely qualifies, add nothing.

## Output (REQUIRED — strict JSON, no markdown fence)
Same schema as the previous pass:
{
  "module_qualified_name": "{module_qualified_name}",
  "candidates": [ ... ]
}

If you genuinely have no changes, return the previous list unchanged. Do NOT pad.
Do NOT explain outside the JSON object.
