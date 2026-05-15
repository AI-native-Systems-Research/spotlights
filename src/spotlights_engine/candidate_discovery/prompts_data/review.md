You are reviewing another agent's candidate list for the same module. Your job
is adversarial: prune false positives, merge duplicates, sharpen rationales, and
add at most 5 *new* candidates the previous pass missed.

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
- Merge candidates that point at the same hot path with overlapping line ranges
  (keep one `id`, drop the others).
- For each KEPT candidate you MAY tighten `rationale` and adjust `line_start` /
  `line_end`. You MUST NOT change `id` or `file`. If you believe a kept
  candidate's `file` is wrong, drop the old `id` and mint a new one instead.
- For each NEW candidate, the rationale must cite a specific code construct
  (function name, loop, alloc call, sync primitive) — not a vague category.
- Mint new ids strictly greater than `{max_seen_candidate_id}`,
  zero-padded to four digits.
- Add at most 5 new candidates per review pass.

## Output (REQUIRED — strict JSON, no markdown fence)
Same schema as the previous pass:
{
  "module_qualified_name": "{module_qualified_name}",
  "candidates": [ ... ]
}

If you genuinely have no changes, return the previous list unchanged. Do NOT pad.
Do NOT explain outside the JSON object.
