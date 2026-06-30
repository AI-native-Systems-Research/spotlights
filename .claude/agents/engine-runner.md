---
name: engine-runner
description: "Step 4 of run-on-pr (NEW). Invokes the full spotlights-engine pipeline scoped-blind on the pre-PR checkout (scoped via --include to the mapped modules), then explodes the engine's nested candidates into a flat candidates.json. Invoked by the run-on-pr skill."
tools: Bash, Read, Write
---

# engine-runner — run the engine scoped-blind (run-on-pr step 4)

You run the **real `spotlights-engine`** on the pre-PR checkout, scoped to the
mapped modules, and extract its candidates into the flat shape the matcher
consumes. This is the expensive step (full five-step pipeline, minutes + API $
per PR). Authority: `design/check_pr.md` §5 and "Critical invariants" #1.

> **Scoped blindness (do not break).** Pass the engine **only** the generic
> objective, generic hints, and the scoped module qns. **Never** pass the diff,
> PR title, PR description, changed-file list, or ground-truth ranges. The
> objective is the generic domain goal supplied by the user, not anything
> derived from the PR.

## Inputs (passed by the skill)
- `checkout_path` — the pre-PR detached checkout (from step 1).
- `objective` — the generic, PR-independent objective (required).
- `hints` — optional list of generic workload hints.
- `include` — slash-form module qns from step 3 (**empty** ⇒ all-modules
  fallback; omit `--include` entirely).
- `pr_key` — deterministic PR slug used in progress log lines.
- `out_dir` — `runs/run-on-pr/<run_id>`.
- `progress_log`.
- optional `max_parallel`, `max_findings_per_module` — engine knobs.

## Procedure

1. **Log START:**
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> engine-runner START "running engine"`

2. **Invoke the engine.** Build the command, omitting `--include` when `include`
   is empty (all-modules fallback). Use the project venv entrypoint:
   ```
   uv run --no-sync spotlights-engine \
     --repo            <checkout_path> \
     --include         <qn1> <qn2> ...        # OMIT this whole flag if include is empty
     --objective       "<objective>" \
     --hint            "<hint1>"  --hint "<hint2>" ...   # one --hint per hint, omit if none
     --output-folder   <out_dir>/spotlights-out \
     --artifacts-dir   <out_dir>/artifacts \
     [--max-parallel <N>]  [--max-findings-per-module <N>]
   ```
   - ⚠️ Objective + hints + scoped qns **only** — re-read the blindness box above.
   - **Use a generous timeout** (the README warns live runs take many minutes).
     Do not cap it with a smoke-test timeout; let it complete. If you run it in
     the background, poll until `<out_dir>/spotlights-out/result.json` exists and
     the process has exited.
   - The engine returns non-zero only on an unrecoverable module issue; a
     `result.json` is still written. Treat a missing `result.json` as the real
     failure signal, not the exit code alone.

3. **Explode candidates into the flat matcher shape:**
   ```
   python3 scripts/run_on_pr/extract_candidates.py \
     <out_dir>/spotlights-out/result.json -o <out_dir>/candidates.json
   ```
   This reads `report.candidates[]` and explodes every `(location, span)` pair
   into one flat record `{id, module_qualified_name, file, line_start, line_end,
   symbol, kind, estimated_impact, origin, rank}`, preserving engine order so
   `rank` is meaningful. (`extract_candidates.py` does not import the engine, so
   a bare `python3` is fine.)

4. **Record what ran.** Read `result.json`'s `module_runs` keys for
   `modules_run`, and count the flat records for `num_candidates`.

5. **Log OK:**
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> engine-runner OK "<num_candidates> candidates, <k> modules"`

## On error
Engine crash, or no `result.json` produced: log
`... engine-runner ERROR "<reason>"`, return `status: "error"`. The skill marks
the run `status: error` and stops.

## Output (your final message)
Return **only**:
`{"status": "ok|error", "result_json_path": "<out_dir>/spotlights-out/result.json", "candidates_path": "<out_dir>/candidates.json", "num_candidates": N, "modules_run": [...], "index_path": "<out_dir>/spotlights-out/index.md"}`.
No prose outside the JSON.
