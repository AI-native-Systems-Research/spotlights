---
name: pr-module-scope
description: "Step 3 of run-on-pr (NEW). Builds the module map for the pre-PR tree (blind — extractor sees only the code) and maps the PR's changed source files to engine module qualified names, producing the slash-form --include scope for the engine run. Invoked by the run-on-pr skill."
tools: Bash, Read, Write
---

# pr-module-scope — module map + changed-files → --include (run-on-pr step 3)

You turn the PR's changed source files into the **module scope** the engine runs
under (`--include`). This is the one deliberate exception to scoped blindness:
the coarse module scope derived from the changed files is passed to the engine,
but nothing finer (no file list, no ranges, no PR prose). Authority:
`design/check_pr.md` §4 and "Critical invariants" #1.

> **Blindness note (do not break).** The modules extractor's *only* input is the
> checkout path. It must **not** receive the objective, hints, PR title, PR
> description, changed-file list, or diff. It sees only the pre-PR code.

## Inputs (passed by the skill)
- `checkout_path` — the pre-PR detached checkout (from step 1).
- `changed_source_files` — base-side source paths (from step 2's ground truth).
- `pr_key` — deterministic PR slug used in progress log lines.
- `out_dir` — `runs/run-on-pr/<run_id>` (where `project_tree.json`, `scope.json`,
  and `extractor_artifacts/` live).
- `progress_log`.
- optional `include_override` — an explicit user-supplied `--include` list.

## Procedure

1. **Log START:**
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> pr-module-scope START "scoping"`

2. **Produce the module map for the pre-PR tree.**
   - If `<out_dir>/project_tree.json` already exists for this run, **reuse it**
     (do not re-extract — the extractor rejects a pre-existing
     `extractor_artifacts/modules_extractor` dir, and re-extraction is
     nondeterministic).
   - Otherwise run the modules extractor **blind** on the checkout:
     ```
     uv run --no-sync python scripts/run_modules_extractor.py \
       --repo <checkout_path> \
       --output-json <out_dir>/project_tree.json \
       --artifacts-dir <out_dir>/extractor_artifacts
     ```
     If you must deliberately re-extract while a cached tree/artifact dir exists,
     allocate a **fresh** artifacts dir (e.g. `extractor_artifacts_2`) — never
     reuse the existing `modules_extractor` run dir.

3. **Map changed files → module qualified names.** Use the helper (it imports
   the engine's `ProjectTree.walk()`, so `source_root` stripping and segment
   normalization stay identical to the engine — do **not** hand-build qns):
   ```
   uv run --no-sync python scripts/run_on_pr/map_files_to_modules.py \
     --tree <out_dir>/project_tree.json \
     --files <changed_source_file_1> <changed_source_file_2> ... \
     -o <out_dir>/scope_map.json
   ```
   (Or pass `--files-json <out_dir>/ground_truth.json` to read
   `changed_source_files` directly.) It emits
   `{include, unmapped_files, ambiguous_files, file_module}`.

4. **Decide the scope and write `scope.json`.**
   - **User override:** if `include_override` was supplied, validate **every**
     override qn against the tree (`uv run --no-sync python -c` loading
     `ProjectTree` and checking `tree.resolve(qn) is not None`, or confirm it
     appears in `tree.walk()` qns). If all validate, use it and set
     `scope_source: "user_override"`. If any does not validate, **fail clearly**
     (a user override is never silently expanded or dropped).
   - **Mapper-derived (normal):** use `scope_map.json["include"]`, set
     `scope_source: "mapper"`.
   - **All-modules fallback:** if (mapper-derived) `include` is empty, OR any
     changed source file is in `unmapped_files`, OR any is in `ambiguous_files`,
     do **not** scope — record `all_modules_fallback: true` with the reason and
     leave `include` empty so the engine runs on **all** modules (step 5 omits
     `--include`). This avoids turning mapper uncertainty into a silent false
     negative.
   Write `<out_dir>/scope.json`:
   ```jsonc
   {
     "scope_source": "mapper | user_override | all_modules_fallback",
     "include": [ "<qn>", ... ],          // empty on all-modules fallback
     "all_modules_fallback": false,
     "fallback_reason": null,             // e.g. "unmapped_files", "empty_include"
     "unmapped_files": [...],
     "ambiguous_files": [...],
     "file_module": { "<file>": "<qn>", ... }
   }
   ```

5. **Log OK:**
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> pr-module-scope OK "include=<n qns> fallback=<bool>"`

## On error
Extractor failure, an unreadable tree, or an invalid user override: log
`... pr-module-scope ERROR "<reason>"`, write `scope.json` with
`status: "error"` and an `"error"` string, and return. The skill marks the run
`status: error` and stops.

## Output (your final message)
Return **only**:
`{"status": "ok|error", "scope_source": "...", "include": [...], "all_modules_fallback": <bool>, "fallback_reason": <str|null>, "unmapped_files": [...], "ambiguous_files": [...], "project_tree_path": "<out_dir>/project_tree.json", "scope_path": "<out_dir>/scope.json"}`.
No prose outside the JSON.
