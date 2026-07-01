---
name: pr-module-scope
description: "Step 3 of run-on-pr. Derives the engine's --include scope DIRECTLY from the PR's changed source-file paths (a module = the folder the code lives in), with no LLM modules extractor in the loop. Produces the slash-form --include scope for the engine run. Invoked by the run-on-pr skill."
tools: Bash, Read, Write
---

# pr-module-scope — changed-file paths → --include (run-on-pr step 3)

You turn the PR's changed source files into the **module scope** the engine runs
under (`--include`). A "module", for scoping, is simply **the folder the changed
source lives in**, expressed as a `source_root`-relative path. This is the one
deliberate exception to scoped blindness: the coarse folder scope derived from
the changed files is passed to the engine, but nothing finer (no file list, no
ranges, no PR prose). Authority: `design/check_pr.md` §4 and "Critical
invariants" #1.

> **No LLM extractor.** Scope is derived deterministically from the changed-file
> *paths* — there is no modules-extractor subprocess, no `project_tree.json`,
> and nothing about the pre-PR code content is read here. The engine treats each
> derived folder qn as a *virtual prefix* and expands it to the real modules
> nested beneath that folder (see `spotlights_manager/filters.py:apply_filter`),
> so the scope works without knowing the module map in advance.

## Inputs (passed by the skill)
- `changed_source_files` — base-side source paths (from step 2's ground truth).
- `pr_key` — deterministic PR slug used in progress log lines.
- `out_dir` — `runs/run-on-pr/<run_id>` (where `scope.json` lives).
- `progress_log`.
- optional `include_override` — an explicit user-supplied `--include` list.
- optional `checkout_path` — only needed to validate a user `include_override`.

## Procedure

1. **Log START:**
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> pr-module-scope START "scoping"`

2. **Derive the folder scope from the changed-file paths.** Use the helper (it
   reuses the engine's own `_normalize_source_root` and
   `_normalize_module_segment`, and infers `source_root` the same way
   `ProjectTree` does — `"src"` iff every changed file is under a top-level
   `src/`, else `""` — so the emitted qns match how the engine parses them; do
   **not** hand-build qns):
   ```
   uv run --no-sync python scripts/run_on_pr/derive_scope_from_paths.py \
     --files-json <out_dir>/ground_truth.json \
     -o <out_dir>/scope_map.json
   ```
   (Or pass `--files <f1> <f2> ...` directly.) It emits
   `{source_root, include, root_level_files, file_module}`.

3. **Decide the scope and write `scope.json`.**
   - **User override:** if `include_override` was supplied, keep it verbatim and
     set `scope_source: "user_override"`. (A user override is passed through
     unchanged — never silently expanded or dropped. If a `checkout_path` is
     available you MAY sanity-check each entry against the engine, but the
     engine itself fails fast on an unknown qn, so validation is optional here.)
   - **Path-derived (normal):** use `scope_map.json["include"]`, set
     `scope_source: "paths"`.
   - **All-modules fallback:** if the derived `include` is empty, OR any changed
     source file is in `root_level_files` (its folder is the source root, so
     there is no sub-folder to scope to), do **not** scope — record
     `all_modules_fallback: true` with the reason and leave `include` empty so
     the engine runs on **all** modules (step 5 omits `--include`). This avoids
     turning a root-level change into a silent under-scope / false negative.
   Write `<out_dir>/scope.json`:
   ```jsonc
   {
     "scope_source": "paths | user_override | all_modules_fallback",
     "source_root": "src | \"\"",
     "include": [ "<qn>", ... ],          // empty on all-modules fallback
     "all_modules_fallback": false,
     "fallback_reason": null,             // e.g. "root_level_files", "empty_include"
     "root_level_files": [...],
     "file_module": { "<file>": "<qn>", ... }
   }
   ```

4. **Log OK:**
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> pr-module-scope OK "include=<n qns> fallback=<bool>"`

## On error
An unreadable `ground_truth.json` or a malformed override: log
`... pr-module-scope ERROR "<reason>"`, write `scope.json` with
`status: "error"` and an `"error"` string, and return. The skill marks the run
`status: error` and stops.

## Output (your final message)
Return **only**:
`{"status": "ok|error", "scope_source": "...", "source_root": "...", "include": [...], "all_modules_fallback": <bool>, "fallback_reason": <str|null>, "root_level_files": [...], "scope_path": "<out_dir>/scope.json"}`.
No prose outside the JSON.
