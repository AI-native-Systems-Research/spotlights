---
name: pr-diff-scope
description: "Step 2 of run-on-pr. Computes one PR's ground truth: changed source files, sub-folders, and BASE-SIDE changed line ranges from a checkout pinned to the merge-base; ALSO extracts the PR's cited paper URLs (the paper-citation ground truth). Invoked by the run-on-pr skill."
tools: Bash, Read, Write
---

# pr-diff-scope — ground-truth line ranges + cited papers (run-on-pr step 2)

You compute the **ground truth** the recall measurement compares against: the
set of code locations a merged PR actually changed, expressed in the **base
file's** line numbers (the pre-PR checkout the engine will see), plus the set of
academic papers the PR's prose cites. You do *no* judgement and *no* code
reading for meaning — the diff side is pure, deterministic diff math, and you
must lean on the helper scripts rather than counting lines or scraping URLs by
eye.

Both outputs stay **on the harness side** and are **never** passed to the
engine — they are label data used only to score after the fact (design "Critical
invariants" #1, #3).

Authority: `design/check_pr.md` §3.

## Inputs (passed by the skill)
- `checkout_path` — local clone, already `git checkout <base_commit>` (detached).
- `base_commit`, `head_commit` — resolved by step 1 (`pr.json`).
- `pr_url` — for fetching prose (cited-paper extraction).
- `pr_key`, `progress_log`, `out_dir` — for logging and where to write output.
- optional `addition_tolerance` — carried through to output for the evaluator
  (default 3; this step only records it, the `match-evaluator` applies it).

## Procedure

### A. Diff → base-side ground truth

1. **Log START.** Run:
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> pr-diff-scope START "diffing <base>..<head>"`
   (logging never aborts the step; ignore its exit status).

2. **Compute the diff in base-side coordinates with the TWO-DOT form.** From
   inside the checkout:
   ```
   git -C <checkout_path> diff <base_commit> <head_commit>
   ```
   ⚠️ Use **two-dot** `git diff <base> <head>`, **never** three-dot `base...head`.
   Two-dot's left side *is* the checked-out tree by construction; three-dot
   re-bases line numbers onto a different merge-base and corrupts every overlap
   test when `base_commit` was overridden. (When `base_commit` is the merge-base
   — the normal case — the two forms are identical, so two-dot is always safe.)

3. **Parse the diff into base-side ranges with the helper** — do not hand-parse
   `@@` headers:
   ```
   git -C <checkout_path> diff <base_commit> <head_commit> \
     | python3 scripts/run_on_pr/diff_ranges.py
   ```
   The helper owns: hunk→range math (modified `[a,a+b-1]`; pure-addition
   zero-width anchor `[max(1,a),max(1,a)]` flagged `addition_only`; top-of-file
   `a==0` clamp), the **source vs non-source filter** (tests/docs/generated/
   vendored/config — the documented knob lives in the script), the sub-folder
   set (deduped parent dirs of changed *source* files), and the `new_files` set
   (diff `--- /dev/null`). Capture its JSON.

4. **Assemble `ground_truth.json`.** The helper already sets `status` (`"ok"`,
   or `"no_source_changes"` when no source files survive filtering). Just add the
   `addition_tolerance` knob (default 3) to its object. When status is
   `no_source_changes`, still write the file (with `excluded_files`) for
   auditability — the skill skips the remaining steps (scope + command emit)
   for that PR.
   Final shape:
   ```jsonc
   {
     "status": "ok | no_source_changes",
     "addition_tolerance": 3,
     "changed_source_files": [...],
     "subfolders": [...],
     "changed_ranges": { "<file>": [ {start,end,addition_only,new_file}, ... ] },
     "addition_only_ranges": [ {file,start,end}, ... ],
     "new_files": [...],
     "excluded_files": [...]
   }
   ```
   Write it to `<out_dir>/ground_truth.json` with the Write tool.

### B. Cited-paper extraction (the paper-citation ground truth)

This is the **only** step that reads PR prose for paper references. Like the
diff, it stays on the harness side and is **never** passed to the engine. An
empty result is fine — the paper signal is then `n/a`. Do **not** let this block
or abort the line-recall path: if prose fetch or extraction fails or finds
nothing, log it and still return a valid result for part A.

5. **Fetch the PR prose** (title + body, plus closing-issue bodies when cheap):
   ```
   gh pr view <pr_url> --json title,body,closingIssuesReferences > <out_dir>/pr_prose.json
   ```
   (If `closingIssuesReferences` is unavailable, `--json title,body` is fine.)

6. **Extract cited papers** conservatively (arxiv / DOI / known paper hosts
   only — the script owns the allow-list and the engine's URL normalizer):
   ```
   uv run --no-sync python scripts/run_on_pr/extract_pr_papers.py \
     --pr-json <out_dir>/pr_prose.json -o <out_dir>/cited_papers.json
   ```
   ⚠️ `extract_pr_papers.py` imports the engine's normalizer (`_normalize_url`),
   which needs the project venv (Python ≥3.10) — always invoke it via
   `uv run --no-sync python ...`, never a bare `python3`.
   Capture the paper count for the log line. If extraction fails, write
   `cited_papers.json` as `{"papers": [], "source_fields": []}` so the
   downstream paper matcher (`compare-pr-run`) can still read it.

7. **Log OK** with a one-line summary, e.g.:
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> pr-diff-scope OK "<N> files, <M> subfolders, <P> cited papers"`

## On error (diff side)
If `git diff` fails (bad commits, not a repo, etc.): log
`... pr-diff-scope ERROR "<reason>"`, write a `ground_truth.json` with
`status: "error"` and an `"error"` string field, and return — do not guess
ranges. The skill treats this PR as `status: error` and stops. A failure in part
B (paper extraction) is **not** an error for the run — keep the `ok`
ground-truth and an empty `cited_papers.json`.

## Output (your final message)
Return **only** a compact JSON object:
`{"status": "...", "ground_truth_path": "<out_dir>/ground_truth.json", "num_source_files": N, "num_subfolders": M, "subfolders": [...], "changed_source_files": [...], "cited_papers_path": "<out_dir>/cited_papers.json", "num_cited_papers": P}`.
The skill reads `changed_source_files` to drive the module scope (step 4) and
`num_cited_papers` to decide whether to run the paper matcher. Do not add prose
outside the JSON.
