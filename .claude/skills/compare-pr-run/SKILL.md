---
name: compare-pr-run
description: "The comparison tail of run-on-pr, run standalone against an existing run dir whose engine step already produced spotlights-out/result.json. Explodes the engine's candidates into the flat matcher shape, matches them against the PR's precomputed ground-truth base-side line ranges (module-scoped line recall) and against the PR's cited papers (paper-citation recall), then writes/updates report.md — including the full information for each matched candidate (description, current approach, evolve rationale, impact explanation, attached proposals) and the candidate's code before vs. after the PR — and prints the headline. Use after `spotlights-engine` has finished on a run-on-pr checkout and you want the recall verdict — i.e. the run-on-pr steps AFTER the engine run."
---

# compare-pr-run — score an already-completed engine run against PR ground truth

The engine has already run (someone kicked off `spotlights-engine` on a
`run-on-pr` checkout and it produced `result.json`). This skill does **only the
steps after the run**: explode candidates → match against the PR's ground truth
and cited papers → report. It reuses the artifacts a prior `run-on-pr` (steps
1–3) already wrote into `$RUN` — it does **not** re-checkout, re-diff, re-scope,
or re-run the engine. Authority: `design/check_pr.md` §5(tail)–§7 and
"Bucketing"; this is a strict subset of the `run-on-pr` skill.

## When to use vs. run-on-pr
- **`run-on-pr`** — prep harness from a bare PR URL (checkout → diff → scope →
  emit the scoped-blind engine command).
- **`compare-pr-run`** (this) — the engine already finished (e.g. an earlier
  run errored mid-engine and you re-ran the command by hand); you just want the
  verdict. Also the right tool to re-score after a manual engine re-run without
  redoing the expensive earlier steps.

## Preconditions (verify, stop clearly if unmet)
Given a run dir `$RUN` (e.g. `runs/run-on-pr/<run_id>`):
- `$RUN/spotlights-out/result.json` **exists and is non-empty** — this is the
  real "engine finished" signal (the engine can exit non-zero yet still write
  it; and an empty `spotlights-out/` means it never completed → re-run the
  engine command first, don't proceed).
- `$RUN/ground_truth.json` exists (from `run-on-pr` step 2). If absent, the
  earlier steps never ran — use `run-on-pr` instead.
- `$RUN/cited_papers.json` exists (may hold an empty `papers` list → paper
  signal `n/a`; that's fine, not an error).
- The engine is callable for the paper matcher:
  `uv run --no-sync python -c "import spotlights_engine"` succeeds.
- Helpers exist: `scripts/run_on_pr/extract_candidates.py`, `overlap.py`,
  `match_papers.py`, `log.sh`.
- `$RUN/checkout` is a git repo with both the base and head commits reachable
  (from `run-on-pr` step 1). Needed only for the **before/after code** section
  (step 6): `git -C $RUN/checkout show <base>:<file>` / `<head>:<file>`. If the
  checkout or the head commit is absent, still write the rest of the report and
  note the before/after section was skipped — it is not a hard precondition.

`runs/` is gitignored scratch — never `git add` anything under it.

## Procedure

### 1. Resolve inputs
Accept the run dir (`$RUN`). If the user gives a PR URL or run_id instead,
resolve to the matching `runs/run-on-pr/<run_id>` (there is exactly one dir per
run_id). Read `$RUN/pr.json` for the PR link/base/head and `$RUN/scope.json`
(if present) for the `--include` set, `scope_source`, `all_modules_fallback`,
`fallback_reason`, and `root_level_files`, so the report can state what was
audited. Echo the resolved `$RUN`, PR, and scope back before proceeding.

### 2. Explode candidates (done inline, no full engine)
The engine already produced `result.json`; only its candidate-extraction tail
is needed, so run the extractor directly — never re-invoke the engine:
```
python3 scripts/run_on_pr/extract_candidates.py \
  $RUN/spotlights-out/result.json -o $RUN/candidates.json
```
This reads `report.candidates[]` and explodes every `(location, span)` pair
into one flat record `{id, module_qualified_name, file, line_start, line_end,
symbol, kind, estimated_impact, origin, rank}`, preserving engine order so
`rank` is meaningful. Then read `result.json`'s `module_runs` keys for
`modules_run` and count the flat records for `num_candidates`. Log it:
`bash scripts/run_on_pr/log.sh $RUN/progress.log <pr_key> compare-pr-run CANDIDATES "<N> candidates, <k> modules"`
(`<pr_key>` is not stored in `pr.json`; use `basename($RUN)` — the run_id, e.g.
`vllm-project__vllm__pr39008__add-fused-moe__bc8d9c70` — as the progress-log
label. It is only a log tag, so the exact PR slug is not required.)

### 3. Match (agent: `match-evaluator`)
Spawn `match-evaluator` with:
`{candidates_path:"$RUN/candidates.json", ground_truth_path:"$RUN/ground_truth.json",
addition_tolerance:3, cited_papers_path:"$RUN/cited_papers.json",
result_json_path:"$RUN/spotlights-out/result.json", pr_key:<pr_key>,
out_dir:"$RUN", progress_log:"$RUN/progress.log"}`.
It writes `$RUN/match.json` (line recall) and — only when papers were cited —
`$RUN/paper_match.json` (paper-citation recall), and returns the verdict JSON.
Do not re-derive ranges or overlaps yourself; the agent owns the deterministic
interval/key arithmetic.

### 4. Persist + report
Write/overwrite `$RUN/report.md` (human-readable). If a prior `report.md`
exists from a failed run, **replace** its verdict sections rather than leaving
the stale `error` bucket. Include:
- PR link, base commit (short), head; the objective the engine was given (from
  `report.md`/`result.json` run header if recorded).
- Changed source files + base-side ranges (from `ground_truth.json`); modules
  run (the `--include` set, or all-modules fallback), the `scope_source`, and
  any `root_level_files` / `fallback_reason` (carry the structural scope caveat
  forward); candidate count.
- **Verdict:** `pr_line_hit` headline + `file_hit`/`folder_hit` diagnostics +
  `new_file_only`. For each hit range, the matching candidate(s) with
  `estimated_impact` and rank ("found, ranked #k"). List `missed_ranges` for
  error analysis.
  > Rank note: `overlap.py` numbers `rank` by flat-record position, which
  > differs from a candidate's discovery rank when a candidate has multiple
  > spans. For the "ranked #k" line, recover the true candidate rank by joining
  > each `matched_pairs[].candidates[].id` back to the candidate-level `rank` in
  > `candidates.json` (which preserves `report.candidates` order).
- **Paper-citation section** when `paper_cited`: the cited paper(s); for each
  `paper_hit` the matched `Finding` title/url/source_type plus which candidate
  it rode in on via `via_candidate_ids` (or "in findings, unattached"). Carry
  the scope caveat: a paper relevant to an *un-scoped* module is a structural
  miss ("not within the audited scope"), not "the engine couldn't find it".
- **Matched candidate — full information** (one subsection per candidate that
  scored a line hit; see step 5 for how to build it): the candidate's complete
  record, not just the matcher fields. `candidates.json` is flat and carries
  only `{id, file, line_start/end, symbol, kind, estimated_impact, origin,
  rank}` — the rich prose fields live **only** in `result.json`'s
  `report.candidates[]`, so read them from there by joining on `id`. For each
  matched candidate include: `id`, `module_qualified_name`, `origin`, true rank
  (`#k` of N — recovered per the rank note above), every `location`
  (`file` + each span's `line_start`–`line_end`, `symbol`, `kind`),
  `estimated_impact`, and the full text of `description`, `current_approach`,
  `evolve_rationale`, and `estimated_impact_explanation`. Then list **every
  attached proposal** (`proposals[]`): for each, its `id`, `source`, `author`,
  the originating `finding_ref_id`/`anomaly_ref_ids` if any, and its `title` +
  `description` (add `mechanism`/`required_changes`/`expected_effect` when they
  sharpen the summary). Flag the proposal that matches what the PR actually did.
- **Code before and after the PR** (step 6): for each matched candidate's hit
  location, show the base-side code the engine flagged and the head-side code
  the PR shipped, so the reader can see what the engine pointed at vs. what
  changed. See step 6 for the extraction commands.
- Pointer to the rendered `$RUN/spotlights-out/index.md` and per-candidate pages.

Log the final line:
`bash scripts/run_on_pr/log.sh $RUN/progress.log <pr_key> compare-pr-run OK "line_hit=<t/f> paper=<hit/miss/n_a> bucket=<bucket>"`

### 5. Assemble the matched-candidate full information
For each candidate `id` that appears in `match.json`'s `matched_pairs[].candidates[]`
(i.e. scored a line hit), open `$RUN/spotlights-out/result.json` and find the
matching object in `report.candidates[]` (join on `id`). That object is the
authoritative source for the rich fields the flat `candidates.json` drops
(`description`, `current_approach`, `evolve_rationale`,
`estimated_impact_explanation`, `locations[]`, `proposals[]`). Recover the true
candidate rank from its 1-based position in `report.candidates` (matches the
`rank` in `candidates.json`). Render one "**`<cand-id>`** — full information"
subsection per matched candidate as described in step 4; if multiple candidates
matched, emit one per candidate.

### 6. Extract the code before and after the PR
For each matched candidate's hit location (`file` + span `line_start`–`line_end`
from `result.json`), pull the code from the `$RUN/checkout` git repo. Read
`base_commit` and `head_commit` from `$RUN/pr.json`:
```
git -C $RUN/checkout show <base_commit>:<file> | sed -n '<start>,<end>p'   # before (what the engine saw)
git -C $RUN/checkout show <head_commit>:<file>                             # after (what the PR shipped)
```
The engine's span is a **base-side** range, so the "before" snippet uses those
exact lines. For "after", the same code has usually moved — locate the
corresponding region in the head-side file (the PR's edit is inside/near the
ground-truth range) and quote enough context to show the change; note the
head-side line numbers. Present both as fenced code blocks under a "**Code
before and after the PR**" heading, and add a one-line note on how the shipped
change relates to the flagged candidate / matched proposal. If the checkout or
head commit is missing, skip this section and say so — do not fail the report.
The `backup-sgl-project__sglang__pr1459__...` run's `report.md` is a worked
example of steps 5–6; mirror its structure.

### 7. Headline to the user
Print: `pr_line_hit` (hit/miss), the matched candidate + true rank, the line
bucket, the **paper signal** (`paper_hit`/`n/a`, with the matched paper title on
a hit), and where the artifacts live (`$RUN`).

## Bucketing
- `new_file_only` — every changed source file is brand-new → structurally
  unrecallable by base-side line overlap (reported, not a plain miss).
- `valid` — matcher ran; `pr_line_hit ∈ {true,false}` is the result.
- `no_source_changes` / `error` are earlier-step buckets — if `ground_truth.json`
  shows no source files, report `no_source_changes` and skip the matcher; a
  missing/empty `result.json` is a precondition failure (re-run the engine),
  not something this skill scores.

The **paper-citation** signal is a separate orthogonal axis: `paper_cited:false`
→ `n/a`; `paper_cited:true,paper_hit:true` → engine independently surfaced the
paper; `paper_cited:true,paper_hit:false` → cited paper not among findings. A PR
can be a line miss but a paper hit, or vice versa — record both. `paper_hit` is
a same-paper match (URL/title dedup keys), not semantic prior-art.

## Notes
- Never re-run the engine here — that's the expensive step this skill
  deliberately skips. If `result.json` is missing, tell the user to run the
  `spotlights-engine` command first (or use `run-on-pr` to regenerate it).
- Re-running this skill on the same `$RUN` is idempotent: it just re-explodes,
  re-matches, and rewrites the report.
- `.claude/` is version-controlled in this repo (its `.gitignore` line is
  commented out); these skill files are committed alongside the code. `runs/`
  stays gitignored scratch — never `git add` anything under it.
