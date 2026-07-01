---
name: compare-pr-run
description: "The comparison tail of run-on-pr, run standalone against an existing run dir whose engine step already produced spotlights-out/result.json. Explodes the engine's candidates into the flat matcher shape, matches them against the PR's precomputed ground-truth base-side line ranges (module-scoped line recall) and against the PR's cited papers (paper-citation recall), then writes/updates report.md and prints the headline. Use after `spotlights-engine` has finished on a run-on-pr checkout and you want the recall verdict — i.e. the run-on-pr steps AFTER the engine run."
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
- **`run-on-pr`** — full harness from a bare PR URL (checkout → diff → scope →
  engine → match → report).
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

`runs/` is gitignored scratch — never `git add` anything under it.

## Procedure

### 1. Resolve inputs
Accept the run dir (`$RUN`). If the user gives a PR URL or run_id instead,
resolve to the matching `runs/run-on-pr/<run_id>` (there is exactly one dir per
run_id). Read `$RUN/pr.json` for the PR link/base/head and `$RUN/scope.json`
(if present) for the `--include` set and `unmapped_files`, so the report can
state what was audited. Echo the resolved `$RUN`, PR, and scope back before
proceeding.

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
(`<pr_key>` is not stored in `pr.json`; derive it as `basename(checkout_path)`,
e.g. `vllm-project__vllm__pr39008`.)

### 3. Match (agent: `match-evaluator` — run-on-pr step 5, verbatim)
Spawn `match-evaluator` with:
`{candidates_path:"$RUN/candidates.json", ground_truth_path:"$RUN/ground_truth.json",
addition_tolerance:3, cited_papers_path:"$RUN/cited_papers.json",
result_json_path:"$RUN/spotlights-out/result.json", pr_key:<pr_key>,
out_dir:"$RUN", progress_log:"$RUN/progress.log"}`.
It writes `$RUN/match.json` (line recall) and — only when papers were cited —
`$RUN/paper_match.json` (paper-citation recall), and returns the verdict JSON.
Do not re-derive ranges or overlaps yourself; the agent owns the deterministic
interval/key arithmetic.

### 4. Persist + report (run-on-pr step 7)
Write/overwrite `$RUN/report.md` (human-readable). If a prior `report.md`
exists from a failed run, **replace** its verdict sections rather than leaving
the stale `error` bucket. Include:
- PR link, base commit (short), head; the objective the engine was given (from
  `report.md`/`result.json` run header if recorded).
- Changed source files + base-side ranges (from `ground_truth.json`); modules
  run (the `--include` set, or all-modules fallback) and any `unmapped_files`
  (carry the known blind spot forward); candidate count.
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
- Pointer to the rendered `$RUN/spotlights-out/index.md` and per-candidate pages.

Log the final line:
`bash scripts/run_on_pr/log.sh $RUN/progress.log <pr_key> compare-pr-run OK "line_hit=<t/f> paper=<hit/miss/n_a> bucket=<bucket>"`

### 5. Headline to the user
Print: `pr_line_hit` (hit/miss), the matched candidate + true rank, the line
bucket, the **paper signal** (`paper_hit`/`n/a`, with the matched paper title on
a hit), and where the artifacts live (`$RUN`).

## Bucketing (same as run-on-pr; this skill only issues line-recall verdicts)
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
  `spotlights-engine` command first (or use `run-on-pr` end-to-end).
- Re-running this skill on the same `$RUN` is idempotent: it just re-explodes,
  re-matches, and rewrites the report.
- `.claude/` is version-controlled in this repo (its `.gitignore` line is
  commented out); these skill files are committed alongside the code. `runs/`
  stays gitignored scratch — never `git add` anything under it.
