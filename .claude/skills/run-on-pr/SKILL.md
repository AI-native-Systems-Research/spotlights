---
name: run-on-pr
description: "PR-grounded recall harness for the full spotlights-engine. Given ONE merged GitHub PR plus a generic objective, re-runs the real spotlights-engine pipeline BLIND on the PR's pre-merge code — scoped via --include to the module(s) containing the changed files — and measures whether the engine independently surfaces a candidate whose code region overlaps the lines the PR changed (module-scoped line recall). Also records a paper-citation hit when the PR cites a paper the engine's deep research independently surfaces. Use when asked to validate/score the engine against a real merged PR, or to run the engine on a specific PR's pre-merge code."
---

# run-on-pr — single-PR recall harness for the full engine

Measure whether the **real `spotlights-engine`** independently flags the code a
merged PR actually changed — and, when the PR cites a paper, whether the
engine's deep research independently surfaces that same paper. Each merged PR is
ground truth ("an expert decided *this* location was worth changing"); the
harness runs the engine **cold on the pre-PR tree**, scoped to the PR-derived
modules but never seeing the diff, and asks: did the engine's candidates land on
the same lines? Full design + rationale: `design/check_pr.md` (read it before
changing behavior).

This is the heavier sibling of the reference `check-prs` skill: the auditor is
no longer a lightweight bootstrap prompt — it is the full five-step engine
invoked exactly as the README documents.

## Critical rules — do not break (design "Critical invariants")
1. **Scoped blindness.** The engine runs on the pre-PR checkout with **no** PR
   title, description, diff, changed-file list, or ground-truth ranges. The one
   deliberate exception is the coarse **module scope** (`--include`) derived from
   the changed files. The only natural-language input is the **generic,
   PR-independent objective** (+ generic hints). The PR prose is read **only** by
   `pr-diff-scope` for cited-paper extraction, and that result is never given to
   the engine.
2. **Base = `merge-base(baseRefOid, headRefOid)`**, never `mergeCommit^1`. Full
   clone, never shallow.
3. **Two-dot diff** `git diff <base> <head>` so ranges are in the pre-PR file's
   line numbers — the same frame the engine's candidates use.
4. **`runs/` is gitignored scratch.** Never `git add` anything under it.

## Architecture (linear pipeline, you drive it)
```
run-on-pr skill (you, main session)        — single-PR orchestrator
  ├─ pr-checkout       (step 1) — pre-PR checkout at the merge-base
  ├─ pr-diff-scope     (step 2) — base-side line ranges + cited-paper URLs
  ├─ pr-module-scope   (step 3) — module map + changed-files→modules → --include
  ├─ engine-runner     (step 4) — run the full engine scoped-blind via --include
  └─ match-evaluator   (step 5) — candidates ∩ ground truth (line recall)
                                 + findings ∩ cited papers (paper recall)
```
Delegate each step to its subagent (one at a time — steps are dependent).
Deterministic math lives in `scripts/run_on_pr/` helpers; agents call them.

## Cost note
A full engine run per PR is the dominant cost (minutes + API $). The `--include`
scoping is the main lever — surface `modules_run` so the user sees what was
audited. This is **one PR per invocation**; there is no list-level fan-out.

## Procedure

### 0. Pre-flight
- `gh auth status` must succeed (clone + PR metadata). Stop clearly if not.
- Confirm the engine is callable: `uv run --no-sync spotlights-engine --help`.
  Stop if absent.
- Confirm helpers exist under `scripts/run_on_pr/`:
  `diff_ranges.py`, `overlap.py`, `log.sh`, `extract_candidates.py`,
  `map_files_to_modules.py`, `extract_pr_papers.py`, `match_papers.py`.

### 1. Parse inputs
Accept a **single PR URL** plus an **objective** (required), and optional:
generic `hints`, an explicit `--include` override, an explicit `base_commit`, and
engine knobs (`max_parallel`, `max_findings_per_module`). Minimal parsing — one
PR, no list grammar.

Derive two deterministic keys (no `Date.now()` — identical inputs must resume,
but a changed objective must not collide with an old engine artifacts dir):
- **`pr_key`** = a slug of the PR URL (e.g. `owner__repo__pr<n>`). Used for the
  reusable clone cache `runs/run-on-pr/_repos/<pr_key>`.
- **`run_id`** = a short hash over (pr_url + objective + sorted hints + include
  override + base override + relevant engine knobs). `RUN=runs/run-on-pr/<run_id>`.

`mkdir -p $RUN`. Echo the parsed inputs (PR, objective, hints, any
override/base, run_id) back to the user for confirmation before the expensive
run.

### 2. Checkout (agent: `pr-checkout`)
Spawn `pr-checkout` with `{repo, pr_url, pr_key, repos_dir:"runs/run-on-pr/_repos",
out_dir:"$RUN", progress_log:"$RUN/progress.log", base_commit?}`. It resolves PR
metadata, full-clones (cached by `pr_key`), fetches `refs/pull/<n>/head` + base,
computes the merge-base, checks out detached, and writes `$RUN/pr.json`.
On `status:error` → stop and report.

### 3. Diff scope / ground truth + cited papers (agent: `pr-diff-scope`)
Spawn `pr-diff-scope` with `{checkout_path, base_commit, head_commit, pr_url,
pr_key, out_dir:"$RUN", progress_log, addition_tolerance:3}`. It writes
`$RUN/ground_truth.json` (base-side `changed_ranges`, `changed_source_files`,
`subfolders`, `new_files`) **and** `$RUN/cited_papers.json` (conservative
arxiv/DOI/paper-host URLs from the PR prose — the only place PR prose is read).
- If `status: no_source_changes` → skip steps 4–6, report bucket
  `no_source_changes` (paper signal `n/a`).
- On `status: error` → stop.
- A paper-extraction failure is **not** fatal — it yields an empty
  `cited_papers.json` and the paper signal becomes `n/a`.

### 4. Module scope (agent: `pr-module-scope`)
Spawn `pr-module-scope` with `{checkout_path, changed_source_files (from step 3),
pr_key, out_dir:"$RUN", progress_log, include_override?}`. It runs the modules
extractor **blind** on the checkout (→ `$RUN/project_tree.json`; reuse if
present), maps changed files → slash-form qns via `map_files_to_modules.py`, and
writes `$RUN/scope.json`.
- Mapper-derived scope is the normal path.
- **All-modules fallback** (record the reason, run with no `--include`) when the
  derived `include` is empty, or any changed source file is unmapped or
  ambiguous — never let mapper uncertainty become a silent false negative.
- A user `--include` override must validate against the tree or fail clearly; it
  is not auto-expanded.
- On `status: error` → stop.

### 5. Engine run (agent: `engine-runner`)
Spawn `engine-runner` with `{checkout_path, objective, hints, include (from
scope.json; empty ⇒ all-modules), pr_key, out_dir:"$RUN", progress_log,
max_parallel?, max_findings_per_module?}`. It runs
`uv run --no-sync spotlights-engine` into `$RUN/spotlights-out` +
`$RUN/artifacts`, then explodes candidates → `$RUN/candidates.json`.
- ⚠️ Pass the **objective + hints + scoped qns only** — never the diff, PR
  title/description, changed-file list, or ranges.
- Use a generous timeout; the run can take many minutes.
- On `status: error` → stop, bucket `error`.

### 6. Match (agent: `match-evaluator`)
Spawn `match-evaluator` with `{candidates_path:"$RUN/candidates.json",
ground_truth_path:"$RUN/ground_truth.json", addition_tolerance:3,
cited_papers_path:"$RUN/cited_papers.json",
result_json_path:"$RUN/spotlights-out/result.json", pr_key, out_dir:"$RUN",
progress_log}`. It writes `$RUN/match.json` (line recall) and — when papers were
cited — `$RUN/paper_match.json` (paper-citation recall).

### 7. Persist + report
Write `$RUN/report.md` (human-readable): PR link, base commit (short), merge
style; changed source files + base-side ranges; modules run (the `--include`
set, or all-modules fallback) and any `unmapped_files`; candidate count;
**verdict** (`pr_line_hit` headline + file/folder diagnostics + `new_file_only`);
for each hit range the matching candidate(s) with `estimated_impact` + rank
("found, ranked #k"); `missed_ranges` for error analysis;
> Rank note: `overlap.py` (verbatim) numbers `rank` by flat-record position,
> which differs from the candidate's discovery rank when a candidate has
> multiple spans. For the "ranked #k" line, recover the true candidate rank by
> joining each `matched_pairs[].candidates[].id` back to the candidate-level
> `rank` in `candidates.json` (which preserves `report.candidates` order).

Include a **paper-citation section** when `paper_cited` (the cited paper(s), and
for each `paper_hit` the matched `Finding` title/url/source_type plus which
candidate it rode in on via `via_candidate_ids`, or "in findings, unattached");
and a pointer to the rendered `$RUN/spotlights-out/index.md` and per-candidate
pages.

Print the headline to the user: `pr_line_hit` (hit/miss), the matched candidate
+ rank, the line bucket, the **paper signal** (`paper_hit` / `n/a`, with the
matched paper title when hit), and where the artifacts live.

## Bucketing (line-recall verdict)
- `error` — any step failed.
- `no_source_changes` — step 3 found no source files (steps 4–6 skipped).
- `new_file_only` — every changed source file is brand-new → structurally
  unrecallable by base-side line overlap (reported, not a plain miss).
- `valid` — reached the matcher; `pr_line_hit ∈ {true,false}` is the result.

The **paper-citation** signal is a separate orthogonal axis (not a bucket value):
- `paper_cited: false` → `n/a` (also `n/a` when the run never reached the engine).
- `paper_cited: true, paper_hit: true` → engine independently surfaced the paper.
- `paper_cited: true, paper_hit: false` → cited paper not among the findings.
A PR can be a line miss but a paper hit, or vice versa — record both. `paper_hit`
is a **same-paper** match (engine's URL/title dedup keys), not semantic prior-art;
`matched_on: "url"|"title"` lets a title-only match be eyeballed.

## Notes
- `runs/` is entirely gitignored — never `git add` anything under it.
- Re-running identical inputs reuses the same `run_id` (addressable, resumable);
  the PR-keyed clone cache avoids recloning across objectives.
- The module map is LLM-produced and may vary across deliberate re-extracts;
  note this where recall numbers are cited.
- **Versioning:** `.claude/` is gitignored in this repo. These skill/agent files
  live directly under `.claude/`. If they should be version-controlled, add a
  parent-chain un-ignore block and call it out before committing.
