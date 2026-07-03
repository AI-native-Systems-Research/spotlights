---
name: run-on-pr
description: "Prep + command emitter for a PR-grounded recall run of the full spotlights-engine. Given ONE merged GitHub PR plus a generic objective, prepares a scoped-blind engine run on the PR's pre-merge code: checks out the merge-base, computes base-side ground-truth line ranges + cited-paper URLs, and derives the --include scope from the changed files' folder paths (deterministic, no LLM extractor). It then RETURNS the exact scoped-blind `spotlights-engine` command for you to run yourself — it does NOT invoke the engine (that expensive step you launch manually). After the engine finishes, use the `compare-pr-run` skill to score candidates against the ground truth (module-scoped line recall) and cited papers (paper-citation recall). Use when asked to set up / prepare an engine run against a real merged PR, or to get the scoped-blind command for a specific PR's pre-merge code."
---

# run-on-pr — single-PR recall prep + engine-command emitter

Prepare a recall run that measures whether the **real `spotlights-engine`**
independently flags the code a merged PR actually changed — and, when the PR
cites a paper, whether the engine's deep research **independently surfaces** that
same paper. When the PR cites one, `--paper-link` is passed to the engine, but it
is a **post-hoc results filter on the deep-research block's output**, not an input
to the blind search: it collapses the findings the research produced *on its own*
down to the single one matching the cited paper (a no-match ⇒ the engine did not
surface it). The paper URL/title never enters the research runners' prompts, so
it does **not** make the paper easier to find — see Critical rule 1. Each merged
PR is ground truth ("an expert decided *this* location was worth changing"); the
harness pins the pre-PR tree, derives the module scope from the changed files (but
never leaks the diff), and produces the exact command to run the engine cold on
that tree.

**This skill does NOT run the engine.** It stops after scoping and **returns the
`spotlights-engine` command** for you to run yourself (the engine run is the
dominant cost — minutes + API $ — so it stays under your explicit control). When
the engine finishes, run the **`compare-pr-run`** skill on the same run dir to
score candidates against the ground truth and cited papers.

This is the heavier sibling of the reference `check-prs` skill: the auditor is
no longer a lightweight bootstrap prompt — it is the full five-step engine, run
exactly as the README documents. (This skill only *emits* that engine command;
you run it yourself — see "This skill does NOT run the engine" above.)

## Critical rules — do not break (design "Critical invariants")
1. **Scoped blindness.** The engine runs on the pre-PR checkout with **no** PR
   title, description, diff, changed-file list, or ground-truth ranges. The one
   deliberate exception is the coarse **module scope** (`--include`) derived from
   the changed files — specifically from the *folders* those files live in, so the
   leaked signal is only "which directories changed", never the diff itself. The
   only natural-language input is the **generic, PR-independent objective**
   (+ generic hints).

   **`--paper-link` does NOT break blindness.** When the PR cites a paper the
   command appends `--paper-link` (+ `--paper-title`) — see step 5 — but this maps
   to a `PaperFilter` that is applied **only after** the deep-research runners have
   already produced their findings on their own. It collapses that independently
   surfaced set down to the single finding matching the cited paper (URL first,
   then title); a no-match yields empty findings. The paper URL/title is **never**
   injected into the research prompts, so it cannot bias what the engine searches
   for or make the paper easier to find. The `compare-pr-run` paper verdict is
   therefore still a genuine **independent-recall** signal: "did the engine's own
   research surface this paper in this module" — the filter only isolates that
   finding from the noise of the other findings. Both the line-recall axis and the
   paper axis stay blind; `--include` carries no ground-truth-specific signal
   beyond the folder.
2. **Base = `merge-base(baseRefOid, headRefOid)`**, never `mergeCommit^1`. Full
   clone, never shallow.
3. **Two-dot diff** `git diff <base> <head>` so ranges are in the pre-PR file's
   line numbers — the same frame the engine's candidates use.
4. **`runs/` is gitignored scratch.** Never `git add` anything under it.

## Architecture (linear prep pipeline, you drive it)
```
run-on-pr skill (you, main session)        — single-PR prep orchestrator
  ├─ pr-checkout       (step 1) — pre-PR checkout at the merge-base
  ├─ pr-diff-scope     (step 2) — base-side line ranges + cited-paper URLs
  ├─ pr-module-scope   (step 3) — changed-file *folders* → --include (path-derived, no LLM)
  └─ emit command      (step 4) — print the spotlights-engine command (blind;
                                   +--paper-link when a paper exists — a post-hoc
                                   filter on research output, not a search bias)
                                   (YOU run it; this skill does not)

   … you run the engine by hand …

compare-pr-run skill (separate)            — the comparison tail
  ├─ explode candidates          — result.json → flat candidates.json
  └─ match-evaluator             — candidates ∩ ground truth (line recall)
                                   + findings ∩ cited papers (paper recall) → report
```
Delegate each prep step to its subagent (one at a time — steps are dependent).
Deterministic math lives in `scripts/run_on_pr/` helpers; agents call them.
The flow emits the command (blind; `--paper-link` is a post-hoc filter on the
research output, not a search input) and hands off to `compare-pr-run`; it never
runs the engine itself.

## Cost note
A full engine run per PR is the dominant cost (minutes + API $) — which is
exactly why this skill **does not run it**: it emits the command and
stops, leaving the run under your control. The `--include` scoping is the main
lever — surface the scoped `include` set so the user sees what will be audited.
This is **one PR per invocation**; there is no list-level fan-out. Each run gets
its own full clone under `$RUN/checkout` — nothing is shared across runs.

## Procedure

### 0. Pre-flight
- `gh auth status` must succeed (clone + PR metadata). Stop clearly if not.
- Confirm the engine is callable: `uv run --no-sync spotlights-engine --help`
  (so the command you emit will actually run). Stop if absent.
- Confirm the prep helpers exist under `scripts/run_on_pr/`:
  `diff_ranges.py`, `log.sh`, `derive_scope_from_paths.py`, `extract_pr_papers.py`.
  (The match/report helpers — `extract_candidates.py`, `overlap.py`,
  `match_papers.py` — are used by `compare-pr-run`, not here.)

### 1. Parse inputs
Accept a **single PR URL** plus an **objective** (required), and optional:
generic `hints`, an explicit `--include` override, an explicit `base_commit`, and
engine knobs (`max_parallel`, `max_findings_per_module`). Minimal parsing — one
PR, no list grammar.

Derive two deterministic keys (no `Date.now()` — identical inputs must resume,
but a changed objective must not collide with an old engine artifacts dir):
- **`pr_key`** = a slug of the PR URL (e.g. `owner__repo__pr<n>`). Used for
  progress-log labels.
- **`run_id`** = `<pr_key>__<hash>`, where `<hash>` is a short hash
  over (pr_url + objective + sorted hints + include override + base override +
  relevant engine knobs). The `pr_key` prefix makes the folder identifiable by
  PR number (e.g. `owner__repo__pr<n>__bc8d9c701211`) without leaking the PR
  title; the hash keeps distinct objectives/knobs from colliding.
  `RUN=runs/run-on-pr/<run_id>`.

`mkdir -p $RUN`. Echo the parsed inputs (PR, objective, hints, any
override/base, run_id) back to the user for confirmation before the expensive
run.

### 2. Checkout (agent: `pr-checkout`)
Spawn `pr-checkout` with `{repo, pr_url, pr_key, out_dir:"$RUN",
progress_log:"$RUN/progress.log", base_commit?}`. It resolves PR metadata,
full-clones into `$RUN/checkout`, fetches `refs/pull/<n>/head` + base, computes
the merge-base, checks out detached, and writes `$RUN/pr.json` (with
`checkout_path` = `$RUN/checkout`). On `status:error` → stop and report.

### 3. Diff scope / ground truth + cited papers (agent: `pr-diff-scope`)
Spawn `pr-diff-scope` with `{checkout_path, base_commit, head_commit, pr_url,
pr_key, out_dir:"$RUN", progress_log, addition_tolerance:3}`. It writes
`$RUN/ground_truth.json` (base-side `changed_ranges`, `changed_source_files`,
`subfolders`, `new_files`) **and** `$RUN/cited_papers.json` (conservative
arxiv/DOI/paper-host URLs — with an optional `title`: the **canonical arxiv API
title** for arxiv refs (never the unreliable link text — absent if the API fetch
fails), or the markdown link text for non-arxiv hosts — from the PR prose; the
only place PR prose is read). The first cited paper is
passed to the engine in step 5 via `--paper-link` as a **post-hoc filter on the
research block's output** (it isolates the matching finding after the research
runs — it does not bias the search; see Critical rule 1), so the paper axis stays
a blind independent-recall signal.
- If `status: no_source_changes` → skip steps 4–5, report bucket
  `no_source_changes` (paper signal `n/a`); there is nothing to run.
- On `status: error` → stop.
- A paper-extraction failure is **not** fatal — it yields an empty
  `cited_papers.json` and the paper signal becomes `n/a`.

### 4. Module scope (agent: `pr-module-scope`)
Spawn `pr-module-scope` with `{changed_source_files (from step 3), pr_key,
out_dir:"$RUN", progress_log, include_override?, checkout_path?}`. It derives the
scope **directly from the changed-file paths** — a module is the *folder* the
changed code lives in — via `derive_scope_from_paths.py`, and writes
`$RUN/scope.json`. No modules extractor, no `project_tree.json`, nothing about
the code content is read. The engine expands each derived folder qn to the real
modules beneath it (virtual-prefix matching in `spotlights_manager/filters.py`).
- Path-derived scope is the normal path.
- **All-modules fallback** (record the reason, run with no `--include`) when the
  derived `include` is empty, or any changed source file sits at the source root
  (no sub-folder to scope to) — never let a root-level change become a silent
  under-scope / false negative.
- A user `--include` override is passed through verbatim; the engine fails fast
  on an unknown qn, so it is never auto-expanded here.
- On `status: error` → stop.

### 5. Emit the engine command (do NOT run it)
Prep is done: `$RUN` now holds `pr.json`, `ground_truth.json`,
`cited_papers.json`, and `scope.json`. Build the exact
command the user should run themselves, reading `include` from `$RUN/scope.json`
(empty ⇒ all-modules fallback: omit `--include` entirely).

**Always append the paper filter when a paper exists.** Read the **first** entry
of `$RUN/cited_papers.json`; if present, append `--paper-link <raw_url>` and,
when that entry has a non-empty `title`, `--paper-title "<title>"`. This is a
**post-hoc filter on step 3's output**: after the deep-research runners produce
their findings on their own, it collapses that set to the single finding matching
that paper (per module), so the paper's presence is isolated from the noise of
the other findings. It does **not** feed the paper into the search — the URL/title
never reaches the research prompts, so it can't make the paper easier to find.
There is **no opt-in flag** — this is the default and only behavior.
- **Fallback:** when `cited_papers.json` has no papers, skip cleanly — emit the
  command with **no** `--paper-link` and note that no cited paper was available
  (the paper axis becomes `n/a`).

Surface it verbatim in a copyable block:
```
uv run --no-sync spotlights-engine \
  --repo            <checkout_path> \
  --include         <qn1> <qn2> ...        # OMIT this whole flag if scope.json's include is empty
  --objective       "<objective>" \
  --hint            "<hint1>"  --hint "<hint2>" ...   # one --hint per hint, omit if none
  --paper-link      "<raw_url>" \          # from cited_papers.json[0]; OMIT (with --paper-title) if no paper
  --paper-title     "<title>" \            # only when cited_papers.json[0].title is non-empty
  --output-folder   $RUN/spotlights-out \
  --artifacts-dir   $RUN/artifacts \
  --max-parallel    7 \
  --max-parallel-pairs 10 \
  --max-findings-per-module 50 \
  --review-iterations 1 \
  --no-agent-proposals
```
- **Always pass `--no-agent-proposals`.** The recall axes this skill measures
  (line recall + paper-citation recall) are decided by the step-2 candidates and
  step-3 research findings; step 5 (agent_proposals) adds only agent-knowledge
  proposals that do not affect either verdict. Skipping it makes each PR run
  cheaper and faster with no loss to the `compare-pr-run` scoring.
- ⚠️ **Scoped blindness:** the command carries **only** the objective + hints +
  scoped qns (plus `--paper-link`/`--paper-title`, which only filter step 3's
  *output* and never enter the search). Never add the diff, PR title/description,
  changed-file list, or ground-truth ranges. Both the line-recall and paper axes
  stay blind.
- ⚠️ **`--paper-link` does not leak the paper into the search.** It maps to a
  `PaperFilter` applied *after* the research runners have already produced their
  findings — it collapses that independently surfaced set to the single matching
  finding (a no-match ⇒ empty). The URL/title never reaches the research prompts.
  So the `compare-pr-run` paper verdict remains a genuine independent-recall
  signal ("did the engine's own research surface this paper in this module"), not
  a "can it locate a known paper" check.
- Tell the user: the run takes many minutes; `result.json` lands at
  `$RUN/spotlights-out/result.json`.
- Log the handoff:
  `bash scripts/run_on_pr/log.sh $RUN/progress.log <pr_key> run-on-pr READY "emitted engine command (paper_filter=<yes|no>); run then use compare-pr-run"`

### 6. Hand off to `compare-pr-run`
Tell the user explicitly: **after the engine finishes**, run the
`compare-pr-run` skill on `$RUN` to explode candidates, match against the ground
truth + cited papers, and write `$RUN/report.md`. Give them the run dir so they
can pass it straight in. This skill stops here — it does not match or report.

## Buckets set by this skill (prep only)
This skill reaches at most the point of emitting the command; the line-recall
and paper verdicts are issued by `compare-pr-run`. The buckets it can set:
- `error` — any prep step (checkout / diff / scope) failed → stop.
- `no_source_changes` — step 3 found no source files → skip the command,
  nothing to run (paper signal `n/a`).
- `ready` — prep succeeded, command emitted, awaiting the manual engine run.

`new_file_only`, `valid`, and the paper-citation axis are decided downstream by
`compare-pr-run` — see that skill's "Bucketing" section.

## Notes
- `runs/` is entirely gitignored — never `git add` anything under it.
- Re-running identical inputs reuses the same `run_id` (addressable, resumable);
  a resumed run reuses its own `$RUN/checkout` clone (fetches instead of
  recloning). Distinct objectives get distinct `run_id`s and thus separate clones.
- The `--include` scope is derived deterministically from the changed-file
  folder paths (no LLM), so it is stable across re-runs. It is coarse by design:
  each folder qn expands to every module the engine finds beneath that folder.
- **Versioning:** `.claude/` is version-controlled in this repo (the `.claude/`
  line in `.gitignore` is commented out). These skill/agent files live directly
  under `.claude/` and are committed alongside the code. `runs/` remains
  gitignored scratch — never `git add` anything under it.
