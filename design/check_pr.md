# Design — `run-on-pr` skill: PR-grounded recall for the full Spotlights engine

## Goal

Build a Claude Code skill that, given **one** merged GitHub PR plus a generic
objective, re-runs the **full `spotlights-engine` pipeline on the PR's
pre-merge code**, scoped to the module(s) containing the PR's changed files, and
measures whether the engine independently surfaces a candidate whose code region
overlaps the lines the PR actually changed.

Each merged PR is treated as ground truth ("an expert decided *this* location
was worth changing"). The harness runs the engine cold on the pre-PR tree,
within the PR-derived module scope but never seeing the diff, and asks: did the
engine's candidates land on the same lines? The headline output is a
**module-scoped line-recall** verdict for that PR.

### Second signal: paper-citation recall
When a PR's prose references an academic paper (e.g. PR
[vllm#39008](https://github.com/vllm-project/vllm/pull/39008) cites
[arxiv 2504.19874](https://arxiv.org/abs/2504.19874)), the engine's deep-research
step produces its own flat set of `report.findings[]` (papers/docs/blogs it
discovered); some of those findings are attached to candidates as proposals,
others sit in the list unattached. So a second, independent ground-truth signal
becomes available: **did the engine's deep research surface the same paper the PR
author cited?** This is a *paper-citation hit* — recorded alongside (not instead
of) the line-recall verdict. A PR can score on neither, either, or both signals;
they are orthogonal (line overlap is about *where* in the code, paper match is
about *what prior art* the engine found). The cited paper is treated as ground
truth exactly like the changed lines: the engine never sees it.

This is the same idea as the reference `check-prs` skill in
`spotlights-paper/.claude`, but the auditor is no longer the lightweight
`design/bootstrap.md` procedure — it is the real engine invoked exactly as the
README documents (`spotlights-engine --repo … --include … --objective …`).

### Scope decisions (confirmed)
- **One PR per invocation** (no list-level fan-out).
- **Scope the run via `--include`**: extract the module map, map the PR's changed
  files to the modules that contain them, and pass only those modules to the
  engine. Falls back to all-modules when the mapper-derived scope is empty,
  ambiguous, or leaves any changed source file unmapped.
- **Standalone `runs/` harness**: skill + agents live in `.claude/`, helper
  scripts in `scripts/run_on_pr/`, all run artifacts under gitignored `runs/`.
  This is a validation tool, not something shipped through `spotlights-engine
  init`.
- **Full pipeline**: run all five engine steps as the user asked. Recall is
  computed from the candidates in `result.json`; the deep-research/agent
  proposals are a bonus for inspecting what a real run produces.

## Why this is different from `check-prs`

| Aspect | `check-prs` (reference) | `run-on-pr` (this design) |
|---|---|---|
| Auditor | `design/bootstrap.md` prompt run by a `bootstrap-runner` subagent, one per sub-folder | The real `spotlights-engine` binary, one run per PR |
| Scope unit | sub-folder (parent dir of each changed file) | engine **module** (qualified name), derived from changed files |
| Candidate source | JSON emitted by the bootstrap prompt | `report.candidates[]` inside the engine's `result.json` (nested `locations[].spans[]`, exploded to flat records) |
| Cost | cheap (prompt only) | full pipeline (~minutes + API \$ per PR) |
| Fan-out | list of PRs → many `pr-checker`s | single PR |

The **engine blindness inside the chosen module scope** and the **base-side
line-overlap** math are carried over — those are what make the recall number
meaningful, and they are the hardest parts to get right.

## Critical invariants (do not break)

1. **Scoped blindness.** The engine must run on the pre-PR checkout with no PR
   title, description, diff, changed-file list, or ground-truth ranges. The
   deliberate exception is the coarse **module scope** selected from the changed
   files and passed via `--include`; the metric is therefore conditional on that
   scoped search space, not whole-repo discovery. The only natural-language
   input is the **generic, PR-independent objective** (plus generic hints). If
   the engine knew the changed files, ranges, or PR prose, recall would be
   meaningless label leakage. Only the harness checkout/diff/scope/match
   bookkeeping sees PR metadata or changed lines.
2. **Correct base commit = `merge-base(baseRefOid, headRefOid)`.** Never
   `mergeCommit^1` (wrong for rebase merges — leaks PR code into the tree). Full
   clone, never shallow (merge-base needs common ancestry).
3. **Two-dot diff in base coordinates.** `git diff <base> <head>` (never
   three-dot) so the changed ranges are expressed in the pre-PR file's line
   numbers — the same coordinate frame the engine's candidates use.
4. **`runs/` is gitignored scratch.** Never `git add` anything under it. (`.gitignore`
   already ignores `runs/`.)

## Key engine facts established during research

- **CLI**: `spotlights-engine --repo <path> --include <qn...> --objective <str>
  [--hint <str>] --output-folder <dir> --artifacts-dir <dir> [--max-parallel N]
  [--max-findings-per-module N]`. No early-stop flag exists; a run always goes
  through all five steps.
- **`--include` takes module *qualified names***, slash-form (e.g.
  `vllm/v1/kv_offload` in a root-layout repo, or `pkg/cache` in a
  `source_root: "src"` repo), **not file paths**. A parent qn selects itself
  plus all modules beneath it. (`cli.py:68-81`)
- **Module map**: produced by step 1 (modules extractor). Each module has a
  `name` and a **repo-relative** `path` (e.g. `vllm/v1/kv_offload`). The
  qualified name is the `path` made **relative to the repo's `source_root`**,
  with every path segment normalized, and joined with slashes. In a src-layout
  repo, `source_root: "src"` and path `src/pkg/cache` → qn `pkg/cache`; in a
  root-layout repo, `source_root: ""` and path `vllm/v1/kv_offload` → qn
  `vllm/v1/kv_offload`. **Everything is slash-form**: both the CLI `--include`
  value and the `result.json` `module_runs` keys (`api.py:90`). There is **no
  dot-form** anywhere. Mapping code should load `ProjectTree` and use
  `tree.walk()` instead of reimplementing `_qualified_name`
  (`project.py:74`).
- **Candidate shape** (the recall payload). Candidates are **nested**, not flat
  (`schemas/candidate.py:70`). ⚠️ There is a legacy `original_schemas/candidate.py`
  with a **different, flat** `Candidate` (top-level `file` + `line_start` /
  `line_end`, id `^cand-\d{4}$`); it is *not* what the engine emits in
  `result.json`. Always read/import from `spotlights_engine.schemas`, never
  `original_schemas`. A `Candidate` carries `id` (pattern
  `^cand-<slug>-\d{4}$`), `module_qualified_name`, `origin`
  (`telemetry_anomaly` | `code_agent`), `estimated_impact`, and
  `locations: [CodeLocation]`. The file lives on each location and the line
  range lives on each span beneath it:
  ```jsonc
  { "id":"cand-vllm_v1_kv_offload-0001",
    "module_qualified_name":"vllm/v1/kv_offload",
    "origin":"code_agent", "estimated_impact":"high",
    "locations":[
      { "file":"vllm/v1/kv_offload/cpu/manager.py",
        "spans":[ { "line_start":19, "line_end":22,
                    "symbol":"_CACHE_POLICIES", "kind":"plugin_seam" } ] } ] }
  ```
  `overlap.py` reads **flat** `file` + `line_start`/`line_end` per record, so
  `extract_candidates.py` must **explode** every `(location, span)` pair into
  one flat record (carrying the parent candidate's `id`,
  `module_qualified_name`, `estimated_impact`, and rank). This is a genuine
  schema translation — not just "flattening across modules."
- **Two candidate sources in `result.json`**: the top-level `report.candidates`
  is already the cross-module flattened list (each entry carrying its
  `module_qualified_name`); `module_runs[<qn>].candidates` is a `Candidates`
  wrapper (`.candidates[]`, or `null` if the module failed before discovery).
  Prefer **`report.candidates`** as the extractor input — simpler and
  rank-stable.
- **Findings / paper references** (the paper-citation payload). The deep-research
  step emits a **flat** `report.findings[]` list (`schemas/pipeline.py:160-176`,
  `SpotlightReport.findings`), each a `Finding`
  (`schemas/finding.py:26-42`):
  ```jsonc
  { "finding_id":"find-preble-0001",   // pattern ^find-<slug>-\d{4}$ (a bare find-0001 is invalid)
    "title":"Preble: Efficient Distributed Prompt Scheduling for LLM Serving",
    "url":"https://arxiv.org/abs/2504.19874",
    "source_type":"paper",          // paper|blog|docs|issue|pr|talk|codebase|other
    "technique_summary":"…",
    "supporting_evidence":"Quote: …" }
  ```
  The **`url`** is the only paper reference, and **`source_type:"paper"`** marks
  academic papers (arxiv lives here). Candidates link to findings indirectly:
  `candidate.proposals[].finding_ref_id` → `Finding.finding_id`
  (`schemas/proposal.py:12-35`). There is **no paper/url field on the candidate
  or proposal itself** — the matcher reads `report.findings[]` directly, not via
  candidates.
- **Arxiv URL normalization already exists in the engine** and must be reused
  verbatim for matching, not reimplemented: `_normalize_url` /
  `_normalize_arxiv_id` in
  `module_deep_research/orchestration.py:243-257` canonicalize
  `https://arxiv.org/{abs,pdf}/<id>` (any version/`.pdf`/query/fragment) to
  `arxiv:<id>`. The PR-cited URL and each `Finding.url` go through the *same*
  normalizer before comparison so e.g. `…/abs/2504.19874v2` and `…/pdf/2504.19874`
  both match. Non-arxiv URLs normalize by lowercasing + trailing-slash strip.
- **Output layout**: `index.md`, `result.json`, `modules/<slug>.md`, and
  `modules/<slug>/<symbol-slug>__<cand-id>.md`, where `<slug>` is the qn with
  `/` (and any non-`[A-Za-z0-9._-]` char) collapsed to `_` — e.g. qn
  `vllm/v1/kv_offload` → `modules/vllm_v1_kv_offload.md`
  (`utils/id_helpers.py:slug_for`,
  `results_renderer/writer.py`). The page links use these slug filenames, not
  the raw slash-form qn.
- The modules extractor is runnable standalone via
  `scripts/run_modules_extractor.py` (or
  `spotlights_engine.modules_extractor.extract_with_telemetry`), writing a
  `ProjectTree` JSON — used by the file→module mapping step.

## Architecture

Single-PR, so the nested-orchestrator topology of `check-prs` collapses to a
**linear pipeline** the skill drives directly, delegating each step to one
subagent:

```
run-on-pr skill (you, main session)        — single-PR orchestrator
  ├─ pr-checkout       (step 1) — pre-PR checkout at the merge-base
  ├─ pr-diff-scope     (step 2) — ground-truth base-side line ranges
  │                              + cited-paper URLs (paper ground truth)
  ├─ pr-module-scope   (step 3) — module map + changed-files→modules → --include
  ├─ engine-runner     (step 4) — run the full engine scoped-blind via --include
  └─ match-evaluator   (step 5) — candidates ∩ ground truth → line-recall verdict
                                 + findings ∩ cited papers → paper-citation verdict
```

The `pr-checkout`, `pr-diff-scope`, and `match-evaluator` agents (and their
helper scripts) are lifted almost verbatim from the reference skill. The
`pr-module-scope` and `engine-runner` agents are new and carry the
engine-specific logic. The paper-citation signal extends the existing
`pr-diff-scope` (PR-cited paper extraction) and `match-evaluator`
(findings↔papers comparison) agents rather than adding new agents.

> Numbering note: the diagram above labels the five agents step 1–5 in pipeline
> order, but the Procedure below numbers its prose sections §0–§7 (pre-flight
> and parse come first). They are offset — diagram "step 4 (engine-runner)" is
> Procedure §5, and diagram "step 5 (match-evaluator)" is Procedure §6. Sections
> cite the reference agent by name, not by the reference's own step number.

## Files to create

### Skill + agents (`.claude/`)
> Note: `.gitignore` currently ignores `.claude/`. For this validation harness
> we author the files directly under `.claude/` (consistent with the chosen
> "standalone runs/ harness" model). If they should be version-controlled, add a
> full parent-chain un-ignore block — `!.claude/`, `!.claude/skills/`,
> `!.claude/skills/run-on-pr/**`, `!.claude/agents/`, and the specific agent
> files — then call this out to the user before committing.

- `.claude/skills/run-on-pr/SKILL.md` — orchestrator + procedure (below).
- `.claude/agents/pr-checkout.md` — **port** of the reference agent, unchanged in
  spirit: resolve PR metadata, full clone, fetch `refs/pull/<n>/head`, compute
  merge-base, detached checkout, write `pr.json`.
- `.claude/agents/pr-diff-scope.md` — **port + extend**: two-dot diff →
  base-side ranges via `diff_ranges.py`, write `ground_truth.json`; **also**
  extract the PR's cited paper URLs (via `extract_pr_papers.py`) into
  `cited_papers.json`. This is the one place PR prose is read for paper refs —
  the engine never sees it.
- `.claude/agents/pr-module-scope.md` — **new**: produce the module map for the
  pre-PR tree and map changed source files → engine module qualified names.
- `.claude/agents/engine-runner.md` — **new**: invoke `spotlights-engine`
  scoped-blind by `--include`, into the run's output/artifacts dirs; extract a
  flat `candidates.json` from `result.json`.
- `.claude/agents/match-evaluator.md` — **port + extend**: run `overlap.py` →
  `match.json` (line recall); **also** run `match_papers.py` over
  `cited_papers.json` × the engine's `report.findings[]` → `paper_match.json`
  (paper-citation recall).

### Helper scripts (`scripts/run_on_pr/`)
Copy/adapt the reference helpers (they are repo-agnostic and already battle-tested):
- `diff_ranges.py` — **copy verbatim** (base-side range parsing, source filter).
- `overlap.py` — **copy verbatim** (candidate ∩ ground-truth math). Reads flat
  `file` + `line_start`/`line_end` per record — which is the shape
  `extract_candidates.py` produces *after* exploding the engine's nested
  `locations[].spans[]` (the engine's candidates are **not** natively flat).
- `log.sh` — **copy verbatim** (progress logging).
- `extract_candidates.py` — **new**: read `result.json`, take
  `report.candidates[]`, and **explode** each candidate's `locations[].spans[]`
  into flat records `{id, module_qualified_name, file, line_start, line_end,
  symbol, kind, estimated_impact, rank}`, writing `candidates.json` in the shape
  `overlap.py` consumes. Deterministic; preserves `report.candidates` order so
  `rank` is meaningful. (One candidate spanning multiple files/spans yields
  multiple flat records sharing its `id` and `rank`.)
- `map_files_to_modules.py` — **new**: given the extractor's `ProjectTree` JSON
  and the changed source file list, return the set of module qualified names
  whose repo-relative `path` contains a changed file, choosing the
  **most-specific (deepest) module** per file. Match paths segment-aware
  (`file == module.path` or `file.startswith(module.path + "/")`), and get qns
  from `ProjectTree.walk()` so `source_root` stripping and segment normalization
  stay identical to the engine. Emits
  `{include:[...], unmapped_files:[...], ambiguous_files:[...]}`.
- `extract_pr_papers.py` — **new**: read the PR title + body + linked-issue
  bodies (fetched by the harness, never given to the engine) and extract
  referenced paper URLs. Scan for arxiv links (`arxiv.org/abs|pdf/<id>`), DOIs,
  and other paper URLs; canonicalize each through the same normalizer used for
  findings (see normalizer-reuse note below) so arxiv refs become `arxiv:<id>`.
  Emits `cited_papers.json`:
  `{papers:[{raw_url, normalized, kind, title?}], source_fields:[…]}`. Capture an
  optional human title where one is cheaply available (e.g. the markdown link
  text around the URL) — the matcher uses it as a fallback key. If no paper is
  cited, emits an empty list (the paper signal is then `n/a` for this PR). Keep
  extraction conservative — false-positive URLs (a link to the repo, CI, etc.)
  should not count as cited papers; prefer arxiv/DOI/known paper hosts.
- `match_papers.py` — **new**: given `cited_papers.json` and the engine's
  `result.json`, match the PR's cited papers against `report.findings[]` using
  the **same key logic the engine uses for its own finding dedup**
  (`_finding_keys` → normalized-URL key *and* normalized-title key,
  `orchestration.py:235-240`). A finding matches a cited paper if their
  normalized-URL keys are equal **or** (when the cited paper carries a title)
  their normalized-title keys are equal. Title-OR-URL is deliberate: the engine
  frequently surfaces a paper at a *different* URL than the PR cites (arxiv abs
  vs. conference-proceedings PDF vs. DOI — e.g. the `llmd` run found Preble at an
  `iclr.cc` proceedings PDF, not its arxiv page), so URL-only matching would
  under-count real hits. Emits `paper_match.json`: `{paper_cited: bool,
  paper_hit: bool, matched:[{finding_id, title, url, source_type, matched_on:
  "url"|"title", via_candidate_ids:[…]}], cited_papers:[…], num_findings:int}`.
  `via_candidate_ids` is filled by reverse-linking each matched `finding_id`
  through `candidate.proposals[].finding_ref_id`, so the report can say *which*
  candidate the matched paper rode in on (or "found in findings but not attached
  to a candidate"). Deterministic.

## Procedure (SKILL.md)

### 0. Pre-flight
- `gh auth status` must succeed (clone + PR metadata). Stop clearly if not.
- Confirm the engine is callable: `spotlights-engine --help` (or
  `uv run spotlights-engine`). Stop if absent.
- Confirm helpers exist under `scripts/run_on_pr/`.

### 1. Parse inputs
Accept a single PR URL plus an objective (and optional hints / explicit
`--include` override / explicit `base_commit`). Minimal parsing — one PR, so no
list grammar. Derive a deterministic **PR key** from the PR URL for clone and
metadata reuse, then derive the actual `run_id` from the PR URL plus objective,
hints, include override, base override, and relevant engine knobs. Do not use
`Date.now()`: identical inputs should resume, but changing the objective must
not collide with an old engine artifacts dir. `RUN=runs/run-on-pr/<run_id>`;
`mkdir -p $RUN`, with reusable clone state under `runs/run-on-pr/_repos/<pr_key>`
or equivalent.

### 2. Checkout (agent: `pr-checkout`)
As reference step 1, but keep the operational contract in this skill so it is
not dependent on memory of the old one:
- Resolve PR metadata with `gh pr view <pr_url> --json
  mergeCommit,baseRefOid,headRefOid,baseRefName` and record `mergeCommit.oid`
  when present.
- Fetch the PR head via `refs/pull/<n>/head`, not the source branch; merged PR
  branches are often deleted while GitHub retains the PR ref.
- Fetch the base ref named by `baseRefName` (or otherwise ensure `baseRefOid` is
  reachable). If explicit `base_commit` was supplied, verify it after these
  fetches and use it as the checkout base. Otherwise compute
  `git merge-base <baseRefOid> <headRefOid>`.
- Create a detached pre-PR checkout at `base_commit`. Writes `$RUN/pr.json`;
  returns `{base_commit, head_commit, checkout_path, ...}`. On error → stop.

### 3. Diff scope / ground truth (agent: `pr-diff-scope`)
As reference step 2. Two-dot diff → `$RUN/ground_truth.json` with base-side
`changed_ranges`, `changed_source_files`, `subfolders`, `new_files`. If
`status: no_source_changes` → skip steps 4–6, report the bucket. On error →
stop.
`changed_ranges` should use range objects rather than bare tuples:
`{"path/to/file.py": [{"start": 21, "end": 23, "addition_only": false,
"new_file": false}]}`. Pure additions anchor to a clamped base-side line and set
`addition_only: true`; brand-new files appear in `new_files` so the matcher can
bucket `new_file_only` correctly.

**Cited-paper extraction (new).** In the same agent, fetch the PR prose with
`gh pr view <pr_url> --json title,body` (plus closing-issue bodies if cheaply
available) and run `extract_pr_papers.py` → `$RUN/cited_papers.json`. This is
the only step that reads PR prose for paper references; like the diff, it stays
on the harness side and is **never** passed to the engine. An empty result is
fine — the paper signal is then `n/a` (see bucketing). Do not block the line
recall on this: paper extraction failing or finding nothing must not abort the
run.

### 4. Module scope (agent: `pr-module-scope`) — NEW
1. If `$RUN/project_tree.json` already exists for this exact run, reuse it.
   Otherwise run the modules extractor on the **pre-PR checkout** (blind —
   extractor sees only the code, never the PR), writing `$RUN/project_tree.json`.
   Reuse
   `scripts/run_modules_extractor.py --repo <checkout_path> --output-json
   $RUN/project_tree.json --artifacts-dir $RUN/extractor_artifacts`.
   Because `extract_with_telemetry` rejects an existing
   `extractor_artifacts/modules_extractor` dir, the harness must either skip the
   extractor when the cached tree exists or allocate a fresh extractor artifact
   directory for a deliberate re-extract.
2. Run `map_files_to_modules.py` over the tree + `changed_source_files` to get
   the slash-form `--include` list. Write `$RUN/scope.json`
   (`{include, unmapped_files, ambiguous_files, all_modules_fallback}`).
   If an explicit `--include` override was supplied, validate it against
   `tree.walk()` and use it instead, recording `scope_source: "user_override"`.
3. Fallback: for mapper-derived scope, if `include` is empty, or any changed
   source file is unmapped or ambiguous, run the engine on **all** modules (omit
   `--include`) and record the reason in `scope.json`. This avoids turning
   mapper uncertainty into a silent false negative. A user override is not
   auto-expanded; it must validate or fail clearly.

> Blindness note: the extractor input is only the checkout path. It must not
> receive the objective, hints, PR title, PR description, changed-file list, or
> diff.

### 5. Engine run (agent: `engine-runner`) — NEW
Invoke the full engine scoped-blind on the pre-PR checkout, scoped to the mapped
modules:
```
spotlights-engine \
  --repo        <checkout_path> \
  --include     <qn1> <qn2> ...        # omitted on the all-modules fallback
  --objective   "<generic objective>" \
  [--hint "<hint>" ...] \
  --output-folder <$RUN/spotlights-out> \
  --artifacts-dir <$RUN/artifacts> \
  --max-parallel <N>  --max-findings-per-module <N>
```
- ⚠️ Pass the **objective + hints + scoped module qns only** — never the diff,
  PR title, PR description, changed-file list, or ground-truth ranges. The
  objective is the generic domain goal supplied by the user.
- The engine writes `result.json`. Then run
  `extract_candidates.py $RUN/spotlights-out/result.json >
  $RUN/candidates.json`.
- Use a generous timeout — README warns live runs can take many minutes; do not
  cap with a smoke-test timeout.
- On engine failure → mark `status: error`, stop.
- Returns `{status, result_json_path, candidates_path, num_candidates,
  modules_run}`.

### 6. Match (agent: `match-evaluator`)
As reference step 4: `overlap.py --candidates $RUN/candidates.json
--ground-truth $RUN/ground_truth.json --addition-tolerance 3` → `$RUN/match.json`.
The flat records `extract_candidates.py` produced (already exploded from the
engine's nested `locations[].spans[]`) feed straight in. Returns
`{pr_line_hit, pr_file_hit, new_file_only, num_candidates, ...}`.

**Paper-citation match (new).** If `$RUN/cited_papers.json` has any papers, also
run `match_papers.py --cited $RUN/cited_papers.json --result
$RUN/spotlights-out/result.json` → `$RUN/paper_match.json`, returning
`{paper_cited, paper_hit, matched:[{…, matched_on}], num_findings}`. If no paper
was cited, skip this and record `paper_cited: false`. This signal is independent
of the engine module scope — the engine's deep research surfaces findings for
whatever modules ran, and a `paper_hit` is meaningful even when `pr_line_hit` is
a miss (and vice versa). Note the scope caveat: the engine only researches the
modules in `--include`, so a cited paper relevant to an *un-scoped* module can be
a structural miss — the report should not read that as "the engine couldn't find
it," only "not within the audited scope."

### 7. Persist + report
- Write `$RUN/report.md` (human-readable, in the spirit of the reference
  `examples/`): PR link, base commit (short), merge style; changed source files
  + base-side ranges; modules run (the `--include` set, or all-modules fallback)
  and any `unmapped_files`; candidate count; **verdict** (`pr_line_hit`
  headline + file/folder diagnostics + `new_file_only`); for each hit range the
  matching candidate(s) with `estimated_impact` + rank ("found, ranked #k");
  `missed_ranges` for error analysis; **paper-citation section** when
  `paper_cited` — the PR's cited paper(s), and for each `paper_hit` the matched
  `Finding` (title, url, source_type) plus which candidate it rode in on
  (`via_candidate_ids`) or "in findings, unattached"; pointer to the rendered
  `spotlights-out/index.md` and per-candidate pages.
- Print the headline to the user: `pr_line_hit` (hit/miss), the matched
  candidate + rank, the bucket, the **paper signal** (`paper_hit` / `n/a`) with
  the matched paper title when hit, and where the artifacts live.

## Bucketing (single-PR verdict)
The bucket describes the **line-recall** signal:
- `error` — any step failed.
- `no_source_changes` — step 3 found no source files (steps 4–6 skipped).
- `new_file_only` — every changed source file is brand-new → structurally
  unrecallable by base-side line overlap (reported, not a plain miss).
- `valid` — reached the matcher; `pr_line_hit ∈ {true,false}` is the result.

The **paper-citation** signal is a separate, orthogonal axis (not a bucket
value), reported alongside the line bucket:
- `paper_cited: false` → paper signal is `n/a` (PR cited no paper).
- `paper_cited: true, paper_hit: true` → the engine's deep research independently
  surfaced the same paper.
- `paper_cited: true, paper_hit: false` → cited paper was not among the engine's
  findings (a paper-recall miss).
A PR can be a line-recall miss but a paper hit, or vice versa — both are recorded
even when one of the engine steps surfaces nothing for the other. The paper
signal is still `n/a` when the run never reached the engine (`error` /
`no_source_changes`).

## Open risks / notes
- **Cost & latency.** A full engine run per PR is the dominant cost (minutes +
  API \$). The `--include` scoping is the main lever; document expected cost in
  SKILL.md and surface `modules_run` so the user sees what was audited.
- **qn derivation, not skew.** Everything is slash-form — both `--include` and
  the `result.json` `module_runs` keys / `candidate.module_qualified_name`
  (`api.py:90`). There is no dot-form to reconcile. The one real transform is in
  `map_files_to_modules.py`: a changed file is matched against the module's
  **repo-relative `path`**, but the emitted qn must be the qn returned by
  `ProjectTree.walk()` for that module, not a hand-built string. That keeps
  `source_root` stripping and segment normalization in lockstep with
  `project.py:_qualified_name`.
- **Module granularity vs ground-truth folder.** The engine module `path` may be
  coarser or finer than the changed file's directory. Recall is judged on
  **line overlap within the file**, so as long as the module containing the file
  is run, the candidate can match — folder/file/line diagnostics from
  `overlap.py` explain near-misses.
- **Extractor determinism.** The module map is LLM-produced and may vary run to
  run; the deterministic `run_id` lets identical inputs reuse the same `$RUN`,
  while the separate PR-keyed clone cache avoids recloning across objectives.
  The extractor/engine outputs themselves are not guaranteed identical across
  deliberate re-extracts. Note this where recall numbers are cited.
- **Versioning the skill.** `.claude/` is gitignored here. Decide with the user
  whether to un-ignore the new skill/agents/scripts before committing.
- **Paper-citation signal is opportunistic.** Most PRs cite no paper, so the
  signal is `n/a` for them — it adds value only on research-grounded PRs. It is a
  *recall* signal on its own axis: a hit means the engine's deep research found
  the same prior art the author cited, independent of whether candidates landed
  on the right lines. It is **not** a blindness leak: the engine never sees the
  PR prose or the cited URL; the harness extracts the paper only to score after
  the fact. Two soft spots to watch: (a) extraction precision — a PR body links
  many non-paper URLs, so keep `extract_pr_papers.py` conservative (arxiv/DOI/
  known paper hosts) and log what it pulled; (b) match strictness — the engine
  may find the *same idea* via a genuinely *different* paper (not just a
  different URL for the same paper — that case is covered by the title-OR-URL
  key), which still counts as a miss. Document that `paper_hit` is same-paper
  match (using the engine's own URL/title dedup keys), not semantic-prior-art
  match, so a miss isn't over-read. Title-matching has its own failure mode:
  near-identical or paraphrased titles can false-match or false-miss, so the
  report surfaces `matched_on: "url"|"title"` to let a title-only match be
  eyeballed.

## Reuse summary
- **Verbatim copies** from `spotlights-paper`: `diff_ranges.py`, `overlap.py`,
  `log.sh`; and the `pr-checkout`, `pr-diff-scope`, `match-evaluator` agents
  (trimmed of `check-prs`-specific framing). `pr-diff-scope` and
  `match-evaluator` are then **extended** for the paper-citation signal.
- **New**: `pr-module-scope` + `engine-runner` agents, `map_files_to_modules.py`,
  `extract_candidates.py`, `extract_pr_papers.py`, `match_papers.py`, and the
  single-PR `SKILL.md`.
- **Normalizer reuse (the matching must mirror the engine's dedup).** The URL/
  title canonicalization lives in `module_deep_research/orchestration.py` as
  module-private helpers (`_normalize_url`, `_normalize_arxiv_id`,
  `_normalize_text`, `_finding_keys`, lines 235-261). The paper matcher must
  produce **the same keys the engine itself uses to dedup findings**, so cited
  papers and engine findings are compared in one coordinate frame. Preferred:
  import those helpers directly even though underscore-prefixed (a known coupling
  to a private symbol — note it, and if the engine maintainer is willing, promote
  them to a small public `findings/keys.py` shared by both the engine and this
  harness). Acceptable fallback only if import is impractical: copy the four
  functions verbatim into a helper and add a test asserting parity on a few
  arxiv/DOI/title fixtures, so drift is caught. Do **not** hand-roll a different
  normalization — divergence from `_finding_keys` silently mis-scores the paper
  signal.
