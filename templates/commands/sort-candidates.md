# Sort Spotlights candidates by estimated impact

Give this prompt to an agent to rank the candidates in a Spotlights
`result.json` by their **estimated impact** and emit a human-readable markdown
report of the sorted candidates with scores.

**UX rule**: Whenever a question has predefined choices, use the `AskUserQuestion` tool to present them as selectable options (not inline text). This gives the user a clean dropdown-style experience.

Count your concrete predefined options before building the question:

- **≥2 concrete options** → rely on the built-in "Other" fallback and do NOT add a "Custom path" / "Custom N" option (avoids a duplicate "Custom" + "Other" chip).
- **<2 concrete options** → add an explicit "Custom path" / "Custom N" option so the call meets `AskUserQuestion`'s ≥2-options minimum. The built-in "Other" is still there for free-text; the explicit option is only there to satisfy the schema.

Either way, if the user picks "Other" (or the explicit Custom option), handle the free-text conversationally.

## Step 0 — Confirm Inputs

Ask the three questions below in this exact order, each as a **separate `AskUserQuestion` call** — do NOT batch them into one call. The built-in "Other" fallback only lets the user type a free-text answer when the question is asked on its own; batching several questions together disables free-text entry per question and forces awkward duplicate "Custom" options. Order: **(1) `RESULT_JSON`, then (2) `OUTPUT_DIR`, then (3) `TOP_N`.** `TOP_N` in particular must always be last, because it only makes sense once the input and output locations are settled.

**Resolve `RESULT_JSON`:**

Present the available choices with a dedicated `AskUserQuestion` call. Build the option list dynamically so the user sees every concrete path that's on offer — the built-in "Other" fallback covers custom paths, so do not add an explicit Custom option yourself:

1. If the user supplied a path in the prompt, include it as an option (label it clearly with the path, e.g. `Use path from prompt: <path>`).
2. If `./spotlights-out/result.json` exists, include it as an option (e.g. `Use default: ./spotlights-out/result.json`).
3. If any other `result.json` exists under a sibling `spotlights-out*/` directory in the CWD, include each as an option (e.g. `./spotlights-my-out/result.json`).

If the user picks "Other", they will type the path in free text. Reject non-existent files and re-ask (re-present the same options). If fewer than 2 concrete options are available, add an explicit `Custom path` option only as a schema filler to meet the ≥2-options minimum — do not add it when you already have 2+ concrete paths.

**Resolve `OUTPUT_DIR`:**

Ask for the output folder with its own `AskUserQuestion` call (separate from the input question — the built-in "Other" fallback needs a solo question to render its free-text input). Offer these two concrete options; the built-in "Other" fallback handles a custom path via free text, so do NOT add an explicit "Custom path" option (it renders as an unfillable chip alongside "Other").

- `<parent-of-RESULT_JSON>` — default (write the output files next to `result.json`)
- `<parent-of-RESULT_JSON>/sorted/` — sibling `sorted/` subdirectory

If the user picks "Other", they will type the path in free text.

Then derive:

- `OUTPUT_MD` = `<OUTPUT_DIR>/sorted_candidates.md`
- `OUTPUT_JSON` = `<OUTPUT_DIR>/sorted_candidates.json`

**Ask for `TOP_N`** with its own `AskUserQuestion` call (defaults to All). Only include the concrete options; the built-in "Other" fallback handles a custom integer via free text:
- All · Top 10 · Top 25

`TOP_N` only affects the markdown details section; the JSON always contains every ranked candidate.

**Echo the resolved values before starting:**

```
RESULT_JSON:  <path>
OUTPUT_MD:    <path>
OUTPUT_JSON:  <path>
TOP_N:        <all | N>
```

## Inputs (summary)

- `RESULT_JSON` — path to the run's `result.json`
  (default: `./spotlights-out/result.json`, or `<parent-of-supplied-result-json>/result.json` when a path is provided).
- `OUTPUT_MD` — path to write the ranked markdown report
  (default: `<result-json-parent>/sorted_candidates.md`).
- `OUTPUT_JSON` — path to write the machine-readable ranked JSON
  (default: `<result-json-parent>/sorted_candidates.json`).
- `TOP_N` — how many candidates to render in full detail in the markdown
  (default: `all`). Does **not** affect the JSON, which always contains every
  ranked candidate.

The candidates live under `report.candidates` — a list of ~100 objects. Each
candidate has: `id`, `module_qualified_name`, `origin`, `locations[]`
(`file` + `spans[]` with `symbol`/`kind`/`line_start`/`line_end`),
`description`, `current_approach`, `evolve_rationale`, `estimated_impact`
(one of `high` / `medium` / `low`), `estimated_impact_explanation`, and
`proposals[]` (each with `title`, `source`, and for `research_finding`
proposals a `finding_ref_id`).

## Task

**Precondition:** Step 0 above is complete and `RESULT_JSON`, `OUTPUT_MD`, `OUTPUT_JSON`, and `TOP_N` are all confirmed. Do not start this section until then.

1. **Load** `report.candidates` from `RESULT_JSON`.

2. **Compute a deterministic pre-score** for every candidate (pure, no model —
   this guarantees a stable tie-break and a fallback if the judge step is
   skipped):

   ```
   impact_weight = {high: 3, medium: 2, low: 1}[estimated_impact] × 2.0
   evidence      = number of research_finding proposals            × 1.0
   pre_score     = impact_weight + evidence
   ```

   Note: do **not** weight `agent_knowledge` proposals — in this data nearly
   every candidate has exactly 2 of them, so the count is a near-constant and
   carries no discriminating signal. `estimated_impact` and the varying
   `research_finding` count are the only pre-score signals that discriminate.

   Deterministic total order (used as tie-break and shard-assignment order):
   `(-pre_score, module_qualified_name, id)`.

3. **Rank by estimated impact using a sub-agent** (preferred). Do NOT rely on a
   pure numeric sort alone — `estimated_impact` is assigned per-module by
   agents that never saw the other modules, so "high" in one module is not
   directly comparable to "high" in another. Use a listwise LLM judge to
   calibrate across modules:

   - Build a compact **ranking card** per candidate (≤ ~150 tokens): `id`,
     `module_qualified_name`, primary `file` + `symbol`, `estimated_impact`,
     a truncated `estimated_impact_explanation`, a truncated `evolve_rationale`,
     the count of `research_finding` proposals, and the `title` of each
     proposal. Build cards from the pre-score order so shards span the quality
     range.
   - **Fan out to sub-agents** (via the `Agent` tool, `general-purpose` or
     `Explore`-style read-only agents). Split the cards into shards of ≤ 40 and
     launch the shard-ranking agents **in parallel** (one message, multiple
     tool calls). Each sub-agent receives the run objective
     (`report.context.objective`), the workload hints
     (`report.context.workload_hints`), and its shard of cards, and returns a
     ranking of its shard as structured JSON:

     ```json
     { "ranked": [ { "candidate_id": "...", "score": 0-100,
                     "rank_rationale": "one or two sentences" } ] }
     ```

     `score` is an objective-relative 0–100 impact estimate; higher = more
     impact on the stated objective/workload.
   - If there is more than one shard, run **one merge sub-agent** over the
     union of the top `ceil(40 / n_shards)` candidates from each shard to
     produce the global top order. Rank the remaining candidates by
     `(shard-rank percentile, pre_score)` — where `shard-rank percentile` is
     the candidate's position within its own shard's ranking normalized to
     0–1 (lower = ranked better by its shard). If two candidates tie on both
     `shard-rank percentile` and `pre_score`, fall back to the deterministic
     total order `(-pre_score, module_qualified_name, id)` from step 2 so the
     tail is fully stable. This keeps it to exactly two agent "layers"
     regardless of candidate count.
   - If a single shard covers everything (≤ 40 candidates), one sub-agent call
     is enough — no merge round.

4. **Repair the ranking so nothing is lost or invented.** Drop any
   `candidate_id` the judge returned that isn't a real candidate; de-dupe
   (keep first occurrence); append any candidate the judge omitted at the end
   in pre-score order with `score: null` and rationale
   `"not ranked by judge; pre-score order"`. The final ranked list must contain
   **every** candidate exactly once. Note in the report whenever repair changed
   anything.

5. **If the sub-agent step fails** (or sub-agents are unavailable), fall back to
   sorting purely by the deterministic pre-score from step 2, and say so
   explicitly in the report header.

## Output — how it is saved

Write **two files** — a human-readable markdown report to `OUTPUT_MD` and a
machine-readable JSON to `OUTPUT_JSON`. Both are derived from the same repaired
ranked list, so their order and scores are identical. Create the output
directory if it does not exist. Do not mutate `result.json`; these are
read-only views over it.

### Markdown (`OUTPUT_MD`)

Structure:

```markdown
# Sorted candidates — <run_id> (<module or repo name>)

**Objective:** <report.context.objective>
**Ranked:** <N> candidates · **Method:** listwise sub-agent judge  (or: pre-score fallback)
**Source:** `<RESULT_JSON>`

## Ranking summary

| # | Candidate | Module | Symbol | Impact | Score | Rationale |
|---|-----------|--------|--------|--------|-------|-----------|
| 1 | [`cand-…`](../modules/pkg/Foo__bar__cand-….md) | pkg/… | Foo.bar | high | 92 | one-line why it ranks here |
| 2 | … | … | … | … | … | … |

## Candidate details

### 1. `cand-…` — <primary symbol> (score 92, impact: high)

- **Module:** pkg/…
- **Location:** `path/to/file.go:838` (`Symbol.method`, config_block)
- **Description:** …
- **Current approach:** …
- **Why this impact / rank:** <judge rank_rationale> — <estimated_impact_explanation>
- **Proposals (N):** short bulleted list of proposal titles + source

### 2. …
```

Rules for the markdown:

- The **summary table** always lists every ranked candidate (all N rows), best
  first. `Score` shows the judge's 0–100 score, or `—` for pre-score-fallback
  rows. Keep the `Rationale` column to one line.
- The **`Candidate` cell must be a clickable link** to that candidate's own
  detail markdown file so the reader can jump straight to it. Each candidate has
  a dedicated md file under `<parent-of-RESULT_JSON>/modules/<module-folder>/`,
  named with the candidate `id` as the filename suffix, e.g.
  `TwoQubitBasisDecomposer__call_inner_best_nbasis__cand-crates_synthesis-0004.md`.
  Resolve the file for each candidate by globbing
  `<parent-of-RESULT_JSON>/modules/**/*__<id>.md` (the `id` suffix is unique, so
  exactly one file matches). Emit the cell as
  ``[`<id>`](<path-relative-to-OUTPUT_DIR>)`` — e.g. when `OUTPUT_DIR` is
  `<parent-of-RESULT_JSON>/sorted/`, the link is
  ``[`cand-crates_synthesis-0004`](../modules/crates_synthesis/TwoQubitBasisDecomposer__call_inner_best_nbasis__cand-crates_synthesis-0004.md)``;
  when `OUTPUT_DIR` is the parent of `result.json` itself, drop the leading
  `../` (`modules/…`). If no md file matches an `id` (none should, but be
  defensive), fall back to the plain `` `<id>` `` in backticks with no link and
  note it in the Method note.
- The **details section** renders the top `TOP_N` candidates in full (default
  all). If `TOP_N` is set and less than N, add a note that the remaining
  candidates appear in the summary table only.
- Use clickable relative links for file locations, e.g.
  `[runner.go:838](../../<repo-relative-path>#L838)` when a repo-relative path
  is known; otherwise just show `file:line` in backticks.
- Include a short **Method note** at the bottom: shard count, whether a merge
  round ran, and how many candidates (if any) were placed by repair rather than
  the judge — so the ranking is honest about which positions came from the
  model.

### JSON (`OUTPUT_JSON`)

Write the full ranked list as a single JSON object. It contains **every**
candidate exactly once, in ranked order, and is the machine-readable companion
to the markdown (`TOP_N` does not truncate it). Shape:

```json
{
  "source_md": "<OUTPUT_MD, repo-relative>",
  "source_result": "<RESULT_JSON, repo-relative>",
  "run_id": "<report.run_id>",
  "objective": "<report.context.objective>",
  "method": "listwise sub-agent judge",
  "total_ranked": 100,
  "candidates": [
    {
      "rank": 1,
      "id": "cand-…",
      "module_qualified_name": "pkg/…",
      "symbol": "<primary symbol>",
      "impact": "high",
      "score": 95,
      "rationale": "<judge rank_rationale, truncated to ~200 chars>",
      "locations": [
        {
          "file": "path/to/file.go",
          "spans": [
            { "line_start": 46, "line_end": 103,
              "symbol": "…", "kind": "config_block" }
          ]
        }
      ]
    }
  ]
}
```

Rules for the JSON:

- `candidates` is ordered by `rank` (1-based, ascending) and matches the
  markdown summary table row-for-row.
- `score` is the judge's 0–100 score, or `null` for pre-score-fallback rows
  (same rule as the markdown `—`).
- `method` is `"listwise sub-agent judge"` normally, or
  `"pre-score fallback"` when step 5's fallback was used.
- `symbol` is the candidate's primary symbol (the first span's `symbol`, or the
  candidate-level symbol when present).
- `locations` is copied verbatim from the candidate in `result.json` (file +
  spans with `line_start`/`line_end`/`symbol`/`kind`).
- `source_md`, `source_result`, and `run_id` let a reader tie the JSON back to
  the markdown report and the originating run.
- Emit the JSON as UTF-8, pretty-printed (2-space indent), and do **not**
  truncate the candidate list — every ranked candidate appears here even when
  `TOP_N` limits the markdown details.

## Notes

- Keep sub-agents read-only and cheap: they rank cards, they do **not**
  re-investigate the code. Give them a tight turn budget.
- The pre-score weights are intentionally simple; they exist for determinism
  and fallback, not to be the primary ranker. The sub-agent judge is what
  performs the cross-module calibration.
