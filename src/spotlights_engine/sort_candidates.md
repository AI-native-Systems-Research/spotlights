# Prompt: Sort Spotlights candidates by estimated impact

Give this prompt to an agent to rank the candidates in a Spotlights
`result.json` by their **estimated impact** and emit a human-readable markdown
report of the sorted candidates with scores.

## Inputs

- `RESULT_JSON` — path to the run's `result.json`
  (default: `output/llmd_router_lsf/result.json`).
- `OUTPUT_MD` — path to write the ranked markdown report
  (default: `output/llmd_router_lsf/sorted_candidates.md`).
- `TOP_N` — how many candidates to render in full detail (default: `all`).

The candidates live under `report.candidates` — a list of ~100 objects. Each
candidate has: `id`, `module_qualified_name`, `origin`, `locations[]`
(`file` + `spans[]` with `symbol`/`kind`/`line_start`/`line_end`),
`description`, `current_approach`, `evolve_rationale`, `estimated_impact`
(one of `high` / `medium` / `low`), `estimated_impact_explanation`, and
`proposals[]` (each with `title`, `source`, and for `research_finding`
proposals a `finding_ref_id`).

## Task

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

Write **one markdown file** to `OUTPUT_MD`. Do not mutate `result.json`; this
is a read-only view over it.

Structure:

```markdown
# Sorted candidates — <run_id> (<module or repo name>)

**Objective:** <report.context.objective>
**Ranked:** <N> candidates · **Method:** listwise sub-agent judge  (or: pre-score fallback)
**Source:** `<RESULT_JSON>`

## Ranking summary

| # | Candidate | Module | Symbol | Impact | Score | Rationale |
|---|-----------|--------|--------|--------|-------|-----------|
| 1 | `cand-…` | pkg/… | Foo.bar | high | 92 | one-line why it ranks here |
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

## Notes

- Keep sub-agents read-only and cheap: they rank cards, they do **not**
  re-investigate the code. Give them a tight turn budget.
- The pre-score weights are intentionally simple; they exist for determinism
  and fallback, not to be the primary ranker. The sub-agent judge is what
  performs the cross-module calibration.
