# candidate_research_proposer — Stage 3c: Novel Proposals (Agent-Novel, Per-Candidate)

Stage 3b is grounded research: every proposal cites one finding (and runs dual-agent per finding for cross-agent agreement). Stage 3c runs **after** Stage 3b finishes: for each candidate, it invokes a coding agent and asks it to propose optimization changes that don't duplicate the proposals Stage 3b already produced for that same candidate. Stage 3b's per-candidate `from_findings[]` proposals are passed in as the "already covered, do not re-propose" set; the agent must read them and produce ideas that sit outside that set.

## 1. Why a separate pass

Stage 3b is bounded by Stage 2's recall — a technique not surfaced by deep research will never appear in `records[]` no matter how obvious it is from the code. Stage 3c puts the agent's own performance priors back in the loop without contaminating the research-grounded record. Sources of novel ideas typical for this pass: language/runtime micro-optimizations (logging-on-hot-path, redundant `dict.get` chains, `str.format` vs `%s`), framework-version-specific tricks (FSDP vs DDP knobs not in the finding's paper, `torch.compile` modes), repo-specific patterns the agent reads in the surrounding code (an existing helper that's faster, a config flag that's silently expensive). These rarely have publishable papers and are exactly what deep research misses.

## 2. Why the exclusion set is Stage 3b's proposals (not raw findings)

Stage 3b's proposals are the *concrete application* of each finding to this candidate's code — far easier for the agent to compare a new idea against than the abstract `technique_summary` of a finding. It also naturally narrows the exclusion to what was actually emitted for *this* candidate (a 2–10-item list) rather than the full ~20–50 findings, most of which were never mapped here. As a side effect, orphan candidates (no Stage 3b proposals — either no finding mapped, or `--orphan-sweep` produced none) get an empty exclusion set, which is the correct behavior: Stage 3c is free to fill that gap.

## 3. Fan-out shape

Per candidate, one Claude Code subprocess and one Codex subprocess run in parallel (`--stage3c-mode dual`, default). Each session is fed:
- the candidate (`id`, `file`, `line_start`, `line_end`, `rationale`),
- Stage 3b's `from_findings[]` proposals already emitted for this candidate (title, description, and the originating finding's id + url for context) — framed as "already covered, do not re-propose,"
- read-only repo access via `--add-dir`.

Optional cheaper mode `--stage3c-mode alternating`: one agent per candidate, round-robin by candidate index; loses the cross-agent agreement signal but halves cost.

```
INPUT:  candidates.json, stage3b_proposals_by_candidate, repo
OUTPUT: novel_records[] (one record per (candidate, agent) pair)

semaphore = asyncio.Semaphore(--max-parallel-agents)             # default 4

async def novel_one(cand, agent_choice):
    async with semaphore:
        already_covered = stage3b_proposals_by_candidate.get(cand.id, [])
        invoke agent_choice with:
            - --add-dir <repo-path>                              # read-only
            - stdin: STAGE3C_NOVEL_PROMPT(cand, already_covered) # 3b proposals for this candidate
            - timeout: --per-candidate-novel-wallclock-s         # default 150s
        returns: {candidate_id, agent, proposals[]}

# dual mode (default)
tasks = [novel_one(c, a) for c in candidates for a in ("claude_code", "codex")]
# alternating mode
agents_cycle = itertools.cycle(["claude_code", "codex"])
tasks = [novel_one(c, next(agents_cycle)) for c in candidates]

results = await asyncio.gather(*tasks, return_exceptions=True)
novel_records = dedupe_and_merge(results, stage3b_proposals_by_candidate)  # see "Dedupe" below
```

## 4. Budgets

- Wallclock per (candidate, agent) session: `--per-candidate-novel-wallclock-s` (default 150s).
- Cost per session: `--per-candidate-novel-budget-usd` (default $0.30).
- Parent stage budget: `--stage3c-budget-usd` (default $10 for dual mode at ~12 candidates × 2 agents × $0.30 with headroom; halved automatically for alternating mode).
- Stall: `--per-candidate-novel-stall-s` (default 75s).

## 5. Dedupe (two layers)

The prompt-time exclusion (Stage 3b proposals for this candidate) is the first line of defense; the merge step is the safety net:

1. **Against Stage 3b proposals for the same candidate (mandatory).** Each emitted novel proposal is scored for overlap with every Stage 3b proposal already emitted for the same `candidate_id` by string-similarity over the `(proposal.title + proposal.description)` blobs — fast first pass — then any borderline case (similarity 0.3–0.7) is sent to a single LLM-judge call via the LiteLLM proxy: "Does novel proposal X re-derive Stage 3b proposal Y for the same candidate?" Reject the proposal if yes, recording it in `stage3c.dedupe.rejected_overlap_with_stage3b` (the research-grounded proposal wins because it cites a source). The model also self-reports via `novelty_check.overlaps_stage3b_proposal_ids[]` in its output; the self-report is used as a prefilter but never trusted alone.
2. **Cross-agent (dual mode only).** When both Claude Code and Codex produce a proposal for the same `candidate_id` with title similarity ≥ 0.7 (or LLM-judge agreement on borderline), they are kept as **one** proposal with `agreed_with_other_agent: true`. The other is dropped and counted in `stage3c.dedupe.rejected_cross_agent_duplicate`. Agreement is a positive quality signal that survives into the final ranking.

## 6. Validation

Parse each session's result with the `novel_changes` pydantic schema. Drop any proposal whose `novelty_check.overlaps_stage3b_proposal_ids[]` self-reports a Stage 3b proposal for this candidate *unless* the dedupe LLM-judge disagrees with the self-report. On parse failure: retry once with strict reminder; on second failure record `{candidate_id, agent, proposals: [], error: "schema_invalid"}` and continue.

## 7. Why dual is the default

This is the *novel* pass — proposals are unbacked, so calibration is harder than for Stage 3b. Running two independent agents lets `agreed_with_other_agent: true` serve as a poor-man's evidence_strength bump; without agreement, a single agent's confident-sounding novel suggestion is the prime hallucination target. M3 ablation compares dual vs alternating on proposal accept-rate (LLM-judge or maintainer-rated).

## 8. Stage 3c prompt (drop-in)

`prompts/stage3c_novel.md`:

````
You are proposing optimization changes for ONE code location. Another pass of the
pipeline already covers proposals derived from a curated list of research findings
(papers / blogs / PRs / issues). Your job is to propose changes the research pass
will NOT produce — language- / runtime- / framework- / repo-specific tricks that
sit BELOW the abstraction level of published techniques, or that the research pass
simply did not surface.

## Code location
- candidate_id: {candidate_id}
- file: {file}
- line range: {line_start}-{line_end}
- rationale from candidate discovery:
  """
  {candidate_rationale}
  """
- excerpt (read the surrounding code in the repo via --add-dir before proposing):
  ```{lang}
  {code_excerpt}
  ```

## Already-emitted proposals for this candidate — DO NOT RE-PROPOSE
The previous pipeline stage (Stage 3b) already produced research-grounded
proposals for this exact code location, each derived from an external finding
(paper / blog / PR / issue / talk). Your job is to propose changes that are NOT
materially the same as any of the proposals listed below. If the most obvious
change at this location is already covered, emit zero proposals here — that's a
valid result.

{for each Stage 3b proposal p emitted for this candidate:}
- proposal_id:   {p.id}
- title:         {p.title}
- description:   {p.description}
- from_finding:  {p.from_finding_id}  ({p.from_finding_url})

(If a description is too thin to judge overlap, fetch the `from_finding` url —
you must not re-derive a technique just because its summary in this list was terse.)

## What "novel" means here
- A different abstraction level (e.g., `logger.debug` arg formatting, redundant
  list-comprehension allocations, `dict.setdefault` vs `defaultdict`, branch
  hoisting out of a hot loop, `__slots__` on a per-step dataclass) — usually NOT
  in a paper.
- A repo-specific observation: an existing helper you found in the codebase that
  is faster than the current call site; a config flag that quietly turns on an
  expensive feature; a redundant copy between two layers.
- A framework- / runtime-version specific trick that the listed proposals don't
  cover (e.g., `torch.compile(mode="reduce-overhead")`, FSDP `use_orig_params=False`).

A proposal is NOT novel if any of the Stage 3b proposals listed above (or the
finding source each cites) already covers it, even with different wording. When
in doubt, do not emit. Better zero proposals than a duplicate of a
research-grounded one.

## Task
Produce 0 to 5 ranked proposals that are novel by the rule above. Each proposal must:
- be implementable as a localized change at this location
- name in `novelty_check.overlaps_stage3b_proposal_ids[]` any Stage 3b proposal
  you considered close and explain in `novelty_check.why_distinct` why you still
  think it's distinct (the merge step will second-guess you with an LLM judge)
- give `expected_impact` with honest `evidence_strength` ∈ {high, medium, low}.
  Without a citation, `evidence_strength: high` is rarely justified; default to
  `medium` for repo-specific reads you verified in the code, `low` for general
  heuristics.
- list `prerequisites`, `risk` ∈ {low, medium, high}, and `effort_estimate`
  ∈ {XS, S, M, L}.

Order by (expected_impact_magnitude × evidence_strength / effort).

## Output (REQUIRED — strict JSON, no prose outside)
{
  "candidate_id": "{candidate_id}",
  "agent": "claude_code | codex",
  "excluded_stage3b_proposal_ids": [/* the Stage 3b proposal ids you read as already covered */],
  "proposals": [
    {
      "rank": 1,
      "source": "agent_novel",
      "from_finding_id": null,
      "title": "...",
      "description": "...",
      "expected_impact": {"metric": "...", "estimate": "...", "evidence_strength": "high|medium|low"},
      "prerequisites": ["..."],
      "effort_estimate": "XS|S|M|L",
      "risk": "low|medium|high",
      "novelty_check": {
        "overlaps_stage3b_proposal_ids": ["prop-XXXX"],
        "why_distinct": "Even though prop-XXXX talks about ring buffers in general, my proposal targets the logging fast-path, not the allocator."
      }
    }
  ]
}

Match the full schema at @novel_changes.schema.json.
````
