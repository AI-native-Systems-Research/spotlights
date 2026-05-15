# candidate_research_proposer — Stage 3b: Change Generation (Research-Grounded Proposals)

Per mapped finding, **both** Claude Code and Codex run as separate subprocesses in parallel (`--stage3b-mode dual`, default). Each session produces ranked, candidate-grouped proposals that apply that one finding's technique to the code locations Stage 3a tied to it.

## 1. Inputs per session

Each session is fed:

- the finding (the 5-field record: `id`, `title`, `url`, `source_type`, `technique_summary`); the agent is expected to fetch `url` before proposing changes when the summary is insufficient,
- the **subset of candidates** that finding mapped to with `confidence ≥ --map-min-confidence` — pivoted from `mapping.json.mappings[]` (one-line group-by on `finding_id`, done once at stage start),
- read-only repo access via `--add-dir`.

Each session returns ranked proposals **grouped by candidate**: for each mapped candidate, 1–5 proposals derived from this one finding's technique. The proposal's supporting finding is implicit (the session's own `finding_id`) — Stage 3b never mixes findings within a session. The two agents' outputs for the same `finding_id` are then merged and cross-agent-deduped into one `records[]` entry.

## 2. Why dual, why per finding

Per finding (vs per candidate) lets the model load one technique's mental model and reason about it across multiple matching code locations — a single research paper applied carefully to N sites beats N shallower applications of N findings to one site each. Dual (vs single-agent) debiases the per-finding read: both models see the same finding and the same candidates, and we observe where they agree on the concrete application, which is a stronger signal than either one alone. Cost is roughly 2× a single-agent pass, capped by `--stage3b-budget-usd`.

## 3. Fan-out shape

```
INPUT:   findings.json, mapping.json, repo
OUTPUT:  records[] (one record per finding, with cross-agent-deduped proposals)

semaphore = asyncio.Semaphore(--max-parallel-agents)             # default 4

# Pivot mapping.json once at stage start.
edges_by_finding = defaultdict(list)
for m in mapping.mappings:
    for e in m.findings:
        edges_by_finding[e.finding_id].append({"candidate_id": m.candidate_id, **e})

async def changegen_one(f, agent_choice):
    mapped_candidates = edges_by_finding[f.id]
    async with semaphore:
        invoke agent_choice with:
            - --add-dir <repo-path>                              # read-only
            - stdin: STAGE3B_CHANGEGEN_PROMPT(f, mapped_candidates)
            - timeout: --per-finding-changegen-wallclock-s       # default 180s
        returns: {finding_id, agent, candidate_proposals[]}

# dual mode (default)
tasks = [changegen_one(f, a) for f in findings_with_edges for a in ("claude_code", "codex")]
# alternating mode
agents_cycle = itertools.cycle(["claude_code", "codex"])
tasks = [changegen_one(f, next(agents_cycle)) for f in findings_with_edges]
# pinned mode
tasks = [changegen_one(f, AGENT) for f in findings_with_edges]   # AGENT ∈ {"claude_code","codex"}

results = await asyncio.gather(*tasks, return_exceptions=True)
records = merge_and_dedupe(results)                              # see "Dedupe" below
```

Other modes via `--stage3b-mode`:
- `alternating` — one agent per finding, round-robin (half the cost; loses agreement signal).
- `claude_code` / `codex` — pin to one agent (smoke tests, ablations).

## 4. Budgets

- Wallclock per session: `--per-finding-changegen-wallclock-s` (default 180s — longer than Stage 3a because the session generates proposals for ≤K candidates, not just classifies).
- Cost per session: `--per-finding-changegen-budget-usd` (default $0.50).
- Parent stage budget: `--stage3b-budget-usd` (default $16 for dual mode at ~20 findings × 2 agents × $0.50 with headroom; halved automatically for `alternating`, halved again for pinned).
- Stall detection: kill the subprocess if no NDJSON output for `--per-finding-changegen-stall-s` (default 90s); record `{error: "stalled"}` and continue.

## 5. Validation

Parse each per-session result with the `finding_changes` pydantic schema. On parse failure, retry once with a strict-mode reminder appended. On second failure: record `{finding_id, agent, candidate_proposals: [], error: "schema_invalid"}` and continue — **do not** fail the stage. Drop any `candidate_id` not in the pivoted edge list for that finding (defense against the model inventing candidates). A finding's overall record is `succeeded` if **at least one** of its agent sessions parsed cleanly.

## 6. Cross-agent dedupe (dual mode only)

Two layers:

1. **Same finding × same candidate × same idea.** When both Claude Code and Codex produce a proposal for the same `(finding_id, candidate_id)` with title similarity ≥ 0.7 or LLM-judge agreement on borderline (0.3–0.7), they collapse to **one** proposal with `proposed_by: ["claude_code", "codex"]` and `agreed_with_other_agent: true`. The merged proposal takes the union of `prerequisites`, max-severity `risk`, and median `effort_estimate`. Dropped duplicates are counted in `stage3b.dedupe.rejected_cross_agent_duplicate`.
2. **Same finding × same candidate × distinct ideas.** Both proposals kept, each with `proposed_by: ["claude_code"]` or `["codex"]` and `agreed_with_other_agent: false`. These signal that the finding admits multiple applications at this site; the maintainer sees both.

Cross-agent dedupe is *only* applied within the same `(finding_id, candidate_id)` — proposals derived from different findings stay separate even if they look similar, because the citation differs.

## 7. Orphan sweep (off by default)

When `--orphan-sweep` is set, after the per-finding fan-out completes, run one additional Claude Code session per orphan candidate with the general-heuristics fallback prompt. (Single-agent because orphan sweep is a fallback, not a primary pass; pair-debiasing isn't worth the extra cost on candidates Stage 2 failed to ground.) Each such session emits proposals tagged `from_finding_id: null` and is accounted in `stage3b.orphan_sweep_*` telemetry. Capped by `--orphan-sweep-budget-usd` (default $2). Disabled by default because orphans typically signal weak Stage-2 retrieval, not a Stage-3b gap.

## 8. Change-generation prompt (drop-in)

`prompts/stage3b_changegen_per_finding.md`:

````
You are proposing concrete, ranked optimization changes for ONE research
technique applied to a small set of code locations. You do NOT write a patch —
you produce a structured proposal list per location that a human (or a future
v2 patch-writer) can act on.

## The technique (this session's single finding)
- finding_id:   {finding_id}
- title:        {f.title}
- url:          {f.url}
- source_type:  {f.source_type}
- summary:      {f.technique_summary}

The 5 fields above are the entire finding record. If the summary is too thin to
ground concrete proposals, FETCH the `url` (or draw on training-data recall of
the cited work) before writing anything — do not invent specifics not present in
the source.

This is the ONLY finding for this session. Every proposal you emit must be a
direct application of THIS technique — not a generic optimization, not a different
technique you happen to know. If a candidate doesn't admit a faithful application
of this technique after you read the code, emit zero proposals for it and say so
in `skip_reason`.

## Candidates the mapping pass tied to this finding
For each candidate the mapping stage rated at confidence ≥ {map_min_confidence}
(read the code via the repo before proposing anything):

{for each c in mapped_candidates:}
- candidate_id:       {c.id}
- file:               {c.file}
- line range:         {c.line_start}-{c.line_end}
- mapping_confidence: {c.confidence}   # one of "low", "medium", "high"
- mapping_reasoning (from Stage 3a — may be wrong, verify):
  """
  {c.mapping_reasoning}
  """
- rationale (from Stage 1 candidate discovery):
  """
  {c.candidate_rationale}
  """

## Repository access
You have read-only access to the repo via `--add-dir`. Open the files for each
candidate above and verify the technique applies before writing proposals; if
the code shape doesn't fit, drop the candidate (emit `proposals: []` with a
`skip_reason`).

## Task — for each mapped_candidate
Produce 0 to 5 ranked proposals. Each proposal must:
- be a faithful application of THIS finding's technique to THIS code location
- be implementable as a localized change (single file or small set of files
  near this location); reject ideas that require a full system redesign
- give an `expected_impact` estimate with an honest `evidence_strength` ∈
  {high, medium, low} (high requires the cited source to report concrete
  benchmarks AND for those benchmarks to translate plausibly to this codebase —
  not just to the paper's setup; if you didn't open the URL, cap at medium)
- list `prerequisites` and `risk` ∈ {low, medium, high}
- estimate `effort_estimate`: XS (<1h), S (≤1d), M (≤1w), L (>1w)

Order proposals within each candidate by
  (expected_impact_magnitude × evidence_strength / effort). Best first.

Across candidates, order `candidate_proposals[]` by the strongest proposal in
each (best candidate first) — this helps the maintainer reading the report.

## Output (REQUIRED — strict JSON, no prose outside)
{
  "finding_id": "{finding_id}",
  "candidate_proposals": [
    {
      "candidate_id": "cand-XXXX",
      "proposals": [
        {
          "rank": 1,
          "title": "...",
          "description": "...",
          "expected_impact": {"metric": "...", "estimate": "...", "evidence_strength": "high|medium|low"},
          "prerequisites": ["..."],
          "effort_estimate": "XS|S|M|L",
          "risk": "low|medium|high"
        }
      ],
      "skip_reason": null
    },
    {
      "candidate_id": "cand-YYYY",
      "proposals": [],
      "skip_reason": "Read scheduler.py:142-211; the technique assumes a fixed-capacity request pool but this code path supports unbounded growth — not a faithful application."
    }
  ]
}

Match the full schema at @finding_changes.schema.json.
````

**Orphan-sweep prompt** (`prompts/stage3b_changegen_orphan_sweep.md`, only used when `--orphan-sweep` is set): same structure but with no `finding`; one orphan candidate per session; the model is told explicitly to apply general performance heuristics for the candidate's code location (read the file at `line_start..line_end` and infer the bottleneck type) and to mark proposals with `from_finding_id: null` and `evidence_strength: low` unless it can cite a concrete prior art URL. (This pass overlaps in spirit with Stage 3c, but stays per-candidate-single-agent and only runs against orphans; Stage 3c is the broader, dual-agent pass that runs against every candidate.)

## 9. v2: produce a runnable patch?

Deferred to v2. The per-finding Stage 3b prompt could be extended with "and produce a unified diff implementing the rank-1 proposal for each non-skipped candidate" gated by `--gen-patch`, validated with `git apply --check`. Per-finding framing is actually friendlier to patch generation than per-candidate would be — the model already has the technique loaded and just specializes it per file. The cost roughly doubles (output tokens grow significantly) and the failure surface area (compile errors, hallucinated APIs) explodes; KernelBench results suggest 30–50% of LLM-generated kernels fail to compile even with explicit signatures. Plan: add only after M6 closes the eval loop with measured improvement on at least 3 modules.
