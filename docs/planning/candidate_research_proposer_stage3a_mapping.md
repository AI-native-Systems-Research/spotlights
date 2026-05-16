# candidate_research_proposer — Stage 3a: Mapping (Findings → Candidates)

Stage 3a iterates over **findings**, not pairs. For each finding the orchestrator spawns **one** coding-agent subprocess that decides which candidates that finding actually applies to. The output is `mapping.json` — a candidate-centric grouping of every per-finding edge above a configurable confidence floor.

## 1. Inputs / outputs

For each finding the orchestrator spawns **one** coding-agent subprocess that receives:

- the single finding (`id`, `title`, `url`, `source_type`, `technique_summary` — the full 5-field record)
- the full `candidates.json` (typical Stage 1 output is 5–50 candidates, well under an agent's context budget)
- read-only access to the repo via `--add-dir`

…and returns the subset of candidates the finding actually applies to, with a per-edge `confidence ∈ {low, medium, high}` and reasoning. Since the finding record itself is minimal, the agent is expected to fetch the `url` directly (or read its training-data memory of the cited work) before deciding — the prompt includes an explicit "read the source before deciding which candidates match" instruction (see §6 below).

Agents alternate Claude Code / Codex by finding index (round-robin: `findings[0]→claude`, `findings[1]→codex`, `findings[2]→claude`, …) to debias single-model failure modes — Stage 3b's dual mode takes this further by running both agents per finding in parallel, but Stage 3a stays alternating to keep the mapping pass cheap.

Once all per-finding subprocesses complete, the orchestrator groups edges by candidate and persists `mapping.json` — every edge with `confidence ≥ --map-min-confidence`, no top-K truncation — so Stage 3b can pivot back to finding-centric losslessly.

## 2. Mechanics

```
INPUT:  candidates.json (N candidates), findings.json (M findings), repo
OUTPUT: mapping.json (candidate-centric, all edges with confidence ≥ --map-min-confidence)

semaphore     = asyncio.Semaphore(--max-parallel-agents)        # default 4
agents_cycle  = itertools.cycle(["claude_code", "codex"])        # alternating

async def map_one_finding(f, agent_choice):
    async with semaphore:
        invoke agent_choice with:
            - --permission-mode plan (claude) / --sandbox workspace-write (codex)
            - --add-dir <repo-path>                  # read-only repo access
            - stdin: STAGE3A_MAP_PROMPT(f, candidates.json)
            - timeout: --per-finding-wallclock-s     # default 90s
        returns: {finding_id, edges: [{candidate_id, confidence ∈ {"low","medium","high"}, reasoning}]}

tasks = [map_one_finding(f, next(agents_cycle)) for f in findings]
per_finding_results = await asyncio.gather(*tasks, return_exceptions=True)

edges = flatten([r.edges for r in per_finding_results
                 if not isinstance(r, Exception) and r.get("edges")])

# Persist a candidate-centric grouping of every edge above --map-min-confidence.
# Confidence ordering: high > medium > low. Stage 3b pivots back to finding-centric in one line.
mapping = group_by_candidate(edges, min_confidence=--map-min-confidence)   # no top-K
# orphans are not persisted; downstream derives them as {c.id for c in candidates} \ mapping.keys()
```

`--mapping-agent {alternating|claude_code|codex}` — default `alternating`. (Stage 3b uses `--stage3b-mode {dual|alternating|claude_code|codex}` instead, default `dual`; see the Stage 3b plan.)

## 3. Per-finding budget, timeouts, and validation

- **Wallclock per finding:** `--per-finding-wallclock-s` (default 90s — bounded task: read one finding + N candidates + verify in code).
- **Cost per finding:** `--per-finding-budget-usd` (default $0.20).
- **Stage-3a parent budget:** `--stage3a-budget-usd` (default $4 — covers ~20 findings × $0.20).
- **Schema validation:** parse each per-finding result with the `mapping_edges` pydantic schema. On parse failure, retry once with a strict-mode reminder appended to the prompt. On second failure: record `{finding_id, edges: [], error: "schema_invalid"}` and continue — **do not** fail the stage.
- **Reference integrity:** drop any edge whose `candidate_id` is not in `candidates.json`; record the drop count in `mapping.json.meta.drops` for diagnostics.
- **Confidence sanity:** confidence must be one of `{"low", "medium", "high"}`; coerce unknown / missing values to `"low"` and warn. Require `reasoning` length ≥ 20 chars (else demote the edge to `"low"` with a warning).
- **Per-finding hang detection:** if a subprocess produces no NDJSON output for `--per-finding-stall-s` (default 60s), kill it and record `{error: "stalled"}` for that finding.

## 4. Optional BM25 pre-filter for large candidate sets

If `len(candidates) > --map-prefilter-threshold` (default 100), each per-finding agent receives a BM25-narrowed shortlist of the top-50 candidates instead of the full list, to keep the prompt under context budget. Disabled by default because typical Stage 1 outputs are well under 100. Configured via `--prefilter {none|bm25}` (default `none`). Pure `rank_bm25` over the tokenized `(file, rationale)` text — no embedding endpoint required.

## 5. Completeness check (mapping output)

Every candidate ideally has ≥1 edge with `confidence ≥ --map-min-confidence` (default `low`). Orphans (candidates with no qualifying finding) are not persisted on `mapping.json`; Stage 3b derives the set as `{c.id for c in candidates.json} \ {m.candidate_id for m in mapping.json.mappings}`. Because Stage 3b iterates findings, orphans get no proposals by default — they are surfaced in the run summary and listed in `changes.json.stage3b.orphan_candidates[]`. Enable `--orphan-sweep` to run one extra Claude Code session per orphan candidate with the general-heuristics fallback prompt (defined in the Stage 3b plan). Orphan rate is surfaced as a CLI summary warning.

**Many-to-many shape preserved.** A finding may map to multiple candidates; a candidate may collect edges from multiple findings. `mapping.json` is the candidate-centric grouping of every per-finding edge at or above `--map-min-confidence`, with no top-K cap — lossless, so Stage 3b can pivot back to finding-centric without dropping edges.

## 6. Concrete Stage-3a prompt (drop-in)

`prompts/stage3a_map.md`:

```
You map ONE research finding to a list of code optimization candidates. You do
NOT propose changes — that happens later. Your only job is to decide which
candidates this finding applies to, and explain why.

## Inputs
- the finding (below) — a technique surfaced by deep-research (paper / blog /
  PR / issue / talk). Five fields only: `id`, `title`, `url`, `source_type`,
  `technique_summary`. If the summary is insufficient to judge applicability,
  fetch the `url` (or recall the work from training data) before scoring.
- `candidates.json` — the full candidate list from Stage 1. Each candidate has
  `id`, `file`, `line_start`, `line_end`, `rationale`.
- the repository (read-only via --add-dir) — open files, read code, verify
  whether the technique actually applies to each candidate's location.

## Rules
- For each candidate, decide whether THIS finding's technique applies to the
  code at THIS location. A topic match is not enough — the code shape must fit.
  Example reject: a "fused-attention CUDA kernel" finding mapped to a
  pure-Python bytecode candidate, even if both involve attention.
- Read the actual code around (`file`, `line_start..line_end`) before assigning
  `high` confidence. Don't extrapolate from the rationale alone.
- Confidence ∈ {"low", "medium", "high"}:
    high   = you read the code and the technique applies as-is (or with trivial adaptation)
    medium = plausible — the code shape fits but real adaptation work is needed,
             or you couldn't fully verify by reading the source
    low    = weak / speculative; still worth surfacing for a maintainer to judge
    (if it's weaker than `low`, do not emit at all)
- Reasoning ≥ 20 chars and grounded in what you read (cite file:line if useful).
- Emit ONLY candidates the finding actually maps to. Don't enumerate rejections.
- If NO candidate fits, emit `"edges": []` — that's a valid result.

## Finding
{finding_json}

## Output (REQUIRED — strict JSON, no prose outside)
{
  "finding_id": "{finding_id}",
  "edges": [
    {"candidate_id": "cand-XXXX", "confidence": "low|medium|high", "reasoning": "..."}
  ]
}
```

**Prior art consulted:** RepoCoder (https://arxiv.org/abs/2303.12570), CodeRAG-Bench (https://arxiv.org/abs/2406.14497), and Agentless (https://arxiv.org/abs/2407.01489) for retrieval and code-to-spec patterns. Stage 3a and Stage 3b share the same per-finding fan-out shape — same semaphore, same NDJSON tailing, same schema-validate-then-retry envelope. They differ in: (a) Stage 3a alternates Claude Code / Codex per finding (one session each), while Stage 3b runs **both** agents in parallel per finding (two sessions each) and cross-agent-dedupes — same dual pattern as Stage 3c, (b) Stage 3a emits categorical-confidence edges while Stage 3b emits ranked proposals grouped by candidate, and (c) Stage 3b is given the repo + the finding's mapped candidates only (not the full candidate list).
