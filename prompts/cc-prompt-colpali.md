You have a repository checked out locally and a single performance objective. Your task is to find the specific code regions that most limit that objective, and for each, propose a concrete, source-backed optimization. Work however you think best. Do not ask me questions — make reasonable assumptions, note them, and produce the full result in one go.

**Objective:** Find the regions that most limit retrieval latency and throughput, and propose concrete optimizations for them.

**Scope:** Consider the entire codebase, including subsystems that are optional or disabled under the default runtime configuration. For a region behind a feature flag or config switch, estimate its impact *assuming that feature is enabled*, and state the enabling condition. Do not deprioritize a region solely because it does not run under default settings.

**Repository root:** `/Users/idanfr/Projects/spotlights/colpali`

**Constraints:**
- This is **static analysis only.** The repo is present for reading, but you cannot build, run, profile, or execute tests. Reason from the source. You may read tests to identify what a correctness oracle would be, but you cannot run them.
- **Do not modify any file in the repository.** Write your output only under `./runs/colpali/16-cc-baseline/candidates/`.
- Decide your own strategy for breaking the repo down, deciding where to look, and going deep where it matters.
- You may use sub-agents, web search, and external sources (papers, docs, issues, PRs, talks) freely. When you rely on an external source for a proposal, it must carry a concrete, transferable method — not just topical relevance — and you must cite it precisely (exact title, URL, and a pointer to the specific section/algorithm), with a short verbatim quote (≤2 sentences). Do not invent titles, URLs, or results.

**Deliverable:** Every region you can justify with a concrete, code-grounded cost rationale, ranked by estimated impact on the objective and tagged high / medium / low. Do not pad to reach a number and do not truncate genuine candidates — the bar is a specific named cost, not a target count. Emit **each one** as its own file `./runs/colpali/16-cc-baseline/candidates/<symbol>__cand-NNNN.md`, in **exactly** this format:

```markdown
# <symbol>

- **File:** `<repo-relative path>` (lines A–B)
- **Symbol:** <fully.qualified.symbol>
- **Kind:** function | method | class
- **Estimated impact:** high | medium | low
- **Id:** cand-NNNN

## Description
<1–3 sentences: what this code does.>

## Current approach
<How it's implemented today, concretely — name the specific construct/op/loop/heuristic that costs, with line references. Must reflect code you actually read.>

## Estimated impact explanation
<Why improving this moves the objective. Tie to a concrete cost: memory/bandwidth/latency scaling, a materialized tensor, a redundant pass, a coarse heuristic, etc. If the region is behind a feature flag or config switch, name the config that activates it and rate impact assuming it is enabled.>

## Evolve rationale
<The specific transform you'd make and why this region is a good target (self-contained, an algebraically equivalent alternative exists, etc.). Then **Correctness oracle:** name the exact existing tests and/or reference behavior any change must preserve, including adversarial cases. (You are naming the oracle, not running it.)>

## Deep research proposals
<Zero or more, each backed by an external source:>
### N. <title of the proposed change>
- **Finding:** find-NNNN — *<exact source title>*
- **Source URL:** <url>

**Detailed description.**
<Concrete change at the named file:lines — what to replace with what, kept scoped.>

**Proposal rationale.**
<Why this source's technique fits this candidate's specific weakness, and how it'd be validated.>

## Agent proposals
### 1. <your own independent proposal>
- **Agent:** claude

**Detailed description.**
<A concrete design you generate yourself, distinct from the deep-research proposals above.>

**Novelty rationale.**
<Why it's non-overlapping with the deep-research proposals.>
```

Also write `./runs/colpali/16-cc-baseline/candidates/_ranking.md`: the full ranked list with one-line justifications and the high/medium/low tag for each, plus a short "assumptions & limitations" note (how you broke down the repo, what you couldn't verify statically, what you skipped, and which candidates are behind feature flags). Then stop.
