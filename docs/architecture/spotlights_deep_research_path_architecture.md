# Spotlights — Architecture for the Deep Research Path

## Project

`spotlight-engine` proposes high-leverage code changes for a target repo. The
deep-research path takes a repo, extracts its module tree, and runs candidate
discovery for each **target module**. For target modules with discovered
candidates, it runs a literature/web survey, maps findings onto candidates, and
then attempts to attach evidence-backed proposals (from findings) and
agent-knowledge proposals (beyond research) to each candidate. Proposal lists
are allowed to be empty: an empty list means the step ran but found nothing
supportable for that candidate.

`SpotlightsManager` orchestrates the flow: it calls `ModulesExtractor` once
per repo, then runs steps 2–6 once per module from the extracted tree.

Callers also supply a `SpotlightContext` (objective, workload hints, validation
plan) alongside the repo path. The context is threaded into the discovery,
deep research, mapping, and proposal steps so they can bias their judgments
toward the caller's goal. It is intentionally withheld from extraction so the
structural map stays objective-agnostic and reusable across runs.

## Block diagram

![Spotlights deep-research path block diagram](spotlights_deep_research_path.png)

## Shared schemas

These types are reused across module signatures below. Most are produced and
consumed across pipeline steps; `SpotlightContext` is the exception — it is a
**caller input** that only flows downward into a subset of steps and is never
mutated by the pipeline.

```python

class File(BaseModel):
    path: str
    role: str

class Module(BaseModel):
    name: str               # ^[a-z][a-z0-9_]*$
    path: str               # repo-relative
    description: str = ""
    depends_on: list[str] = []
    main_files: list[File] = []
    submodules: list[Module] = []

class Repository(BaseModel):
    name: str
    summary: str
    external_dependencies: list[str] = []

class ProjectModules(BaseModel):
    repository: Repository
    modules: list[Module] = []

class SpotlightContext(BaseModel):
    objective: str
    workload_hints: list[str] = []
    validation_plan: list[str] = []

CandidateKind = Literal[
    "function", "method", "loop", "region",
    "kernel", "config_block", "plugin_seam",
]
EstimatedImpact = Literal["high", "medium", "low"]
MappingConfidence = Literal["high", "medium", "low"]
FindingSourceType = Literal[
    "paper", "blog", "docs", "issue", "pr", "talk", "codebase", "other",
]

CandidateState = Literal[
    "DISCOVERED",
    "FINDINGS_MAPPED",
    "FINDING_PROPOSALS_CREATED",
    "AGENT_PROPOSALS_CREATED",
]

PipelineStep = Literal[
    "modules_extractor",
    "candidate_discovery",
    "module_deep_research",
    "finding_to_candidates_mapper",
    "proposal_from_finding_creator",
    "agent_proposals",
]
ModuleRunStatus = Literal["SUCCEEDED", "DEGRADED", "SKIPPED", "FAILED"]

class Finding(BaseModel):
    finding_id: str         # ^find-\d{4}$
    title: str
    url: str
    source_type: FindingSourceType
    technique_summary: str
    supporting_evidence: str = ""     # short excerpt, paraphrase, or source note

class FindingMatch(BaseModel):
    finding: Finding
    confidence: MappingConfidence
    rationale: str
    mapped_by: str                    # agent or mapper identifier

class DeepResearchProposal(BaseModel):
    title: str
    detailed_description: str
    finding_id: str
    proposal_rationale: str
    created_by: str

class AgentProposal(BaseModel):
    title: str
    detailed_description: str
    agent_name: str
    novelty_rationale: str            # why this is not covered by research proposals

class Candidate(BaseModel):
    id: str                 # ^cand-\d{4}$
    file: str
    line_start: int         # >= 1
    line_end: int           # >= line_start
    symbol: str
    kind: CandidateKind
    description: str
    current_approach: str
    evolve_rationale: str
    estimated_impact: EstimatedImpact
    estimated_impact_explanation: str
    state: CandidateState = "DISCOVERED"
    finding_matches: list[FindingMatch] = []        # filled by step 4
    deep_research_proposals: list[DeepResearchProposal] = []  # step 5
    agent_proposals: list[AgentProposal] = []       # step 6

class Candidates(BaseModel):
    module_qualified_name: str
    candidates: list[Candidate] = []

class StepIssue(BaseModel):
    step: PipelineStep
    severity: Literal["warning", "error"]
    message: str
    recoverable: bool = True

class ModuleRun(BaseModel):
    module_qualified_name: str
    status: ModuleRunStatus
    candidates: Candidates | None = None   # None only when the module failed before discovery
    issues: list[StepIssue] = []
```

### SpotlightContext semantics

`SpotlightContext` carries the caller's intent for a run:

- `objective` — the high-level goal of this run (e.g. "reduce decode latency
  for long-context serving"). Anchors what counts as a worthwhile candidate
  or proposal. Required: a run without an objective is a smell, and the field
  is cheap to provide.
- `workload_hints` — free-form workload / deployment notes (batch sizes,
  traffic shape, hardware) that bias relevance judgments downstream.
- `validation_plan` — how a proposed change should be validated in this
  environment; lets proposal and mapping steps prefer changes whose evidence
  is testable here.

The `list[str]` shape for `workload_hints` and `validation_plan` is a
deliberate v1 trade-off: it pushes structure into prose so consuming steps
can read it as natural language. Revisit if a downstream step ever needs to
filter or branch on a specific hint kind — at that point, promote the
relevant entries to typed fields.

`SpotlightContext` is **caller-supplied**. The pipeline does not derive,
mutate, or persist it onto candidates. `Candidate`, `Finding`, `ModuleRun`,
and `Candidates` are unchanged by its introduction; the only existing schema
that records context is `SpotlightsResult`, which echoes it back so a result
is self-describing for audit and repro.

Example:

```python
SpotlightContext(
    objective="reduce decode latency for long-context serving",
    workload_hints=[
        "batch size 1–8, prompts up to 32k tokens",
        "single-node 8xH100, NVLink",
    ],
    validation_plan=[
        "compare tokens/sec on the existing vLLM benchmark harness",
        "verify generations match reference within tolerance",
    ],
)
```

A **module qualified name** is the dot-joined chain of `Module.name` values from
a top-level entry in `ProjectModules.modules` down through nested `submodules` to
the target module. It uniquely identifies a module within a `ProjectModules`
(`Module.name` alone is only locally unique among siblings).

For example, given a tree with a top-level module `inference` that contains a
submodule `attention` with a submodule `paged_kv`, the qualified name of the
deepest module is `inference.attention.paged_kv`.

The canonical way to address a module across this pipeline is the pair
`(ProjectModules, module_qualified_name)`: the tree carries the module data,
the qualified name selects which node. All per-module steps follow this
convention — they take or carry the qualified name rather than embedding a
`Module` object — and `module_runs` is keyed by it.

### Target modules and lifecycle semantics

`SpotlightsManager` starts the steps 2–6 pipeline only for **target modules**.
A target module is a leaf `Module` with no `submodules`. Non-leaf modules
provide hierarchy, qualified-name prefixes, descriptions, and dependency
context; they are not processed directly. If a non-leaf area needs direct
analysis, its directly owned files should be modeled as a child leaf module so
target scopes remain non-overlapping.

Candidate lists may be empty. If `candidate_discovery` completes with zero
candidates, the module run is marked `SKIPPED` and downstream research,
mapping, and proposal steps are not run for that module.

`Candidate.state` records the furthest pipeline step that completed for that
candidate, not whether the corresponding lists are non-empty:

- `DISCOVERED`: emitted by candidate discovery.
- `FINDINGS_MAPPED`: mapping completed; `finding_matches` may be empty.
- `FINDING_PROPOSALS_CREATED`: finding-based proposal generation completed;
  `deep_research_proposals` may be empty.
- `AGENT_PROPOSALS_CREATED`: agent proposal generation completed;
  `agent_proposals` may be empty.

External research and agent calls can fail partially. Recoverable failures are
recorded as `StepIssue` entries and the module run continues with the best
available data. Unrecoverable failures mark that module `FAILED`; the manager
continues with other target modules unless configured otherwise.

## Modules

### 0. SpotlightsManager

Top-level orchestrator. Calls `ModulesExtractor` once on the repo, derives leaf
target modules, then starts the per-module pipeline with candidate discovery.
Steps 3–6 run only when candidates exist. A module with no discovered
candidates is marked `SKIPPED`. A module that finishes with recoverable issues
is marked `DEGRADED`; a module with an unrecoverable issue is marked `FAILED`.

**Input**

```python
class SpotlightsManagerInput(BaseModel):
    repo_path: Path
    context: SpotlightContext
    max_findings_per_module: int = 10
    continue_on_module_failure: bool = True
```

The manager forwards `context` unchanged into steps 2, 3, 4, 5, and 6, and
copies it onto `SpotlightsResult`. It is intentionally not passed to step 1.

**Output**

```python
class SpotlightsResult(BaseModel):
    project_tree: ProjectModules
    context: SpotlightContext
    module_runs: dict[str, ModuleRun]   # keyed by module qualified name
```

### 1. ModulesExtractor

Produces a structured map of the repo's modules: top-level `Repository`
metadata plus a tree of `Module` nodes (name, path, description, `depends_on`,
`main_files`, nested `submodules`). Synchronous and deterministic for a given
input.

`SpotlightContext` is intentionally not passed in: the structural map must not
be biased by objective so its output is reusable across runs with different
objectives.

**Input**

```python
class ModulesExtractorInput(BaseModel):
    repo_path: Path
```

**Output** — `ProjectModules` (see *Shared schemas*).

### 2. candidate_discovery

Already implemented. Claude ⇄ Codex alternating review of one module emits
candidate code locations worth optimizing, each with a rationale and an impact
rating. Read-only over the module.

Uses `context.objective` and `context.workload_hints` to rank candidate kinds
and filter out low-leverage locations that are off-objective.

**Input**

```python
class CandidateDiscoveryInput(BaseModel):
    project_tree: ProjectModules
    module_qualified_name: str
    context: SpotlightContext
```

**Output** — `Candidates` (see *Shared schemas*). At this stage
`finding_matches`, `deep_research_proposals`, and `agent_proposals` are empty.
If `candidates` is empty, the manager marks the module run `SKIPPED`.

### 3. module_deep_research

Per-module literature/web survey. The module builds its own prompt internally
from the repo info (`Repository`) and the target `Module` fields (name, path,
description, `main_files`, `depends_on`), keeping prompt construction out of
the orchestrator.

Uses `context.objective`, `context.workload_hints`, and
`context.validation_plan` to bias the survey toward sources relevant to the
caller's goal, deployment shape, and available validation path, and to filter
out findings that are clearly off-objective.

An empty `findings` list is valid. It means no relevant source survived the
survey and filtering pass; it only marks the module `DEGRADED` when accompanied
by a recoverable `StepIssue`.

**Input**

```python
class ModuleDeepResearchInput(BaseModel):
    project_tree: ProjectModules
    module_qualified_name: str
    context: SpotlightContext
    max_findings_per_module: int = 10
```

**Output**

```python
class ModuleDeepResearchOutput(BaseModel):
    findings: list[Finding] = []  # most relevant findings
    issues: list[StepIssue] = []
```

### 4. finding_to_candidates_mapper

Loops over findings; for each finding starts a Claude session whose input is
the finding plus the full candidate list, and returns the subset of candidates
the finding applies to. Edges are inverted into a candidate-centric mapping:
each candidate is filled with its related matches on
`Candidate.finding_matches`.

The match edge is explicit because applicability is a judgment, not just a
foreign-key join. Each edge carries confidence, rationale, and mapper identity.

Uses `context.objective` to score finding↔candidate confidence and to filter
matches that are clearly off-objective. The target module is read off
`candidates.module_qualified_name`; no parallel argument is added.

**Input**

```python
class FindingToCandidatesMapperInput(BaseModel):
    findings: list[Finding]            # from module_deep_research
    candidates: Candidates             # from candidate_discovery
    context: SpotlightContext
```

**Output**

```python
class FindingToCandidatesMapperOutput(BaseModel):
    candidates: Candidates             # finding_matches populated; state -> FINDINGS_MAPPED
    issues: list[StepIssue] = []
```

Empty `finding_matches` on a candidate is valid.

### 5. proposal_from_finding_creator

For every candidate/finding-match pair the candidate carries, drafts one
`DeepResearchProposal`. The module may decide a finding does not contain enough
information to support a proposal and skip it.

Uses `context.validation_plan` to bias toward proposals whose claims can be
tested in the caller's environment, and `context.objective` to keep the
proposal aligned with the run's goal.

**Input**

```python
class ProposalFromFindingCreatorInput(BaseModel):
    candidates: Candidates             # with finding_matches, from step 4
    context: SpotlightContext
```

**Output**

```python
class ProposalFromFindingCreatorOutput(BaseModel):
    candidates: Candidates             # deep_research_proposals populated;
                                       # state -> FINDING_PROPOSALS_CREATED
    issues: list[StepIssue] = []
```

Empty `deep_research_proposals` on a candidate is valid.

### 6. agent_proposals

Per-candidate agent pass that proposes additional changes **not** already
covered by `deep_research_proposals`. This pass still runs for candidates with
no research-backed proposals, because those candidates may have useful
agent-knowledge ideas.

Uses `context.objective` and `context.validation_plan` to keep agent-knowledge
proposals aligned with the run's goal and verifiable in the caller's
environment. Receives `project_tree` so proposals can situate the candidate
within sibling and parent modules rather than reasoning from the candidate
site alone.

**Input**

```python
class AgentProposalsInput(BaseModel):
    project_tree: ProjectModules
    candidates: Candidates             # from step 5
    context: SpotlightContext
```

**Output**

```python
class AgentProposalsOutput(BaseModel):
    candidates: Candidates             # agent_proposals populated;
                                       # state -> AGENT_PROPOSALS_CREATED
    issues: list[StepIssue] = []
```

Empty `agent_proposals` on a candidate is valid.
