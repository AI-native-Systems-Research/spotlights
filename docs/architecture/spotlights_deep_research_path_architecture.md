# Spotlights — Architecture for the Deep Research Path

## Project

`spotlights-engine` proposes high-leverage code changes for a target repo. The
deep-research path takes a repo, extracts its module tree, and runs candidate
discovery for each **target module**. For target modules with discovered
candidates, it runs a literature/web survey, then considers each
(candidate, finding) pair directly to attach evidence-backed proposals (from
findings) and finally agent-knowledge proposals (beyond research) to each
candidate. Proposal lists are allowed to be empty: an empty list means the
step ran but found nothing supportable for that candidate.

`SpotlightsManager` orchestrates the flow: it calls `ModulesExtractor` once
per repo, then runs steps 2, 3, 4, and 5 once per module from the extracted
tree.

Callers also supply a `SpotlightContext` (objective, workload hints, validation
plan) alongside the repo path. The context is threaded into the discovery,
deep research, and proposal steps so they can bias their judgments toward the
caller's goal. It is intentionally withheld from extraction so the structural
map stays objective-agnostic and reusable across runs.

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
    name: str               # ^[a-z][a-z0-9_]*$; == normalized basename of path
    path: str               # repo-relative
    description: str = ""
    depends_on: list[str] = []
    main_files: list[File] = []
    submodules: list[Module] = []

class Repository(BaseModel):
    name: str
    summary: str
    source_root: str = ""   # repo-relative dir the package(s) live under;
                            # "" = repo root. Qualified name = path relative
                            # to source_root (see below).
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
FindingSourceType = Literal[
    "paper", "blog", "docs", "issue", "pr", "talk", "codebase", "other",
]

CandidateState = Literal[
    "DISCOVERED",
    "FINDING_PROPOSALS_CREATED",
    "AGENT_PROPOSALS_CREATED",
]

PipelineStep = Literal[
    "modules_extractor",
    "candidate_discovery",
    "module_deep_research",
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
    deep_research_proposals: list[DeepResearchProposal] = []  # step 4
    agent_proposals: list[AgentProposal] = []       # step 5

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
    findings: list[Finding] = []           # populated by step 3 (module_deep_research)
    issues: list[StepIssue] = []
```

`findings` carries the step-3 output for that module. It is populated even
though step 4 also consumes findings as an input — keeping them on the
`ModuleRun` makes the per-module record self-describing for audit, lets a
run that stops before step 4 (partial pipeline, slice rollout) still surface
what step 3 produced, and matches the lifecycle already used for
`candidates`.

### SpotlightContext semantics

`SpotlightContext` carries the caller's intent for a run:

- `objective` — the high-level goal of this run (e.g. "reduce decode latency
  for long-context serving"). Anchors what counts as a worthwhile candidate
  or proposal. Required: a run without an objective is a smell, and the field
  is cheap to provide.
- `workload_hints` — free-form workload / deployment notes (batch sizes,
  traffic shape, hardware) that bias relevance judgments downstream.
- `validation_plan` — how a proposed change should be validated in this
  environment; lets proposal steps prefer changes whose evidence is
  testable here.

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

A **module qualified name** is the module's `path` taken relative to
`Repository.source_root`, joined with `/` (slash-form), with each segment
normalized to `^[a-z][a-z0-9_]*$`. It uniquely identifies a module within a `ProjectModules`.
By construction its last segment equals the target `Module.name`, and within
the emitted tree the qualified name of a child equals its parent's qualified
name plus the child's name (child paths are validated to nest under their
parent). The runtime tree class is `ProjectTree` in
[schemas/project.py](../../src/spotlights_engine/schemas/project.py); its
`_validate_paths_and_qns` validator enforces these invariants and rejects
normalization collisions.

For example, in a src-layout repo with `source_root = "src"`, a module at
`src/spotlights_engine/modules_extractor/agent` has qualified name
`spotlights_engine/modules_extractor/agent`. In a root-layout repo with
`source_root = ""`, a module at `vllm/attention` has qualified name
`vllm/attention`.

The canonical way to address a module across this pipeline is the pair
`(ProjectModules, module_qualified_name)`: the tree carries the module data,
the qualified name selects which node. All per-module steps follow this
convention — they take or carry the qualified name rather than embedding a
`Module` object — and `module_runs` is keyed by it.

### Target modules and lifecycle semantics

`SpotlightsManager` starts the steps 2–5 pipeline only for **target modules**.
A target module is a leaf `Module` with no `submodules`. Non-leaf modules
provide hierarchy, qualified-name prefixes, descriptions, and dependency
context; they are not processed directly. If a non-leaf area needs direct
analysis, its directly owned files should be modeled as a child leaf module so
target scopes remain non-overlapping.

Candidate lists may be empty. If `candidate_discovery` completes with zero
candidates, the module run is marked `SKIPPED` and downstream research and
proposal steps are not run for that module.

`Candidate.state` records the furthest pipeline step that completed for that
candidate, not whether the corresponding lists are non-empty:

- `DISCOVERED`: emitted by candidate discovery.
- `FINDING_PROPOSALS_CREATED`: finding-proposal phase complete for this
  candidate; `deep_research_proposals` may be empty. When step 3 produced
  zero findings and step 4 was skipped, the manager still advances the
  candidate to this state with `deep_research_proposals=[]` so the state
  machine remains a strict prefix relation.
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
Steps 3, 4, and 5 run only when candidates exist. A module with no discovered
candidates is marked `SKIPPED`. If step 3 produces zero findings, step 4 is
skipped and the manager advances every candidate to
`FINDING_PROPOSALS_CREATED` with `deep_research_proposals=[]` before step 5.
A module that finishes with recoverable issues is marked `DEGRADED`; a module
with an unrecoverable issue is marked `FAILED`.

The manager owns persistence and resume for the run. After every step it
checkpoints that step's output to an `artifacts_dir` so a crash mid-run can
be recovered without redoing completed work; per-module pipelines write
into per-module subdirectories so they don't contend on shared files. The
manager also bounds in-flight work with a `max_parallel_sessions` gate
across per-module pipelines. Persistence layout, atomicity rules, and
resume semantics are deliberately left out of the architectural contract
— they live with the manager implementation
([spotlights_persist_impl_plan.md](spotlights_persist_impl_plan.md)) so
the per-step contracts remain pure data shapes.

**Input**

```python
class SpotlightsManagerInput(BaseModel):
    repo_path: Path
    context: SpotlightContext
    max_findings_per_module: int = 10
    continue_on_module_failure: bool = True
```

The manager forwards `context` unchanged into steps 2, 3, 4, and 5, and
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
`deep_research_proposals` and `agent_proposals` are empty. If `candidates`
is empty, the manager marks the module run `SKIPPED`.

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

The codex agent driving the survey runs with its working directory set to
`repo_path`, so it can open target-module and adjacent files directly via
their repo-relative paths when grounding findings in current code.

**Paper filter (optional).** When `paper_filter` is set, the merged/deduped
findings are collapsed to **at most one** finding — the one matching that paper.
Matching reuses the same normalized key space as dedup (`select_paper_finding`):
first by normalized URL (arxiv `abs`/`pdf`/versioned ids compare equal), then by
title when a `title` is supplied and the URLs differ. The filter runs per module
before renumber/prefix, so the survivor renumbers cleanly to `find-<segment>-0001`
and downstream steps naturally focus on that paper. A no-match yields empty
findings plus a recoverable `StepIssue`. The prompt is **unchanged** — agents
still do a blind survey; the filter is applied to their output. `run-on-pr`
drives this to focus the paper-citation axis on a specific known paper.

**Input**

```python
class ModuleDeepResearchInput(BaseModel):
    project_tree: ProjectModules
    module_qualified_name: str
    context: SpotlightContext
    repo_path: Path
    max_findings_per_module: int = 10
    paper_filter: PaperFilter | None = None  # ≤1 finding matching this paper
```

**Output**

```python
class ModuleDeepResearchOutput(BaseModel):
    findings: list[Finding] = []  # most relevant findings
    issues: list[StepIssue] = []
```

### 4. proposal_from_finding_creator

For every `(candidate, finding)` pair in the cartesian product of
`candidates.candidates × findings`, the step decides whether the finding
provides enough information to support a concrete change to the candidate.
If yes, it emits one `DeepResearchProposal` with `finding_id` set to the
finding's id; if no, it emits nothing for that pair. A single candidate may
collect zero, one, or many proposals (one per supporting finding).

Applicability and proposal drafting are folded into one judgment here:
without a separate mapping step, `context.objective` is what keeps the step
from drafting proposals from off-objective findings. `context.validation_plan`
biases toward proposals whose claims can be tested in the caller's
environment.

The contract imposes no dependency between pairs: each `(candidate, finding)`
pair has its own emit-or-not decision tied to that finding's id, and the
output schema does not require ordering or aggregation across pairs. Whether
an implementation evaluates each pair in isolation, batches all findings per
candidate, or runs the full cartesian product as one session is an
implementation choice. The target module is read off
`candidates.module_qualified_name`; no parallel argument is added.

**Input**

```python
class ProposalFromFindingCreatorInput(BaseModel):
    candidates: Candidates             # from candidate_discovery; state == DISCOVERED
    findings: list[Finding]            # from module_deep_research
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

### 5. agent_proposals

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
    candidates: Candidates             # from step 4
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
