# Spotlights — Architecture for the Deep Research Path 

## Project

`spotlight-engine` proposes high-leverage code changes for a target repo. The
deep-research path takes a repo, extracts its module tree, and for **each
module** runs candidate discovery and a literature/web survey, maps findings
onto candidates, then attaches both evidence-backed proposals (from findings)
and agent-knowledge proposals (beyond research) to every candidate.

`SpotlightsManager` orchestrates the flow: it calls `ModulesExtractor` once
per repo, then runs steps 2–6 once per module from the extracted tree.

## Block diagram

![Spotlights deep-research path block diagram](spotlights_deep_research_path.png)

## Shared schemas

These types are reused across module signatures below.

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

class ProjectTree(BaseModel):
    repository: Repository
    modules: list[Module] = []

CandidateKind = Literal[
    "function", "method", "loop", "region",
    "kernel", "config_block", "plugin_seam",
]
EstimatedImpact = Literal["high", "medium", "low"]

CandidateState = Literal[
    "DISCOVERED", "FINDINGS", "FINDING_PROPOSALS", "AGENT_PROPOSALS",
]

class Finding(BaseModel):
    finding_id: str
    title: str
    url: str
    short_description: str

class DeepResearchProposal(BaseModel):
    title: str
    detailed_description: str
    finding_id: str

class AgentProposal(BaseModel):
    title: str
    detailed_description: str
    agent_name: str

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
    state: CandidateState = "DISCOVERED"                   # advanced by steps 4–6
    findings: list[Finding] = []                          # filled by step 4
    deep_research_proposals: list[DeepResearchProposal] = []  # filled by step 5
    agent_proposals: list[AgentProposal] = []             # filled by step 6

class Candidates(BaseModel):
    module_qualified_name: str
    candidates: list[Candidate]
```

A **module qualified name** is the dot-joined chain of `Module.name` values from
a top-level entry in `ProjectTree.modules` down through nested `submodules` to
the target module. It uniquely identifies a module within a `ProjectTree` and is
used as the key in `candidates_by_module` and as the module selector passed into
steps 2–6.

For example, given a tree with a top-level module `inference` that contains a
submodule `attention` with a submodule `paged_kv`, the qualified name of the
deepest module is `inference.attention.paged_kv`.

## Modules

### 0. SpotlightsManager

Top-level orchestrator. Calls `ModulesExtractor` once on the repo, then runs
steps 2–6 sequentially per module discovered.

**Input**

```python
class SpotlightsManagerInput(BaseModel):
    repo_path: Path
    max_findings_per_module: int = 10
```

**Output**

```python
class SpotlightsResult(BaseModel):
    project_tree: ProjectTree
    candidates_by_module: dict[str, Candidates]   # keyed by module qualified name
```

### 1. ModulesExtractor

Produces a structured map of the repo's modules: top-level `Repository`
metadata plus a tree of `Module` nodes (name, path, description, `depends_on`,
`main_files`, nested `submodules`). Synchronous and deterministic for a given
input.

**Input**

```python
class ModulesExtractorInput(BaseModel):
    repo_path: Path
```

**Output** — `ProjectTree` (see *Shared schemas*).

### 2. candidate_discovery

Already implemented. Claude ⇄ Codex alternating review of one module emits
candidate code locations worth optimizing, each with a rationale and an impact
rating. Read-only over the module.

**Input**

```python
class CandidateDiscoveryInput(BaseModel):
    project_tree: ProjectTree
    module_qualified_name: str
```

**Output** — `Candidates` (see *Shared schemas*). At this stage `findings`,
`deep_research_proposals`, and `agent_proposals` are empty; they are filled by
steps 4–6.

### 3. module_deep_research

Per-module literature/web survey. The prompt is built by the module from the
repo info (`Repository`) and the target `Module` fields (name, path,
description, `main_files`, `depends_on`).

**Input**

```python
class ModuleDeepResearchInput(BaseModel):
    prompt: str                        # built from repo + module info
    max_output_results: int = 10
```

**Output**

```python
class ModuleDeepResearchOutput(BaseModel):
    findings: list[Finding]            # most relevant findings for the module
```

### 4. finding_to_candidates_mapper

Loops over findings; for each finding starts a Claude session whose input is
the finding plus the full candidate list, and returns the subset of candidates
the finding applies to. Edges are inverted into a candidate-centric mapping:
each candidate is filled with its related findings on `Candidate.findings`.

**Input**

```python
class FindingToCandidatesMapperInput(BaseModel):
    findings: list[Finding]            # from module_deep_research
    candidates: Candidates             # from candidate_discovery
```

**Output** — `Candidates` with `Candidate.findings` populated.

### 5. proposal_from_finding_creator

For every (candidate, finding) pair the candidate carries, drafts one
`DeepResearchProposal`. The module may decide a finding does not contain
enough information to support a proposal and skip it.

**Input**

```python
class ProposalFromFindingCreatorInput(BaseModel):
    candidates: Candidates             # with findings, from step 4
```

**Output** — `Candidates` with `Candidate.deep_research_proposals` populated.

### 6. agent_proposals

Per-candidate agent pass that proposes additional changes **not** already
covered by `deep_research_proposals`.

**Input**

```python
class AgentProposalsInput(BaseModel):
    candidates: Candidates             # from step 5
```

**Output** — `Candidates` with `Candidate.agent_proposals` populated.
