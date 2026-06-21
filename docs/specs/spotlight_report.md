# `SpotlightReport` — unified pipeline output

> **Source-of-truth spec for the unification work.** Status: PR 1
> (schemas) implementing as of 2026-06-18; PR 2 (signal pipeline
> repair) and PR 3 (DR pipeline repair) follow per §5.
>
> **Revision history.** Saved 2026-06-15. Revised 2026-06-16 (initial
> review). Revised 2026-06-17 (post-walkthrough). Revised 2026-06-18
> (replace-in-place strategy). Revised 2026-06-18 (findings/anomalies
> restored; §4.4 reversed). Promoted to `docs/specs/` 2026-06-18.
>
> Proposes the `SpotlightReport` unified output schema for both
> pipelines. The new types **replace** today's `Candidate`,
> `DeepResearchProposal`, `AgentProposal`, signal `Change`, and
> friends in place under `src/spotlights_engine/schemas/`. The old
> types are not preserved under a `legacy/` subpackage — they're
> deleted, and each pipeline owner repairs their pipeline to use the
> new schemas (signal: user; DR: Ophir). The integration branch
> `unification/schemas` will show expected red CI between repair
> merges; that's "the work isn't done yet," not "tests are deferred."
>
> Adapter-at-output-boundary translation (the prior strategy) is
> preserved as **§2 — deferred**. It may come back later as an
> optimization once the dust settles, but is explicitly out of scope
> for the current implementation.
>
> Sibling docs:
> - [`2026-06-14_pipelines_unification_plan.md`](2026-06-14_pipelines_unification_plan.md)
>   — broader unification plan (PR #A / PR #B).
> - [`2026-06-15_change_schema_v1_v2.md`](2026-06-15_change_schema_v1_v2.md)
>   — historical (background rationale only — naming, structure, and
>   several fields have been superseded).

---

## What this session decides

1. **New shared output type** (`SpotlightReport`). Both pipelines emit
   it directly — no adapter layer.
2. **Existing schemas are replaced in place.** Today's `Candidate`,
   `DeepResearchProposal`, `AgentProposal`, signal `Change`, and
   friends are deleted from `src/spotlights_engine/schemas/` and the
   new types take their place. Each pipeline owner repairs their
   pipeline to use the new types directly. No `legacy/` subpackage,
   no adapter shim.
3. **One file per top-level schema + its helpers.** The new schema
   module splits into three files:
   - `schemas/pipeline.py` — `SpotlightReport`, `RunInfo`
   - `schemas/proposal.py` — `SpotlightProposal`, `ProposalSource`
   - `schemas/candidate.py` — `SpotlightCandidate`, `CandidateOrigin`,
     `CodeLocation`, `CodeSpan`, `CodeKind`
4. **Flat top-level lists** — `SpotlightReport.candidates`. No
   per-module nesting; each candidate carries its
   `module_qualified_name` linking back to `project_tree.modules[*]`.
5. **Findings and anomalies ARE in the report (revised 2026-06-18).**
   Top-level `findings: list[Finding]` and
   `anomalies: list[Anomaly]` lists. Each proposal references upstream
   evidence by id (`finding_ref_id: str | None`,
   `anomaly_ref_ids: list[str] | None`) which joins to entries in
   those lists. Earlier draft (saved here as §4.4 history) dropped them
   to keep the report "what to do" only; that decision was reversed
   because the legacy artifacts are going away under unification, so
   the ref-ids would dangle.
6. **`SpotlightCandidate.proposals: list[SpotlightProposal]`** —
   proposals stay *embedded* under their candidate (the association
   is tight; a proposal isn't meaningful without the candidate's
   symbol / evolve_rationale).
7. **Today's DR shape (`Candidate.deep_research_proposals[] +
   agent_proposals[]`) folds cleanly** into one `proposals[]` per
   `SpotlightCandidate`, each carrying provenance
   (`source: research_finding | agent_knowledge | telemetry_anomaly`).
   The DR pipeline-repair work makes this fold inside the
   `spotlights_manager` step modules, not in an adapter.
8. **Signal pipeline today is 1:1** — each candidate gets a
   single-element `proposals`. Future signal sources (multiple
   proposals per anomaly, multi-mechanism candidates) get N-element
   lists for free.
9. **Pipeline owners decide per dropped field whether to keep it on
   an internal type or refactor it away** — see §5 for the
   per-field-decision protocol. The unified `SpotlightReport` only
   carries what consumers need; internal pipeline state lives on
   pipeline-internal types as needed.
10. **IDs are globally unique within a report and follow a segmented
    `<type>-<segment>-NNNN` pattern** (ratified in #28, schema enforced
    in PR #25 / commit `c2af407f`). Schema patterns:

    ```
    Candidate.id    : ^cand-[A-Za-z0-9._-]+-\d{4}$
    Proposal.id     : ^prop-[A-Za-z0-9._-]+-\d{4}$
    Finding.finding_id (and DeepResearchProposal.finding_id):
                      ^find-[A-Za-z0-9._-]+-\d{4}$
    ```

    The `<segment>` slot is per-pipeline:

    - **Deep-research pipeline** — uses the module slug (e.g.
      `cand-auth_login.s2-0001`). Module ids are the natural
      segmentation since DR runs per-module by construction; this also
      gives parallel-module uniqueness without a global counter and
      keeps resume/persistence directory layout aligned with the id.
    - **Signal pipeline** — uses the fixed string `signal` for every
      candidate / proposal (e.g. `cand-signal-0001`,
      `prop-signal-0001`). Signal-pipeline candidates don't have a
      natural per-module segmentation (`module_qualified_name` is
      optional and resolved post-hoc), so a fixed segment keeps the
      schema valid without inventing one.

    Slug character class `[A-Za-z0-9._-]` matches `_SLUG_SAFE` in
    [`utils/id_helpers.py`](../../src/spotlights_engine/utils/id_helpers.py).
    Pipeline-repair work normalizes ids at the boundary where the
    report is built. Traceability back to source artifacts is via
    content fingerprints (see §4.3) — no `original_id` field.

    *Note: this is a pragmatic compromise, not a principled choice —
    see #28 for the full rationale and the option to revisit (flat
    ids with a renumber-at-assembly step) once both pipelines have
    settled.*

---

## 1. New schema

Three files under `src/spotlights_engine/schemas/`, **replacing**
today's `candidate.py`, `common.py`, `finding.py`, `pipeline.py`,
`project.py`, and `proposals.py`. The deleted-and-replaced types are
recorded in the comparison table below; pipeline owners adapt their
code to the new types per §5.

### `schemas/candidate.py`

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

# CodeKind, CodeSpan, CodeLocation, CandidateOrigin, EstimatedImpact,
# and SpotlightCandidate live in this file because they are the
# candidate sub-schema and its helpers; consumers only need
# `from schemas import SpotlightCandidate` to get the rooted shape.

CodeKind = Literal[
    "function", "method", "loop", "region", "kernel",
    "config_block", "plugin_seam"
]
# ↑ describes "kind of code" at the span level. Same vocabulary as
#   today's CandidateKind in legacy schemas; reused verbatim. Bumping
#   the value set is a schema-version bump per §4.8.


class CodeSpan(BaseModel):
    """One labelled chunk of code at a (line_start, line_end) range
    within a file. The candidate's locations are built out of these.
    """
    model_config = ConfigDict(extra="forbid")

    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    symbol: str = Field(min_length=1, max_length=200)
    # ↑ free-form readable label naming what's in this range
    #   (function/method/class name, or a labelled region/config
    #   block). Not strictly a Python symbol.
    kind: CodeKind


class CodeLocation(BaseModel):
    """A file the candidate touches plus the spans within it.

    A candidate's `locations: list[CodeLocation]` carries one entry
    per distinct file. Within each, `spans` lists one-or-more labelled
    ranges; spans within one CodeLocation may have different kinds
    (e.g. a class declaration span plus a method span within the same
    file).
    """
    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1)
    spans: list[CodeSpan] = Field(min_length=1)


CandidateOrigin = Literal["telemetry_anomaly", "code_agent"]
# ↑ where the candidate's *target* was discovered (the discovery
#   channel). Distinct from SpotlightProposal.source (what motivates
#   each proposal on the candidate). Today: signal pipeline →
#   "telemetry_anomaly"; DR pipeline → "code_agent". Future origins
#   ("github_signal", "feature_request") are a schema-version bump
#   per §4.8.


EstimatedImpact = Literal["high", "medium", "low"]


class SpotlightCandidate(BaseModel):
    """A candidate-shaped record in `SpotlightReport.candidates`.

    Denormalizes identity/context fields of legacy
    `schemas.legacy.candidate.Candidate`, adds `module_qualified_name`
    (link back to `project_tree.modules[*]`), `origin` (where it was
    discovered), and carries embedded `proposals`. Not a replacement
    for legacy Candidate; the legacy schema is unchanged and still
    drives both pipelines internally.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^cand-[A-Za-z0-9._-]+-\d{4}$")  # globally unique; segmented per §10

    # link to project_tree.modules[*].qualified_name.
    # DR always populates (it runs per-module by construction).
    # Signal may leave None when the file doesn't match any extracted module.
    module_qualified_name: str | None = None

    # where this candidate was discovered (channel, not evidence)
    origin: CandidateOrigin

    # the code locations this candidate covers — one CodeLocation per
    # distinct file, each with one-or-more labelled CodeSpans
    locations: list[CodeLocation] = Field(min_length=1)

    # candidate context (mirrors legacy Candidate)
    description: str = Field(min_length=1)
    current_approach: str = Field(min_length=1)
    evolve_rationale: str = Field(min_length=1)
    estimated_impact: EstimatedImpact
    estimated_impact_explanation: str = Field(min_length=1)

    # proposals against this candidate (embedded — see §4.2)
    proposals: list["SpotlightProposal"] = Field(default_factory=list)


# Forward import to resolve the type reference at module load.
from spotlights_engine.schemas.proposal import SpotlightProposal  # noqa: E402
SpotlightCandidate.model_rebuild()
```

### `schemas/proposal.py`

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


ProposalSource = Literal[
    "research_finding", "agent_knowledge", "telemetry_anomaly"
]


class SpotlightProposal(BaseModel):
    """The unified proposal shape. Replaces today's
    DeepResearchProposal, AgentProposal, and signal-pipeline `Change`
    for any consumer reading `SpotlightReport`. The legacy source
    schemas stay where they are; the adapter folds them into
    SpotlightProposal records.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^prop-[A-Za-z0-9._-]+-\d{4}$")  # globally unique; segmented per §10

    source: ProposalSource

    # External-identifier references to the upstream provenance items.
    # Both nullable; populated based on `source`:
    #   source="research_finding" → finding_ref_id is the upstream
    #     Finding.finding_id (single); anomaly_ref_ids is None.
    #   source="telemetry_anomaly" → anomaly_ref_ids is the list of
    #     upstream Anomaly.anomaly_id values; finding_ref_id is None.
    #   source="agent_knowledge"  → both are None.
    # Treat these as opaque strings — the report does NOT carry the
    # finding/anomaly metadata. Consumers needing upstream content
    # read the legacy artifact (SpotlightsResult / SignalPipelineResult).
    anomaly_ref_ids: list[str] | None = None
    finding_ref_id: str | None = None

    # author of the underlying proposal — meaningful only when
    # source="agent_knowledge", where the agent IS the source of value
    # (e.g. "claude", "codex"). Free-form str so the field stays open
    # to new agents / human contributors without a schema bump.
    # None for research_finding and telemetry_anomaly (the synthesizing
    # LLM is pipeline plumbing, not a meaningful author).
    author: str | None = None

    # narrative — always populated
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    # structured — populated when known; None otherwise
    mechanism: str | None = None
    required_changes: str | None = None
    expected_effect: str | None = None
    evaluation_metric: str | None = None
```

### `schemas/pipeline.py`

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.candidate import SpotlightCandidate
from spotlights_engine.schemas.legacy.common import SpotlightContext, StepIssue
from spotlights_engine.schemas.legacy.project import ProjectTree


class RunInfo(BaseModel):
    """Info about the run that produced this report."""
    model_config = ConfigDict(extra="forbid")

    pipeline: Literal["deep_research", "signal"]
    # ↑ which pipeline produced the report. Future "unified" value
    #   is deferred until pipelines actually unify; widening this
    #   Literal is a schema-version bump per §4.8.

    run_id: str = Field(min_length=1)
    started_at: str                       # ISO-8601
    finished_at: str | None = None
    cost_usd: float | None = None


class SpotlightReport(BaseModel):
    """Cross-pipeline output. Both pipelines emit one of these
    alongside their existing pipeline-internal results. Flat — no
    per-module nesting; each candidate carries its
    module_qualified_name. No top-level findings/anomalies lists;
    proposals carry external-id references only.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"

    # inputs echoed back, for self-describing audit/repro
    project_tree: ProjectTree
    context: SpotlightContext

    # the work — flat list; group by module_qualified_name client-side
    candidates: list[SpotlightCandidate] = Field(default_factory=list)

    # upstream evidence referenced by each proposal's *_ref_id field(s)
    findings: list[Finding] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)

    # cross-pipeline run info + diagnostics
    run: RunInfo
    issues: list[StepIssue] = Field(default_factory=list)
```

`Anomaly` is defined in `schemas/anomaly.py` with the locked field set
from the prompt + design doc: `anomaly_id`, `type`, `description`,
`severity` (Literal["high","medium","low"] | None), `confidence`
(float 0..1 | None), `evidence_pointer`, `magnitude`. `Finding` is the
existing type from `schemas/finding.py`.

### `schemas/__init__.py`

Re-exports every public new type from the bare `schemas` namespace
so `from spotlights_engine.schemas import SpotlightReport, ...` works.
Today's broader re-export set (`Candidate`, `Finding`, etc.) is gone —
existing pipeline code will fail to import until repaired per §5.

### Naming map

| Today | In `SpotlightReport` |
|---|---|
| `Candidate` | `SpotlightCandidate` |
| `DeepResearchProposal` / `AgentProposal` / signal `Change` | `SpotlightProposal` |
| `Finding` | Reused as-is; carried inline on `SpotlightReport.findings` and referenced by `Proposal.finding_ref_id`. |
| `AnomalyLite` (signal-pipeline-internal) | Replaced by new `Anomaly` (closed shape, see §4.4). Carried inline on `SpotlightReport.anomalies` and referenced by `Proposal.anomaly_ref_ids`. |
| (file/line_start/line_end/symbol/kind on Candidate) | `CodeLocation` + `CodeSpan` + `CodeKind` |
| — | `RunInfo` |
| — | `SpotlightReport` |

### Comparison with what's already there

| Existing artifact | Status under this proposal |
|---|---|
| `schemas/candidate.py` (`Candidate`, `Candidates`, `CandidateKind`, etc.) | **Deleted** and replaced by the new `schemas/candidate.py` (different contents). DR pipeline-repair work translates today's `Candidate` shape into `SpotlightCandidate` directly inside the orchestrator/step modules. |
| `schemas/finding.py` (`Finding`, `FindingSourceType`) | **Survives** (reused as-is by `SpotlightReport.findings`). Each `Proposal.finding_ref_id` joins to a `findings[i].finding_id`. |
| `schemas/proposals.py` (`DeepResearchProposal`, `AgentProposal`) | **Deleted.** DR pipeline emits `SpotlightProposal` directly. |
| `schemas/pipeline.py` (`SpotlightsResult`, `ModuleRun`, etc.) | **Deleted** and replaced by the new `schemas/pipeline.py` (`SpotlightReport`, `RunInfo`). DR pipeline-repair work emits a `SpotlightReport` directly instead of a `SpotlightsResult`. |
| `schemas/common.py` (`SpotlightContext`, `StepIssue`, `ModuleRunStatus`, `PipelineStep`) | **Mostly deleted.** `SpotlightContext` and `StepIssue` survive (referenced by `SpotlightReport`); `ModuleRunStatus` and `PipelineStep` go away unless a pipeline owner chooses to keep them on an internal type per §5. |
| `schemas/project.py` (`ProjectTree`, `Module`, `File`, `Repository`) | **Survives** (reused as-is by `SpotlightReport.project_tree`). |
| `signal_pipeline/schemas.py` (signal `Change`) | **Stays as a pipeline-internal type.** Stage 04 still emits it; signal pipeline-repair work translates it into `SpotlightProposal` at the boundary where the `SpotlightReport` is built. |
| `SpotlightProposal`, `SpotlightCandidate`, `CodeLocation`, `CodeSpan`, `CodeKind`, `RunInfo`, `SpotlightReport` | **New.** Three files: `schemas/pipeline.py`, `schemas/proposal.py`, `schemas/candidate.py`. |

---

## 2. Two adapters (deferred)

> **Deferred 2026-06-18.** Adapter-at-output-boundary translation is
> NOT in scope for the current implementation. Each pipeline owner
> repairs their pipeline to produce `SpotlightReport` directly (see
> §5). The adapter design below is preserved as a reference: if the
> direct-emit approach turns out to be too invasive in any pipeline,
> we can fall back to translating from a pipeline-internal type at
> the output boundary using the steps below.

### 2.1 Deep-research adapter

Input: a `SpotlightsResult` (today's output of `SpotlightsManager.run`).
Output: a `SpotlightReport`.

Renumber + flatten + reshape:

1. Walk `module_runs` in deterministic order (sorted by
   `module_qualified_name`).
2. **Candidates** — within each module, assign a global `cand-NNNN`
   id to every legacy `Candidate`; remember the mapping for cross-
   references.
3. **Locations** — for each Candidate, build
   `locations=[CodeLocation(file=cand.file, spans=[CodeSpan(line_start=cand.line_start, line_end=cand.line_end, symbol=cand.symbol, kind=cand.kind)])]`.
   Today's pipelines emit single-location candidates, so this is one
   CodeLocation with one CodeSpan; the schema is ready for
   multi-file / multi-span when it materializes.
4. **Origin** — `origin = "code_agent"` (DR's discovery is the
   candidate-discovery LLM).
5. **Proposals** — assign global `prop-NNNN` ids in walk order;
   concatenate the candidate's two source lists:
   - For each `cand.deep_research_proposals[i]`:
     ```
     SpotlightProposal(
         id=f"prop-{global_idx:04d}",
         source="research_finding",
         finding_ref_id=drp.finding_id,    # opaque external id
         anomaly_ref_ids=None,
         author=None,
         title=drp.title,
         description=drp.detailed_description,
         rationale=drp.proposal_rationale,
         # structured fields (mechanism, ...) left None
     )
     ```
   - For each `cand.agent_proposals[j]`:
     ```
     SpotlightProposal(
         id=f"prop-{global_idx:04d}",
         source="agent_knowledge",
         finding_ref_id=None,
         anomaly_ref_ids=None,
         author=ap.agent_name,           # e.g. "claude" / "codex"
         title=ap.title,
         description=ap.detailed_description,
         rationale=(
             f"Novel (not covered by literature): {ap.novelty_rationale}\n\n"
             f"{ap.detailed_description}"
         ),
     )
     ```
6. Issues from each `module_run.issues` are concatenated into top-level
   `SpotlightReport.issues`.

`RunInfo.pipeline = "deep_research"`. The adapter populates
`run_id`, `started_at`, `finished_at` (when available), and
`cost_usd` (sum of pipeline-side cost envelopes).

The adapter is pure (no I/O, no LLM calls); it can run as a
post-processing step inside `SpotlightsManager.run` or as a standalone
helper. Either way the existing `SpotlightsResult` is still
persisted; the new `SpotlightReport` is written next to it
(e.g. `<artifacts_dir>/spotlight_report.json`).

### 2.2 Signal-pipeline adapter

Input: a `SignalPipelineResult` plus the run-dir artifacts on disk
(`01_signals/anomalies.json`, `03_candidates/candidates.json`,
`04_changes/<cand-id>.json`).
Output: a `SpotlightReport`.

1. Load the flat `Candidates` from stage 03.
2. **Module assignment** — for each Candidate, set
   `module_qualified_name` by matching `Candidate.file` against
   `ProjectTree.modules[*]`. Use the deepest module path that
   prefixes the file. Leave `None` when no module matches.
3. **Renumber** candidates to global `cand-NNNN` ids in deterministic
   traversal order. (Today's signal pipeline already produces a
   single flat list, so renumbering is mostly a no-op; the adapter
   normalizes for consumers.)
4. **Locations** — same shape as DR:
   `locations=[CodeLocation(file=cand.file, spans=[CodeSpan(line_start, line_end, cand.symbol, kind=cand.kind)])]`.
5. **Origin** — `origin = "telemetry_anomaly"`.
6. **Proposals** — for each Candidate, load its stage-04 `Change`(s)
   (today: one per Candidate; loop handles N) and emit:
   ```
   SpotlightProposal(
       id=f"prop-{global_idx:04d}",
       source="telemetry_anomaly",
       anomaly_ref_ids=list(cand.anomaly_refs),
       finding_ref_id=None,
       author=None,
       title=f"{sc.change_type.replace('_', ' ').title()} in {cand.symbol}",
       description=f"{sc.mechanism}\n\nExpected: {sc.expected_effect}",
       rationale=cand.evolve_rationale,
       mechanism=sc.mechanism,
       required_changes=sc.required_changes,
       expected_effect=sc.expected_effect,
       evaluation_metric=sc.evaluation_metric,
       # proposal_type intentionally omitted from the unified shape
       # (see §4.10); the categorical label stays on the legacy
       # `Change.change_type` for anyone reading the legacy artifact.
   )
   ```
7. Stage-level issues map to top-level `SpotlightReport.issues`.

`RunInfo.pipeline = "signal"`.

The adapter writes `runs/<id>/spotlight_report.json`.

### 2.3 What stays untouched

- DR's per-module `Candidate.deep_research_proposals[] +
  agent_proposals[]` still get written exactly as today.
- Signal pipeline's `04_changes/<id>.json` files still get written
  exactly as today.
- `SpotlightsResult` and `SignalPipelineResult` still get returned
  from their respective entry points.
- The legacy renderer keeps reading the legacy artifacts.

The `SpotlightReport` is *additional* output. New consumers
(validation backend, archive, future renderer) read it. Old consumers
ignore it.

---

## 3. Worked example — vllm_subset reshaped

`examples/vllm_subset/result.json` is today's `SpotlightsResult`.
Below is what a `SpotlightReport` populated from that same run
looks like — for two of its candidates from the `v1.kv_offload`
module (one with DR + agent proposals; one agent-only). Truncated
for readability — descriptions / rationales are the originals.

In this DR run, `v1.kv_offload` was the first module walked, so its
candidates take the head of the global numbering (`cand-0001`,
`cand-0002`) and the proposals as `prop-0001..prop-0008`.

> *Note (2026-06-21):* the ids below show the pre-ratification flat
> format. Under the segmented scheme adopted in #28 (see §10), DR ids
> would be `cand-v1.kv_offload-0001` / `prop-v1.kv_offload-NNNN`. The
> example is left as-is for now — only the format differs; the
> renumbering / global-uniqueness story is unchanged.

```json
{
  "schema_version": "1",
  "project_tree": { "...": "echoed unchanged" },
  "context": {
    "objective": "reduce the media TTFT and median TPOT (Time Per Output Token)",
    "workload_hints": ["multi-turn agentic workload"],
    "validation_plan": []
  },
  "candidates": [
    {
      "id": "cand-0001",
      "module_qualified_name": "v1.kv_offload",
      "origin": "code_agent",
      "locations": [
        {
          "file": "vllm/v1/kv_offload/cpu/manager.py",
          "spans": [
            {
              "line_start": 19, "line_end": 22,
              "symbol": "_CACHE_POLICIES",
              "kind": "plugin_seam"
            }
          ]
        }
      ],
      "description": "Registration table mapping the runtime eviction-policy string to the CachePolicy implementation used by CPUOffloadingManager.",
      "current_approach": "The registry exposes \"lru\" -> LRUCachePolicy ...",
      "evolve_rationale": "A replacement policy is a contained extension ...",
      "estimated_impact": "high",
      "estimated_impact_explanation": "Eviction policy directly controls offloaded KV hit rate ...",
      "proposals": [
        {
          "id": "prop-0001",
          "source": "research_finding",
          "anomaly_ref_ids": null,
          "finding_ref_id": "find-0008",
          "author": null,
          "title": "Add S3-FIFO CachePolicy with probationary FIFO admission to CPU offload registry",
          "description": "Implement a new S3FIFOCachePolicy under vllm/v1/kv_offload/cpu/policies/s3fifo.py ...",
          "rationale": "The candidate is explicitly a plugin seam that already exposes \"lru\" and \"arc\" ...",
          "mechanism": null, "required_changes": null,
          "expected_effect": null, "evaluation_metric": null
        },
        {
          "id": "prop-0002",
          "source": "research_finding",
          "anomaly_ref_ids": null,
          "finding_ref_id": "find-0009",
          "author": null,
          "title": "Add TinyLFU admission policy as a new CachePolicy in _CACHE_POLICIES",
          "description": "Implement a new CachePolicy subclass (TinyLFUCachePolicy ...) ...",
          "rationale": "...",
          "mechanism": null, "required_changes": null,
          "expected_effect": null, "evaluation_metric": null
        },
        {
          "id": "prop-0003",
          "source": "research_finding",
          "anomaly_ref_ids": null,
          "finding_ref_id": "find-0010",
          "author": null,
          "title": "Add GreedyDual-Size cache policy for cost/size-aware KV block eviction",
          "description": "...",
          "rationale": "...",
          "mechanism": null, "required_changes": null,
          "expected_effect": null, "evaluation_metric": null
        },
        {
          "id": "prop-0004",
          "source": "research_finding",
          "anomaly_ref_ids": null,
          "finding_ref_id": "find-0011",
          "author": null,
          "title": "Add SIEVE eviction policy to CPU offload _CACHE_POLICIES registry",
          "description": "...",
          "rationale": "...",
          "mechanism": null, "required_changes": null,
          "expected_effect": null, "evaluation_metric": null
        },
        {
          "id": "prop-0005",
          "source": "agent_knowledge",
          "anomaly_ref_ids": null,
          "finding_ref_id": null,
          "author": "claude",
          "title": "Add prefix-chain-aware CachePolicy that evicts orphans/leaves first ...",
          "description": "Implement a new PrefixChainCachePolicy ...",
          "rationale": "Novel (not covered by literature): All four existing deep_research_proposals ...",
          "mechanism": null, "required_changes": null,
          "expected_effect": null, "evaluation_metric": null
        },
        {
          "id": "prop-0006",
          "source": "agent_knowledge",
          "anomaly_ref_ids": null,
          "finding_ref_id": null,
          "author": "codex",
          "title": "Add cross-KV-group stripe-aware CachePolicy that evicts partial group stripes first",
          "description": "Implement a new KVGroupStripeCachePolicy ...",
          "rationale": "Novel: ...",
          "mechanism": null, "required_changes": null,
          "expected_effect": null, "evaluation_metric": null
        }
      ]
    },
    {
      "id": "cand-0002",
      "module_qualified_name": "v1.kv_offload",
      "origin": "code_agent",
      "locations": [
        {
          "file": "vllm/v1/kv_offload/cpu/gpu_worker.py",
          "spans": [
            {
              "line_start": 39, "line_end": 86,
              "symbol": "compute_sub_block_ptrs",
              "kind": "function"
            }
          ]
        }
      ],
      "description": "Builds the flat int64 byte-pointer array for sub-block copies ...",
      "current_approach": "...",
      "evolve_rationale": "...",
      "estimated_impact": "medium",
      "estimated_impact_explanation": "...",
      "proposals": [
        {
          "id": "prop-0007",
          "source": "agent_knowledge",
          "anomaly_ref_ids": null,
          "finding_ref_id": null,
          "author": "claude",
          "title": "Hoist sub-block offset computation out of the per-data_ref loop in transfer_async",
          "description": "Refactor `compute_sub_block_ptrs` ...",
          "rationale": "Novel: The candidate lists no existing deep_research_proposals ...",
          "mechanism": null, "required_changes": null,
          "expected_effect": null, "evaluation_metric": null
        },
        {
          "id": "prop-0008",
          "source": "agent_knowledge",
          "anomaly_ref_ids": null,
          "finding_ref_id": null,
          "author": "codex",
          "title": "Compact contiguous copy spans before calling swap_blocks_batch",
          "description": "After `compute_sub_block_ptrs` fills `all_src` and `all_dst` ...",
          "rationale": "Novel: There are no listed deep_research_proposals ...",
          "mechanism": null, "required_changes": null,
          "expected_effect": null, "evaluation_metric": null
        }
      ]
    }
  ],
  "run": {
    "pipeline": "deep_research",
    "run_id": "vllm_subset_2026-06-14",
    "started_at": "2026-06-14T...",
    "finished_at": "2026-06-14T...",
    "cost_usd": 1.5969
  },
  "issues": []
}
```

`finding_ref_id` values (`find-0008..find-0011`) are the **upstream**
finding ids from the legacy `SpotlightsResult` artifact — opaque
strings. The report does not carry `Finding` records; consumers
needing the finding's URL / technique_summary / supporting_evidence
read the legacy artifact next to the report.

A signal-pipeline `SpotlightReport` against the same vllm tree would
have identical *shape* — only with `run.pipeline = "signal"`,
candidate `origin = "telemetry_anomaly"`, `anomaly_ref_ids` populated
on each proposal, `finding_ref_id` always None, and `proposals` of
length one each with `source = "telemetry_anomaly"` + the structured
fields (`mechanism`, `expected_effect`, …) filled in.

---

## 4. Trade-offs and rationale

### 4.1 Flat top-level lists vs per-module nesting

`SpotlightReport.candidates` is flat; each candidate carries
`module_qualified_name` linking back to `project_tree.modules[*]`.

**Why flat:** module hierarchy isn't a consumer concern (it's a
pipeline organization detail); filtering / sorting / dedupe is a
one-liner; group-by-module is cheap to recover via `groupby(...)`.

**What we lose:** module identity is denormalized into every record
(short string, acceptable cost).

### 4.2 Embedded proposals vs top-level proposals

We flatten *modules* out, but keep `proposals` *embedded* under each
candidate.

**Why:** a proposal is meaningless without its candidate's symbol /
evolve_rationale. A candidate with just `module_qualified_name` is
meaningful on its own.

**What we lose:** flat-iteration consumers pay a small walk —
`[p for c in report.candidates for p in c.proposals]` — one line.

### 4.3 ID renumbering — global ids; trace via fingerprints

Today's `cand-NNNN` ids are scoped per-module-run; different modules
each have their own `cand-0001`. A flat top-level list collides on
those, so the adapter renumbers globally. `prop-NNNN` is also
globally unique; proposal ids are decoupled from candidate ids — the
embedding under `SpotlightCandidate.proposals` carries the linkage.

The format of those renumbered ids is segmented (`<type>-<segment>-NNNN`)
per §10 — the `<segment>` slot disambiguates DR's module slugs from
signal's fixed `signal` segment without changing the global-uniqueness
or determinism story below.

**Determinism:** the adapter walks `module_runs` alphabetically by
`module_qualified_name`, candidates within a module in their stored
order, proposals per candidate in `deep_research_proposals` then
`agent_proposals` order. Same `SpotlightsResult` re-converted at any
time produces an identical `SpotlightReport`.

**Traceability without `original_id`.** Earlier draft kept
`original_id` on every record. Dropped — traceability is intrinsic
via content fingerprints:

- **Candidates** — `(file, spans[].symbol, spans[].line_start,
  spans[].line_end)` is effectively unique per run. A debugger
  matching `cand-0042` in the report against `cand-0001` in a
  per-module legacy artifact does so by that tuple.
- **Proposals** — `(title, author)` is highly distinctive; `title`
  alone usually suffices.
- **External ids on proposals** (`finding_ref_id`,
  `anomaly_ref_ids`) — preserved verbatim from the upstream
  artifact, so they double as the trace-back handle for proposal
  provenance.

### 4.4 Top-level `findings`/`anomalies` lists (reversed 2026-06-18)

`SpotlightReport.findings: list[Finding]` and
`SpotlightReport.anomalies: list[Anomaly]` carry the upstream evidence
inline. Each `Proposal.finding_ref_id` joins to a `findings[i].finding_id`;
`Proposal.anomaly_ref_ids` joins to `anomalies[i].anomaly_id`.

**Why inline (current):** the legacy `SpotlightsResult` and signal-pipeline
run-dir artifacts are going away under unification — once gone, opaque
ref-ids on a proposal would dangle. The report has to be self-describing
end-to-end, not partially (echoing `project_tree` and `context` but
sending consumers elsewhere for finding/anomaly text). One-file
consumption also matches how a downstream renderer wants to work.

**Earlier draft (dropped 2026-06-17, restored 2026-06-18):** the report
referenced upstream evidence by id only, on the argument that "the
report's job is what to do, not the full citation context." That
argument failed on (a) legacy artifacts disappearing, and (b)
consumer-side friction of joining two files.

**What we lose by carrying inline:** report payload grows by the size
of findings + anomalies (typically small — findings are URL +
title + technique_summary; anomalies are id + type + description +
~5 short fields). Acceptable cost for self-containment.

**Why two fields, not one `source_refs: list[str]`:** the previous
draft had a single open-typed `source_refs` whose meaning depended
on `source`. Splitting into two named fields with their natural
shapes (`anomaly_ref_ids: list[str] | None`, `finding_ref_id: str |
None`) is more readable for consumers and lets the type system carry
the distinction. Both fields default to `None` for "absent" — same
sentinel across the wire.

### 4.5 Signal pipeline → module mapping

The signal pipeline today produces a single flat `Candidates` object.
The signal adapter assigns each candidate a `module_qualified_name`
by matching `Candidate.file` against `ProjectTree.modules[*]`
(deepest match). Files outside any extracted module leave
`module_qualified_name = None`.

DR is unaffected — it runs per-module by construction, so its
adapter always populates the field.

`None` over a sentinel string (`"_unassigned"`) because Python's
idiomatic missing value is `None`, the type system tells consumers
to handle the absent case, and a sentinel string can collide with
real module names.

### 4.6 Where the `SpotlightReport` is written

- DR adapter writes `<artifacts_dir>/spotlight_report.json` next to
  the existing `result.json`.
- Signal adapter writes `runs/<id>/spotlight_report.json` next to
  the existing artifacts.

Both adapters can also be exposed as standalone CLI subcommands
(`spotlights-engine report <result.json>`,
`signal-pipeline report <run-dir>`) for re-converting historical
artifacts without re-running the pipeline.

### 4.7 DR diagnostic envelopes — dropped from report

Today's `SpotlightsResult` carries three DR-only diagnostic envelopes
(`renderer_result`, `extractor_invocation`, `per_module_telemetry`).
Operator diagnostics, not "the work" the report exists to convey.
Signal pipeline has no analogues. **Drop all three.**

The aggregate signal a consumer most likely wants — total cost — is
on `RunInfo.cost_usd`. The detailed breakdowns stay where they are
today (separate JSON files next to the legacy `result.json`).

`duration_s` is also dropped; consumers can compute it from
`finished_at - started_at`.

### 4.8 Schema versioning

`schema_version: Literal["1"] = "1"` is a single envelope version
covering the whole report. Bump on breaking changes; per-field
versioning is overkill.

**Who's responsible for bumping:** whoever edits the new schema files
decides if their change is breaking.

- *Additive* (new optional field with a default) — no bump.
- *Breaking* (rename/remove a field, change a type, change a field's
  meaning, **widen a `Literal` like `CandidateOrigin`,
  `ProposalSource`, `RunInfo.pipeline`, or `CodeKind`**) — bump
  `Literal["1"]` to `Literal["1", "2"]` for reads; default new
  writes to `"2"`; update both adapters.

Enforced by code review and a frozen-fixture test in
`tests/unit/schemas/`. There's no automation that detects breaking
changes from a diff; it's a discipline call.

### 4.9 Naming — `SpotlightReport` and the `Code*` family

`SpotlightReport` (singular) — chosen on 2026-06-15 over
`SpotlightsResult` (collides with today's DR pipeline result),
`DiscoveryReport` (too generic), `Spotlights` (collides with the
package name), `EngineReport` (loses the domain noun). "Report" also
signals consumer-facing artifact, distinguishing from the existing
`*Result` types.

Top-level domain types take the `Spotlight*` prefix for consistency:
`SpotlightCandidate`, `SpotlightProposal`. Plumbing types nested
inside a domain type take the `Code*` family for clarity:
`CodeLocation`, `CodeSpan`, `CodeKind`. `RunInfo` is unprefixed —
it's not a domain entity.

### 4.10 Proposal-type field dropped

Earlier draft carried `proposal_type: ProposalType | None` (a closed
`Literal["prefetch", "reorder", "replace", "tune", "add_cache",
"fuse", "other"]`) inherited from the signal-pipeline stage-04
schema.

**Dropped on 2026-06-17.** Reasons:

- **One-pipeline field.** Only signal produces it; DR leaves None. A
  field that's None on one of the two pipelines is a weak signal it
  doesn't belong on a *cross-pipeline* shape.
- **Domain-locked vocabulary.** The values are vLLM/inference-perf
  flavored (`prefetch`, `add_cache`, `fuse`). They don't generalize
  to other subject systems. Keeping them freezes a wrong-for-most-
  cases taxonomy into the unified schema.
- **Legacy artifact still has it.** Anyone needing stage-04's
  classification reads `legacy/signal_pipeline.Change.change_type`
  directly from the legacy artifact.

The remaining four structured fields (`mechanism`, `required_changes`,
`expected_effect`, `evaluation_metric`) stay — they're narrative free-
form strings, not closed taxonomies, and DR could fill them when its
prompts are updated. Aspirational-when-known, not domain-locked.

### 4.11 Author identity on `SpotlightProposal`

Today's `AgentProposal.agent_name` carries which agent produced a
proposal (`"claude" | "codex"`). Without an explicit field on the
unified type, that identity would be lost in the conversion — and
audit / review traceability with it.

**`SpotlightProposal.author: str | None`** — populated from
`AgentProposal.agent_name` for `source = "agent_knowledge"`; None
otherwise.

**Why free-form `str`, not `Literal`:** the agent value space will
grow (more agents, possibly humans). Closed enum forces a schema
bump per agent; matches the open-typed pattern.

### 4.12 Locations design — `CodeLocation` + `CodeSpan` + `CodeKind`

Today's legacy `Candidate` is single-location: one `file`, one
`(line_start, line_end)`, one `symbol`, one `kind`. Today's pipelines
all emit single-location candidates.

The unified shape uses `locations: list[CodeLocation]` where each
`CodeLocation` carries `file` plus a list of `CodeSpan`s, and `kind`
lives on each `CodeSpan`. Multi-file proposals (e.g. refactor a base
class + N implementations), multi-range targets within one file (two
key blocks), and multi-symbol-multi-kind within one file (a class
declaration + a method body) all land naturally.

**Why file-as-grouping:** matches how renderers, IDEs, and humans
think about code targets ("this candidate touches files A and B; in
A look at these spans; in B look at these spans").

**Why kind on `CodeSpan` rather than `CodeLocation` or
`SpotlightCandidate`:** spans are the atomic unit; per-span kind is
more flexible and doesn't force same-file-different-kind cases into a
degenerate shape. (A class declaration span with kind=`region` and a
method span with kind=`method` can sit under one CodeLocation.)

### 4.13 `SpotlightCandidate.origin` — discovery channel

`origin: Literal["telemetry_anomaly", "code_agent"]` records where
the candidate's *target* was discovered. Distinct from
`SpotlightProposal.source` (what motivates each proposal on the
candidate).

The two layers carry different information:

- **Origin** = discovery channel (anomaly detection, code-agent
  candidate discovery, future github-issue surface, feature-request
  intake, etc.).
- **Source** = evidence kind for a proposal (literature finding,
  agent knowledge, telemetry anomaly).

Today's pipelines have a 1:1 mapping (signal → all
`telemetry_anomaly` proposals; DR → `research_finding` +
`agent_knowledge` proposals). In plausible futures the layers
diverge: a `github_signal` candidate could carry `research_finding`
proposals; an anomaly-derived candidate could later get a
`research_finding` proposal added by an enrichment step.

### 4.14 `anomaly_refs` on candidate — dropped

Today's legacy `Candidate.anomaly_refs` lists which anomalies
motivated the candidate as a whole. Dropped from `SpotlightCandidate`
— the information lives at the proposal level (`anomaly_ref_ids`),
where it's source-tagged.

For signal candidates, every proposal has
`source = "telemetry_anomaly"` with the anomaly ids in
`anomaly_ref_ids`. The candidate-level field repeats that info.
Consumers wanting the union compute `set().union(*[p.anomaly_ref_ids
or [] for p in c.proposals])`.

The candidate-level `origin` (§4.13) carries the categorical "this
candidate was anomaly-discovered" label without pinning specific
anomaly ids.

### 4.15 `state` on candidate — dropped

Today's legacy `Candidate.state` is a pipeline-internal lifecycle
label (`DISCOVERED → FINDING_PROPOSALS_CREATED →
AGENT_PROPOSALS_CREATED`). It belongs on the in-pipeline schema,
not on the consumer-facing report. Dropped.

### 4.16 `RunInfo` field set

Earlier draft had `model`, `duration_s`, `parameters`. All dropped on
2026-06-17:

- **`model`** — a single string misrepresents reality (DR uses
  multiple models per run; signal supports per-stage overrides).
  Dropped rather than promoted to a list/dict; per-step model usage
  stays in the legacy diagnostic artifacts.
- **`duration_s`** — redundant with `finished_at - started_at`.
- **`parameters`** — open-typed dict was a "where to put things we
  don't have a typed home for" escape hatch. No real consumer
  programs against it. Add back when a real consumer demands a
  specific knob.

What remains: `pipeline`, `run_id`, `started_at`, `finished_at`,
`cost_usd`. Tight; just enough to identify and audit a run.

---

## 5. Implementation plan — replace in place

> **Strategy locked 2026-06-18.** Existing schemas are deleted and
> replaced with the new types in place. Each pipeline owner repairs
> their pipeline to use the new types directly. No `legacy/` folder,
> no adapter shim. Adapter-at-output-boundary translation (§2) stays
> in the spec as a deferred fallback.

### Integration branch

All work lands on `unification/schemas` (already on origin, frozen at
commit `248ed289` = `origin/main` HEAD as of 2026-06-18). The branch
does NOT chase `main` while the work is in flight; one final rebase
to current `main` happens after both pipeline-repair PRs have merged
into `unification/schemas`.

PRs target `unification/schemas` (NOT `main`). Final
`unification/schemas` → `main` merge is one event at the end.

### PR 1 — schemas land alone

Single PR replacing today's six schema files with the new three +
their tests. Touches **only** `src/spotlights_engine/schemas/` and
`tests/unit/schemas/`. No pipeline-code edits.

1. Delete `src/spotlights_engine/schemas/{candidate,common,finding,pipeline,project,proposals}.py`.
2. Add the new files per spec §1:
   - `schemas/pipeline.py` — `RunInfo`, `SpotlightReport`.
   - `schemas/proposal.py` — `SpotlightProposal`, `ProposalSource`.
   - `schemas/candidate.py` — `SpotlightCandidate`, `CandidateOrigin`,
     `EstimatedImpact`, `CodeLocation`, `CodeSpan`, `CodeKind`.
   - `schemas/__init__.py` — re-exports of every new public type so
     `from spotlights_engine.schemas import SpotlightReport, ...` works.
3. Keep `schemas/project.py` (`ProjectTree` etc.) — `SpotlightReport`
   imports it. The new `schemas/__init__.py` re-exports it too.
4. Salvage the small surviving subset of today's `common.py` —
   `SpotlightContext`, `StepIssue` — into a new `schemas/common.py`
   (referenced by `SpotlightReport`). Drop `ModuleRunStatus` and
   `PipelineStep` unless either pipeline owner asks for them in their
   repair work.
5. Tests under `tests/unit/schemas/test_spotlight_report.py`:
   - `extra="forbid"` rejection per type
   - id-pattern checks (`cand-NNNN`, `prop-NNNN`)
   - every Literal's accept set + one rejected non-member
   - `module_qualified_name: str | None` semantics
   - `min_length=1` on locations and spans
   - `schema_version` pinned to "1"
   - frozen-fixture JSON round-trip backstop for the §4.8 schema-
     version policy
6. PR title: `schemas: replace existing types with SpotlightReport`.

After PR 1 merges, `unification/schemas` has the new schemas in place.
The pipeline tests on this branch will be **red** because pipeline
code still imports the old types — that's the expected state until
the pipeline-repair PRs land.

### PR 2 + PR 3 — pipeline repairs (parallel)

Two independent PRs against `unification/schemas`, one per pipeline
owner. They can be developed in parallel.

#### PR 2 — Signal pipeline repair (owner: user)

Sub-branch: `unification/schemas-signal-repair` off `unification/schemas`.

Scope: refactor everything under `src/spotlights_engine/signal_pipeline/`
and its tests so the pipeline produces a `SpotlightReport` directly.

1. **Identify dropped fields.** As you touch signal-pipeline code,
   list every today-used field that doesn't exist on the new schemas
   (e.g. `Candidate.anomaly_refs`, signal `Change.change_type`,
   `Anomaly` body, etc.).
2. **Per-field decision** (per §6 protocol — drop vs keep
   internal). For each dropped field, decide:
   - *Drop:* refactor the pipeline to stop producing/reading it. The
     prompt change (if any) lands in this PR.
   - *Keep internal:* move the field onto a pipeline-internal type
     (e.g. extend `signal_pipeline/schemas.py`'s `Change` /
     `AnomalyLite`). Translate at the boundary where the
     `SpotlightReport` is built.
3. **Build SpotlightReport at the runner's output.** Where
   `run_pipeline` finishes, assemble a `SpotlightReport` with
   `RunInfo.pipeline = "signal"`, walk parsed signals + candidates +
   stage-04 changes, populate proposals with
   `source = "telemetry_anomaly"`, `anomaly_ref_ids` from upstream
   anomaly ids, and the structured fields from the stage-04 `Change`.
4. **Locations, ids:** wrap each candidate's location into one
   `CodeLocation` + one `CodeSpan`; assign global `cand-NNNN` /
   `prop-NNNN` ids in deterministic walk order.
5. **Module assignment** for each candidate:
   `module_qualified_name` by deepest-prefix match against
   `ProjectTree.modules[*]`; `None` on no match.
6. Write `runs/<id>/spotlight_report.json` next to the existing
   artifacts. Existing emission paths unchanged.
7. **Tests must be green** for `tests/unit/signal_pipeline/` at PR
   merge time. `tests/unit/schemas/` stays green throughout.

#### PR 3 — DR pipeline repair (owner: Ophir)

Sub-branch: `unification/schemas-dr-repair` off `unification/schemas`.

Scope: refactor everything under `src/spotlights_engine/spotlights_manager/`,
`src/spotlights_engine/{candidate_discovery,module_deep_research,proposal_from_finding_creator,agent_proposals,results_renderer,modules_extractor}/`,
and their tests so the pipeline produces a `SpotlightReport` directly.

1. **Identify dropped fields.** Walk DR's existing types — `Candidate`
   (with `state`, `anomaly_refs`, `kind`, embedded
   `deep_research_proposals`/`agent_proposals` lists),
   `DeepResearchProposal` (with `created_by`, `finding_id`),
   `AgentProposal` (with `agent_name`, `novelty_rationale`), `Finding`
   body, `SpotlightsResult` (with `module_runs`, `renderer_result`,
   `extractor_invocation`, `per_module_telemetry`).
2. **Per-field decision** (per §6 protocol). For each dropped field,
   drop or keep internal. Notable likely-load-bearing fields: `Finding`
   body (step 4 reads to write proposals — keep internal);
   `Candidate.state` (orchestrator step transitions — keep internal,
   or refactor away).
3. **Build SpotlightReport at the manager's output.** Where
   `SpotlightsManager.run` finishes, walk each module's candidates
   and proposals, fold `deep_research_proposals + agent_proposals`
   into one `proposals[]` per `SpotlightCandidate`. For research-
   finding proposals: `source = "research_finding"`, `finding_ref_id`
   = upstream `Finding.finding_id` (single opaque string),
   `anomaly_ref_ids = None`, `author = None`. For agent proposals:
   `source = "agent_knowledge"`, `author = ap.agent_name` (preserve
   `"claude"` / `"codex"`).
4. **Locations, ids:** as in signal repair (one CodeLocation + one
   CodeSpan per candidate today; assign global ids in deterministic
   walk order — alphabetical by `module_qualified_name`, then stored
   order within a module).
5. **Renderer:** today's `results_renderer` reads the legacy
   `SpotlightsResult` shape. Either:
   - (a) Update the renderer to read `SpotlightReport` (probably
     doable; group by `module_qualified_name` client-side).
   - (b) Keep the renderer reading a pipeline-internal shape and
     produce `SpotlightReport` only at the very end.
   Ophir picks per the §6 per-field-decision spirit.
6. Write `<artifacts_dir>/spotlight_report.json` next to existing
   artifacts. Existing per-module artifacts and `result.json` (if
   anyone still depends on them) remain unless dropped explicitly.
7. **Tests must be green** for `tests/unit/spotlights_manager/`,
   `tests/unit/candidate_discovery/`, `tests/unit/module_deep_research/`,
   `tests/unit/proposal_from_finding_creator/`,
   `tests/unit/agent_proposals/`, `tests/unit/results_renderer/`,
   `tests/unit/module_knowledge/` at PR merge time. Schemas tests
   stay green throughout.

### Final merge — `unification/schemas` → `main`

Once both PR 2 and PR 3 have merged into `unification/schemas` and
the branch is fully green, rebase the integration branch onto current
`main` (handling whatever drift accumulated during the broken window),
run the full unit suite to confirm green, and merge to `main`.

Do not merge to `main` until both pipeline repairs are in. Partial
merges leave `main` broken.

### CI expectations during the broken window

| When | `tests/unit/schemas/` | Signal pipeline tests | DR pipeline tests |
|---|---|---|---|
| After PR 1 (schemas land alone) | green | red | red |
| After PR 2 (signal repair) | green | green | red |
| After PR 3 (DR repair) | green | green | green |

The "red" cells are expected — work isn't done yet, not "tests
deferred." Schemas-themselves tests stay green throughout as the
backstop for "is the schema itself sound."

### Adapter fallback (deferred per §2 banner)

If during PR 2 or PR 3 a pipeline owner finds the direct-emit refactor
is too invasive (or that they need a smaller-scope first pass), the
adapter design in §2 is a valid fallback for that specific pipeline:
keep the pipeline-internal type as today, write an adapter at the
output boundary, defer the deeper refactor. This is a per-pipeline
escape hatch, not a default path.
