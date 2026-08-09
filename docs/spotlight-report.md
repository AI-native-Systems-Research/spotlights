# The `SpotlightReport` format

`SpotlightReport` is the machine-readable output of a run: a single JSON envelope holding the structural map, the candidates, and the evidence (findings and anomalies) they were built from. Both pipelines — deep-research and the telemetry `signal-pipeline` — emit the **same** shape, so a consumer written against it works regardless of which signal source produced the run. The browsable Markdown tree (`index.md`, module pages, candidate pages) is a human-facing rendering of this same data.

← Back to [README](../README.md)

The schema is defined as Pydantic v2 models under [`src/spotlights_engine/schemas/`](../src/spotlights_engine/schemas/) with `extra="forbid"` on every model (unknown keys are rejected). Import it as:

```python
from spotlights_engine.schemas import SpotlightReport
```

## Top-level envelope

`SpotlightReport` ([`schemas/pipeline.py`](../src/spotlights_engine/schemas/pipeline.py)):

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `"1"` | Pinned envelope version. |
| `project_tree` | `ProjectTree` | The structural map — repository + nested modules. |
| `context` | `SpotlightContext` | The run's objective and workload hints. |
| `candidates` | `list[Candidate]` | The code regions worth optimizing, each with its proposals. |
| `findings` | `list[Finding]` | Source-backed literature findings referenced by proposals. |
| `anomalies` | `list[Anomaly]` | Telemetry anomalies referenced by proposals (signal pipeline). |
| `run` | `RunInfo` | Which pipeline produced this, run id, timing, cost. |
| `issues` | `list[StepIssue]` | Recoverable/fatal problems observed during the run. |

`candidates`, `findings`, and `anomalies` are flat top-level lists; proposals join back to their evidence by id (see [References](#references-how-the-pieces-join)).

## The pieces

### `ProjectTree` — the structural map
[`schemas/project.py`](../src/spotlights_engine/schemas/project.py). A `Repository` (`name`, `summary`, `source_root`, `external_dependencies`) plus a recursive tree of `Module`s. Each `Module` has a `name`, `path`, `description`, `depends_on`, `main_files` (`File` = `path` + `role`), and `submodules`. A module's **qualified name** is its source-root-relative path — the same slash-form id you pass to `--include` and see in `index.md`.

### `SpotlightContext` — the run's goal
[`schemas/common.py`](../src/spotlights_engine/schemas/common.py). `objective` (the optimization goal), `workload_hints` (from `--hint`), and `validation_plan`.

### `Candidate` — a place worth optimizing
[`schemas/candidate.py`](../src/spotlights_engine/schemas/candidate.py):

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | Pattern `cand-<slug>-NNNN`. |
| `module_qualified_name` | `str \| None` | The module this candidate belongs to. |
| `origin` | `telemetry_anomaly \| code_agent` | How the candidate was surfaced. |
| `locations` | `list[CodeLocation]` | Files + `CodeSpan`s (`line_start`, `line_end`, `symbol`, `kind`). |
| `description` | `str` | What the code does. |
| `current_approach` | `str` | How it's implemented today. |
| `evolve_rationale` | `str` | Why it's worth investigating for the objective. |
| `estimated_impact` | `high \| medium \| low` | Agent-estimated impact. |
| `estimated_impact_explanation` | `str` | The reasoning behind the estimate. |
| `proposals` | `list[Proposal]` | Concrete change proposals for this candidate. |

`CodeSpan.kind` is one of `function`, `method`, `loop`, `region`, `kernel`, `config_block`, `plugin_seam`.

### `Proposal` — a concrete change
[`schemas/proposal.py`](../src/spotlights_engine/schemas/proposal.py). Carried under `Candidate.proposals`:

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | Pattern `prop-<slug>-NNNN`. |
| `source` | `research_finding \| agent_knowledge \| telemetry_anomaly` | Where the proposal came from. |
| `finding_ref_id` | `str \| None` | Join key into `findings` (for `research_finding`). |
| `anomaly_ref_ids` | `list[str] \| None` | Join keys into `anomalies` (for `telemetry_anomaly`). |
| `author` | `str \| None` | Agent name for `agent_knowledge` (e.g. `claude`, `codex`). |
| `title`, `description`, `rationale` | `str` | The proposal itself. |
| `mechanism`, `required_changes`, `expected_effect`, `evaluation_metric` | `str \| None` | Optional detail. |

### `Finding` — a literature source
[`schemas/finding.py`](../src/spotlights_engine/schemas/finding.py). `finding_id` (`find-<slug>-NNNN`), `title`, `url`, `source_type` (`paper`, `blog`, `docs`, `issue`, `pr`, `talk`, `codebase`, `other`), `technique_summary`, `supporting_evidence`.

### `Anomaly` — a telemetry signal
[`schemas/anomaly.py`](../src/spotlights_engine/schemas/anomaly.py). `anomaly_id` (`anom-<slug>-NNNN`), `type`, `description`, `estimated_severity` (`high \| medium \| low \| None`), `confidence` (`0.0`–`1.0`), `evidence_pointer`, `magnitude`.

### `RunInfo` and `StepIssue`
`RunInfo` ([`schemas/pipeline.py`](../src/spotlights_engine/schemas/pipeline.py)): `pipeline` (`deep_research \| signal`), `run_id`, `started_at`, `finished_at`, `cost_usd`. `StepIssue` ([`schemas/common.py`](../src/spotlights_engine/schemas/common.py)): `step`, `severity` (`warning \| error`), `message`, `recoverable`.

## References — how the pieces join

Proposals do not embed their evidence; they reference it by id, and the referenced records live once in the top-level `findings`/`anomalies` lists:

- `Proposal.finding_ref_id` → `Finding.finding_id`
- `Proposal.anomaly_ref_ids` → `Anomaly.anomaly_id`
- `Candidate.module_qualified_name` → a module's qualified name in `project_tree`

## Example

An abbreviated `SpotlightReport` (full round-trip fixture at [`tests/unit/schemas/fixtures/spotlight_report_v1.json`](../tests/unit/schemas/fixtures/spotlight_report_v1.json)):

```json
{
  "schema_version": "1",
  "project_tree": {
    "repository": { "name": "demo", "summary": "…", "source_root": "", "external_dependencies": [] },
    "modules": [ { "name": "core", "path": "src/core", "depends_on": [], "main_files": [], "submodules": [] } ]
  },
  "context": { "objective": "reduce p99 latency", "workload_hints": ["batch=1-8"], "validation_plan": [] },
  "candidates": [
    {
      "id": "cand-core-0001",
      "module_qualified_name": "core",
      "origin": "code_agent",
      "locations": [ { "file": "src/core/x.py", "spans": [ { "line_start": 10, "line_end": 42, "symbol": "core.x.run", "kind": "function" } ] } ],
      "description": "Entry point doing work.",
      "current_approach": "Sequential.",
      "evolve_rationale": "Could parallelize.",
      "estimated_impact": "high",
      "estimated_impact_explanation": "Hot path.",
      "proposals": [
        { "id": "prop-core-0001", "source": "research_finding", "finding_ref_id": "find-core-0001",
          "title": "Parallelize core.x.run", "description": "Use a thread pool.", "rationale": "Hot path." }
      ]
    }
  ],
  "findings": [
    { "finding_id": "find-core-0001", "title": "Parallel-friendly hot path",
      "url": "https://example.com/…", "source_type": "paper", "technique_summary": "…" }
  ],
  "anomalies": [],
  "run": { "pipeline": "deep_research", "run_id": "demo-2026-06-18",
           "started_at": "2026-06-18T12:00:00Z", "finished_at": "2026-06-18T12:05:00Z", "cost_usd": 1.23 },
  "issues": []
}
```

## On disk

- The **deep-research** run writes the full structured output to `<output-folder>/result.json`.
- The **signal pipeline** (preview) writes the bare report to `report/spotlight_report.json` in its run directory.
- Either way, the human-readable Markdown tree (`index.md` + module/candidate pages) is rendered from the same run data. See [What you get](../README.md#what-you-get) for the on-disk layout, and [Cost and the run manifest](cost-and-manifest.md) for the separate `run_manifest.json`.
