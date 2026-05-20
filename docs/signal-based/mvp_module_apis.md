# MVP Module APIs

*Concrete inputs, outputs, and interactions for each module in the MVP signal-discovery pipeline. Companion to `plans/signal_discovery_flow.md` (visual + sequential view) and `contracts/signal_interface_spec.md` (the locked Bundle A → Bundle C contract).*

## TL;DR — pipeline I/O at a glance

End-to-end:
- **Inputs:** raw telemetry + repo root.
- **Outputs:** list of `(Change spec, ExecutionResult)` pairs.

Per module (full schemas below):

| Module | Input | Output |
|---|---|---|
| **Signal extraction** (Bundle A) | `RawTelemetry` | `WorkloadProfile`, `TraceSummary[]`, `Anomaly[]` |
| **ProjectTree extractor** | repo root path | `ProjectTree` (modules, files, deps, descriptions) |
| **Candidate generation** (Bundle C) | signals + `ProjectTree` + source access | `Candidate[]` |
| **Change generation + handoff** (Bundle D) | `Candidate` | `Change` spec → `ExecutionResult` |
| **Execution backend** | `Change` spec | `ExecutionResult` (file edits + rationale) |

---

This document covers MVP scope. Bundle B (retrieval), Bundle E (validation), and Bundle F (orchestration) are deferred; their charters are in `plans/bundle_charters.md`. Schemas marked **locked** are pinned by other docs and reproduced here for convenience; schemas marked **proposed** are this doc's contribution and need the bundle owner's review.

---

## Module map

| Module | Repo / package | Public surface |
|---|---|---|
| Bundle A — Signal extraction | `spotlights-observability` | `process(telemetry)`, `get_raw_trace(pointer)` |
| ProjectTree extractor | `spotlights-engine.modules_extractor` | `extract(root)`, HTTP `/v1/projecttrees/{map_id}/...` |
| Bundle C — Candidate generation | `spotlights-engine.candidates` | `build_candidates(...)` |
| Bundle D — Change gen + handoff | `spotlights-engine.changes` | `generate_change(c)`, `handoff(change, backend)` |
| Execution backend | external (Claude Code, OpenEvolve, ...) | `run(change)` |

Naming: per `[[project-discovery-to-spotlights-rename]]`, `discovery-engine` is the same repo as `spotlights-engine`. The discovery-docs corpus uses the old name; live code uses the new.

---

## Bundle A — Signal extraction

**Status:** locked schemas. See `contracts/signal_interface_spec.md`.

**Public functions:**

```python
def process(telemetry: RawTelemetry) -> tuple[WorkloadProfile, list[TraceSummary], list[Anomaly]]:
    """Heuristics + summarization. No interpretation."""

def get_raw_trace(pointer: str) -> bytes:
    """Pull the underlying raw capture (PyTorch Profiler JSON / Nsight / OTel)."""
```

**Inputs**
- `RawTelemetry` — deployment-specific bundle of profiles + traces; shape is Bundle A's call.

**Outputs**
- `WorkloadProfile` — captured workload's class, concurrency, request pattern, hardware, software version (locked).
- `list[TraceSummary]` — top components, stalls, resource utilization, end-to-end metrics, with `raw_trace_pointer` for drill-down (locked).
- `list[Anomaly]` — typed observations with magnitude, evidence pointer, confidence (locked).

**Interactions**
- Called by Bundle C at the start of every run (full process).
- `get_raw_trace` invoked by Bundle C on demand when summaries aren't enough.

---

## ProjectTree extractor

**Status:** locked. See `spotlights-engine/docs/projecttree/agent_projecttree_guide.md`.

**Library:**

```python
from spotlights_engine.modules_extractor import extract
from spotlights_engine.schemas.modules import ProjectTree

tree: ProjectTree = extract(root=Path("/path/to/repo"))
```

**HTTP:**

```
GET  /v1/projecttrees/{map_id}/repository
GET  /v1/projecttrees/{map_id}/modules?depth=N
GET  /v1/projecttrees/{map_id}/modules/by-path?path=X[&subtree&depth=N]
POST /v1/projecttrees/{map_id}/contract-check
POST /v1/projecttrees/{map_id}/explain-module
```

**Schema** (locked): `ProjectTree { repository, modules: list[Module] }` where `Module = {name, path, description, depends_on, main_files[{path, role}], submodules: list[Module]}`. Full Pydantic source: `spotlights_engine.schemas.modules`.

**What it does NOT provide:** raw source bytes. To read code, resolve `File.path` relative to the parent module's path. Canonical absolute path is `subject_root / module.path / file.path` — `file.path` is **module-relative**, not repo-relative. (Per `spotlights_engine.schemas.modules.File`: "Repo-relative path under the parent module's `path`.")

**Interactions**
- Called by Bundle C: full tree at start; `by-path` for targeted module drills; `explain-module` for reviewer summaries when prose is preferable to schema.

---

## Bundle C — Candidate generation

**Status:** function signature **proposed**; `Candidate` schema reproduced from proposal §4.1.

**Public function:**

```python
def build_candidates(
    workload: WorkloadProfile,
    traces: list[TraceSummary],
    anomalies: list[Anomaly],
    project_tree: ProjectTree,
    *,
    # injected dependencies for on-demand drills
    bundle_a: BundleAClient,
    project_tree_api: ProjectTreeAPI,
    subject_root: Path,
    knowledge: KnowledgeQuery | None = None,   # Bundle B, deferred for MVP
) -> list[Candidate]:
    """Reason over signals + structure to produce findings.
    Findings only — no proposed fixes (that's Bundle D's job).
    """
```

**Candidate schema (proposed, mirrors proposal §4.1):**

```python
class Evidence(BaseModel):
    kind: Literal["trace_pointer", "metric", "comparison", "passage"]
    pointer: str                          # anomaly_id, trace_id, retrieval result id, ...
    summary: str                          # one-line description

class Candidate(BaseModel):
    candidate_id: str
    locations: list[str]                  # qualified module names or code regions
    observation: str                      # what the signal indicates (NOT a hypothesized fix)
    evidence: list[Evidence]
    significance: str                     # why this matters
    workload_ref: str                     # WorkloadProfile.workload_id
```

**Internal stages** (not public — Bundle C owner's call; see `signal_discovery_flow.md` Bundle C internals):
1. *Form insight* — LLM call over signals + project_tree → internal hypothesis (no schema).
2. *Drill module* — `project_tree_api.by_path(path)` + read `main_files` from `subject_root`.
3. *Build candidate* — LLM call over insight + drilled context → `Candidate`.

**Inputs**
- The three Bundle A objects.
- A `ProjectTree` (extracted upfront).
- Live clients for on-demand calls: `BundleAClient` (for `get_raw_trace`), `ProjectTreeAPI` (for `by-path`), `subject_root` path for file reads.
- *(Deferred)* `KnowledgeQuery` for technique-driven discovery.

**Outputs**
- `list[Candidate]`.

**Interactions**
- Pulls structured signals from Bundle A (call once per run).
- `bundle_a.get_raw_trace(anomaly.evidence_pointer)` — on demand.
- `project_tree_api.by_path(path)` — on demand.
- File reads from disk at `subject_root / module.path / main_files[i].path` (paths are module-relative — see ProjectTree section).
- *(Deferred)* `knowledge.query(query, mode)`.

---

## Bundle D — Change generation + execution handoff

**Status:** **proposed.** `Change` schema mirrors proposal §4.1; `ExecutionResult` is new.

**Public functions:**

```python
def generate_change(candidate: Candidate) -> Change:
    """LLM produces a Change spec from a Candidate. Spec only — no file edits."""

def handoff(change: Change, backend_id: str) -> ExecutionResult:
    """Route the Change spec to an execution backend; return the backend's output."""
```

**Change schema (proposed, mirrors proposal §4.1):**

```python
class Change(BaseModel):
    change_id: str
    candidate_ref: str                          # Candidate.candidate_id
    change_type: Literal["prefetch", "reorder", "replace", "tune", "add_cache", "fuse", "other"]
    mechanism: str                              # why this should work
    expected_effect: str                        # latency↓ / throughput↑ / memory↓
    required_changes: str                       # code surface (paths, modules)
    evaluation_metric: str                      # how success is measured
```

**ExecutionResult schema (proposed):**

```python
class FileEdit(BaseModel):
    path: str                                   # relative to subject_root
    format: Literal["unified_diff", "after_content"]
    payload: str                                # interpretation governed by `format`:
                                                #   unified_diff   → standard `diff -u` text
                                                #   after_content  → full new file contents

class ExecutionResult(BaseModel):
    result_id: str
    change_ref: str                             # Change.change_id
    backend_id: str
    status: Literal["applied", "failed", "partial"]
    file_edits: list[FileEdit]
    rationale: str                              # backend's explanation
    artifacts: dict[str, str] = {}              # log paths, intermediate state pointers
```

A given backend should pick **one** `format` per result; mixing both within one `ExecutionResult` is allowed but discouraged. Consumers (Bundle E, Bundle F) parse based on `format`.

**Inputs**
- `generate_change`: a `Candidate`.
- `handoff`: a `Change` + a `backend_id`.

**Outputs**
- `Change` (from `generate_change`).
- `ExecutionResult` (from `handoff`).

**Interactions**
- Consumes `Candidate`s from Bundle C.
- Calls `ExecutionBackend.run(change)` against the chosen backend.
- *(Future)* Hands `(Change, ExecutionResult)` to Bundle E (validation) and Bundle F (archive).

---

## Execution backend (abstract)

**Status:** **proposed.**

```python
class ExecutionBackend(Protocol):
    backend_id: str

    def run(self, change: Change) -> ExecutionResult:
        """Take a Change spec, produce concrete file edits + rationale."""
```

**MVP backend:** `ClaudeCodeBackend` — wraps Claude Code with file-edit tools; passes the Change as the brief.

**Future backends** (per Bundle D charter): `OpenEvolveBackend`, `ShinkaEvolveBackend`, `NousBackend`. Backends self-register against `backend_id`; the registry is owned by Bundle D.

---

## Interaction walks

Three concrete sequences showing the call structure.

### 1. End-to-end pipeline run (Bundle F's job in production)

```python
workload, traces, anomalies = bundle_a.process(telemetry)
project_tree = projecttree.extract(subject_root)

candidates = bundle_c.build_candidates(
    workload, traces, anomalies, project_tree,
    bundle_a=bundle_a, project_tree_api=projecttree_api, subject_root=subject_root,
)

for c in candidates:
    change = bundle_d.generate_change(c)
    result = bundle_d.handoff(change, backend_id="claude_code")
    # → Bundle E (validation), Bundle F (archive) — out of MVP scope
```

### 2. Bundle C drilling into a module mid-reasoning

```python
# inside Bundle C, after forming an insight pointing at vllm/v1/kv_offload
module = project_tree_api.by_path("vllm/v1/kv_offload")     # → Module metadata
# f.path is module-relative; absolute path = subject_root / module.path / f.path
sources = {
    f.path: (subject_root / module.path / f.path).read_text()
    for f in module.main_files
}
# now feed `module` + `sources` into the build-candidate LLM call
```

### 3. Bundle C pulling a raw trace on demand

```python
# inside Bundle C, when an Anomaly's TraceSummary is too coarse
raw = bundle_a.get_raw_trace(anomaly.evidence_pointer)
# decode (PyTorch Profiler JSON / Nsight / OTel) per the locked spec
```

---

## Open contracts

Items that this doc surfaces but does not pin — they need the relevant bundle owner's signoff:

1. **`Evidence` schema** (Bundle C owner). Sketched above; finalize.
2. **`Change.change_type` literal set** (Bundle D owner). Proposal §4.1 lists examples but not a closed set.
3. **`ExecutionResult` schema** (Bundle D owner). The shape above is the doc's proposal.
4. **`backend_id` registry** (Bundle D owner). Who maintains the canonical list; how backends self-register.
5. **`RawTelemetry` shape** (Bundle A owner). Bundle A's input; not yet specified anywhere.
6. **Commit pinning** — see `plans/signal_discovery_flow.md` Open Q2(b). `WorkloadProfile.software_version` captures the runtime commit; nothing currently requires the source Bundle C reads to match.
7. **Bundle B knowledge query interface** — `bundle_charters.md` defines `query(query, mode) → list[Result]`. The `Result` shape isn't pinned. Deferred for MVP.

---

*Once a row in "Open contracts" is resolved, move it into the relevant module section above and remove it from this list.*

