"""Schemas owned by the signal pipeline.

`Change`, `ExecutionResult`, and `CandidateDraft` are pipeline-internal
types -- they carry the fields the signal pipeline produces today but the
unified `SpotlightReport` schema doesn't surface (e.g. `Change.change_type`,
`CandidateDraft.anomaly_refs`). The runner translates them into the unified
`Candidate` / `Proposal` / `Anomaly` types at the boundary where
`SpotlightReport` is built.

`Signals` / `WorkloadProfileLite` / `TraceSummaryLite` / `AnomalyLite` are
**placeholders** for the locked Bundle A schemas in
`spotlight_observability.signals`. The sibling repo isn't a published
dependency (per `pyproject.toml` lines 43–46), so importing it at module top
would break clean clones. Stage 01 lazy-imports the real types when present;
this file gives us a typed surface in their absence so the runner, schema
checks, and tests don't need the sibling installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.candidate import CandidateKind, EstimatedImpact
from spotlights_engine.schemas.common import SpotlightContext


# ── Bundle A placeholder schemas ─────────────────────────────────────────
# Replace by importing from `spotlight_observability.signals` once that
# sibling is available. The fields here are the minimum the rest of the
# pipeline reads — not the full locked contract.


class WorkloadProfileLite(BaseModel):
    model_config = ConfigDict(extra="allow")
    workload_id: str = Field(min_length=1)
    description: str = ""


class TraceSummaryLite(BaseModel):
    model_config = ConfigDict(extra="allow")
    trace_id: str = Field(min_length=1)
    summary: str = ""
    raw_trace_pointer: str | None = None


class AnomalyLite(BaseModel):
    model_config = ConfigDict(extra="allow")
    anomaly_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    description: str = ""


class Signals(BaseModel):
    """Stage 01 output bundle — placeholder for Bundle A's locked contract."""

    model_config = ConfigDict(extra="forbid")

    workload: WorkloadProfileLite
    traces: list[TraceSummaryLite] = Field(default_factory=list)
    anomalies: list[AnomalyLite] = Field(default_factory=list)


# CandidateDraft -- pipeline-internal flat-shape candidate produced by stage 03
# and consumed by stage 04. Carries the fields the LLM emits naturally
# (file/line_start/line_end/symbol/kind/anomaly_refs) plus the discovery-
# context fields that survive into the unified `Candidate`.
#
# The runner translates a list[CandidateDraft] + list[Change] + Signals.anomalies
# into the unified `SpotlightReport` shape (Candidate with locations[],
# Proposal with anomaly_ref_ids, Anomaly with the locked closed shape) at
# the boundary where the report is built. Keeping the LLM seam shape flat
# avoids forcing the model to nest spans/locations in its structured output.


class CandidateDraft(BaseModel):
    """Stage 03 output -- one entry per anomaly-rooted finding.

    `anomaly_refs` lists upstream `AnomalyLite.anomaly_id` values that motivated
    this candidate; the runner promotes these to `Proposal.anomaly_ref_ids`
    when building the `SpotlightReport`. Empty when the candidate is not
    anomaly-rooted (rare in this pipeline today).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^cand-\d{4}$")
    file: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    symbol: str = Field(min_length=1, max_length=200)
    kind: CandidateKind
    description: str = Field(min_length=1)
    current_approach: str = Field(min_length=1)
    evolve_rationale: str = Field(min_length=1)
    estimated_impact: EstimatedImpact
    estimated_impact_explanation: str = Field(min_length=1)
    anomaly_refs: list[str] = Field(default_factory=list)


# ── Stage 04 / 05 schemas ────────────────────────────────────────────────
# Mirror `docs/signal-based/mvp_module_apis.md` "Change schema (proposed)"
# and "ExecutionResult schema (proposed)". Open contracts §2–§3 in that doc.


ChangeType = Literal[
    "prefetch", "reorder", "replace", "tune", "add_cache", "fuse", "other"
]
"""Closed set per `mvp_module_apis.md` Open contract §2 — owner: Bundle D.
Treated as proposed; widen here when more types are agreed."""

FileEditFormat = Literal["unified_diff", "after_content"]


class Change(BaseModel):
    """Bundle D part 1 output — one per `Candidate`."""

    model_config = ConfigDict(extra="forbid")

    change_id: str = Field(min_length=1)
    candidate_ref: str = Field(min_length=1)  # Candidate.id from main's schema
    change_type: ChangeType
    mechanism: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    required_changes: str = Field(min_length=1)
    evaluation_metric: str = Field(min_length=1)


class FileEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)  # relative to subject_root
    format: FileEditFormat
    payload: str  # interpretation per `format`


ExecutionStatus = Literal["applied", "failed", "partial"]


class ExecutionResult(BaseModel):
    """Bundle D part 2 output — one per `Change`, produced by an execution backend."""

    model_config = ConfigDict(extra="forbid")

    result_id: str = Field(min_length=1)
    change_ref: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    status: ExecutionStatus
    file_edits: list[FileEdit] = Field(default_factory=list)
    rationale: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)


# ── Pipeline I/O ─────────────────────────────────────────────────────────


class SignalPipelineInput(BaseModel):
    """Architectural input to `run_pipeline`."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    subject_root: Path
    telemetry_from: Path | None = None
    # Backend id forwarded to stage 05. Default mirrors `mvp_module_apis.md` §"MVP backend".
    backend_id: str = "claude_code"
    # When True (default), stage 02 may short-circuit using the cross-run
    # ProjectTree cache keyed by (subject_root, git HEAD, dirty hash).
    # Set False to force a fresh extraction (the result still updates the
    # cache). CLI: `--no-projecttree-cache`.
    projecttree_cache: bool = True
    # Soft cap on the number of candidates stage 03 should produce. None =
    # the prompt's default ("multiple expected for rich workloads"). When
    # set, the prompt instructs the model to keep the top-N by signal
    # strength × significance. Not a hard schema cap — the model may still
    # exceed it; intent is to nudge concise outputs. CLI: `--max-candidates N`.
    max_candidates: int | None = Field(default=None, ge=1)
    # Run-level model override. None = fall back to the project default
    # (`signal_pipeline.DEFAULT_MODEL`) and finally to `claude -p`'s own
    # default. A stage's pinned `SPEC.model` still wins over this — pin
    # there if you want the choice durable across runs. CLI: `--model <id>`.
    model: str | None = None
    # Optional `SpotlightContext` for the emitted `SpotlightReport`. When
    # None (today's signal-pipeline default), the runner synthesizes one
    # from `signals.workload` via `_default_context`. The unified runner
    # passes a caller-supplied context so signal- and DR-side reports
    # agree on `objective` / `workload_hints`.
    context: SpotlightContext | None = None


class SignalPipelineResult(BaseModel):
    """Architectural output from `run_pipeline`."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_dir: Path
    output_folder: Path
    completed_stages: list[str] = Field(default_factory=list)
    skipped_stages: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


__all__ = [
    "AnomalyLite",
    "CandidateDraft",
    "Change",
    "ChangeType",
    "ExecutionResult",
    "ExecutionStatus",
    "FileEdit",
    "FileEditFormat",
    "Signals",
    "SignalPipelineInput",
    "SignalPipelineResult",
    "TraceSummaryLite",
    "WorkloadProfileLite",
]
