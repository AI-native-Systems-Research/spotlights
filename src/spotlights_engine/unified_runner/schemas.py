"""Public types for the unified runner.

The unified runner orchestrates the deep-research and telemetry pipelines from
the top-level `spotlights-engine` CLI. Module extraction runs once at the top;
sub-pipelines selected via `UnifiedInput.pipelines` receive the same
`ProjectTree` via their own resume mechanism. The output is a single merged
`SpotlightReport` carrying every contributor's pipeline name in
`run.pipelines`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import SpotlightReport
from spotlights_engine.spotlights_manager.filters import ModuleFilter


# Closed enum of pipeline names selectable via `--pipelines`.  Matches the
# `RunInfo.pipelines` Literal in `schemas/pipeline.py` so a `UnifiedInput`
# `pipelines` list flows verbatim into the emitted report.
PipelineName = Literal["deep_research", "telemetry"]


class UnifiedInput(BaseModel):
    """Architectural input for one unified run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    repo_path: Path
    context: SpotlightContext
    # Which pipelines this run should execute.  Single-element list runs that
    # one pipeline alone; multi-element list runs them concurrently and merges
    # results.  Order is irrelevant — the unified runner sorts the union into
    # `RunInfo.pipelines`.
    pipelines: list[PipelineName] = Field(min_length=1, default_factory=lambda: ["deep_research"])

    # Telemetry-pipeline knobs forwarded into `SignalPipelineInput`.
    telemetry_from: Path | None = None
    backend_id: str = "claude_code"
    max_candidates: int | None = Field(default=None, ge=1)
    model: str | None = None

    # DR-pipeline knobs forwarded into `SpotlightsManagerInput`.
    max_findings_per_module: int | None = Field(default=None, ge=0)
    continue_on_module_failure: bool = True


class UnifiedConfig(BaseModel):
    """Runtime/infra knobs (filesystem layout, parallelism, caps).

    Mirrors the split between `SpotlightsManagerInput` (architectural) and
    `SpotlightsManagerConfig` (infra) on the DR side.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifacts_dir: Path
    output_folder: Path
    resume: bool = True

    # Module-target filter applied to the DR per-module fan-out (signal
    # pipeline doesn't filter by module today).
    module_filter: ModuleFilter | None = None

    # DR per-step parallelism + debug caps.
    max_parallel_sessions: int = Field(default=1, ge=1)
    max_parallel_pairs: int | None = None
    max_parallel_candidates: int | None = None
    debug_first_n_pairs: int | None = None
    debug_first_n_candidates: int | None = None


class UnifiedRunSummary(BaseModel):
    """Compact per-pipeline summary persisted alongside the merged report.

    `telemetry_run_dir` / `dr_artifacts_dir` are `None` when the corresponding
    sub-pipeline did not run (e.g. `pipelines=["telemetry"]` skips DR).
    `cost_usd` follows the merge rule: `None` only if every contributor was
    `None`; otherwise non-None values sum.
    """

    model_config = ConfigDict(extra="forbid")

    pipelines: list[PipelineName] = Field(min_length=1)
    started_at: str
    finished_at: str | None = None
    cost_usd: float | None = None
    telemetry_run_dir: Path | None = None
    dr_artifacts_dir: Path | None = None
    issues: list[str] = Field(default_factory=list)


class UnifiedResult(BaseModel):
    """Top-level result returned by `run_unified`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    report: SpotlightReport
    run_dir: Path
    summary: UnifiedRunSummary


__all__ = [
    "PipelineName",
    "UnifiedConfig",
    "UnifiedInput",
    "UnifiedResult",
    "UnifiedRunSummary",
]
