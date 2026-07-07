"""Public types and entrypoints for the spotlights manager.

`run` is the cross-pipeline entrypoint
(`SpotlightsManagerInput` -> `SpotlightReport`); `run_with_telemetry` is
the runtime-rich variant returning the report alongside per-module
discovery telemetry, deep-research wallclock, and the manager's view of
step issues.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.candidate_discovery.api import (
    DiscoveryConfig,
    IterationTelemetry,
)
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.modules_extractor import ExtractorConfig
from spotlights_engine.modules_extractor.agent import ExtractionInvocation
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import (
    ModuleRun,
    SpotlightReport,
    SpotlightsManagerInput,
)
from spotlights_engine.spotlights_manager.filters import ModuleFilter


class SpotlightsManagerConfig(BaseModel):
    """Runtime/infra knobs for the manager.

    The architectural input fields stay on `SpotlightsManagerInput`; this
    class is for filesystem layout, parallelism, and per-step infra
    forwarded to steps 1-3.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifacts_dir: Path
    output_folder: Path
    max_parallel_sessions: int = Field(default=1, ge=1)

    module_filter: ModuleFilter | None = None

    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    discovery: DiscoveryConfig | None = None
    deep_research: CodexExecOptions | None = None
    proposal_from_finding: ProposalFromFindingConfig | None = None
    agent_proposals: AgentProposalsConfig | None = None

    resume: bool = True


class ModuleTelemetry(BaseModel):
    """Per-module runtime telemetry (not part of the architectural contract)."""

    model_config = ConfigDict(extra="forbid")

    discovery_iterations: list[IterationTelemetry] = Field(default_factory=list)
    discovery_total_duration_s: float | None = None
    discovery_total_cost_usd: float | None = None
    deep_research_duration_s: float | None = None
    proposal_from_finding_duration_s: float | None = None
    proposal_from_finding_per_pair_durations_s: dict[str, float] = Field(
        default_factory=dict
    )
    agent_proposals_duration_s: float | None = None
    agent_proposals_per_candidate_durations_s: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )
    issues: list[StepIssue] = Field(default_factory=list)


class SpotlightsManagerResult(BaseModel):
    """Runtime-rich manager result wrapping the cross-pipeline report.

    The `report` (`SpotlightReport`) owns the architectural payload
    (project_tree, context, candidates, findings, anomalies, run, issues).
    This wrapper adds the manager's runtime view: per-target `module_runs`,
    extractor invocation, per-module telemetry, manager-level issues, and
    the renderer result.

    `module_runs` keys are module qualified names in slash form (e.g.
    `v1/kv_offload`).

    `ExtractionInvocation` is a dataclass (not a pydantic model), so this
    model sets `arbitrary_types_allowed=True` while keeping `extra="forbid"`.
    The persisted JSON sidecar is written via `dataclasses.asdict`.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    report: SpotlightReport
    module_runs: dict[str, ModuleRun] = Field(default_factory=dict)

    extractor_invocation: ExtractionInvocation
    per_module_telemetry: dict[str, ModuleTelemetry] = Field(default_factory=dict)
    manager_issues: list[StepIssue] = Field(default_factory=list)
    renderer_result: RendererResult | None = None


def run(
    input: SpotlightsManagerInput, *, config: SpotlightsManagerConfig
) -> SpotlightReport:
    """Cross-pipeline entrypoint: returns the assembled `SpotlightReport`.

    For the runtime-rich result (per-module telemetry, extractor
    invocation, renderer result, module_runs), call `run_with_telemetry`.
    """
    return run_with_telemetry(input, config=config).report


def run_with_telemetry(
    input: SpotlightsManagerInput, *, config: SpotlightsManagerConfig
) -> SpotlightsManagerResult:
    """Runtime-rich entrypoint; thin shim over `orchestrator.run`."""
    # Late import keeps the public `api` import cheap and breaks any cycle.
    from spotlights_engine.spotlights_manager.orchestrator import (
        run as _orchestrator_run,
    )

    return _orchestrator_run(input, config=config)


# Resolve forward reference to `RendererResult`. The renderer package imports
# `spotlights_manager.persistence`, not `spotlights_manager.api`, so the cycle
# is avoided by deferring this import to module bottom.
from spotlights_engine.results_renderer.api import (  # noqa: E402
    RendererResult,
)

SpotlightsManagerResult.model_rebuild()


__all__ = [
    "ModuleTelemetry",
    "SpotlightsManagerConfig",
    "SpotlightsManagerResult",
    "run",
    "run_with_telemetry",
]
