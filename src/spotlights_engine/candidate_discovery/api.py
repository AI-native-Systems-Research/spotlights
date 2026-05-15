"""Public types and `discover()` entrypoint for Stage 1."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.candidate_discovery.errors import DiscoverySetupError
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.modules import Module


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_path: Path
    module_qualified_name: str
    module: Module
    artifacts_dir: Path
    num_review_iterations: int = Field(default=3, ge=0)
    per_iteration_wallclock_s: int = Field(default=900, ge=1)
    claude_max_turns: int = Field(default=30, ge=1)
    codex_model: str = Field(default="gpt-5.5", pattern=r"^[\w.\-/]+$")
    codex_reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] = "high"


class IterationTelemetry(BaseModel):
    n: int
    agent: Literal["claude_code", "codex"]
    session_id: str | None = None
    duration_s: float
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    candidate_count: int
    schema_retries: int = 0
    dropped_outside_module: int = 0
    dropped_missing_file: int = 0
    dropped_invalid_ranges: int = 0
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)


class DiscoveryResult(BaseModel):
    candidates: Candidates
    iterations: list[IterationTelemetry]
    total_duration_s: float
    total_cost_usd: float | None = None


def discover(config: DiscoveryConfig) -> DiscoveryResult:
    """Thin wrapper around `Orchestrator.run()` — the sole public entrypoint.

    Pre-construction setup checks live here so a missing CLI or a bad
    `artifacts_dir` fails before the run dir is minted.
    """
    repo = config.repo_path
    artifacts = config.artifacts_dir

    if not repo.exists() or not repo.is_dir():
        raise DiscoverySetupError(
            f"repo_path does not exist or is not a directory: {repo}",
            repo_path=str(repo),
        )

    resolved_repo = repo.resolve(strict=False)
    resolved_artifacts = artifacts.resolve(strict=False)
    try:
        resolved_artifacts.relative_to(resolved_repo)
    except ValueError:
        pass
    else:
        raise DiscoverySetupError(
            f"artifacts_dir resolves inside repo_path: {artifacts}",
            artifacts_dir=str(artifacts),
            repo_path=str(repo),
        )

    run_dir = artifacts / "candidate_discovery"
    if run_dir.exists():
        raise DiscoverySetupError(
            f"pre-existing candidate_discovery run dir; resume is not supported: {run_dir}",
            run_dir=str(run_dir),
        )

    from spotlights_engine.candidate_discovery.orchestrator import Orchestrator

    return Orchestrator(config).run()


__all__ = [
    "DiscoveryConfig",
    "DiscoveryResult",
    "IterationTelemetry",
    "discover",
]
