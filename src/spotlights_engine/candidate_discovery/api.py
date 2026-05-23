"""Public types and entrypoints for step 2 of the deep-research pipeline.

The architecture (`docs/architecture/spotlights_deep_research_path_architecture.md`)
defines step 2 as `discover_candidates(CandidateDiscoveryInput) -> Candidates`.
The `input` carries the architectural triplet (`project_tree`,
`module_qualified_name`, `context`); the `config` carries infra-only knobs
(`repo_path`, `artifacts_dir`, agent budgets, model identifiers) that have no
place on the public per-step contract.

`discover_candidates(input, config=...)` returns the contract `Candidates`.
`discover(input, config=...)` is the runtime-rich variant that additionally
returns telemetry and per-iteration cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.candidate_discovery.errors import DiscoverySetupError
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.pipeline import CandidateDiscoveryInput
from spotlights_engine.schemas.project import Module


class DiscoveryConfig(BaseModel):
    """Runtime/infra config for candidate discovery.

    Holds only fields outside the architectural `CandidateDiscoveryInput`
    contract: filesystem layout (`repo_path`, `artifacts_dir`), agent budgets,
    and model identifiers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # `repo_path` and `artifacts_dir` are required at step entry but accept
    # `None` at construction time so callers (e.g. `SpotlightsManager`) can
    # build a config once and fill the per-module paths via `model_copy`.
    # `discover()` raises `DiscoverySetupError` if either is still `None`.
    repo_path: Path | None = None
    artifacts_dir: Path | None = None
    repo_context_markdown: str | None = Field(
        default=None, min_length=1, max_length=20_000
    )
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


def resolve_target_module(input: CandidateDiscoveryInput) -> Module:
    """Look up the target `Module` inside `input.project_tree`.

    Accepts both architecture dot-form (`a.b.c`) and `ProjectTree.walk`
    slash-form (`a/b/c`).
    """
    qn = input.module_qualified_name
    resolved = input.project_tree.resolve(qn)
    if resolved is None and "." in qn:
        resolved = input.project_tree.resolve(qn.replace(".", "/"))
    if resolved is None:
        raise ValueError(
            f"module_qualified_name {qn!r} not found in project_tree"
        )
    return resolved


def discover(
    input: CandidateDiscoveryInput, *, config: DiscoveryConfig
) -> DiscoveryResult:
    """Runtime-rich variant of step 2: returns telemetry alongside `Candidates`.

    Pre-construction setup checks live here so a missing CLI or a bad
    `artifacts_dir` fails before the run dir is minted.
    """
    repo = config.repo_path
    artifacts = config.artifacts_dir

    if repo is None:
        raise DiscoverySetupError("DiscoveryConfig.repo_path is required at step entry")
    if artifacts is None:
        raise DiscoverySetupError("DiscoveryConfig.artifacts_dir is required at step entry")

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

    module = resolve_target_module(input)

    from spotlights_engine.candidate_discovery.orchestrator import Orchestrator

    return Orchestrator(input=input, config=config, module=module).run()


def discover_candidates(
    input: CandidateDiscoveryInput, *, config: DiscoveryConfig
) -> Candidates:
    """Architecture-shaped entrypoint for step 2.

    Mirrors the per-step contract: input is `CandidateDiscoveryInput`, output
    is `Candidates`. Telemetry and per-iteration cost flow through `discover()`
    for callers that want them.
    """
    return discover(input, config=config).candidates


__all__ = [
    "DiscoveryConfig",
    "DiscoveryResult",
    "IterationTelemetry",
    "discover",
    "discover_candidates",
    "resolve_target_module",
]
