"""Public types and entrypoints for step 2 of the deep-research pipeline.

`discover_candidates(input)` is the architecture-shaped entrypoint: it takes
a `CandidateDiscoveryInput` (project tree + module qualified name + spotlight
context) and returns a `Candidates` whose entries are at state `DISCOVERED`
with empty match / proposal lists. Step 4–6 modules later populate those.

`discover(config)` is the runtime-rich entrypoint that the orchestrator
itself drives. It takes the architectural inputs *plus* infra knobs
(`artifacts_dir`, agent timeouts, model identifiers, etc.) that have no place
on the public per-step contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spotlights_engine.candidate_discovery.errors import DiscoverySetupError
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import CandidateDiscoveryInput
from spotlights_engine.schemas.project import Module, ProjectTree


class DiscoveryConfig(BaseModel):
    """Runtime config for the orchestrator.

    Extends the architectural `CandidateDiscoveryInput` (`project_tree`,
    `module_qualified_name`, `context`) with infra-only knobs that don't
    belong on the public step contract: filesystem layout (`repo_path`,
    `artifacts_dir`), agent budgets, and model identifiers.

    `module: Module` is accepted directly for backward compatibility with
    existing call sites; when omitted the module is resolved from
    `project_tree` (if provided).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_path: Path
    module_qualified_name: str
    artifacts_dir: Path
    project_tree: ProjectTree | None = None
    module: Module | None = None
    context: SpotlightContext | None = None
    repo_context_markdown: str | None = Field(
        default=None, min_length=1, max_length=20_000
    )
    num_review_iterations: int = Field(default=3, ge=0)
    per_iteration_wallclock_s: int = Field(default=900, ge=1)
    claude_max_turns: int = Field(default=30, ge=1)
    codex_model: str = Field(default="gpt-5.5", pattern=r"^[\w.\-/]+$")
    codex_reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] = "high"

    @model_validator(mode="after")
    def _resolve_module(self) -> DiscoveryConfig:
        if self.module is None:
            if self.project_tree is None:
                raise ValueError(
                    "DiscoveryConfig requires either `module` or `project_tree`"
                )
            resolved = self.project_tree.resolve(self.module_qualified_name)
            if resolved is None and "." in self.module_qualified_name:
                # Architecture spec (dot-joined) ↔ ProjectTree.walk (slash-joined).
                resolved = self.project_tree.resolve(
                    self.module_qualified_name.replace(".", "/")
                )
            if resolved is None:
                raise ValueError(
                    "module_qualified_name "
                    f"{self.module_qualified_name!r} not found in project_tree"
                )
            self.module = resolved
        return self


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
    """Thin wrapper around `Orchestrator.run()` — the runtime-rich entrypoint.

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


def discover_candidates(
    input: CandidateDiscoveryInput,
    *,
    repo_path: Path,
    artifacts_dir: Path,
    repo_context_markdown: str | None = None,
    num_review_iterations: int = 3,
    per_iteration_wallclock_s: int = 900,
    claude_max_turns: int = 30,
    codex_model: str = "gpt-5.5",
    codex_reasoning_effort: Literal[
        "minimal", "low", "medium", "high", "xhigh"
    ] = "high",
) -> Candidates:
    """Architecture-shaped entrypoint for step 2.

    Returns the `Candidates` object the per-step contract specifies. Telemetry
    and per-iteration cost go through `discover()` for callers that want them.
    """
    config = DiscoveryConfig(
        repo_path=repo_path,
        artifacts_dir=artifacts_dir,
        project_tree=input.project_tree,
        module_qualified_name=input.module_qualified_name,
        context=input.context,
        repo_context_markdown=repo_context_markdown,
        num_review_iterations=num_review_iterations,
        per_iteration_wallclock_s=per_iteration_wallclock_s,
        claude_max_turns=claude_max_turns,
        codex_model=codex_model,
        codex_reasoning_effort=codex_reasoning_effort,
    )
    return discover(config).candidates


__all__ = [
    "DiscoveryConfig",
    "DiscoveryResult",
    "IterationTelemetry",
    "discover",
    "discover_candidates",
]
