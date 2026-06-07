"""Top-level orchestration for the modules extractor.

Implements step 1 of the deep-research pipeline (see
`docs/architecture/spotlights_deep_research_path_architecture.md`):
take a `ModulesExtractorInput`, drive a Claude Code subprocess over the repo
with the `extraction.md` prompt, and return a `ProjectTree` validated against
the shared schema.

Two entrypoints, mirroring `candidate_discovery`:

- `extract(input)` is the architecture-shaped contract — synchronous,
  takes `ModulesExtractorInput`, returns a `ProjectTree`.
- `extract_with_telemetry(input, *, config)` is the runtime-rich form: same
  logic, plus an optional `artifacts_dir` for run logs and tunable agent
  budgets, returning telemetry alongside the tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.agent import (
    ExtractionInvocation,
    ExtractionRunResult,
    run_extraction,
)
from spotlights_engine.modules_extractor.errors import ExtractorSetupError
from spotlights_engine.modules_extractor.prompts import EXTRACTION_PROMPT
from spotlights_engine.schemas.pipeline import ModulesExtractorInput
from spotlights_engine.schemas.project import ProjectTree


class ExtractorConfig(BaseModel):
    """Runtime/infra config for the modules extractor.

    Holds only fields outside the architectural `ModulesExtractorInput`
    contract: optional `artifacts_dir` for persisting prompt/schema/raw
    output, plus tunable agent budgets.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifacts_dir: Path | None = None
    claude_bin: str = "claude"
    max_turns: int = Field(default=60, ge=1)
    timeout_s: int = Field(default=1800, ge=1)


class ExtractorResult(BaseModel):
    """Architectural output (`project_tree`) plus run telemetry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_tree: ProjectTree
    invocation: ExtractionInvocation


def extract_with_telemetry(
    input: ModulesExtractorInput,
    *,
    config: ExtractorConfig | None = None,
    on_event: Callable[[str], None] | None = None,
) -> ExtractorResult:
    """Runtime-rich extraction. Persists artifacts when `artifacts_dir` is set."""
    cfg = config or ExtractorConfig()
    repo = input.repo_path
    if not repo.exists() or not repo.is_dir():
        raise ExtractorSetupError(
            f"repo_path does not exist or is not a directory: {repo}",
            repo_path=str(repo),
        )

    artifacts = cfg.artifacts_dir
    if artifacts is not None:
        resolved_repo = repo.resolve(strict=False)
        resolved_artifacts = artifacts.resolve(strict=False)
        try:
            resolved_artifacts.relative_to(resolved_repo)
        except ValueError:
            pass
        else:
            raise ExtractorSetupError(
                f"artifacts_dir resolves inside repo_path: {artifacts}",
                artifacts_dir=str(artifacts),
                repo_path=str(repo),
            )
        run_dir = artifacts / "modules_extractor"
        if run_dir.exists():
            raise ExtractorSetupError(
                f"pre-existing modules_extractor run dir; resume is not supported: {run_dir}",
                run_dir=str(run_dir),
            )
        run_dir.mkdir(parents=True)
    else:
        run_dir = None

    run_kwargs: dict = dict(
        repo_path=repo,
        prompt=EXTRACTION_PROMPT,
        claude_bin=cfg.claude_bin,
        max_turns=cfg.max_turns,
        timeout_s=cfg.timeout_s,
        artifacts_dir=run_dir,
    )
    if on_event is not None:
        run_kwargs["on_event"] = on_event
    run: ExtractionRunResult = run_extraction(**run_kwargs)

    return ExtractorResult(project_tree=run.project_tree, invocation=run.invocation)


def extract(
    input: ModulesExtractorInput,
    *,
    config: ExtractorConfig | None = None,
    on_event: Callable[[str], None] | None = None,
) -> ProjectTree:
    """Architecture-shaped entrypoint for step 1 (`ModulesExtractor`).

    Mirrors the per-step contract: input is `ModulesExtractorInput`, output is
    the structural `ProjectTree`. Telemetry flows through `extract_with_telemetry`
    for callers that want it.

    `on_event`, when given, is forwarded to the underlying `claude -p` reader
    thread (one summary line per stream-json event). Default is silent.
    """
    return extract_with_telemetry(
        input, config=config, on_event=on_event
    ).project_tree


__all__ = [
    "ExtractorConfig",
    "ExtractorResult",
    "extract",
    "extract_with_telemetry",
]
