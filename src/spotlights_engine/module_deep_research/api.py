"""Public entrypoint for the module deep-research step."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import CliUsage
from spotlights_engine.module_deep_research.agent_exec import ModuleResearchRunner
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.module_deep_research.orchestration import (
    merge_outcomes,
    module_deep_research_issue,
    run_runners,
    select_runners,
)
from spotlights_engine.module_deep_research.prompts import render_module_deep_research_prompt
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import (
    ModuleDeepResearchInput,
    ModuleDeepResearchOutput,
)
from spotlights_engine.schemas.project import Module, ProjectTree
from spotlights_engine.utils.id_helpers import slug_for


def _issue(message: str, *, recoverable: bool) -> StepIssue:
    return StepIssue(
        step="module_deep_research",
        severity="error",
        message=message,
        recoverable=recoverable,
    )


def resolve_target_module(project_tree: ProjectTree, module_qualified_name: str) -> Module | None:
    """Resolve a module by its slash-form qualified name (the canonical key)."""
    return project_tree.resolve(module_qualified_name)


class ModuleDeepResearchResult(BaseModel):
    """Runtime-rich result: contract output plus per-runner usage captures."""

    model_config = ConfigDict(extra="forbid")

    output: ModuleDeepResearchOutput
    usages: list[CliUsage] = Field(default_factory=list)


def _cli_for_agent(agent_name: str) -> str | None:
    name = agent_name.lower()
    if "claude" in name:
        return "claude"
    if "codex" in name:
        return "codex"
    return None


def research_module_with_telemetry(
    request: ModuleDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
    segment: str | None = None,
) -> ModuleDeepResearchResult:
    """Runtime-rich variant used by the manager for usage/cost accounting."""
    seg = segment if segment is not None else slug_for(request.module_qualified_name)
    module = resolve_target_module(request.project_tree, request.module_qualified_name)
    if module is None:
        return ModuleDeepResearchResult(
            output=ModuleDeepResearchOutput(
                findings=[],
                issues=[
                    module_deep_research_issue(
                        f"module not found in project tree: {request.module_qualified_name}",
                        recoverable=False,
                    )
                ],
            )
        )

    prompt = render_module_deep_research_prompt(request, module)
    active_runners = select_runners(
        repo_path=request.repo_path,
        codex_options=codex_options,
        runner=runner,
        runners=runners,
    )
    outcomes = run_runners(
        prompt=prompt,
        runners=active_runners,
        check=check,
        module_qualified_name=request.module_qualified_name,
    )
    usages: list[CliUsage] = []
    for outcome in outcomes:
        if outcome.result is None or outcome.result.usage is None:
            continue
        cli = _cli_for_agent(outcome.agent_name)
        if cli is None:
            continue
        usages.append(CliUsage(cli=cli, usage=outcome.result.usage))
    return ModuleDeepResearchResult(
        output=merge_outcomes(
            outcomes,
            max_findings_per_module=request.max_findings_per_module,
            segment=seg,
        ),
        usages=usages,
    )


def research_module(
    request: ModuleDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
    segment: str | None = None,
) -> ModuleDeepResearchOutput:
    """Run module deep research and return the architecture output contract.

    `segment` is the module id segment (decision D3) used to prefix finding ids
    to `find-<segment>-NNNN`; the manager supplies the resolved value. Standalone
    callers may omit it, in which case the module slug is used.
    """
    return research_module_with_telemetry(
        request,
        codex_options=codex_options,
        check=check,
        runner=runner,
        runners=runners,
        segment=segment,
    ).output


__all__ = [
    "ModuleDeepResearchResult",
    "ModuleResearchRunner",
    "research_module",
    "research_module_with_telemetry",
    "resolve_target_module",
]
