"""Public entrypoint for the module deep-research step."""

from __future__ import annotations

from collections.abc import Sequence

from spotlights_engine.module_deep_research.agent_exec import ModuleResearchRunner
from spotlights_engine.module_deep_research.antigravity_exec import AntigravityExecOptions
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.module_deep_research.orchestration import (
    merge_outcomes,
    module_deep_research_issue,
    resolve_target_module,
    run_runners,
    select_runners,
)
from spotlights_engine.module_deep_research.prompts import render_module_deep_research_prompt
from spotlights_engine.schemas.pipeline import ModuleDeepResearchInput, ModuleDeepResearchOutput


def research_module(
    request: ModuleDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    antigravity_options: AntigravityExecOptions | None = None,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
) -> ModuleDeepResearchOutput:
    """Run module deep research and return the architecture output contract."""
    module = resolve_target_module(request.project_tree, request.module_qualified_name)
    if module is None:
        return ModuleDeepResearchOutput(
            findings=[],
            issues=[
                module_deep_research_issue(
                    f"module not found in project tree: {request.module_qualified_name}",
                    recoverable=False,
                )
            ],
        )

    prompt = render_module_deep_research_prompt(request, module)
    active_runners = select_runners(
        repo_path=request.repo_path,
        codex_options=codex_options,
        antigravity_options=antigravity_options,
        runner=runner,
        runners=runners,
    )
    outcomes = run_runners(
        prompt=prompt,
        runners=active_runners,
        check=check,
        module_qualified_name=request.module_qualified_name,
    )
    return merge_outcomes(
        outcomes,
        max_findings_per_module=request.max_findings_per_module,
    )


__all__ = ["ModuleResearchRunner", "research_module", "resolve_target_module"]
