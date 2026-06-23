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
from spotlights_engine.utils.id_helpers import slug_for


def research_module(
    request: ModuleDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    antigravity_options: AntigravityExecOptions | None = None,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
    segment: str | None = None,
) -> ModuleDeepResearchOutput:
    """Run module deep research and return the architecture output contract.

    `segment` is the module id segment (decision D3) used to prefix finding ids
    to `find-<segment>-NNNN`; the manager supplies the resolved value. Standalone
    callers may omit it, in which case the module slug is used.
    """
    seg = segment if segment is not None else slug_for(request.module_qualified_name)
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
        target_module=module,
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
        segment=seg,
    )


__all__ = ["ModuleResearchRunner", "research_module", "resolve_target_module"]
