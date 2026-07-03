"""Public entrypoint for the module deep-research step."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from spotlights_engine.module_deep_research.agent_exec import ModuleResearchRunner
from spotlights_engine.module_deep_research.arxiv_exec import (
    ArxivSearchClient,
    ArxivSearchOptions,
)
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.module_deep_research.debug import write_prompt
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


def research_module(
    request: ModuleDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
    arxiv_options: ArxivSearchOptions | None = None,
    segment: str | None = None,
    debug_dir: Path | None = None,
) -> ModuleDeepResearchOutput:
    """Run module deep research and return the architecture output contract.

    `segment` is the module id segment (decision D3) used to prefix finding ids
    to `find-<segment>-NNNN`; the manager supplies the resolved value. Standalone
    callers may omit it, in which case the module slug is used.

    When `debug_dir` is set, the rendered prompt, each runner's raw output, and
    the merged findings (before any paper filter) are written there as
    diagnostics. Nothing downstream reads them.
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
    if debug_dir is not None:
        write_prompt(debug_dir, prompt)
    # The arXiv runner needs structured inputs (module model + candidates) to
    # build queries, so it is constructed here — where `request`/`module` are in
    # scope — and appended to the default set. It is ignored when the caller
    # passes an explicit `runner`/`runners`.
    extra_runners: tuple[ModuleResearchRunner, ...] = ()
    if arxiv_options is not None:
        extra_runners = (
            ArxivSearchClient(request, module, arxiv_options, debug_dir=debug_dir),
        )
    active_runners = select_runners(
        repo_path=request.repo_path,
        codex_options=codex_options,
        runner=runner,
        runners=runners,
        extra_runners=extra_runners,
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
        paper_filter=request.paper_filter,
        debug_dir=debug_dir,
    )


__all__ = ["ModuleResearchRunner", "research_module", "resolve_target_module"]
