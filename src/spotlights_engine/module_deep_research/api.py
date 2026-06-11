"""Public entrypoint for the module deep-research step."""

from __future__ import annotations

from typing import Protocol

from spotlights_engine.module_deep_research.codex_exec import (
    CodexExecClient,
    CodexExecOptions,
    CodexExecResult,
)
from spotlights_engine.module_deep_research.expanded import (
    ExpandedModuleResearchOrchestrator,
    ExpandedResearchConfig,
)
from spotlights_engine.module_deep_research.prompts import render_module_deep_research_prompt
from spotlights_engine.module_deep_research.validation import parse_module_deep_research_output
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import (
    ModuleDeepResearchInput,
    ModuleDeepResearchOutput,
)
from spotlights_engine.schemas.project import Module, ProjectTree


class ModuleResearchRunner(Protocol):
    """Minimal runner protocol used by `research_module`."""

    def run(self, prompt: str, *, check: bool = True) -> CodexExecResult: ...


def _issue(message: str, *, recoverable: bool) -> StepIssue:
    return StepIssue(
        step="module_deep_research",
        severity="error",
        message=message,
        recoverable=recoverable,
    )


def resolve_target_module(project_tree: ProjectTree, module_qualified_name: str) -> Module | None:
    """Resolve slash-qualified names and dot-qualified names used by the architecture."""
    direct = project_tree.resolve(module_qualified_name)
    if direct is not None:
        return direct

    if "." in module_qualified_name and "/" not in module_qualified_name:
        return project_tree.resolve(module_qualified_name.replace(".", "/"))

    return None


def research_module(
    request: ModuleDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    runner: ModuleResearchRunner | None = None,
    expanded_config: ExpandedResearchConfig | None = None,
) -> ModuleDeepResearchOutput:
    """Run module deep research and return the architecture output contract."""
    module = resolve_target_module(request.project_tree, request.module_qualified_name)
    if module is None:
        return ModuleDeepResearchOutput(
            findings=[],
            issues=[
                _issue(
                    f"module not found in project tree: {request.module_qualified_name}",
                    recoverable=False,
                )
            ],
        )

    if expanded_config is not None and expanded_config.enabled:
        baseline = (
            _run_vanilla_module_research(
                request=request,
                module=module,
                codex_options=codex_options,
                check=check,
                runner=runner,
            )
            if expanded_config.run_vanilla_baseline
            else ModuleDeepResearchOutput()
        )
        return ExpandedModuleResearchOrchestrator(
            request=request,
            module=module,
            baseline_output=baseline,
            config=expanded_config,
        ).run()

    return _run_vanilla_module_research(
        request=request,
        module=module,
        codex_options=codex_options,
        check=check,
        runner=runner,
    )


def _run_vanilla_module_research(
    *,
    request: ModuleDeepResearchInput,
    module: Module,
    codex_options: CodexExecOptions | None,
    check: bool,
    runner: ModuleResearchRunner | None,
) -> ModuleDeepResearchOutput:
    """Run the vanilla Codex-only implementation used as the control group."""
    prompt = render_module_deep_research_prompt(request, module)
    active_runner = runner or CodexExecClient(
        codex_options or CodexExecOptions(cwd=request.repo_path)
    )

    try:
        result = active_runner.run(prompt, check=check)
    except Exception as exc:
        return ModuleDeepResearchOutput(
            findings=[],
            issues=[_issue(f"module_deep_research execution failed: {exc}", recoverable=True)],
        )

    response_text = result.final_message or result.stdout
    output = parse_module_deep_research_output(
        response_text,
        max_findings_per_module=request.max_findings_per_module,
    )
    if not result.ok:
        output.issues.append(
            _issue(
                "module_deep_research runner exited with code "
                f"{result.returncode}: {result.stderr.strip() or '(no stderr)'}",
                recoverable=True,
            )
        )
    return output


__all__ = ["ModuleResearchRunner", "research_module", "resolve_target_module"]
