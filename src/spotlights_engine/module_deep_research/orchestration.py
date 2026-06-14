"""Internal runner fan-out and result merging for module deep research."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult, ModuleResearchRunner
from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient, ClaudeExecOptions
from spotlights_engine.module_deep_research.codex_exec import CodexExecClient, CodexExecOptions
from spotlights_engine.module_deep_research.gemini_exec import GeminiExecClient, GeminiExecOptions
from spotlights_engine.module_deep_research.validation import (
    normalize_module_deep_research_output,
    parse_module_deep_research_output,
)
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput
from spotlights_engine.schemas.project import Module, ProjectTree


@dataclass(frozen=True)
class RunnerOutcome:
    """Normalized success/failure state for one research runner."""

    agent_name: str
    result: AgentExecResult | None = None
    error: str | None = None


def module_deep_research_issue(message: str, *, recoverable: bool) -> StepIssue:
    """Build a step-scoped issue for this pipeline stage."""
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


def select_runners(
    *,
    repo_path: Path,
    codex_options: CodexExecOptions | None,
    runner: ModuleResearchRunner | None,
    runners: Sequence[ModuleResearchRunner] | None,
) -> tuple[ModuleResearchRunner, ...]:
    """Resolve caller-provided runners or create the default Codex/Claude/Gemini set."""
    if runner is not None and runners is not None:
        raise ValueError("pass either runner or runners, not both")
    if runner is not None:
        return (runner,)
    if runners is not None:
        return tuple(runners)

    return (
        CodexExecClient(codex_options or CodexExecOptions(cwd=repo_path)),
        ClaudeExecClient(ClaudeExecOptions(cwd=repo_path)),
        GeminiExecClient(GeminiExecOptions(cwd=repo_path)),
    )


def run_runners(
    *, prompt: str, runners: Sequence[ModuleResearchRunner], check: bool
) -> list[RunnerOutcome]:
    """Run research agents concurrently and capture recoverable runner failures."""
    if not runners:
        return [
            RunnerOutcome(
                agent_name="none",
                error="no module_deep_research runners configured",
            )
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(runners)) as executor:
        futures = [
            executor.submit(_run_one_runner, runner=runner, prompt=prompt, check=check)
            for runner in runners
        ]
        return [future.result() for future in futures]


def merge_outcomes(
    outcomes: Sequence[RunnerOutcome], *, max_findings_per_module: int
) -> ModuleDeepResearchOutput:
    """Merge agent outputs into the stable module deep-research contract."""
    findings = []
    issues: list[StepIssue] = []
    seen: set[str] = set()

    for outcome in outcomes:
        if outcome.error is not None:
            issues.append(module_deep_research_issue(outcome.error, recoverable=True))
            continue
        if outcome.result is None:
            continue

        result = outcome.result
        response_text = result.final_message or result.stdout
        output = parse_module_deep_research_output(
            response_text,
            max_findings_per_module=max_findings_per_module,
        )
        issues.extend(_agent_issues(outcome.agent_name, output.issues))
        if not result.ok:
            issues.append(
                module_deep_research_issue(
                    f"module_deep_research {outcome.agent_name} runner exited with code "
                    f"{result.returncode}: {result.stderr.strip() or '(no stderr)'}",
                    recoverable=True,
                )
            )

        for finding in output.findings:
            keys = _finding_keys(finding.title, finding.url)
            if seen.intersection(keys):
                continue
            seen.update(keys)
            findings.append(finding)

    merged = ModuleDeepResearchOutput(findings=findings, issues=issues)
    return normalize_module_deep_research_output(
        merged,
        max_findings_per_module=max_findings_per_module,
    )


def _run_one_runner(*, runner: ModuleResearchRunner, prompt: str, check: bool) -> RunnerOutcome:
    agent_name = getattr(runner, "name", type(runner).__name__)
    try:
        return RunnerOutcome(
            agent_name=agent_name,
            result=runner.run(prompt, check=check),
        )
    except Exception as exc:
        return RunnerOutcome(
            agent_name=agent_name,
            error=f"module_deep_research {agent_name} execution failed: {exc}",
        )


def _agent_issues(agent_name: str, issues: Sequence[StepIssue]) -> list[StepIssue]:
    return [
        issue.model_copy(update={"message": f"{agent_name}: {issue.message}"})
        for issue in issues
    ]


def _finding_keys(title: str, url: str) -> set[str]:
    keys = {f"title:{_normalize_text(title)}"}
    normalized_url = _normalize_url(url)
    if normalized_url:
        keys.add(f"url:{normalized_url}")
    return keys


def _normalize_url(url: str) -> str:
    value = url.strip().lower().rstrip("/")
    if not value or value == "unknown":
        return ""
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/"):
        if value.startswith(prefix):
            return "arxiv:" + value.removeprefix(prefix).removesuffix(".pdf")
    for prefix in ("https://arxiv.org/pdf/", "http://arxiv.org/pdf/"):
        if value.startswith(prefix):
            return "arxiv:" + value.removeprefix(prefix).removesuffix(".pdf")
    return value


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())
