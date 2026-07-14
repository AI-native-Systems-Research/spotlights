"""Internal runner fan-out and result merging for module deep research."""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult, ModuleResearchRunner
from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient, ClaudeExecOptions
from spotlights_engine.module_deep_research.codex_exec import CodexExecClient, CodexExecOptions
from spotlights_engine.module_deep_research.opencode_exec import (
    OpenCodeExecClient,
    OpenCodeExecOptions,
)
from spotlights_engine.module_deep_research.validation import (
    AgentModuleDeepResearchOutput,
    normalize_module_deep_research_output,
    parse_agent_output,
)
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput
from spotlights_engine.schemas.project import Module, ProjectTree

_log = logging.getLogger(__name__)


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
    """Resolve caller-provided runners or create the default Codex/Claude/OpenCode set."""
    if runner is not None and runners is not None:
        raise ValueError("pass either runner or runners, not both")
    if runner is not None:
        return (runner,)
    if runners is not None:
        return tuple(runners)

    return (
        CodexExecClient(codex_options or CodexExecOptions(cwd=repo_path)),
        ClaudeExecClient(ClaudeExecOptions(cwd=repo_path)),
        OpenCodeExecClient(OpenCodeExecOptions(cwd=repo_path)),
    )


def run_runners(
    *,
    prompt: str,
    runners: Sequence[ModuleResearchRunner],
    check: bool,
    module_qualified_name: str | None = None,
) -> list[RunnerOutcome]:
    """Run research agents concurrently and capture recoverable runner failures."""
    if not runners:
        return [
            RunnerOutcome(
                agent_name="none",
                error="no module_deep_research runners configured",
            )
        ]

    agent_names = [getattr(r, "name", type(r).__name__) for r in runners]
    log_prefix = f"[{module_qualified_name}] " if module_qualified_name else ""
    _log.info(
        "%sdeep_research: dispatching %d runner(s): %s",
        log_prefix,
        len(runners),
        ", ".join(agent_names),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(runners)) as executor:
        futures = [
            executor.submit(
                _run_one_runner,
                runner=runner,
                prompt=prompt,
                check=check,
                log_prefix=log_prefix,
            )
            for runner in runners
        ]
        return [future.result() for future in futures]


def merge_outcomes(
    outcomes: Sequence[RunnerOutcome],
    *,
    max_findings_per_module: int,
    segment: str,
) -> ModuleDeepResearchOutput:
    """Merge agent outputs into the stable module deep-research contract.

    Per-runner outputs are parsed into the lenient wire shape (bare ids), then
    deduped and merged; the single promotion to persisted `Finding`s — capping,
    renumbering, and prefixing each id to `find-<segment>-NNNN` (D3) — happens
    once here via `normalize_module_deep_research_output`."""
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
        output = parse_agent_output(response_text)
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

    merged = AgentModuleDeepResearchOutput(findings=findings, issues=issues)
    merged_findings_cap = max_findings_per_module * len(outcomes)
    return normalize_module_deep_research_output(
        merged,
        max_findings_per_module=merged_findings_cap,
        segment=segment,
    )


def _run_one_runner(
    *,
    runner: ModuleResearchRunner,
    prompt: str,
    check: bool,
    log_prefix: str = "",
) -> RunnerOutcome:
    agent_name = getattr(runner, "name", type(runner).__name__)
    started = time.monotonic()
    _log.info("%sdeep_research: %s start", log_prefix, agent_name)
    try:
        result = runner.run(prompt, check=check)
    except Exception as exc:
        elapsed = time.monotonic() - started
        _log.warning(
            "%sdeep_research: %s failed in %.1fs: %s",
            log_prefix,
            agent_name,
            elapsed,
            exc,
        )
        return RunnerOutcome(
            agent_name=agent_name,
            error=f"module_deep_research {agent_name} execution failed: {exc}",
        )

    elapsed = time.monotonic() - started
    response_text = result.final_message or result.stdout
    parsed = parse_agent_output(response_text)
    finding_count = len(parsed.findings)
    parse_issue_count = len(parsed.issues)
    if not result.ok:
        _log.warning(
            "%sdeep_research: %s exited with code %d in %.1fs — "
            "%d findings, %d parse issues, stderr=%s",
            log_prefix,
            agent_name,
            result.returncode,
            elapsed,
            finding_count,
            parse_issue_count,
            result.stderr.strip() or "(no stderr)",
        )
    else:
        _log.info(
            "%sdeep_research: %s complete in %.1fs — %d findings%s",
            log_prefix,
            agent_name,
            elapsed,
            finding_count,
            f", {parse_issue_count} parse issue(s)" if parse_issue_count else "",
        )
    return RunnerOutcome(agent_name=agent_name, result=result)


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
            return "arxiv:" + _normalize_arxiv_id(value.removeprefix(prefix))
    for prefix in ("https://arxiv.org/pdf/", "http://arxiv.org/pdf/"):
        if value.startswith(prefix):
            return "arxiv:" + _normalize_arxiv_id(value.removeprefix(prefix))
    return value


def _normalize_arxiv_id(value: str) -> str:
    return re.sub(r"v\d+$", "", value.removesuffix(".pdf").split("?", 1)[0].split("#", 1)[0])


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())
