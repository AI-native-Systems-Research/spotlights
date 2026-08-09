"""Internal runner fan-out and result merging for module deep research."""

from __future__ import annotations

import concurrent.futures
import logging
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult, ModuleResearchRunner
from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient, ClaudeExecOptions
from spotlights_engine.module_deep_research.codex_exec import CodexExecClient, CodexExecOptions
from spotlights_engine.module_deep_research.validation import (
    AgentFinding,
    AgentModuleDeepResearchOutput,
    normalize_module_deep_research_output,
    parse_agent_output,
)
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput
from spotlights_engine.schemas.project import Module, ProjectTree
from spotlights_engine.schemas.search import SearchQueryLog, SearchResult

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
    enable_claude_search: bool = False,
) -> tuple[ModuleResearchRunner, ...]:
    """Resolve caller-provided runners or create the default runner set.

    The default set is Codex only; when `enable_claude_search` is true, Claude
    is added so step 3 fans out to Codex + Claude.
    """
    if runner is not None and runners is not None:
        raise ValueError("pass either runner or runners, not both")
    if runner is not None:
        return (runner,)
    if runners is not None:
        return tuple(runners)

    default_runners: list[ModuleResearchRunner] = [
        CodexExecClient(codex_options or CodexExecOptions(cwd=repo_path)),
    ]
    if enable_claude_search:
        default_runners.append(ClaudeExecClient(ClaudeExecOptions(cwd=repo_path)))
    return tuple(default_runners)


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


def resolve_consensus_threshold(k: int, threshold: int | None) -> int:
    """ceil(k/2) when unset; clamped to [1, k] when set.

    An explicit threshold above K is treated as "unanimous" (`= K`) rather than
    silently discarding every finding."""
    if threshold is None:
        return math.ceil(k / 2)
    return max(1, min(threshold, k))


@dataclass(frozen=True)
class RunResult:
    """One search run's within-run merged output.

    `findings` are the enabled runners of a *single* run unioned and deduped by
    `_finding_keys` (wire shape, bare ids, no cap, no renumber). `issues` and
    `search_logs` are that run's issues and per-runner tagged query logs."""

    findings: list[AgentFinding] = field(default_factory=list)
    issues: list[StepIssue] = field(default_factory=list)
    search_logs: list[SearchQueryLog] = field(default_factory=list)


def merge_run(outcomes: Sequence[RunnerOutcome]) -> RunResult:
    """Union the enabled runners of ONE run, dedup by `_finding_keys`, keep order.

    No cap. No renumber. This is the per-runner collection loop of the old
    `merge_outcomes`, minus the normalize/cap tail: it returns the deduped wire
    findings for one run plus that run's issues and query logs.

    Search queries are collected and tagged with the issuing runner
    (`agent=outcome.agent_name`). Unlike findings they are **not** deduped —
    identical query strings from two agents are meaningful signal — and their
    order is preserved (outcomes in fixed `futures` order, queries in the
    agent's emitted order) so run-to-run markdown diffs are stable."""
    findings: list[AgentFinding] = []
    issues: list[StepIssue] = []
    search_logs: list[SearchQueryLog] = []
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

        for query in output.search_queries:
            search_logs.append(
                SearchQueryLog(
                    agent=outcome.agent_name,
                    query=query.query,
                    tool=query.tool,
                    results=[
                        SearchResult(
                            title=r.title, url=r.url, snippet=r.snippet
                        )
                        for r in query.results
                    ],
                )
            )

    return RunResult(findings=findings, issues=issues, search_logs=search_logs)


def consensus_merge(
    runs: Sequence[RunResult],
    *,
    k: int,
    threshold: int | None,
    segment: str,
) -> ModuleDeepResearchOutput:
    """Keep findings that recur across `>= threshold` of the K merged runs.

    The vote is computed over the K merged run-outputs, so a source cited by
    both codex and claude *in the same run* counts as one vote for that run (it
    was deduped inside the run first). A source must recur across different runs
    to accumulate votes.

    The threshold defaults to `ceil(k_effective/2)` where `k_effective` is the
    number of runs that produced >= 1 finding, so a single infra failure that
    empties one run does not silently raise the bar. An explicit threshold is
    honored literally (clamped to `[1, k_effective]`).

    Survivors are ordered by descending vote count, then first-occurrence run
    index and position; `normalize_module_deep_research_output` mints the
    `find-<segment>-NNNN` ids in that order, with no cap."""
    all_issues: list[StepIssue] = []
    all_search_logs: list[SearchQueryLog] = []
    for run in runs:
        all_issues.extend(run.issues)
        all_search_logs.extend(run.search_logs)

    k_effective = sum(1 for run in runs if run.findings)
    if k_effective < k:
        all_issues.append(
            module_deep_research_issue(
                f"consensus: {k - k_effective} of {k} search run(s) produced no "
                f"findings; threshold resolved against {k_effective} effective run(s)",
                recoverable=True,
            )
        )
    thr = resolve_consensus_threshold(max(k_effective, 1), threshold)

    votes: dict[str, int] = {}
    first_seen: dict[str, tuple[int, int]] = {}  # key -> (run_idx, pos_in_run)
    repr_finding: dict[str, AgentFinding] = {}  # key -> representative wire finding
    for run_idx, run in enumerate(runs):
        # one vote per key PER RUN: collapse this run's keys to a set first
        run_keys_seen: set[str] = set()
        for pos, finding in enumerate(run.findings):
            keys = _finding_keys(finding.title, finding.url)
            canonical = _canonical_vote_key(keys)
            if canonical in run_keys_seen:
                continue  # already voted this run
            run_keys_seen.add(canonical)
            votes[canonical] = votes.get(canonical, 0) + 1
            if canonical not in first_seen:
                first_seen[canonical] = (run_idx, pos)
                repr_finding[canonical] = finding

    survivors = [key for key, v in votes.items() if v >= thr]
    survivors.sort(key=lambda key: (-votes[key], first_seen[key]))
    kept = [repr_finding[key] for key in survivors]

    merged = AgentModuleDeepResearchOutput(findings=kept, issues=all_issues)
    return normalize_module_deep_research_output(
        merged,
        segment=segment,
        search_queries=all_search_logs,
    )


def merge_outcomes(
    outcomes: Sequence[RunnerOutcome],
    *,
    segment: str,
) -> ModuleDeepResearchOutput:
    """Back-compat shim: merge a single run's outcomes (K=1, threshold=1).

    Equivalent to today's union-with-first-occurrence-dedup, minus the per-unit
    cap. New callers should use the `merge_run` + `consensus_merge` pair to run
    K-run consensus."""
    return consensus_merge(
        [merge_run(outcomes)], k=1, threshold=1, segment=segment
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


def _canonical_vote_key(keys: set[str]) -> str:
    """Prefer the URL key (stronger identity) over the title key.

    URL identity is stronger and already arxiv-normalized; the title is the
    fallback when a finding has no usable URL (`_normalize_url` returns "" for
    empty/`unknown`)."""
    url_keys = sorted(k for k in keys if k.startswith("url:"))
    if url_keys:
        return url_keys[0]
    return sorted(keys)[0]  # title-only fallback


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
