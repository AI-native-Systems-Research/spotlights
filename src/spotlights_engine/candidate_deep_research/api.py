"""Public entrypoint for the candidate deep-research step (candidate mode).

Sibling of `module_deep_research.api`: instead of one module-wide survey this
runs one survey **per candidate**, tagging every finding and search-query log
with the candidate it belongs to. The pure helpers (`select_runners`,
`merge_outcomes`, `parse_agent_output` via the merge, the runner clients) are
imported from `module_deep_research`; only the timed fan-out is copied, because
`RunnerOutcome` carries no duration and must not gain one.

Issues are reported under the existing `module_deep_research` `PipelineStep`
and usage records under the existing `module_deep_research` `UsageStep`: both
literal unions are closed and candidate mode deliberately reuses the module
mode's sidecars and checkpoint states.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.candidate_deep_research.prompts import (
    render_candidate_deep_research_prompt,
)
from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.module_deep_research.agent_exec import ModuleResearchRunner
from spotlights_engine.module_deep_research.api import (
    _cli_for_agent,
    resolve_target_module,
)
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.module_deep_research.orchestration import (
    RunnerOutcome,
    merge_outcomes,
    module_deep_research_issue,
    select_runners,
)
from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import (
    CandidateDeepResearchInput,
    ModuleDeepResearchOutput,
)
from spotlights_engine.schemas.project import Module
from spotlights_engine.utils.id_helpers import parse_id, slug_for

_log = logging.getLogger(__name__)


class CandidateCliUsage(BaseModel):
    """One CLI invocation's usage, attributed to the candidate it surveyed.

    Richer than `CliUsage` (which carries only `cli` + `usage`) because
    candidate mode needs the candidate id for the usage record's
    `invocation_id` and the per-run wallclock as the duration fallback — see
    the step-3 usage-write branch in the manager.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    cli: str
    usage: AgentUsage
    duration_s: float | None = None


class CandidateDeepResearchResult(BaseModel):
    """Runtime-rich result: contract output plus candidate-tagged usages."""

    model_config = ConfigDict(extra="forbid")

    output: ModuleDeepResearchOutput
    usages: list[CandidateCliUsage] = Field(default_factory=list)


def _run_one_runner_timed(
    *,
    runner: ModuleResearchRunner,
    prompt: str,
    check: bool,
    log_prefix: str = "",
) -> tuple[RunnerOutcome, float]:
    """Run one research runner and return its outcome **with** its wallclock.

    Copy of `module_deep_research.orchestration._run_one_runner` narrowed to the
    logging this step needs, plus the elapsed time in the return value.
    `RunnerOutcome` is a frozen dataclass with no duration field and the module
    path must stay untouched, so the duration is carried alongside instead.
    """
    agent_name = getattr(runner, "name", type(runner).__name__)
    started = time.monotonic()
    _log.info("%scandidate_deep_research: %s start", log_prefix, agent_name)
    try:
        result = runner.run(prompt, check=check)
    except Exception as exc:
        elapsed = time.monotonic() - started
        _log.warning(
            "%scandidate_deep_research: %s failed in %.1fs: %s",
            log_prefix,
            agent_name,
            elapsed,
            exc,
        )
        return (
            RunnerOutcome(
                agent_name=agent_name,
                error=f"module_deep_research {agent_name} execution failed: {exc}",
            ),
            elapsed,
        )

    elapsed = time.monotonic() - started
    if not result.ok:
        _log.warning(
            "%scandidate_deep_research: %s exited with code %d in %.1fs — stderr=%s",
            log_prefix,
            agent_name,
            result.returncode,
            elapsed,
            result.stderr.strip() or "(no stderr)",
        )
    else:
        _log.info(
            "%scandidate_deep_research: %s complete in %.1fs",
            log_prefix,
            agent_name,
            elapsed,
        )
    return RunnerOutcome(agent_name=agent_name, result=result), elapsed


def _run_runners_timed(
    *,
    prompt: str,
    runners: Sequence[ModuleResearchRunner],
    check: bool,
    log_prefix: str = "",
) -> list[tuple[RunnerOutcome, float]]:
    """Timed variant of `run_runners`: same `ThreadPoolExecutor` fan-out shape,
    but each outcome is paired with the wallclock of its own invocation."""
    if not runners:
        return [
            (
                RunnerOutcome(
                    agent_name="none",
                    error="no module_deep_research runners configured",
                ),
                0.0,
            )
        ]

    agent_names = [getattr(r, "name", type(r).__name__) for r in runners]
    _log.info(
        "%scandidate_deep_research: dispatching %d runner(s): %s",
        log_prefix,
        len(runners),
        ", ".join(agent_names),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(runners)) as executor:
        futures = [
            executor.submit(
                _run_one_runner_timed,
                runner=runner,
                prompt=prompt,
                check=check,
                log_prefix=log_prefix,
            )
            for runner in runners
        ]
        return [future.result() for future in futures]


def _candidate_segment(candidate: Candidate, *, segment: str) -> str:
    """Build the per-candidate finding-id segment `<segment>-<counter:04d>`.

    The candidate counter is parsed out of the candidate id (never split on
    `-`; `parse_id` is right-anchored so slugs containing `-` survive). The
    resulting finding ids `find-<segment>-<counter>-NNNN` still match
    `Finding.finding_id` and still round-trip through `parse_id`.
    """
    _, candidate_segment, counter = parse_id(candidate.id)
    if candidate_segment != segment:
        raise ValueError(
            f"candidate id {candidate.id!r} carries segment "
            f"{candidate_segment!r}, expected {segment!r}"
        )
    return f"{segment}-{counter:04d}"


def _candidate_options(
    codex_options: CodexExecOptions | None,
    *,
    repo_path: Path,
    candidate: Candidate,
    last_message_dir: Path | None,
) -> CodexExecOptions | None:
    """Derive candidate-local codex options with a private last-message file.

    A single shared `output_last_message` would let one candidate's survey parse
    another's JSON (the runner reads the file back as `final_message`), so each
    candidate gets `<last_message_dir>/<candidate_id>.md`. `Candidate.id` has no
    path separators by construction.

    With no `last_message_dir` there is nothing to isolate *unless* the caller's
    own options already pin a single `output_last_message` — that one file would
    otherwise be shared by every concurrently-running candidate, so a
    per-candidate sibling is derived next to it. Options carrying no file at all
    (or no options) pass through unchanged: `CodexExecClient.build_command`
    mints a private temp file per invocation, and `select_runners` builds its
    own default when options are `None`.
    """
    if last_message_dir is None:
        if codex_options is None or codex_options.output_last_message is None:
            return codex_options
        shared = Path(codex_options.output_last_message)
        sibling = shared.with_name(f"{shared.stem}.{candidate.id}{shared.suffix}")
        return codex_options.model_copy(update={"output_last_message": sibling})
    path = last_message_dir / f"{candidate.id}.md"
    if codex_options is None:
        return CodexExecOptions(
            cwd=repo_path,
            output_last_message=path,
            json_events=True,
        )
    return codex_options.model_copy(update={"output_last_message": path})


def _dedup_candidates(
    candidates: Sequence[Candidate],
) -> tuple[list[Candidate], list[StepIssue]]:
    """Drop repeated candidate ids, reporting each drop as a recoverable issue.

    The per-candidate finding segment is `<segment>-<counter>` derived from the
    candidate id, so two entries sharing an id would mint *identical* finding
    ids and (in step 4) attach duplicate proposals to the same code site. First
    occurrence wins so the surviving order is the input order.
    """
    seen: set[str] = set()
    unique: list[Candidate] = []
    issues: list[StepIssue] = []
    for candidate in candidates:
        if candidate.id in seen:
            issues.append(
                module_deep_research_issue(
                    f"duplicate candidate id {candidate.id} in candidate_deep_research "
                    "input; surveying it once",
                    recoverable=True,
                )
            )
            continue
        seen.add(candidate.id)
        unique.append(candidate)
    return unique, issues


def research_candidates_with_telemetry(
    request: CandidateDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
    segment: str | None = None,
    last_message_dir: Path | None = None,
    max_parallel_candidates: int = 2,
) -> CandidateDeepResearchResult:
    """Runtime-rich variant used by the manager for usage/cost accounting.

    Runs one survey per `request.candidates` entry, at most
    `max_parallel_candidates` at a time, and concatenates the per-candidate
    findings, issues, and search-query logs into a single
    `ModuleDeepResearchOutput`. Every emitted finding and query log carries
    `candidate_id`.

    Repeated candidate ids are surveyed once: the per-candidate finding segment
    is derived from the id, so a duplicate would mint duplicate finding ids and
    (in step 4) duplicate proposals for one code site. Discovery already rejects
    duplicates within an iteration, so this is a defensive de-dup.

    Synchronous, like `research_module_with_telemetry`: the manager wraps it in
    `asyncio.to_thread`, so the candidate fan-out uses a `ThreadPoolExecutor`
    rather than an `asyncio.Semaphore`.
    """
    seg = segment if segment is not None else slug_for(request.module_qualified_name)
    module = resolve_target_module(request.project_tree, request.module_qualified_name)
    if module is None:
        return CandidateDeepResearchResult(
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

    if not request.candidates:
        return CandidateDeepResearchResult(output=ModuleDeepResearchOutput())

    unique_candidates, duplicate_issues = _dedup_candidates(request.candidates)

    workers = max(1, min(max_parallel_candidates, len(unique_candidates)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _research_one_candidate,
                request=request,
                module=module,
                candidate=candidate,
                segment=seg,
                codex_options=codex_options,
                check=check,
                runner=runner,
                runners=runners,
                last_message_dir=last_message_dir,
            )
            for candidate in unique_candidates
        ]
        per_candidate = [future.result() for future in futures]

    findings = []
    issues: list[StepIssue] = list(duplicate_issues)
    search_queries = []
    usages: list[CandidateCliUsage] = []
    for output, candidate_usages in per_candidate:
        findings.extend(output.findings)
        issues.extend(output.issues)
        search_queries.extend(output.search_queries)
        usages.extend(candidate_usages)

    return CandidateDeepResearchResult(
        output=ModuleDeepResearchOutput(
            findings=findings,
            issues=issues,
            search_queries=search_queries,
        ),
        usages=usages,
    )


def _research_one_candidate(
    *,
    request: CandidateDeepResearchInput,
    module: Module,
    candidate: Candidate,
    segment: str,
    codex_options: CodexExecOptions | None,
    check: bool,
    runner: ModuleResearchRunner | None,
    runners: Sequence[ModuleResearchRunner] | None,
    last_message_dir: Path | None,
) -> tuple[ModuleDeepResearchOutput, list[CandidateCliUsage]]:
    """Survey one candidate; a failure degrades only this candidate.

    Per-runner failures are already turned into recoverable issues by
    `merge_outcomes`; this wrapper additionally covers prompt/options/id
    derivation *and* merge/tagging errors so one bad candidate leaves the module
    `DEGRADED` rather than `FAILED` (D7). Everything that can raise must stay
    inside the `try`: this runs in a worker thread and an escaping exception
    resurfaces from `future.result()` and fails the whole module.
    """
    try:
        candidate_segment = _candidate_segment(candidate, segment=segment)
        options = _candidate_options(
            codex_options,
            repo_path=request.repo_path,
            candidate=candidate,
            last_message_dir=last_message_dir,
        )
        prompt = render_candidate_deep_research_prompt(request, module, candidate)
        active_runners = select_runners(
            repo_path=request.repo_path,
            codex_options=options,
            runner=runner,
            runners=runners,
            enable_claude_search=request.enable_claude_search,
        )
        timed = _run_runners_timed(
            prompt=prompt,
            runners=active_runners,
            check=check,
            log_prefix=f"[{request.module_qualified_name} {candidate.id}] ",
        )

        outcomes = [outcome for outcome, _ in timed]
        usages: list[CandidateCliUsage] = []
        for outcome, duration in timed:
            if outcome.result is None or outcome.result.usage is None:
                continue
            cli = _cli_for_agent(outcome.agent_name)
            if cli is None:
                continue
            usages.append(
                CandidateCliUsage(
                    candidate_id=candidate.id,
                    cli=cli,
                    usage=outcome.result.usage,
                    duration_s=duration,
                )
            )

        merged = merge_outcomes(
            outcomes,
            max_findings_per_module=request.max_findings_per_candidate,
            segment=candidate_segment,
        )
        # `merge_outcomes`/`_renumber_findings` build bare `Finding`s and
        # un-tagged `SearchQueryLog`s, so the candidate tag is stamped on
        # post-merge.
        tagged = merged.model_copy(
            update={
                "findings": [
                    f.model_copy(update={"candidate_id": candidate.id}) for f in merged.findings
                ],
                "search_queries": [
                    q.model_copy(update={"candidate_id": candidate.id})
                    for q in merged.search_queries
                ],
            }
        )
    except Exception as exc:
        _log.warning(
            "candidate_deep_research: candidate %s failed: %s",
            candidate.id,
            exc,
            exc_info=True,
        )
        return (
            ModuleDeepResearchOutput(
                issues=[
                    module_deep_research_issue(
                        f"candidate_deep_research failed for {candidate.id}: {exc}",
                        recoverable=True,
                    )
                ]
            ),
            [],
        )

    return tagged, usages


def research_candidates(
    request: CandidateDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
    segment: str | None = None,
    last_message_dir: Path | None = None,
    max_parallel_candidates: int = 2,
) -> ModuleDeepResearchOutput:
    """Run per-candidate deep research and return the output contract.

    `segment` is the module id segment; per-candidate finding ids become
    `find-<segment>-<candidate_counter>-NNNN`. Standalone callers may omit it,
    in which case the module slug is used.
    """
    return research_candidates_with_telemetry(
        request,
        codex_options=codex_options,
        check=check,
        runner=runner,
        runners=runners,
        segment=segment,
        last_message_dir=last_message_dir,
        max_parallel_candidates=max_parallel_candidates,
    ).output


__all__ = [
    "CandidateCliUsage",
    "CandidateDeepResearchResult",
    "ModuleResearchRunner",
    "research_candidates",
    "research_candidates_with_telemetry",
]
