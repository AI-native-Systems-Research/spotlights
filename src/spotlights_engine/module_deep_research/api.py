"""Public entrypoint for the module deep-research step.

Step 3 surveys the literature **per candidate** (decision D1/D2): the public
entrypoint loops over `request.candidates`, builds a candidate-focused prompt for
each, fans out the runners, and merges/dedups within that candidate. Findings are
renumbered with a per-candidate segment (`find-<module_segment>-<candidate_counter>-NNNN`,
decision D3) and stamped with `candidate_id` before being concatenated into the
single flat `ModuleDeepResearchOutput.findings` list the manager persists.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
from spotlights_engine.module_deep_research.prompts import (
    render_candidate_deep_research_prompt,
)
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import (
    ModuleDeepResearchInput,
    ModuleDeepResearchOutput,
)
from spotlights_engine.schemas.project import Module, ProjectTree
from spotlights_engine.schemas.search import SearchQueryLog
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
    """Runtime-rich result: contract output plus per-candidate usage captures.

    Because step 3 now runs one survey per candidate (decision D1), usages and
    durations are attributed per candidate (decision D7). `usages_by_candidate`
    maps each candidate id to the `CliUsage`s its runners reported;
    `per_candidate_durations_s` maps each candidate id to a `{cli: wall_seconds}`
    dict used as the per-invocation fallback duration so N candidates do not each
    claim the whole step-3 wall time.
    """

    model_config = ConfigDict(extra="forbid")

    output: ModuleDeepResearchOutput
    usages_by_candidate: dict[str, list[CliUsage]] = Field(default_factory=dict)
    per_candidate_durations_s: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )


def _cli_for_agent(agent_name: str) -> str | None:
    name = agent_name.lower()
    if "claude" in name:
        return "claude"
    if "codex" in name:
        return "codex"
    return None


def _per_candidate_codex_options(
    codex_options: CodexExecOptions | None,
    *,
    last_message_dir: Path | None,
    candidate_id: str,
) -> CodexExecOptions | None:
    """Derive a per-candidate `CodexExecOptions` with its own last-message file (D8).

    The manager hands step 3 a last-message *directory* via
    `codex_options.output_last_message`; each candidate survey gets a distinct
    `<dir>/<candidate_id>.md` so two candidates can never parse each other's
    codex response. When no directory is configured (standalone use) the caller's
    options pass through unchanged.
    """
    if codex_options is None:
        return None
    if last_message_dir is None:
        return codex_options
    return codex_options.model_copy(
        update={"output_last_message": last_message_dir / f"{candidate_id}.md"}
    )


def research_module_with_telemetry(
    request: ModuleDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
    segment: str | None = None,
) -> ModuleDeepResearchResult:
    """Runtime-rich variant used by the manager for usage/cost accounting.

    Loops over `request.candidates` (decision D1): each candidate gets its own
    survey prompt, runner fan-out, merge/dedup, and per-candidate finding segment
    `f"{segment}-{candidate_counter:04d}"` (decision D3). Findings, issues, and
    search-query logs are concatenated across candidates into one
    `ModuleDeepResearchOutput`; usages and durations are attributed per candidate
    (decision D7).
    """
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

    # The manager passes the per-candidate last-message *directory* on the shared
    # codex options (D8); when only runners/runner are supplied (tests, standalone)
    # there is nothing to derive.
    last_message_dir: Path | None = None
    if codex_options is not None and codex_options.output_last_message is not None:
        last_message_dir = Path(codex_options.output_last_message)

    all_findings = []
    all_issues: list[StepIssue] = []
    all_search: list[SearchQueryLog] = []
    usages_by_candidate: dict[str, list[CliUsage]] = {}
    per_candidate_durations_s: dict[str, dict[str, float]] = {}

    for counter, candidate in enumerate(request.candidates, start=1):
        candidate_segment = f"{seg}-{counter:04d}"
        prompt = render_candidate_deep_research_prompt(request, module, candidate)
        cand_codex_options = _per_candidate_codex_options(
            codex_options,
            last_message_dir=last_message_dir,
            candidate_id=candidate.id,
        )
        active_runners = select_runners(
            repo_path=request.repo_path,
            codex_options=cand_codex_options,
            runner=runner,
            runners=runners,
            enable_claude_search=request.enable_claude_search,
        )
        outcomes = run_runners(
            prompt=prompt,
            runners=active_runners,
            check=check,
            module_qualified_name=f"{request.module_qualified_name} :: {candidate.id}",
        )

        cand_usages: list[CliUsage] = []
        cand_durations: dict[str, float] = {}
        for outcome in outcomes:
            cli = _cli_for_agent(outcome.agent_name)
            if cli is None:
                continue
            if outcome.duration_s is not None:
                cand_durations[cli] = outcome.duration_s
            if outcome.result is None or outcome.result.usage is None:
                continue
            cand_usages.append(CliUsage(cli=cli, usage=outcome.result.usage))
        if cand_usages:
            usages_by_candidate[candidate.id] = cand_usages
        if cand_durations:
            per_candidate_durations_s[candidate.id] = cand_durations

        merged = merge_outcomes(
            outcomes,
            max_findings_per_candidate=request.max_findings_per_candidate,
            segment=candidate_segment,
            candidate_id=candidate.id,
        )
        all_findings.extend(merged.findings)
        all_issues.extend(merged.issues)
        all_search.extend(merged.search_queries)

    return ModuleDeepResearchResult(
        output=ModuleDeepResearchOutput(
            findings=all_findings,
            issues=all_issues,
            search_queries=all_search,
        ),
        usages_by_candidate=usages_by_candidate,
        per_candidate_durations_s=per_candidate_durations_s,
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

    `segment` is the module id segment (decision D3); each candidate's findings
    are prefixed to `find-<segment>-<candidate_counter>-NNNN`. The manager
    supplies the resolved value; standalone callers may omit it, in which case the
    module slug is used.
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
