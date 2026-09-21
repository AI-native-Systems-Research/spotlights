"""Public entrypoint for the module deep-research step."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import CliUsage
from spotlights_engine.module_deep_research.agent_exec import ModuleResearchRunner
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.module_deep_research.orchestration import (
    RunnerOutcome,
    merge_outcomes,
    module_deep_research_issue,
    run_runners,
    select_runners,
)
from spotlights_engine.module_deep_research.prompts import render_module_deep_research_prompt
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.finding import Finding
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
    if "openalex" in name:
        return "openalex"
    return None


def _usages_from(outcomes: Sequence[RunnerOutcome]) -> list[CliUsage]:
    """Collect per-runner CLI usage from a batch of outcomes."""
    usages: list[CliUsage] = []
    for outcome in outcomes:
        if outcome.result is None or outcome.result.usage is None:
            continue
        cli = _cli_for_agent(outcome.agent_name)
        if cli is None:
            continue
        usages.append(CliUsage(cli=cli, usage=outcome.result.usage))
    return usages


def _combine_per_candidate(
    outputs: Sequence[ModuleDeepResearchOutput], *, segment: str
) -> ModuleDeepResearchOutput:
    """Concatenate per-candidate research outputs into one module output.

    Each input was merged/capped/renumbered on its own, so finding ids restart at
    `find-<segment>-0001` per candidate. Re-sequence the concatenated findings
    once here to keep ids globally unique while preserving each finding's
    `candidate_id` (and every other field). Search-query logs and issues are
    concatenated in candidate order for stable, per-candidate-attributable output."""
    findings: list[Finding] = []
    search_queries = []
    issues: list[StepIssue] = []
    for out in outputs:
        findings.extend(out.findings)
        search_queries.extend(out.search_queries)
        issues.extend(out.issues)

    renumbered = [
        f.model_copy(update={"finding_id": f"find-{segment}-{idx:04d}"})
        for idx, f in enumerate(findings, start=1)
    ]
    return ModuleDeepResearchOutput(
        findings=renumbered,
        issues=issues,
        search_queries=search_queries,
    )


def research_module_with_telemetry(
    request: ModuleDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
    segment: str | None = None,
    claude_model: str | None = None,
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

    active_runners = select_runners(
        repo_path=request.repo_path,
        codex_options=codex_options,
        runner=runner,
        runners=runners,
        enable_claude_search=request.enable_claude_search,
        claude_model=claude_model,
        enable_openalex=request.enable_openalex,
        openalex_model=request.openalex_model,
        openalex_query_mode=request.openalex_query_mode,
    )
    # Per-candidate mode: one scoped research pass per hot spot (each pass sees a
    # single candidate). Each pass's outcomes are merged/deduped/capped on their
    # own and tagged with that candidate's id, then the per-candidate outputs are
    # combined (findings kept scoped, ids re-sequenced globally). Step 4 pairs a
    # candidate only with its own findings. Any other case (flag off, or no
    # candidates) runs a single module-wide pass with untagged findings.
    usages: list[CliUsage] = []
    if request.per_candidate_deep_research and request.candidates:
        per_candidate_outputs: list[ModuleDeepResearchOutput] = []
        for candidate in request.candidates:
            prompt = render_module_deep_research_prompt(
                request.model_copy(
                    update={
                        "candidates": [candidate],
                        "include_candidate_hotspots": True,
                    }
                ),
                module,
            )
            outcomes = run_runners(
                prompt=prompt,
                runners=active_runners,
                check=check,
                module_qualified_name=request.module_qualified_name,
            )
            usages.extend(_usages_from(outcomes))
            per_candidate_outputs.append(
                merge_outcomes(
                    outcomes,
                    max_findings_per_module=request.max_findings_per_module,
                    segment=seg,
                    candidate_id=candidate.id,
                )
            )
        output = _combine_per_candidate(per_candidate_outputs, segment=seg)
    else:
        outcomes = run_runners(
            prompt=render_module_deep_research_prompt(request, module),
            runners=active_runners,
            check=check,
            module_qualified_name=request.module_qualified_name,
        )
        usages.extend(_usages_from(outcomes))
        output = merge_outcomes(
            outcomes,
            max_findings_per_module=request.max_findings_per_module,
            segment=seg,
        )

    return ModuleDeepResearchResult(output=output, usages=usages)


def research_module(
    request: ModuleDeepResearchInput,
    codex_options: CodexExecOptions | None = None,
    *,
    check: bool = False,
    runner: ModuleResearchRunner | None = None,
    runners: Sequence[ModuleResearchRunner] | None = None,
    segment: str | None = None,
    claude_model: str | None = None,
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
        claude_model=claude_model,
    ).output


__all__ = [
    "ModuleDeepResearchResult",
    "ModuleResearchRunner",
    "research_module",
    "research_module_with_telemetry",
    "resolve_target_module",
]
