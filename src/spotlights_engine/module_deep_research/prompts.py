"""Prompt assembly for the module deep-research step."""

from __future__ import annotations

import json

from spotlights_engine.schemas.deep_research import (
    ModuleDeepResearchInput,
    ModuleDeepResearchOutput,
)
from spotlights_engine.schemas.modules import File, Module, Repository


def _format_main_files(files: list[File]) -> str:
    if not files:
        return "(none)"
    return "\n".join(f"- {f.path}: {f.role}" for f in files)


def _format_list(values: list[str]) -> str:
    if not values:
        return "(none)"
    return "\n".join(f"- {value}" for value in values)


def _format_repository(repository: Repository) -> str:
    dependencies = ", ".join(repository.external_dependencies) or "(none)"
    return (
        f"Name: {repository.name}\n"
        f"Summary: {repository.summary}\n"
        f"External dependencies: {dependencies}"
    )


def _format_module(module: Module, module_qualified_name: str) -> str:
    depends_on = ", ".join(module.depends_on) or "(none)"
    return (
        f"Qualified name: {module_qualified_name}\n"
        f"Name: {module.name}\n"
        f"Path: {module.path}\n"
        f"Description: {module.description or '(none)'}\n"
        f"Depends on: {depends_on}\n"
        f"Main files:\n{_format_main_files(module.main_files)}"
    )


def render_module_deep_research_prompt(
    request: ModuleDeepResearchInput,
    module: Module,
) -> str:
    """Render the survey prompt from repository, target module, and context fields."""
    schema_json = json.dumps(ModuleDeepResearchOutput.model_json_schema(), indent=2)
    return f"""You are running the Spotlights module_deep_research pipeline step.
Do not modify files. Do not ask questions.

Goal:
Run a focused literature/web survey for the target module and return only findings
that are relevant to the caller objective and workload hints.

Repository:
{_format_repository(request.project_tree.repository)}

Target module:
{_format_module(module, request.module_qualified_name)}

Caller context:
Objective: {request.context.objective}
Workload hints:
{_format_list(request.context.workload_hints)}
Validation plan:
{_format_list(request.context.validation_plan)}

Output rules:
- Return a single bare JSON object matching the ModuleDeepResearchOutput schema
  below. Do not wrap it in Markdown and do not include explanatory prose.
- Include at most {request.max_findings_per_module} findings.
- Use finding IDs find-0001, find-0002, ... ordered by expected impact on the
  caller objective multiplied by the likelihood the change lands cleanly inside
  the target module. Highest expected-impact-and-applicable first.
- Empty findings are valid when no relevant source survives filtering.
- Add StepIssue entries only for warnings or errors encountered during the survey.
- Use source_type values only from: paper, blog, docs, issue, pr, talk, codebase, other.
- title must be the exact title of the cited source (paper, blog post, doc page,
  issue, PR, talk, etc.). Put the proposed local change in technique_summary.
- technique_summary is the handoff to later change-producing steps. For each
  finding, write it as a compact local-change sketch using these labels:
  "Where: <local file/symbol/contract>; Change: <concrete change>; Why:
  <source-backed rationale tied to objective/workload>; Validate: <test,
  benchmark, or check, preferring the caller validation plan when relevant>."
- supporting_evidence must include a short verbatim quote from the source
  (at most two sentences) plus a pointer into the source (section, figure,
  algorithm number, timestamp, or commit/line). Paraphrase only when the
  source is not quotable (for example, a video without a transcript), and
  in that case still give a precise pointer.

Local relevance gate:
- First inspect the target module files and nearby files under the target module path.
- Build a quick map of current behavior and gaps before surveying external sources.
- Keep a finding only when it passes all of these checks:
  1. It is about an owned responsibility of the target module, not just the
     repository or broad technology area.
  2. It suggests a concrete implementation, policy, data-layout, scheduling,
     API, or validation idea that could plausibly change one of the target
     module files or main-file contracts.
  3. It is aligned with the caller objective and workload hints.
  4. It is not merely background, prior art, or a technique already present in
     the local code unless the source supports a specific local gap or variant.
- A later implementation agent should be able to open one named local file and
  start a patch from technique_summary alone.
- Each finding's technique_summary must name the local
  file, symbol, policy, handler, or contract it applies to.
- Prefer narrow, actionable findings over general surveys. Foundational sources
  are valid only when they directly justify a specific local change.
- Prefer findings implementable mostly inside the target module. If a change
  needs an adjacent module, name the boundary or contract and keep the target
  module's edit explicit.
- Filter out findings that are merely adjacent to KV caches, inference serving,
  GPU systems, or caching but do not pass the local relevance gate.

Source quality:
- Prefer primary or implementation-rich sources: papers with algorithms,
  official docs, accepted design docs, issues/PRs with concrete patches, or
  talks/blogs by maintainers.
- Prefer sources whose system constraints resemble the workload hints.
- Reject surveys, position papers, vision/roadmap pieces, and high-level
  overviews as the source for a finding unless they cite a specific algorithm,
  patch, or implementation that you are reusing as the actual technique. The
  cited concrete artifact then becomes the source; do not list the survey itself.
- Do not invent titles, URLs, benchmark results, or local code behavior.
- If web/source retrieval is unavailable or too incomplete, return empty
  findings with a recoverable StepIssue instead of unsourced suggestions.

ModuleDeepResearchOutput JSON schema:
{schema_json}
""".strip()


__all__ = ["render_module_deep_research_prompt"]
