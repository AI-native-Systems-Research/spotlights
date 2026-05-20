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

Output rules:
- Return a single JSON object matching the ModuleDeepResearchOutput schema below.
- Include at most {request.max_findings_per_module} findings.
- Use finding IDs find-0001, find-0002, ... in relevance order.
- Empty findings are valid when no relevant source survives filtering.
- Add StepIssue entries only for warnings or errors encountered during the survey.
- Use source_type values only from: paper, blog, docs, issue, pr, talk, codebase, other.
- Keep supporting_evidence brief: a short excerpt, paraphrase, or source note.
- Filter out findings that are clearly off-objective.

ModuleDeepResearchOutput JSON schema:
{schema_json}
""".strip()


__all__ = ["render_module_deep_research_prompt"]
