"""Prompt assembly for the module deep-research step."""

from __future__ import annotations

import json

from spotlights_engine.module_deep_research.validation import (
    AgentModuleDeepResearchOutput,
)
from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.pipeline import ModuleDeepResearchInput
from spotlights_engine.schemas.project import File, Module, Repository


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


def _format_candidate_location(candidate: Candidate) -> str:
    """Render `symbol (file)` for the candidate's first span, the primary hot
    spot. Candidates always carry at least one location with one span."""
    location = candidate.locations[0]
    span = location.spans[0]
    return f"`{span.symbol}` ({location.file})"


def _format_candidates(candidates: list[Candidate]) -> str:
    if not candidates:
        return "(none)"
    blocks: list[str] = []
    for candidate in candidates:
        blocks.append(
            f"- Symbol: {_format_candidate_location(candidate)}\n"
            f"  Estimated impact: {candidate.estimated_impact}\n"
            f"  Description: {candidate.description}\n"
            f"  Evolve rationale: {candidate.evolve_rationale}"
        )
    return "\n\n".join(blocks)


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
    # Show the agent the bare-id wire schema (`find-NNNN`); the manager prefixes
    # finding ids with the module slug after parsing (decision D3, option A).
    schema_json = json.dumps(
        AgentModuleDeepResearchOutput.model_json_schema(), indent=2
    )
    repo_path = str(request.repo_path)
    hotspots_section = ""
    if request.include_candidate_hotspots and request.candidates:
        hotspots_section = (
            "\nIdentified hot spots (from candidate_discovery):\n"
            "These are the symbols the discovery step flagged as worth\n"
            "evolving in this module. Use them to steer your search toward\n"
            "sources that address them; they are hints, not a strict scope.\n"
            f"{_format_candidates(request.candidates)}\n"
        )
    return f"""You are running the Spotlights module_deep_research pipeline step.
Do not modify files. Do not ask questions.

Goal:
Find external sources (papers, blogs, docs, talks, issues, PRs) that carry a
concrete method, algorithm, technique, or design idea that could be used to
improve the target module. Every finding must contribute an actionable idea
the module could adopt or adapt - not just be topically related. This step
surfaces improvement-bearing related work for the module; it does not produce
per-file change recipes.

Repository:
{_format_repository(request.project_tree.repository)}

Repository working directory (the codex sandbox is rooted here; read files
directly with relative paths from this root, e.g. the target module path
below): {repo_path}

Target module:
{_format_module(module, request.module_qualified_name)}
{hotspots_section}
Caller context:
Objective: {request.context.objective}
Workload hints:
{_format_list(request.context.workload_hints)}
Validation plan:
{_format_list(request.context.validation_plan)}

Workflow:
1. First understand the target module before searching. Open the module's
   main files (and any nearby files needed to make sense of them) to learn
   what the module does, its responsibilities, key abstractions, data flow,
   and the techniques it already uses. The codex sandbox cwd is the
   repository working directory shown above, so open files via their
   repo-relative paths (e.g. the target module Path) using your
   file-reading tools.
2. Form a short mental model of the module's scope and the gaps,
   bottlenecks, or open questions relative to the caller objective and
   workload hints. These gaps define what an "improvement" looks like for
   this module. Use this model to steer the search; do not emit it in the
   output.
3. Run a focused literature/web search for sources that propose a concrete
   method, algorithm, technique, or design idea the module could adopt to
   close one of those gaps. You do not need to map findings to specific
   files, symbols, or contracts inside the module - but you must be able to
   name the transferable idea.

Output rules:
- Return a single bare JSON object matching the ModuleDeepResearchOutput schema
  below. Do not wrap it in Markdown and do not include explanatory prose.
- Include at most {request.max_findings_per_module} findings.
- Use finding IDs find-0001, find-0002, ... ordered by expected relevance to
  the module and the caller objective. Most relevant first.
- Empty findings are valid when no relevant source survives filtering.
- Add StepIssue entries only for warnings or errors encountered during the
  survey. Set StepIssue.step to "module_deep_research".
- Use source_type values only from: paper, blog, docs, issue, pr, talk, codebase, other.
- title must be the exact title of the cited source (paper, blog post, doc page,
  issue, PR, talk, etc.).
- technique_summary is a short description (1-3 sentences) that names the
  concrete method, algorithm, or design idea the source contributes and
  states how it could improve the target module relative to the caller
  objective (e.g., what gap or bottleneck it addresses). Do not name
  specific local files, symbols, or contracts.
- supporting_evidence must include a short verbatim quote from the source
  (at most two sentences) plus a pointer into the source (section, figure,
  algorithm number, timestamp, or commit/line). Paraphrase only when the
  source is not quotable (for example, a video without a transcript), and
  in that case still give a precise pointer.

Module relevance gate:
- Keep a finding only when it passes all of these checks:
  1. It is about an owned responsibility of the target module, not just the
     repository or a broad technology area.
  2. It is aligned with the caller objective and workload hints.
  3. It is not merely background or prior art that the module already
     implements, unless the source meaningfully extends or contrasts with
     the current approach.
  4. It carries a concrete, transferable method, algorithm, technique, or
     design idea that the module could plausibly adopt or adapt to improve
     itself. Sources that only describe a problem, narrate experience, or
     restate what the module already does do not pass. Operational test:
     if you cannot state the transferable idea in a single sentence, the
     finding is too diffuse - drop it.
- Topical adjacency is not relevance. A source that shares vocabulary with
  the module's domain, sits in a neighboring technology area, or addresses
  the surrounding ecosystem is not automatically on-topic. Keep it only if
  it speaks to the module's owned responsibilities as identified in step 1
  of the workflow.

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
