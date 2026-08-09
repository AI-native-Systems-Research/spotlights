"""Prompt assembly for the candidate deep-research step (candidate mode).

Sibling of `module_deep_research.prompts`: the survey is scoped to one
candidate's code site rather than to the whole module, so the prompt centers on
the candidate block (file/lines/symbol, `current_approach`,
`evolve_rationale`, impact) with module/repository context supplied only for
grounding. The agent-facing wire schema is unchanged — the same
`AgentModuleDeepResearchOutput` with bare `find-NNNN` ids.
"""

from __future__ import annotations

import json

from spotlights_engine.module_deep_research.validation import (
    AgentModuleDeepResearchOutput,
)
from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.pipeline import CandidateDeepResearchInput
from spotlights_engine.schemas.project import File, Module, Repository
from spotlights_engine.utils.schema_compat import primary_file, primary_span


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


def _format_candidate(candidate: Candidate) -> str:
    span = primary_span(candidate)
    return (
        f"- id: {candidate.id}\n"
        f"- file: {primary_file(candidate)}\n"
        f"- lines: {span.line_start}-{span.line_end}\n"
        f"- symbol: {span.symbol}\n"
        f"- kind: {span.kind}\n"
        f"- description: {candidate.description}\n"
        f"- current_approach: {candidate.current_approach}\n"
        f"- evolve_rationale: {candidate.evolve_rationale}\n"
        f"- estimated_impact: {candidate.estimated_impact}\n"
        f"- estimated_impact_explanation: {candidate.estimated_impact_explanation}"
    )


def render_candidate_deep_research_prompt(
    request: CandidateDeepResearchInput,
    module: Module,
    candidate: Candidate,
) -> str:
    """Render the per-candidate survey prompt.

    Unlike the module-mode prompt the target of the survey is a single code
    site: findings must carry an idea applicable to *this* candidate, not to the
    module at large.
    """
    # Show the agent the bare-id wire schema (`find-NNNN`); the caller prefixes
    # finding ids with `<module_segment>-<candidate_counter>` after parsing.
    schema_json = json.dumps(AgentModuleDeepResearchOutput.model_json_schema(), indent=2)
    repo_path = str(request.repo_path)
    return f"""You are running the Spotlights candidate_deep_research pipeline step.
Do not modify files. Do not ask questions.

Goal:
Find external sources (papers, blogs, docs, talks, issues, PRs) that carry a
concrete method, algorithm, technique, or design idea that could be used to
improve ONE specific candidate code site. Every finding must contribute an
actionable idea that this candidate could adopt or adapt - not just be
topically related to its module or its domain.

Repository:
{_format_repository(request.project_tree.repository)}

Repository working directory (the codex sandbox is rooted here; read files
directly with relative paths from this root, e.g. the candidate file below):
{repo_path}

Containing module (context for grounding only - the survey target is the
candidate below, not the module):
{_format_module(module, request.module_qualified_name)}

Target candidate:
{_format_candidate(candidate)}

Caller context:
Objective: {request.context.objective}
Workload hints:
{_format_list(request.context.workload_hints)}
Validation plan:
{_format_list(request.context.validation_plan)}

Workflow:
1. First understand the candidate before searching. Open the candidate file at
   the line range shown above (and any nearby files needed to make sense of
   it) to learn what this code does, what technique it currently uses, and
   which constraints it operates under. The codex sandbox cwd is the
   repository working directory shown above, so open files via their
   repo-relative paths using your file-reading tools.
2. Form a short mental model of the candidate's gap, bottleneck, or open
   question relative to `evolve_rationale` and the caller objective. That gap
   defines what an "improvement" looks like for this candidate. Use this model
   to steer the search; do not emit it in the output.
3. Run a focused literature/web search for sources that propose a concrete
   method, algorithm, technique, or design idea that this candidate could
   adopt to close that gap. You must be able to name the transferable idea and
   say why it applies to this code site.

Output rules:
- Return a single bare JSON object matching the ModuleDeepResearchOutput schema
  below. Do not wrap it in Markdown and do not include explanatory prose.
- Your FINAL message must be exactly that JSON object and nothing else - it is
  parsed by machine, not read by a human. Emitting only tool-call/step events
  with no final JSON text is a failure: even when you found nothing, still emit
  the JSON object (empty `findings`, with a StepIssue explaining why).
- Report every source that passes the relevance gate; do not artificially limit the count.
- Use finding IDs find-0001, find-0002, ... ordered by expected relevance to
  this candidate and the caller objective. Most relevant first.
- Empty findings are valid when no relevant source survives filtering.
- Add StepIssue entries only for warnings or errors encountered during the
  survey. Set StepIssue.step to "module_deep_research".
- Use source_type values only from: paper, blog, docs, issue, pr, talk, codebase, other.
- title must be the exact title of the cited source (paper, blog post, doc page,
  issue, PR, talk, etc.).
- technique_summary is a short description (1-3 sentences) that names the
  concrete method, algorithm, or design idea the source contributes and
  states how it could improve THIS candidate relative to the caller
  objective (e.g., what gap or bottleneck it addresses).
- supporting_evidence must include a short verbatim quote from the source
  (at most two sentences) plus a pointer into the source (section, figure,
  algorithm number, timestamp, or commit/line). Paraphrase only when the
  source is not quotable (for example, a video without a transcript), and
  in that case still give a precise pointer.

Search transparency:
- Record every web/literature search query you issued in search_queries, in
  the order you issued them - including queries that produced no kept finding.
  This is required for debugging run-to-run variance.
- For each query, list the top results the search returned as
  {{title, url, snippet}}. snippet is a short (<=1 sentence) excerpt or the
  result's own description. If a query returned nothing, emit an empty results
  list - do not omit the query.
- tool is the tool you used (e.g. web_search, web_fetch); best-effort, leave
  "" if unsure.
- Do not invent queries or results. Report only searches you actually ran.
- Cap the reported results at ~5 per query (top results) to bound output size.
- Note: this relies on your honest self-report. Capture intent-level queries
  reliably; exhaustive fidelity to the raw tool-call stream is not required.

Candidate relevance gate:
- Keep a finding only when it passes all of these checks:
  1. It speaks to what this candidate actually does, not merely to its module,
     its repository, or a broad technology area.
  2. It is aligned with the caller objective and workload hints.
  3. It is not merely background or prior art that the candidate's
     `current_approach` already implements, unless the source meaningfully
     extends or contrasts with that approach.
  4. It carries a concrete, transferable method, algorithm, technique, or
     design idea that this candidate could plausibly adopt or adapt.
     Sources that only describe a problem, narrate experience, or restate
     what the candidate already does do not pass. Operational test: if you
     cannot state the transferable idea in a single sentence and name the part
     of this candidate it would change, the finding is too diffuse - drop it.
- Topical adjacency is not relevance. A source that shares vocabulary with the
  candidate's domain, sits in a neighboring technology area, or addresses the
  surrounding ecosystem is not automatically on-topic.

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


__all__ = ["render_candidate_deep_research_prompt"]
