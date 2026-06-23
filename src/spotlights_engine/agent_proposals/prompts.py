"""Per-candidate prompt builders for step 5 (`agent_proposals`).

Two prompts:

- `build_claude_prompt` — agent A. Asked to emit one `AgentProposal` that is
  not already covered by the candidate's `deep_research_proposals`.
- `build_codex_prompt` — agent B. Same goal but additionally must not
  duplicate the proposal Claude (agent A) emitted, if any.

Both prompts include a project-tree breadcrumb (sibling/parent module names)
so the agent can situate the candidate's module within the project.
"""

from __future__ import annotations

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.project import Module, ProjectTree
from spotlights_engine.schemas.proposal import Proposal
from spotlights_engine.schemas.proposals import AgentProposal
from spotlights_engine.utils.schema_compat import (
    primary_file,
    primary_span,
    proposals_from,
)


def _format_list(values: list[str]) -> str:
    if not values:
        return "(none)"
    return "\n".join(f"- {value}" for value in values)


def _format_research_proposals(proposals: list[Proposal]) -> str:
    """Render the candidate's research-backed proposals (the unified
    `Proposal`s whose `source == "research_finding"`) into the human-readable
    block the prompt previously built from `deep_research_proposals`."""
    if not proposals:
        return "(none)"
    lines: list[str] = []
    for i, p in enumerate(proposals, 1):
        lines.append(
            f"{i}. finding_id={p.finding_ref_id}\n"
            f"   title: {p.title}\n"
            f"   detailed_description: {p.description}\n"
            f"   proposal_rationale: {p.rationale}"
        )
    return "\n".join(lines)


def _format_claude_proposal(proposal: AgentProposal | None) -> str:
    if proposal is None:
        return "(no proposal — agent A did not emit one)"
    return (
        f"agent_name: {proposal.agent_name}\n"
        f"title: {proposal.title}\n"
        f"detailed_description: {proposal.detailed_description}\n"
        f"novelty_rationale: {proposal.novelty_rationale}"
    )


def _project_tree_breadcrumb(
    tree: ProjectTree, module_qualified_name: str
) -> str:
    """Return a short breadcrumb of parent + sibling module names.

    The qualified name is the module's slash-form path relative to `source_root`
    (see `ProjectTree.walk`) — the single canonical form throughout. Resolve the
    target by its full qualified name and read its parent/siblings off the actual
    tree structure. Source-root prefix segments (e.g. `spotlights_engine`) have no
    corresponding node, so we cannot match segment-by-segment on `Module.name`.
    """
    qn = module_qualified_name
    if not qn:
        return "(unknown)"

    # Qualified name per module identity, so we can read parent/siblings off
    # the actual tree structure without parsing prefix segments.
    qn_by_id: dict[int, str] = {id(m): node_qn for node_qn, m in tree.walk()}
    ancestors_of: dict[str, list[Module]] = {}
    siblings_of: dict[str, list[Module]] = {}

    def _index(modules: list[Module], chain: list[Module]) -> None:
        for m in modules:
            node_qn = qn_by_id.get(id(m))
            if node_qn is not None:
                ancestors_of[node_qn] = list(chain)
                siblings_of[node_qn] = [s for s in modules if s is not m]
            _index(m.submodules, chain + [m])

    _index(tree.modules, [])

    if qn not in ancestors_of:
        return "(unknown)"

    parents = ancestors_of[qn]
    siblings = siblings_of[qn]
    parent_chain = " > ".join(p.name for p in parents) or "(top-level)"
    sibling_names = ", ".join(sorted(m.name for m in siblings)) or "(none)"
    return f"parents: {parent_chain}\nsiblings: {sibling_names}"


def _format_candidate_block(candidate: Candidate) -> str:
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


def _format_context_block(context: SpotlightContext) -> str:
    return (
        f"Objective: {context.objective}\n"
        f"Workload hints:\n{_format_list(context.workload_hints)}\n"
        f"Validation plan:\n{_format_list(context.validation_plan)}"
    )


def build_claude_prompt(
    *,
    candidate: Candidate,
    project_tree: ProjectTree,
    module_qualified_name: str,
    context: SpotlightContext,
    agent_name: str,
) -> str:
    """Render the per-candidate prompt for the Claude pass (agent A)."""
    breadcrumb = _project_tree_breadcrumb(project_tree, module_qualified_name)
    return f"""You are running the Spotlights agent_proposals pipeline step (agent A).
Do not modify files. Do not ask questions. You may read repository files
under the current working directory to ground your reasoning.

Goal:
Look at the supplied candidate and propose at most ONE concrete, actionable
change ("AgentProposal") that draws on your own knowledge and is NOT already
covered by the candidate's existing deep_research_proposals. If you have no
novel idea that meets the bar, emit no proposal — quality over quantity.

Target module: {module_qualified_name}
Project tree breadcrumb:
{breadcrumb}

Candidate:
{_format_candidate_block(candidate)}

Existing deep_research_proposals on this candidate:
{_format_research_proposals(proposals_from(candidate, "research_finding"))}

Caller context:
{_format_context_block(context)}

Output rules:
- Return a JSON object with a single property `proposals` whose value is a
  JSON array of length 0 or 1.
- An empty array (`[]`) means: you have no novel idea worth emitting.
  Prefer emptiness when your idea overlaps a listed deep_research_proposal
  or is only topically adjacent.
- A 1-element array means: emit one AgentProposal.
  - title: short, action-oriented (1 line).
  - detailed_description: concrete description of the change applied to
    this specific candidate. May reference the candidate's file/lines/symbol;
    do not invent unrelated locations.
  - agent_name: must be exactly "{agent_name}".
  - novelty_rationale: explicitly explain why this proposal is not already
    covered by any of the listed deep_research_proposals.
- Do not wrap the object in Markdown. Do not include explanatory prose.
""".strip()


def build_codex_prompt(
    *,
    candidate: Candidate,
    project_tree: ProjectTree,
    module_qualified_name: str,
    context: SpotlightContext,
    agent_name: str,
    claude_proposal: AgentProposal | None,
) -> str:
    """Render the per-candidate prompt for the Codex pass (agent B).

    Agent A's output (if any) is included verbatim so agent B can avoid
    duplicating it.
    """
    breadcrumb = _project_tree_breadcrumb(project_tree, module_qualified_name)
    return f"""You are running the Spotlights agent_proposals pipeline step (agent B).
Do not modify files. Do not ask questions. You may read repository files
under the current working directory to ground your reasoning.

Goal:
Look at the supplied candidate and propose at most ONE concrete, actionable
change ("AgentProposal") that draws on your own knowledge and is NOT already
covered by EITHER the candidate's deep_research_proposals OR the proposal
agent A (Claude) just emitted for this same candidate. If you have no novel
idea that meets the bar, emit no proposal — quality over quantity.

Target module: {module_qualified_name}
Project tree breadcrumb:
{breadcrumb}

Candidate:
{_format_candidate_block(candidate)}

Existing deep_research_proposals on this candidate:
{_format_research_proposals(proposals_from(candidate, "research_finding"))}

Agent A (Claude) proposal for this candidate (avoid duplicating):
{_format_claude_proposal(claude_proposal)}

Caller context:
{_format_context_block(context)}

Output rules:
- Return a JSON object with a single property `proposals` whose value is a
  JSON array of length 0 or 1.
- An empty array (`[]`) means: you have no novel idea worth emitting.
  Prefer emptiness when your idea overlaps a listed deep_research_proposal
  or agent A's proposal, or is only topically adjacent.
- A 1-element array means: emit one AgentProposal.
  - title: short, action-oriented (1 line).
  - detailed_description: concrete description of the change applied to
    this specific candidate. May reference the candidate's file/lines/symbol;
    do not invent unrelated locations.
  - agent_name: must be exactly "{agent_name}".
  - novelty_rationale: explicitly explain why this proposal is not already
    covered by any listed deep_research_proposal AND not already covered by
    agent A's proposal.
- Do not wrap the object in Markdown. Do not include explanatory prose.
""".strip()


__all__ = ["build_claude_prompt", "build_codex_prompt"]
