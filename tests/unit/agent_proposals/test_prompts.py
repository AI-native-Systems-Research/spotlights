"""Prompt builders include the novelty context each agent needs."""

from __future__ import annotations

from spotlights_engine.agent_proposals.prompts import (
    _project_tree_breadcrumb,
    build_claude_prompt,
    build_codex_prompt,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.schemas.proposal import Proposal
from spotlights_engine.schemas.proposals import AgentProposal
from tests.unit.agent_proposals._fakes import make_candidate, make_project_tree


def test_claude_prompt_includes_candidate_research_context_and_agent_name() -> None:
    candidate = make_candidate().model_copy(
        update={
            "proposals": [
                Proposal(
                    id="prop-v1_kv_offload-0001",
                    source="research_finding",
                    finding_ref_id="find-v1_kv_offload-0001",
                    author="proposal_from_finding_creator",
                    title="Batch eviction",
                    description="Batch evictions during decode.",
                    rationale="Research-backed rationale.",
                )
            ]
        }
    )

    prompt = build_claude_prompt(
        candidate=candidate,
        project_tree=make_project_tree(),
        module_qualified_name="v1/kv_offload",
        context=SpotlightContext(
            objective="reduce latency",
            workload_hints=["decode-heavy"],
            validation_plan=["benchmark p95"],
        ),
        agent_name="claude-a",
    )

    assert "agent A" in prompt
    assert "Target module: v1/kv_offload" in prompt
    assert "parents: v1" in prompt
    assert "finding_id=find-v1_kv_offload-0001" in prompt
    assert "Batch eviction" in prompt
    assert "Objective: reduce latency" in prompt
    assert "- decode-heavy" in prompt
    assert '- agent_name: must be exactly "claude-a".' in prompt


def test_codex_prompt_includes_claude_proposal_to_avoid() -> None:
    claude_proposal = AgentProposal(
        title="Pin hot blocks",
        detailed_description="Keep hot decode blocks resident.",
        agent_name="claude",
        novelty_rationale="Not covered by research.",
    )

    prompt = build_codex_prompt(
        candidate=make_candidate(),
        project_tree=make_project_tree(),
        module_qualified_name="v1/kv_offload",
        context=SpotlightContext(objective="reduce latency"),
        agent_name="codex-b",
        claude_proposal=claude_proposal,
    )

    assert "agent B" in prompt
    assert "Agent A (Claude) proposal for this candidate" in prompt
    assert "Pin hot blocks" in prompt
    assert "Keep hot decode blocks resident." in prompt
    assert '- agent_name: must be exactly "codex-b".' in prompt


def test_codex_prompt_marks_absent_claude_proposal() -> None:
    prompt = build_codex_prompt(
        candidate=make_candidate(),
        project_tree=make_project_tree(),
        module_qualified_name="v1/kv_offload",
        context=SpotlightContext(objective="reduce latency"),
        agent_name="codex",
        claude_proposal=None,
    )

    assert "(no proposal" in prompt


def _prefixed_tree() -> ProjectTree:
    """src-layout tree whose qualified names carry a package prefix that has
    no corresponding tree node (`spotlights_engine`), the §4 break case."""
    return ProjectTree(
        repository=Repository(name="se", summary="x", source_root="src"),
        modules=[
            Module(
                name="modules_extractor",
                path="src/spotlights_engine/modules_extractor",
                submodules=[
                    Module(path="src/spotlights_engine/modules_extractor/agent"),
                    Module(path="src/spotlights_engine/modules_extractor/errors"),
                ],
            ),
            Module(name="schemas", path="src/spotlights_engine/schemas"),
        ],
    )


def test_breadcrumb_resolves_through_source_root_prefix() -> None:
    """Regression for §4: prefix segments (`spotlights_engine`) have no node,
    so segment-by-segment matching on `Module.name` would break. The
    breadcrumb must still resolve parents/siblings off the tree structure."""
    tree = _prefixed_tree()

    # Top-level module: prefix `spotlights_engine` is not a node.
    top = _project_tree_breadcrumb(tree, "spotlights_engine/modules_extractor")
    assert "parents: (top-level)" in top
    assert "siblings: schemas" in top

    # Nested leaf: parent is the modules_extractor node; sibling is `errors`.
    nested = _project_tree_breadcrumb(
        tree, "spotlights_engine/modules_extractor/agent"
    )
    assert "parents: modules_extractor" in nested
    assert "siblings: errors" in nested


def test_breadcrumb_unknown_qn_is_safe() -> None:
    tree = _prefixed_tree()
    assert _project_tree_breadcrumb(tree, "nope/nada") == "(unknown)"
