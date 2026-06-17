"""Prompt builders include the novelty context each agent needs."""

from __future__ import annotations

from spotlights_engine.agent_proposals.prompts import (
    build_claude_prompt,
    build_codex_prompt,
)
from spotlights_engine.schemas.legacy.common import SpotlightContext
from spotlights_engine.schemas.legacy.proposals import AgentProposal, DeepResearchProposal
from tests.unit.agent_proposals._fakes import make_candidate, make_project_tree


def test_claude_prompt_includes_candidate_research_context_and_agent_name() -> None:
    candidate = make_candidate().model_copy(
        update={
            "deep_research_proposals": [
                DeepResearchProposal(
                    title="Batch eviction",
                    detailed_description="Batch evictions during decode.",
                    finding_id="find-0001",
                    proposal_rationale="Research-backed rationale.",
                    created_by="proposal_from_finding_creator",
                )
            ]
        }
    )

    prompt = build_claude_prompt(
        candidate=candidate,
        project_tree=make_project_tree(),
        module_qualified_name="v1.kv_offload",
        context=SpotlightContext(
            objective="reduce latency",
            workload_hints=["decode-heavy"],
            validation_plan=["benchmark p95"],
        ),
        agent_name="claude-a",
    )

    assert "agent A" in prompt
    assert "Target module: v1.kv_offload" in prompt
    assert "parents: v1" in prompt
    assert "finding_id=find-0001" in prompt
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
        module_qualified_name="v1.kv_offload",
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
        module_qualified_name="v1.kv_offload",
        context=SpotlightContext(objective="reduce latency"),
        agent_name="codex",
        claude_proposal=None,
    )

    assert "(no proposal" in prompt
