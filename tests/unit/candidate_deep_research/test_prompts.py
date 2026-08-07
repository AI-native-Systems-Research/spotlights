"""Candidate-mode step-3 prompt assembly."""

from __future__ import annotations

from spotlights_engine.candidate_deep_research import (
    render_candidate_deep_research_prompt,
)
from spotlights_engine.module_deep_research.api import resolve_target_module
from tests.unit.candidate_deep_research._fakes import (
    MODULE_QN,
    SEGMENT,
    make_candidate,
    make_request,
)


def _render(*, candidate_idx: int = 0, max_findings_per_candidate: int = 5) -> str:
    request = make_request(1, max_findings_per_candidate=max_findings_per_candidate)
    module = resolve_target_module(request.project_tree, MODULE_QN)
    assert module is not None
    return render_candidate_deep_research_prompt(request, module, make_candidate(candidate_idx))


def test_prompt_names_the_target_candidate() -> None:
    prompt = _render()

    assert f"cand-{SEGMENT}-0001" in prompt
    assert "Target candidate:" in prompt
    assert "src/inference/attention/core.py" in prompt
    assert "current approach 1" in prompt
    assert "evolve rationale 1" in prompt


def test_prompt_keeps_module_and_repository_context_for_grounding() -> None:
    prompt = _render()

    assert "Attention implementation." in prompt
    assert "Demo repository." in prompt
    assert MODULE_QN in prompt


def test_prompt_uses_the_per_candidate_cap() -> None:
    assert "Include at most 3 findings." in _render(max_findings_per_candidate=3)


def test_prompt_pins_the_shared_step_name_and_wire_schema() -> None:
    prompt = _render()

    # Both the `PipelineStep` literal and the agent-facing wire schema are
    # deliberately reused from module mode.
    assert 'StepIssue.step to "module_deep_research"' in prompt
    assert "ModuleDeepResearchOutput JSON schema:" in prompt
    assert "find-0001" in prompt


def test_prompt_states_the_candidate_relevance_gate() -> None:
    prompt = _render()

    assert "Candidate relevance gate:" in prompt
    assert "Topical adjacency is not relevance." in prompt
