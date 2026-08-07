"""Candidate-mode step-4 per-pair prompt assembly."""

from __future__ import annotations

from spotlights_engine.proposal_from_candidate_finding_creator.prompts import (
    build_prompt,
)
from spotlights_engine.schemas.common import SpotlightContext
from tests.unit.proposal_from_candidate_finding_creator._fakes import (
    CREATED_BY,
    make_finding,
    make_input,
)


def _prompt() -> str:
    inp = make_input(n_candidates=1)
    return build_prompt(
        candidate=inp.candidates.candidates[0],
        finding=make_finding(0),
        context=SpotlightContext(
            objective="reduce latency", validation_plan=["compare tokens/sec"]
        ),
        module_qualified_name="v1/kv_offload",
        created_by=CREATED_BY,
    )


def test_prompt_carries_the_candidate_and_finding_blocks() -> None:
    prompt = _prompt()

    assert "cand-v1_kv_offload-0001" in prompt
    assert "find-v1_kv_offload-0001-0001" in prompt
    assert "src/v1/kv_offload/core.py" in prompt
    assert "compare tokens/sec" in prompt


def test_prompt_pins_finding_id_and_created_by() -> None:
    prompt = _prompt()

    assert 'must be exactly "find-v1_kv_offload-0001-0001"' in prompt
    assert f'must be exactly "{CREATED_BY}"' in prompt


def test_prompt_requires_the_four_structured_fields() -> None:
    prompt = _prompt()

    for field in (
        "mechanism:",
        "required_changes:",
        "expected_effect:",
        "evaluation_metric:",
    ):
        assert f"- {field}" in prompt


def test_prompt_states_the_stronger_relevance_gate() -> None:
    prompt = _prompt()

    assert "Relevance gate (apply before writing anything):" in prompt
    assert "topical\n  overlap is expected and is NOT evidence of relevance" in prompt
    assert "Emitting `[]` is a correct, expected outcome." in prompt
