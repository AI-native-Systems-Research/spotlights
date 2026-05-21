"""Unit tests for `spotlights_engine.schemas.candidate`."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.candidate import Candidate, Candidates


def _valid_candidate(**overrides):
    payload = {
        "id": "cand-0001",
        "file": "src/foo.py",
        "line_start": 1,
        "line_end": 10,
        "symbol": "module.foo.handle_request",
        "kind": "function",
        "description": "Handles inbound requests.",
        "current_approach": "Linear scan over the request body.",
        "evolve_rationale": "Hot loop with simple structure; oracle is unit tests in test_foo.py.",
        "estimated_impact": "high",
        "estimated_impact_explanation": "Reduces request_latency_ms; the loop dominates the profile and re-allocates per item.",
    }
    payload.update(overrides)
    return payload


def test_id_pattern_accepts_four_digit_zero_padded():
    Candidate.model_validate(_valid_candidate(id="cand-0001"))


@pytest.mark.parametrize("bad_id", ["cand-1", "cand-00001", "candidate-0001", "CAND-0001", "cand-001a"])
def test_id_pattern_rejects(bad_id):
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(id=bad_id))


def test_line_end_must_be_ge_line_start():
    with pytest.raises(ValidationError, match="line_end must be >= line_start"):
        Candidate.model_validate(_valid_candidate(line_start=10, line_end=9))


def test_line_end_equal_to_line_start_is_ok():
    c = Candidate.model_validate(_valid_candidate(line_start=10, line_end=10))
    assert c.line_start == c.line_end == 10


def test_line_start_must_be_ge_1():
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(line_start=0, line_end=10))


def test_evolve_rationale_empty_rejected():
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(evolve_rationale=""))


def test_description_empty_rejected():
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(description=""))


def test_current_approach_empty_rejected():
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(current_approach=""))


def test_symbol_empty_rejected():
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(symbol=""))


def test_kind_must_be_enum():
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(kind="frobnicate"))


def test_estimated_impact_must_be_enum():
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(estimated_impact="huge"))


def test_estimated_impact_explanation_required():
    payload = _valid_candidate()
    payload.pop("estimated_impact_explanation")
    with pytest.raises(ValidationError):
        Candidate.model_validate(payload)


def test_estimated_impact_explanation_empty_rejected():
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(estimated_impact_explanation=""))


def test_estimated_impact_explanation_accepts_freeform_prose():
    text = (
        "High because cache_hit_rate would jump: the current LRU evicts hot keys "
        "under bursty access patterns; switching to TinyLFU keeps the working set."
    )
    c = Candidate.model_validate(_valid_candidate(estimated_impact_explanation=text))
    assert c.estimated_impact_explanation == text


def test_estimated_impact_explanation_in_json_schema():
    cand_schema = Candidate.model_json_schema()
    assert "estimated_impact_explanation" in cand_schema["properties"]
    assert "estimated_impact_explanation" in cand_schema["required"]


def test_verbose_rationale_fields_are_accepted():
    Candidate.model_validate(
        _valid_candidate(
            description="x" * 600,
            current_approach="y" * 600,
            evolve_rationale="z" * 600,
        )
    )


def test_candidates_list_can_be_empty():
    """Per architecture: an empty candidate list is valid output. The manager
    marks such a module run `SKIPPED` and does not call downstream steps."""
    obj = Candidates.model_validate({"module_qualified_name": "v1/foo", "candidates": []})
    assert obj.candidates == []
    assert obj.module_qualified_name == "v1/foo"


def test_candidate_default_state_and_empty_attachments():
    """Per architecture, a freshly discovered candidate is at `DISCOVERED`
    with empty match/proposal lists; later steps populate these fields."""
    c = Candidate.model_validate(_valid_candidate())
    assert c.state == "DISCOVERED"
    assert c.finding_matches == []
    assert c.deep_research_proposals == []
    assert c.agent_proposals == []


def test_candidate_state_accepts_pipeline_progression():
    for state in (
        "DISCOVERED",
        "FINDINGS_MAPPED",
        "FINDING_PROPOSALS_CREATED",
        "AGENT_PROPOSALS_CREATED",
    ):
        c = Candidate.model_validate(_valid_candidate(state=state))
        assert c.state == state


def test_candidate_rejects_unknown_state():
    with pytest.raises(ValidationError):
        Candidate.model_validate(_valid_candidate(state="MAPPED"))


def test_candidates_model_json_schema_round_trips_through_json_dumps():
    schema = Candidates.model_json_schema()
    text = json.dumps(schema)
    reloaded = json.loads(text)
    assert reloaded == schema


def test_candidates_schema_fits_claude_json_schema_argv():
    """Pin the current schema size so a future addition is loud.

    Claude inlines this schema via `--json-schema <text>`; Linux's
    `MAX_ARG_STRLEN` is 131_072 bytes per argument, and the orchestrator
    enforces a 120_000-byte ceiling with headroom. If a future schema
    change pushes us anywhere close to that limit, this test will fail
    before the production check does.
    """
    schema_bytes = len(json.dumps(Candidates.model_json_schema(), indent=2).encode("utf-8"))
    assert schema_bytes < 10_000, (
        f"Candidates schema is now {schema_bytes} bytes; update the pin or "
        f"reconsider passing the schema by path instead of inline."
    )


def test_candidates_round_trips_through_model_validate_json():
    payload = {
        "module_qualified_name": "v1/engine/core",
        "candidates": [_valid_candidate()],
    }
    obj = Candidates.model_validate(payload)
    text = obj.model_dump_json()
    again = Candidates.model_validate_json(text)
    assert again == obj
