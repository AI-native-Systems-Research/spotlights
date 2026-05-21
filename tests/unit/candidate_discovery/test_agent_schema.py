"""Unit tests for the discovery-only agent output schema."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from spotlights_engine.candidate_discovery.agent_schema import (
    AgentCandidate,
    AgentCandidates,
)


def _valid_agent_candidate(**overrides) -> dict:
    payload = {
        "id": "cand-0001",
        "file": "src/foo/x.py",
        "line_start": 1,
        "line_end": 10,
        "symbol": "module.x.run",
        "kind": "function",
        "description": "does work",
        "current_approach": "linear scan",
        "evolve_rationale": "hot loop; oracle is test_x.py",
        "estimated_impact": "medium",
        "estimated_impact_explanation": "cuts request_latency_us; loop dominates",
    }
    payload.update(overrides)
    return payload


def test_agent_candidates_strict_schema_marks_every_property_required():
    """OpenAI structured-output strict mode requires every property be in
    `required` when `additionalProperties: false`. Pin that invariant for
    both the top-level shape and each candidate."""
    schema = AgentCandidates.model_json_schema()

    top_props = set(schema["properties"])
    top_required = set(schema["required"])
    assert top_required == top_props, top_props - top_required

    candidate_def = schema["$defs"]["AgentCandidate"]
    cand_props = set(candidate_def["properties"])
    cand_required = set(candidate_def["required"])
    assert cand_required == cand_props, cand_props - cand_required


def test_agent_candidates_schema_omits_downstream_step_fields():
    """The agent must not be asked to fill `state`, `finding_matches`,
    `deep_research_proposals`, or `agent_proposals` — those belong to later
    pipeline steps."""
    candidate_def = AgentCandidates.model_json_schema()["$defs"]["AgentCandidate"]
    forbidden = {
        "state",
        "finding_matches",
        "deep_research_proposals",
        "agent_proposals",
    }
    assert forbidden.isdisjoint(candidate_def["properties"])


def test_agent_candidates_allows_empty_list():
    parsed = AgentCandidates.model_validate_json(
        '{"module_qualified_name": "v1/foo", "candidates": []}'
    )
    promoted = parsed.to_candidates()
    assert promoted.candidates == []


def test_agent_candidates_to_candidates_sets_default_state_and_empty_attachments():
    parsed = AgentCandidates(
        module_qualified_name="v1/foo",
        candidates=[AgentCandidate.model_validate(_valid_agent_candidate())],
    )
    promoted = parsed.to_candidates()
    assert len(promoted.candidates) == 1
    c = promoted.candidates[0]
    assert c.state == "DISCOVERED"
    assert c.finding_matches == []
    assert c.deep_research_proposals == []
    assert c.agent_proposals == []


def test_agent_candidate_rejects_extra_field():
    payload = _valid_agent_candidate(state="FINDINGS_MAPPED")
    with pytest.raises(ValidationError):
        AgentCandidate.model_validate(payload)


def test_agent_candidates_schema_size_under_argv_ceiling():
    schema_bytes = len(json.dumps(AgentCandidates.model_json_schema(), indent=2).encode("utf-8"))
    assert schema_bytes < 10_000, schema_bytes
