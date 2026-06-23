"""Validation tests for `parse_candidate_payload`."""

from __future__ import annotations

from spotlights_engine.agent_proposals.validation import parse_candidate_payload


_VALID = {
    "title": "T",
    "detailed_description": "D",
    "agent_name": "claude",
    "novelty_rationale": "R",
}


def test_wrapper_object_with_one_proposal_validates() -> None:
    out = parse_candidate_payload(
        {"proposals": [_VALID]},
        candidate_id="cand-v1_kv_offload-0001",
        agent_name="claude",
    )
    assert len(out.proposals) == 1
    assert out.warnings == []
    assert out.proposals[0].title == "T"


def test_bare_list_with_one_proposal_validates() -> None:
    out = parse_candidate_payload(
        [_VALID],
        candidate_id="cand-v1_kv_offload-0001",
        agent_name="claude",
    )
    assert len(out.proposals) == 1
    assert out.warnings == []


def test_empty_proposals_array_returns_no_warnings() -> None:
    out = parse_candidate_payload(
        {"proposals": []},
        candidate_id="cand-v1_kv_offload-0001",
        agent_name="claude",
    )
    assert out.proposals == []
    assert out.warnings == []


def test_more_than_one_proposal_keeps_first_and_warns() -> None:
    second = dict(_VALID, title="T2")
    out = parse_candidate_payload(
        {"proposals": [_VALID, second]},
        candidate_id="cand-v1_kv_offload-0001",
        agent_name="claude",
    )
    assert len(out.proposals) == 1
    assert out.proposals[0].title == "T"
    assert any("emitted 2 proposals" in w for w in out.warnings)


def test_agent_name_mismatch_normalizes_and_warns() -> None:
    raw = dict(_VALID, agent_name="something-else")
    out = parse_candidate_payload(
        {"proposals": [raw]},
        candidate_id="cand-v1_kv_offload-0001",
        agent_name="claude",
    )
    assert len(out.proposals) == 1
    assert out.proposals[0].agent_name == "claude"
    assert any("agent_name mismatch" in w for w in out.warnings)


def test_wrapper_missing_proposals_returns_empty_with_warning() -> None:
    out = parse_candidate_payload(
        {"foo": []},
        candidate_id="cand-v1_kv_offload-0001",
        agent_name="claude",
    )
    assert out.proposals == []
    assert any("missing list 'proposals'" in w for w in out.warnings)


def test_non_object_non_array_payload_returns_empty_with_warning() -> None:
    out = parse_candidate_payload(
        "not an object",
        candidate_id="cand-v1_kv_offload-0001",
        agent_name="claude",
    )
    assert out.proposals == []
    assert any("was not an object or array" in w for w in out.warnings)


def test_invalid_proposal_fields_drop_with_warning() -> None:
    bad = dict(_VALID, title="")  # min_length=1 violated
    out = parse_candidate_payload(
        {"proposals": [bad]},
        candidate_id="cand-v1_kv_offload-0001",
        agent_name="claude",
    )
    assert out.proposals == []
    assert any("schema validation" in w for w in out.warnings)


def test_none_payload_returns_empty_with_warning() -> None:
    out = parse_candidate_payload(
        None,
        candidate_id="cand-v1_kv_offload-0001",
        agent_name="claude",
    )
    assert out.proposals == []
    assert any("was not an object or array" in w for w in out.warnings)
