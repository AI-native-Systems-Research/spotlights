"""Per-pair payload validation."""

from __future__ import annotations

from spotlights_engine.proposal_from_finding_creator.validation import (
    parse_pair_payload,
)


_OK = {
    "title": "Use technique X",
    "detailed_description": "A detailed plan",
    "finding_id": "find-0001",
    "proposal_rationale": "Because of Y",
    "created_by": "proposal_from_finding_creator",
}


def test_zero_length_array_yields_no_proposals_and_no_warnings() -> None:
    result = parse_pair_payload(
        [],
        candidate_id="cand-0001",
        finding_id="find-0001",
        created_by="proposal_from_finding_creator",
    )
    assert result.proposals == []
    assert result.warnings == []


def test_single_valid_proposal_is_kept() -> None:
    result = parse_pair_payload(
        [_OK],
        candidate_id="cand-0001",
        finding_id="find-0001",
        created_by="proposal_from_finding_creator",
    )
    assert len(result.proposals) == 1
    assert result.proposals[0].finding_id == "find-0001"
    assert result.warnings == []


def test_multiple_entries_warn_and_keep_first() -> None:
    second = dict(_OK, title="Other")
    result = parse_pair_payload(
        [_OK, second],
        candidate_id="cand-0001",
        finding_id="find-0001",
        created_by="proposal_from_finding_creator",
    )
    assert len(result.proposals) == 1
    assert result.proposals[0].title == "Use technique X"
    assert any("emitted 2 proposals" in w for w in result.warnings)


def test_finding_id_mismatch_drops_proposal_with_warning() -> None:
    bad = dict(_OK, finding_id="find-0099")
    result = parse_pair_payload(
        [bad],
        candidate_id="cand-0001",
        finding_id="find-0001",
        created_by="proposal_from_finding_creator",
    )
    assert result.proposals == []
    assert any("finding_id mismatch" in w for w in result.warnings)


def test_created_by_mismatch_normalizes_with_warning() -> None:
    bad = dict(_OK, created_by="someone-else")
    result = parse_pair_payload(
        [bad],
        candidate_id="cand-0001",
        finding_id="find-0001",
        created_by="proposal_from_finding_creator",
    )
    assert len(result.proposals) == 1
    assert result.proposals[0].created_by == "proposal_from_finding_creator"
    assert any("created_by mismatch" in w for w in result.warnings)


def test_non_array_payload_returns_warning() -> None:
    result = parse_pair_payload(
        {"oops": "not an array"},
        candidate_id="cand-0001",
        finding_id="find-0001",
        created_by="proposal_from_finding_creator",
    )
    assert result.proposals == []
    assert any("not a JSON array" in w for w in result.warnings)


def test_non_object_entry_returns_warning() -> None:
    result = parse_pair_payload(
        ["string-not-object"],
        candidate_id="cand-0001",
        finding_id="find-0001",
        created_by="proposal_from_finding_creator",
    )
    assert result.proposals == []
    assert any("not an object" in w for w in result.warnings)


def test_invalid_schema_yields_warning() -> None:
    bad = dict(_OK)
    bad.pop("title")
    result = parse_pair_payload(
        [bad],
        candidate_id="cand-0001",
        finding_id="find-0001",
        created_by="proposal_from_finding_creator",
    )
    assert result.proposals == []
    assert any("schema validation" in w for w in result.warnings)
