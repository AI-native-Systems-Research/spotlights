"""Candidate-mode per-pair payload validation.

Behaviour is a verbatim fork of the module-mode validator; what is new is that
the four structured fields survive `DeepResearchProposal.model_validate`.
"""

from __future__ import annotations

from spotlights_engine.proposal_from_candidate_finding_creator.validation import (
    parse_pair_payload,
)
from tests.unit.proposal_from_candidate_finding_creator._fakes import (
    CREATED_BY,
    make_proposal_payload,
)

_FINDING_ID = "find-v1_kv_offload-0001-0001"
_CANDIDATE_ID = "cand-v1_kv_offload-0001"


def _parse(payload):
    return parse_pair_payload(
        payload,
        candidate_id=_CANDIDATE_ID,
        finding_id=_FINDING_ID,
        created_by=CREATED_BY,
    )


def test_structured_fields_survive_validation() -> None:
    result = _parse(make_proposal_payload(finding_id=_FINDING_ID))

    assert result.warnings == []
    p = result.proposals[0]
    assert p.mechanism == "Swap the loop for a pool"
    assert p.required_changes == "Rewrite core.py hot_1"
    assert p.expected_effect == "2x on the hot path"
    assert p.evaluation_metric == "p99 latency"


def test_missing_structured_fields_still_validate_as_none() -> None:
    # The `--json-schema` layer requires them; the validator must not add a
    # second, redundant hard failure if a runner returns them absent.
    payload = make_proposal_payload(finding_id=_FINDING_ID)
    for field in (
        "mechanism",
        "required_changes",
        "expected_effect",
        "evaluation_metric",
    ):
        payload[0].pop(field)

    result = _parse(payload)

    assert result.warnings == []
    assert result.proposals[0].mechanism is None


def test_empty_structured_field_is_rejected() -> None:
    payload = make_proposal_payload(finding_id=_FINDING_ID)
    payload[0]["mechanism"] = ""

    result = _parse(payload)

    assert result.proposals == []
    assert any("schema validation" in w for w in result.warnings)


def test_zero_length_array_yields_no_proposals_and_no_warnings() -> None:
    result = _parse([])

    assert result.proposals == []
    assert result.warnings == []


def test_finding_id_mismatch_drops_proposal_with_warning() -> None:
    result = _parse(make_proposal_payload(finding_id="find-v1_kv_offload-0099-0001"))

    assert result.proposals == []
    assert any("finding_id mismatch" in w for w in result.warnings)


def test_created_by_mismatch_normalizes_with_warning() -> None:
    result = _parse(make_proposal_payload(finding_id=_FINDING_ID, created_by="someone-else"))

    assert result.proposals[0].created_by == CREATED_BY
    assert any("created_by mismatch" in w for w in result.warnings)


def test_multiple_entries_warn_and_keep_first() -> None:
    first = make_proposal_payload(finding_id=_FINDING_ID, title="First")
    second = make_proposal_payload(finding_id=_FINDING_ID, title="Second")

    result = _parse([*first, *second])

    assert [p.title for p in result.proposals] == ["First"]
    assert any("emitted 2 proposals" in w for w in result.warnings)


def test_non_array_payload_returns_warning() -> None:
    result = _parse({"oops": "not an array"})

    assert result.proposals == []
    assert any("not a JSON array" in w for w in result.warnings)
