"""The candidate-mode per-pair `--json-schema` contract."""

from __future__ import annotations

import json

from spotlights_engine.proposal_from_candidate_finding_creator.agent_schema import (
    build_per_pair_schema_text,
)
from spotlights_engine.proposal_from_finding_creator.claude_exec import (
    _unwrap_proposals,
)


def _schema() -> dict:
    return json.loads(
        build_per_pair_schema_text(
            finding_id="find-v1_kv_offload-0001-0001",
            created_by="proposal_from_candidate_finding_creator",
        )
    )


def test_wrapper_shape_is_unchanged_so_unwrapping_still_works() -> None:
    schema = _schema()

    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["proposals"]
    assert schema["properties"]["proposals"]["maxItems"] == 1
    assert schema["properties"]["proposals"]["minItems"] == 0
    # The wrapper the runner strips before validation.
    assert _unwrap_proposals({"proposals": []}) == []


def test_the_four_structured_fields_are_required_and_non_empty() -> None:
    items = _schema()["properties"]["proposals"]["items"]

    for field in (
        "mechanism",
        "required_changes",
        "expected_effect",
        "evaluation_metric",
    ):
        assert field in items["required"]
        assert items["properties"][field] == {"type": "string", "minLength": 1}


def test_finding_id_and_created_by_stay_pinned_with_const() -> None:
    props = _schema()["properties"]["proposals"]["items"]["properties"]

    assert props["finding_id"]["const"] == "find-v1_kv_offload-0001-0001"
    assert props["created_by"]["const"] == "proposal_from_candidate_finding_creator"


def test_no_extra_keys_are_accepted() -> None:
    items = _schema()["properties"]["proposals"]["items"]

    assert items["additionalProperties"] is False
    assert set(items["properties"]) == set(items["required"])
