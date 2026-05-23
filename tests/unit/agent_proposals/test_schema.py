"""Tests for the per-candidate JSON schema text."""

from __future__ import annotations

import json

from spotlights_engine.agent_proposals.agent_schema import (
    build_per_candidate_schema_text,
)


def test_schema_is_object_with_proposals_array() -> None:
    schema = json.loads(build_per_candidate_schema_text(agent_name="claude"))
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["proposals"]
    proposals = schema["properties"]["proposals"]
    assert proposals["type"] == "array"
    assert proposals["minItems"] == 0
    assert proposals["maxItems"] == 1


def test_schema_pins_agent_name_via_const() -> None:
    schema = json.loads(build_per_candidate_schema_text(agent_name="codex"))
    item_props = schema["properties"]["proposals"]["items"]["properties"]
    assert item_props["agent_name"]["const"] == "codex"


def test_schema_inner_object_has_required_fields() -> None:
    schema = json.loads(build_per_candidate_schema_text(agent_name="claude"))
    item = schema["properties"]["proposals"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {
        "title",
        "detailed_description",
        "agent_name",
        "novelty_rationale",
    }
