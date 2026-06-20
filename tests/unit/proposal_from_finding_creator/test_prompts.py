"""Prompt builder tests for proposal_from_finding_creator."""

from __future__ import annotations

from spotlights_engine.proposal_from_finding_creator.prompts import build_prompt
from spotlights_engine.schemas.common import SpotlightContext
from tests.unit.proposal_from_finding_creator._fakes import (
    make_candidate,
    make_finding,
)


def test_prompt_matches_schema_wrapper_shape() -> None:
    prompt = build_prompt(
        candidate=make_candidate(),
        finding=make_finding(),
        context=SpotlightContext(objective="reduce latency"),
        module_qualified_name="v1/kv_offload",
        created_by="proposal_from_finding_creator",
    )

    assert "single property `proposals`" in prompt
    assert "Do not wrap the object in Markdown" in prompt
    assert "- Return a JSON array" not in prompt
