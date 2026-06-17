"""Unit tests for cross-cutting schema types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.common import SpotlightContext


def test_context_objective_default_is_empty_string() -> None:
    # `objective` may be empty; some pipelines (e.g. signal) have no
    # caller-supplied objective and pass through whatever they have.
    SpotlightContext()
    SpotlightContext(objective="")
    SpotlightContext(objective="reduce TTFT")


def test_context_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SpotlightContext.model_validate(
            {"objective": "x", "extra": "nope"}
        )
