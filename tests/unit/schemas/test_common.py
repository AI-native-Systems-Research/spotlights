"""Unit tests for cross-cutting schema types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.common import SpotlightContext


def test_context_objective_is_required_and_non_empty() -> None:
    with pytest.raises(ValidationError):
        SpotlightContext(objective="")
