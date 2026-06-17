"""Unit tests for finding-related schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.legacy.finding import Finding


def test_finding_id_pattern_is_strict() -> None:
    Finding(
        finding_id="find-0001",
        title="Paged attention",
        url="https://example.com/paged-attention",
        source_type="paper",
        technique_summary="Use paged KV allocation to reduce fragmentation.",
    )

    with pytest.raises(ValidationError):
        Finding(
            finding_id="finding-1",
            title="Paged attention",
            url="https://example.com/paged-attention",
            source_type="paper",
            technique_summary="Use paged KV allocation to reduce fragmentation.",
        )
