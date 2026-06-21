"""Unit tests for finding-related schemas."""

from __future__ import annotations

from spotlights_engine.schemas.finding import Finding


def test_finding_id_accepts_module_prefixed() -> None:
    # `finding_id` is a free-form string (the schema no longer enforces a
    # pattern); the manager mints the module-prefixed `find-<segment>-NNNN` form.
    finding = Finding(
        finding_id="find-attention-0001",
        title="Paged attention",
        url="https://example.com/paged-attention",
        source_type="paper",
        technique_summary="Use paged KV allocation to reduce fragmentation.",
    )
    assert finding.finding_id == "find-attention-0001"
