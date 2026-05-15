"""Unit tests for `spotlights_engine.candidate_discovery.telemetry`."""

from __future__ import annotations

from spotlights_engine.candidate_discovery.api import IterationTelemetry
from spotlights_engine.candidate_discovery.telemetry import (
    _compute_diff,
    render_diff_markdown,
)
from spotlights_engine.schemas.candidate import Candidate, Candidates


def _cs(*ids_with_meta, qn: str = "m") -> Candidates:
    """Helper: build a Candidates list from a sequence of (id, file, ls, le, rat)."""
    items = [
        Candidate(id=i, file=f, line_start=ls, line_end=le, rationale=rat)
        for (i, f, ls, le, rat) in ids_with_meta
    ]
    return Candidates(module_qualified_name=qn, candidates=items)


def test_telemetry_round_trips_via_json():
    t = IterationTelemetry(
        n=2,
        agent="claude_code",
        session_id="s",
        duration_s=1.5,
        cost_usd=0.05,
        input_tokens=10,
        output_tokens=20,
        candidate_count=3,
        schema_retries=1,
        dropped_outside_module=2,
        dropped_missing_file=0,
        dropped_invalid_ranges=0,
        added=["cand-0003"],
        removed=["cand-0001"],
        modified=["cand-0002"],
    )
    blob = t.model_dump_json()
    again = IterationTelemetry.model_validate_json(blob)
    assert again == t


def test_diff_n0_all_added():
    current = _cs(("cand-0001", "a.py", 1, 5, "r"), ("cand-0002", "a.py", 6, 9, "s"))
    added, removed, modified = _compute_diff(None, current)
    assert added == ["cand-0001", "cand-0002"]
    assert removed == []
    assert modified == []


def test_diff_rename_becomes_removed_and_added():
    prev = _cs(("cand-0001", "a.py", 1, 5, "r"))
    current = _cs(("cand-0002", "a.py", 1, 5, "r"))
    added, removed, modified = _compute_diff(prev, current)
    assert added == ["cand-0002"]
    assert removed == ["cand-0001"]
    assert modified == []


def test_diff_line_end_change_is_modified():
    prev = _cs(("cand-0001", "a.py", 1, 5, "r"))
    current = _cs(("cand-0001", "a.py", 1, 9, "r"))
    added, removed, modified = _compute_diff(prev, current)
    assert modified == ["cand-0001"]
    assert added == []
    assert removed == []


def test_diff_whitespace_only_rationale_not_modified():
    prev = _cs(("cand-0001", "a.py", 1, 5, "  hot loop  "))
    current = _cs(("cand-0001", "a.py", 1, 5, "hot loop"))
    _, _, modified = _compute_diff(prev, current)
    assert modified == []


def test_render_diff_markdown_sections_present():
    prev = _cs(
        ("cand-0001", "a.py", 1, 5, "old"),
        ("cand-0002", "b.py", 10, 20, "scheduler"),
    )
    current = _cs(
        ("cand-0001", "a.py", 1, 7, "old"),
        ("cand-0003", "c.py", 1, 1, "alloc"),
    )
    md = render_diff_markdown(prev, current)
    assert "## Added" in md
    assert "## Removed" in md
    assert "## Modified" in md
    assert "cand-0003" in md
    assert "cand-0002" in md
    assert "cand-0001" in md


def test_render_diff_markdown_none_sections():
    current = _cs(("cand-0001", "a.py", 1, 5, "r"))
    md = render_diff_markdown(None, current)
    assert "_(none)_" in md  # Removed and Modified are empty for n=0
    assert "cand-0001" in md
