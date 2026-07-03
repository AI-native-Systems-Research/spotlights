"""Unit tests for the pure bootstrap-merge helpers (dedup + renumber).

These exercise `_dedup_by_overlap` and `_renumber_candidate` directly, away
from the orchestrator, covering overlap vs. adjacency, different files, and the
empty edge cases the plan calls out.
"""

from __future__ import annotations

from spotlights_engine.candidate_discovery.orchestrator import (
    _dedup_by_overlap,
    _renumber_candidate,
)
from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.utils.schema_compat import make_location, primary_file, primary_span


def _cand(cid: str, file: str, line_start: int, line_end: int) -> Candidate:
    return Candidate(
        id=cid,
        module_qualified_name="v1/foo",
        origin="code_agent",
        locations=[
            make_location(
                file=file,
                line_start=line_start,
                line_end=line_end,
                symbol="sym",
                kind="function",
            )
        ],
        description="d",
        current_approach="c",
        evolve_rationale="e",
        estimated_impact="medium",
        estimated_impact_explanation="x",
        proposals=[],
    )


def test_dedup_overlapping_same_file_keeps_first():
    a = _cand("cand-seg-0001", "a.py", 1, 10)
    b = _cand("cand-seg-0002", "a.py", 5, 15)  # overlaps a
    kept, dropped = _dedup_by_overlap([a, b])
    assert [c.id for c in kept] == ["cand-seg-0001"]
    assert dropped == 1


def test_dedup_adjacent_non_overlapping_kept_separately():
    a = _cand("cand-seg-0001", "a.py", 1, 10)
    b = _cand("cand-seg-0002", "a.py", 11, 20)  # adjacent, no overlap
    kept, dropped = _dedup_by_overlap([a, b])
    assert [c.id for c in kept] == ["cand-seg-0001", "cand-seg-0002"]
    assert dropped == 0


def test_dedup_touching_boundary_overlaps():
    # inclusive ranges: [1,10] and [10,20] share line 10.
    a = _cand("cand-seg-0001", "a.py", 1, 10)
    b = _cand("cand-seg-0002", "a.py", 10, 20)
    kept, dropped = _dedup_by_overlap([a, b])
    assert [c.id for c in kept] == ["cand-seg-0001"]
    assert dropped == 1


def test_dedup_same_range_different_files_kept():
    a = _cand("cand-seg-0001", "a.py", 1, 10)
    b = _cand("cand-seg-0002", "b.py", 1, 10)  # same range, different file
    kept, dropped = _dedup_by_overlap([a, b])
    assert [c.id for c in kept] == ["cand-seg-0001", "cand-seg-0002"]
    assert dropped == 0


def test_dedup_empty():
    kept, dropped = _dedup_by_overlap([])
    assert kept == []
    assert dropped == 0


def test_dedup_three_way_overlap_collapses_to_one():
    a = _cand("cand-seg-0001", "a.py", 1, 10)
    b = _cand("cand-seg-0002", "a.py", 8, 12)
    c = _cand("cand-seg-0003", "a.py", 11, 20)  # overlaps b but not a
    kept, dropped = _dedup_by_overlap([a, b, c])
    # a kept; b dropped (overlaps a); c kept (overlaps only the dropped b).
    assert [x.id for x in kept] == ["cand-seg-0001", "cand-seg-0003"]
    assert dropped == 1


def test_renumber_assigns_prefixed_id_and_preserves_content():
    a = _cand("cand-other-0009", "a.py", 3, 7)
    out = _renumber_candidate(a, counter=1, segment="v1_foo")
    assert out.id == "cand-v1_foo-0001"
    # Content (location, spans) is preserved.
    assert primary_file(out) == "a.py"
    assert primary_span(out).line_start == 3
    assert primary_span(out).line_end == 7


def test_renumber_contiguous_sequence():
    cands = [
        _cand("cand-a-0005", "a.py", 1, 2),
        _cand("cand-b-0003", "b.py", 1, 2),
        _cand("cand-c-0009", "c.py", 1, 2),
    ]
    renumbered = [
        _renumber_candidate(c, counter=i, segment="seg")
        for i, c in enumerate(cands, start=1)
    ]
    assert [c.id for c in renumbered] == [
        "cand-seg-0001",
        "cand-seg-0002",
        "cand-seg-0003",
    ]
