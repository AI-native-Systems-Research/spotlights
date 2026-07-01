"""Unit tests for `spotlights_engine.candidate_discovery.validation`."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from spotlights_engine.candidate_discovery.validation import Validator
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.utils.schema_compat import make_location


def _validator(
    repo_path: Path, module_path: str, submodule_paths: Sequence[str] = ()
) -> Validator:
    return Validator(
        repo_path=repo_path,
        module_path=module_path,
        submodule_paths=submodule_paths,
    )


def _cand(
    *,
    id: str = "cand-mod-0001",
    file: str = "src/foo/x.py",
    line_start: int = 1,
    line_end: int = 10,
    symbol: str = "module.x.run",
    kind: str = "function",
    description: str = "does work",
    current_approach: str = "linear scan",
    evolve_rationale: str = "hot loop; oracle is test_x.py",
    estimated_impact: str = "medium",
    estimated_impact_explanation: str = "cuts request_latency_us; loop dominates the profile",
) -> Candidate:
    return Candidate(
        id=id,
        origin="code_agent",
        locations=[
            make_location(
                file=file,
                line_start=line_start,
                line_end=line_end,
                symbol=symbol,
                kind=kind,
            )
        ],
        description=description,
        current_approach=current_approach,
        evolve_rationale=evolve_rationale,
        estimated_impact=estimated_impact,
        estimated_impact_explanation=estimated_impact_explanation,
        proposals=[],
    )


def _make_file(root: Path, rel: str, lines: int = 50) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(f"line {i}" for i in range(1, lines + 1)) + "\n")
    return p


def test_drops_candidate_outside_module(tmp_path):
    _make_file(tmp_path, "src/foo/x.py")
    _make_file(tmp_path, "src/bar/y.py")
    v = _validator(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-mod-0001", file="src/foo/x.py"),
            _cand(id="cand-mod-0002", file="src/bar/y.py"),
        ],
    )
    survivors, counters = v.run(parsed)
    assert [c.id for c in survivors] == ["cand-mod-0001"]
    assert counters.dropped_outside_module == 1


def test_drops_candidate_inside_submodule(tmp_path):
    _make_file(tmp_path, "src/foo/x.py")
    _make_file(tmp_path, "src/foo/sub/y.py")
    v = _validator(tmp_path, "src/foo", submodule_paths=["src/foo/sub"])
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-mod-0001", file="src/foo/x.py"),
            _cand(id="cand-mod-0002", file="src/foo/sub/y.py"),
        ],
    )
    survivors, counters = v.run(parsed)
    assert [c.id for c in survivors] == ["cand-mod-0001"]
    assert counters.dropped_in_submodule == 1
    assert counters.dropped_outside_module == 0


def test_no_submodule_paths_keeps_nested_files(tmp_path):
    # Leaf module (no submodule_paths): files in nested dirs are still kept.
    _make_file(tmp_path, "src/foo/sub/y.py")
    v = _validator(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[_cand(id="cand-mod-0001", file="src/foo/sub/y.py")],
    )
    survivors, counters = v.run(parsed)
    assert [c.id for c in survivors] == ["cand-mod-0001"]
    assert counters.dropped_in_submodule == 0


def test_drops_absolute_path(tmp_path):
    _make_file(tmp_path, "src/foo/x.py")
    v = _validator(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-mod-0001", file="src/foo/x.py"),
            _cand(id="cand-mod-0002", file="/etc/passwd"),
        ],
    )
    survivors, counters = v.run(parsed)
    assert [c.id for c in survivors] == ["cand-mod-0001"]
    assert counters.dropped_outside_module == 1


def test_drops_symlink_escape(tmp_path):
    _make_file(tmp_path, "src/foo/x.py")
    outside = tmp_path / "outside.py"
    outside.write_text("hi\n")
    sym = tmp_path / "src" / "foo" / "linked.py"
    os.symlink(outside, sym)

    v = _validator(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[_cand(file="src/foo/linked.py")],
    )
    survivors, counters = v.run(parsed)
    assert survivors == []
    assert counters.dropped_outside_module == 1


def test_drops_missing_file(tmp_path):
    _make_file(tmp_path, "src/foo/x.py")
    v = _validator(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-mod-0001", file="src/foo/x.py"),
            _cand(id="cand-mod-0002", file="src/foo/missing.py"),
        ],
    )
    survivors, counters = v.run(parsed)
    assert [c.id for c in survivors] == ["cand-mod-0001"]
    assert counters.dropped_missing_file == 1


def test_drops_line_end_past_file_end(tmp_path):
    _make_file(tmp_path, "src/foo/x.py", lines=10)
    v = _validator(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-mod-0001", file="src/foo/x.py", line_start=1, line_end=10),
            _cand(id="cand-mod-0002", file="src/foo/x.py", line_start=1, line_end=11),
        ],
    )
    survivors, counters = v.run(parsed)
    assert [c.id for c in survivors] == ["cand-mod-0001"]
    assert counters.dropped_invalid_ranges == 1


def test_same_file_read_once(tmp_path):
    target = _make_file(tmp_path, "src/foo/x.py", lines=20)
    v = _validator(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-mod-0001", file="src/foo/x.py", line_start=1, line_end=5),
            _cand(id="cand-mod-0002", file="src/foo/x.py", line_start=6, line_end=12),
        ],
    )
    real_open = Path.open
    open_calls = {"count": 0}

    def counting_open(self, *args, **kwargs):
        if self == target.resolve(strict=False):
            open_calls["count"] += 1
        return real_open(self, *args, **kwargs)

    with patch.object(Path, "open", counting_open):
        survivors, _ = v.run(parsed)
    assert len(survivors) == 2
    assert open_calls["count"] == 1


def test_counters_independent(tmp_path):
    _make_file(tmp_path, "src/foo/x.py", lines=5)
    v = _validator(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-mod-0001", file="src/bar/y.py"),
            _cand(id="cand-mod-0002", file="src/foo/missing.py"),
            _cand(id="cand-mod-0003", file="src/foo/x.py", line_end=99, line_start=1),
        ],
    )
    survivors, counters = v.run(parsed)
    assert survivors == []
    assert counters.dropped_outside_module == 1
    assert counters.dropped_missing_file == 1
    assert counters.dropped_invalid_ranges == 1


def test_keeps_valid_candidate(tmp_path):
    _make_file(tmp_path, "src/foo/x.py", lines=50)
    v = _validator(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[_cand(file="src/foo/x.py", line_start=10, line_end=20)],
    )
    survivors, counters = v.run(parsed)
    assert len(survivors) == 1
    assert counters.dropped_outside_module == 0
    assert counters.dropped_missing_file == 0
    assert counters.dropped_invalid_ranges == 0
