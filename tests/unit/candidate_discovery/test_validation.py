"""Unit tests for `spotlights_engine.candidate_discovery.validation`."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from spotlights_engine.candidate_discovery.validation import Validator
from spotlights_engine.schemas.candidate import Candidate, Candidates


def _config(repo_path: Path, module_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        repo_path=repo_path,
        module=SimpleNamespace(path=module_path),
    )


def _cand(**overrides) -> Candidate:
    payload = {
        "id": "cand-0001",
        "file": "src/foo/x.py",
        "line_start": 1,
        "line_end": 10,
        "rationale": "loop",
    }
    payload.update(overrides)
    return Candidate.model_validate(payload)


def _make_file(root: Path, rel: str, lines: int = 50) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(f"line {i}" for i in range(1, lines + 1)) + "\n")
    return p


def test_drops_candidate_outside_module(tmp_path):
    _make_file(tmp_path, "src/foo/x.py")
    _make_file(tmp_path, "src/bar/y.py")
    cfg = _config(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-0001", file="src/foo/x.py"),
            _cand(id="cand-0002", file="src/bar/y.py"),
        ],
    )
    survivors, counters = Validator(cfg).run(parsed)
    assert [c.id for c in survivors] == ["cand-0001"]
    assert counters.dropped_outside_module == 1


def test_drops_absolute_path(tmp_path):
    _make_file(tmp_path, "src/foo/x.py")
    cfg = _config(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-0001", file="src/foo/x.py"),
            _cand(id="cand-0002", file="/etc/passwd"),
        ],
    )
    survivors, counters = Validator(cfg).run(parsed)
    assert [c.id for c in survivors] == ["cand-0001"]
    assert counters.dropped_outside_module == 1


def test_drops_symlink_escape(tmp_path):
    _make_file(tmp_path, "src/foo/x.py")
    outside = tmp_path / "outside.py"
    outside.write_text("hi\n")
    sym = tmp_path / "src" / "foo" / "linked.py"
    os.symlink(outside, sym)

    cfg = _config(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[_cand(file="src/foo/linked.py")],
    )
    survivors, counters = Validator(cfg).run(parsed)
    assert survivors == []
    assert counters.dropped_outside_module == 1


def test_drops_missing_file(tmp_path):
    _make_file(tmp_path, "src/foo/x.py")
    cfg = _config(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-0001", file="src/foo/x.py"),
            _cand(id="cand-0002", file="src/foo/missing.py"),
        ],
    )
    survivors, counters = Validator(cfg).run(parsed)
    assert [c.id for c in survivors] == ["cand-0001"]
    assert counters.dropped_missing_file == 1


def test_drops_line_end_past_file_end(tmp_path):
    _make_file(tmp_path, "src/foo/x.py", lines=10)
    cfg = _config(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-0001", file="src/foo/x.py", line_start=1, line_end=10),
            _cand(id="cand-0002", file="src/foo/x.py", line_start=1, line_end=11),
        ],
    )
    survivors, counters = Validator(cfg).run(parsed)
    assert [c.id for c in survivors] == ["cand-0001"]
    assert counters.dropped_invalid_ranges == 1


def test_same_file_read_once(tmp_path):
    target = _make_file(tmp_path, "src/foo/x.py", lines=20)
    cfg = _config(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-0001", file="src/foo/x.py", line_start=1, line_end=5),
            _cand(id="cand-0002", file="src/foo/x.py", line_start=6, line_end=12),
        ],
    )
    real_open = Path.open
    open_calls = {"count": 0}

    def counting_open(self, *args, **kwargs):
        if self == target.resolve(strict=False):
            open_calls["count"] += 1
        return real_open(self, *args, **kwargs)

    with patch.object(Path, "open", counting_open):
        survivors, _ = Validator(cfg).run(parsed)
    assert len(survivors) == 2
    assert open_calls["count"] == 1


def test_counters_independent(tmp_path):
    _make_file(tmp_path, "src/foo/x.py", lines=5)
    cfg = _config(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[
            _cand(id="cand-0001", file="src/bar/y.py"),
            _cand(id="cand-0002", file="src/foo/missing.py"),
            _cand(id="cand-0003", file="src/foo/x.py", line_end=99, line_start=1),
        ],
    )
    survivors, counters = Validator(cfg).run(parsed)
    assert survivors == []
    assert counters.dropped_outside_module == 1
    assert counters.dropped_missing_file == 1
    assert counters.dropped_invalid_ranges == 1


def test_keeps_valid_candidate(tmp_path):
    _make_file(tmp_path, "src/foo/x.py", lines=50)
    cfg = _config(tmp_path, "src/foo")
    parsed = Candidates(
        module_qualified_name="m",
        candidates=[_cand(file="src/foo/x.py", line_start=10, line_end=20)],
    )
    survivors, counters = Validator(cfg).run(parsed)
    assert len(survivors) == 1
    assert counters.dropped_outside_module == 0
    assert counters.dropped_missing_file == 0
    assert counters.dropped_invalid_ranges == 0
