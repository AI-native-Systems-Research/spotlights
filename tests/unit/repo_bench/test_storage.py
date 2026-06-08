"""Storage helpers — paths, atomic writes, JSONL."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from spotlights_engine.repo_bench import storage


class _Row(BaseModel):
    a: int
    b: str


def test_window_id_for_date_objects():
    assert storage.window_id_for(date(2025, 12, 3), date(2026, 6, 3)) == (
        "2025-12-03__2026-06-03"
    )


def test_window_id_for_iso_strings():
    assert storage.window_id_for("2025-12-03", "2026-06-03") == (
        "2025-12-03__2026-06-03"
    )


def test_window_id_strips_time_from_datetime():
    dt = datetime(2025, 12, 3, 14, 30)
    assert storage.window_id_for(dt, "2026-06-03") == "2025-12-03__2026-06-03"


def test_data_root_respects_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))
    assert storage.data_root() == tmp_path


def test_raw_dir_uses_data_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))
    assert storage.raw_dir("w1") == tmp_path / "raw" / "w1"


def test_atomic_write_text_creates_dirs_and_file(tmp_path: Path):
    target = tmp_path / "nested" / "deep" / "file.txt"
    storage.atomic_write_text(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_atomic_write_text_no_temp_files_left(tmp_path: Path):
    target = tmp_path / "f.txt"
    storage.atomic_write_text(target, "x")
    leftovers = [p for p in tmp_path.iterdir() if p.name != "f.txt"]
    assert leftovers == []


def test_write_jsonl_pydantic(tmp_path: Path):
    target = tmp_path / "rows.jsonl"
    rows = [_Row(a=1, b="x"), _Row(a=2, b="y")]
    n = storage.write_jsonl(target, rows)
    assert n == 2
    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == [
        {"a": 1, "b": "x"}, {"a": 2, "b": "y"},
    ]


def test_write_jsonl_dicts(tmp_path: Path):
    target = tmp_path / "rows.jsonl"
    storage.write_jsonl(target, [{"a": 1}, {"a": 2}])
    assert target.read_text(encoding="utf-8") == '{"a":1}\n{"a":2}\n'


def test_write_jsonl_empty(tmp_path: Path):
    target = tmp_path / "rows.jsonl"
    n = storage.write_jsonl(target, [])
    assert n == 0
    assert target.read_text(encoding="utf-8") == ""


def test_write_json_pydantic(tmp_path: Path):
    target = tmp_path / "x.json"
    storage.write_json(target, _Row(a=7, b="z"))
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 7, "b": "z"}


def test_append_jsonl_creates_and_appends(tmp_path: Path):
    target = tmp_path / "log.jsonl"
    storage.append_jsonl(target, _Row(a=1, b="x"))
    storage.append_jsonl(target, _Row(a=2, b="y"))
    storage.append_jsonl(target, {"a": 3, "b": "z"})
    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == [
        {"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"},
    ]


def test_read_jsonl_lenient_missing_file_returns_empty(tmp_path: Path):
    assert storage.read_jsonl_lenient(tmp_path / "absent.jsonl") == []


def test_read_jsonl_lenient_drops_torn_final_line(tmp_path: Path):
    target = tmp_path / "torn.jsonl"
    target.write_text(
        '{"a": 1, "b": "ok"}\n{"a": 2, "b": "tru',  # second line truncated
        encoding="utf-8",
    )
    assert storage.read_jsonl_lenient(target) == [{"a": 1, "b": "ok"}]


def test_read_jsonl_lenient_raises_on_mid_corruption(tmp_path: Path):
    target = tmp_path / "bad.jsonl"
    target.write_text(
        '{"a": 1}\n{"a": NOPE\n{"a": 3}\n',  # middle line bad
        encoding="utf-8",
    )
    with pytest.raises(json.JSONDecodeError):
        storage.read_jsonl_lenient(target)
