"""Bench spec — JSON canonical + MD rendered from JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spotlights_engine.repo_bench import storage
from spotlights_engine.repo_bench.bench_spec import (
    _strip_leading_timestamp,
    load_spec,
    render_md,
    write_spec,
)
from spotlights_engine.repo_bench.schemas import SnapshotPin


def _pin(**overrides) -> SnapshotPin:
    defaults = {
        "window_id": "2026-06-01__2026-06-10",
        "view_id": "abc12345",
        "snapshot_sha": "0" * 39 + "f",
        "snapshot_pr_number": 12345,
        "snapshot_merged_at": datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        "earliest_in_view_pr_number": 67890,
        "earliest_in_view_merged_at": datetime(2026, 6, 3, 12, tzinfo=timezone.utc),
        "n_view": 23,
        "rationale": "test rationale here",
        "pinned_at": datetime(2026, 6, 4, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return SnapshotPin(**defaults)


# ── _strip_leading_timestamp ──────────────────────────────────────────


def test_strip_timestamp_real_bundle_name():
    assert _strip_leading_timestamp(
        "20260525T202105Z_util0.4_mem16_lru"
    ) == "util0.4_mem16_lru"


def test_strip_timestamp_no_timestamp_returns_unchanged():
    assert _strip_leading_timestamp("just_a_label") == "just_a_label"


def test_strip_timestamp_empty():
    assert _strip_leading_timestamp("") == ""


# ── write_spec / load_spec round-trip ─────────────────────────────────


def test_write_spec_creates_json_and_md(tmp_path: Path):
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    json_path, md_path = write_spec(
        snapshot=_pin(),
        reference_bundle_name="20260525T202105Z_util0.4_mem16_lru",
        run_dir=run_dir,
        config_notes="LSF cluster",
    )
    assert json_path.exists()
    assert md_path.exists()
    # Both land inside the run dir
    assert json_path.parent == run_dir
    assert md_path.parent == run_dir
    # JSON validates back into BenchSpec
    spec = load_spec(json_path)
    assert spec.snapshot.snapshot_sha.endswith("f")
    assert spec.config.reference_bundle_name == "20260525T202105Z_util0.4_mem16_lru"
    assert spec.config.notes == "LSF cluster"


def test_md_render_contains_critical_fields(tmp_path: Path):
    pin = _pin(
        snapshot_sha="abcdef0123456789" + "0" * 24,
        snapshot_pr_number=99999,
    )
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    _, md_path = write_spec(
        snapshot=pin,
        reference_bundle_name="20260525T202105Z_util0.4_mem16_lru",
        run_dir=run_dir,
    )
    md = md_path.read_text(encoding="utf-8")
    assert "abcdef0123456789" in md  # full SHA
    assert "abcdef0" in md  # short SHA
    assert "PR #99999" in md
    assert "test rationale" in md
    assert "20260525T202105Z_util0.4_mem16_lru" in md
    assert "23 PRs all merged" in md  # n_view formatted into §5


def test_md_render_handles_empty_config_notes(tmp_path: Path):
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    _, md_path = write_spec(
        snapshot=_pin(),
        reference_bundle_name="bundle",
        run_dir=run_dir,
        config_notes="",
    )
    assert "(no extra notes)" in md_path.read_text(encoding="utf-8")


def test_render_md_is_pure(tmp_path: Path):
    """Two calls with same input → byte-identical output."""
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    json_path, _ = write_spec(
        snapshot=_pin(),
        reference_bundle_name="bundle",
        run_dir=run_dir,
    )
    spec = load_spec(json_path)
    a = render_md(spec)
    b = render_md(spec)
    assert a == b


def test_artifacts_named_canonically(tmp_path: Path):
    """JSON is `bench_spec.json`; MD is `OBSERVABILITY_BENCH_SPEC.md`."""
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    json_path, md_path = write_spec(
        snapshot=_pin(),
        reference_bundle_name="bundle",
        run_dir=run_dir,
    )
    assert json_path.name == "bench_spec.json"
    assert md_path.name == "OBSERVABILITY_BENCH_SPEC.md"


def test_load_spec_validates_on_load(tmp_path: Path):
    """Load fails with helpful error if JSON is malformed."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"window_id": "x"}), encoding="utf-8")
    with pytest.raises(Exception):  # pydantic ValidationError
        load_spec(bad)


def test_md_includes_full_sha_not_just_short(tmp_path: Path):
    """Common confusion: the spec must show the FULL SHA, not just first 7."""
    full = "0" * 39 + "f"
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    _, md_path = write_spec(
        snapshot=_pin(snapshot_sha=full),
        reference_bundle_name="bundle",
        run_dir=run_dir,
    )
    md = md_path.read_text(encoding="utf-8")
    # Full SHA appears at least twice (in §1 and in commit URL)
    assert md.count(full) >= 2
