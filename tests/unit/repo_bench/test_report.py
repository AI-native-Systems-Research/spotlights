"""Run report — JSON canonical + MD rendered from JSON."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spotlights_engine.repo_bench.report import (
    make_run_id,
    render_md,
    write_report,
)
from spotlights_engine.repo_bench.schemas import (
    RunReport,
    SnapshotPin,
    StageReport,
)


def _pin() -> SnapshotPin:
    return SnapshotPin(
        window_id="2026-06-01__2026-06-10",
        view_id="abc12345",
        snapshot_sha="0" * 39 + "f",
        snapshot_pr_number=12345,
        snapshot_merged_at=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        earliest_in_view_pr_number=67890,
        earliest_in_view_merged_at=datetime(2026, 6, 3, 12, tzinfo=timezone.utc),
        n_view=23,
        rationale="test rationale",
        pinned_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
    )


def _stage(stage: str, status: str = "done", **counts) -> StageReport:
    return StageReport(stage=stage, status=status, counts=counts)


# ── make_run_id ──────────────────────────────────────────────────────


def test_make_run_id_format():
    rid = make_run_id("win-2026-06-01", "view-abc")
    # `bench-<window>__<view>__<UTC-timestamp>`
    assert rid.startswith("bench-")
    parts = rid[len("bench-"):].split("__")
    assert parts[0] == "win-2026-06-01"
    assert parts[1] == "view-abc"
    assert re.fullmatch(r"\d{8}T\d{6}Z", parts[2])


# ── write_report ──────────────────────────────────────────────────────


def test_write_report_creates_json_and_md(tmp_path: Path):
    started = datetime(2026, 6, 4, 9, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 6, 4, 9, 30, tzinfo=timezone.utc)
    json_path, md_path = write_report(
        run_id="r1",
        window_id="w1", view_id="v1",
        started_at=started, finished_at=finished,
        stages=[
            _stage("aggregate", "skipped"),
            _stage("filter", "done", n_input=100, n_kept=24),
        ],
        snapshot=_pin(),
        bench_spec_path=tmp_path / "spec.md",
        out_root=tmp_path,
    )
    assert json_path.exists()
    assert md_path.exists()
    # JSON validates
    data = json.loads(json_path.read_text(encoding="utf-8"))
    report = RunReport.model_validate(data)
    assert report.run_id == "r1"
    assert len(report.stages) == 2


def test_md_includes_stage_table(tmp_path: Path):
    _, md_path = write_report(
        run_id="r1", window_id="w1", view_id="v1",
        started_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        finished_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        stages=[
            _stage("aggregate", "skipped"),
            _stage("filter", "done", n_input=100, n_kept=24),
            _stage("snapshot", "done"),
        ],
        snapshot=_pin(),
        bench_spec_path=tmp_path / "spec.md",
        out_root=tmp_path,
    )
    md = md_path.read_text(encoding="utf-8")
    # Stage table rows
    assert "| 1 | `aggregate` |" in md
    assert "| 2 | `filter` |" in md
    assert "| 3 | `snapshot` |" in md
    # Status markers
    assert "[--] skipped" in md
    assert "[OK] done" in md


def test_md_includes_summary_extras_for_known_counts(tmp_path: Path):
    _, md_path = write_report(
        run_id="r1", window_id="w1", view_id="v1",
        started_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        finished_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        stages=[
            _stage("filter", "done", n_input=5329, n_kept=24),
            _stage("snapshot", "done"),
        ],
        snapshot=_pin(),
        bench_spec_path=tmp_path / "spec.md",
        out_root=tmp_path,
    )
    md = md_path.read_text(encoding="utf-8")
    assert "| Filtered PRs | 24 kept of 5329 |" in md
    # snapshot SHA short form should appear
    assert "| Snapshot SHA |" in md


def test_md_status_ok_when_all_done_or_skipped(tmp_path: Path):
    _, md_path = write_report(
        run_id="r1", window_id="w1", view_id="v1",
        started_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        finished_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        stages=[_stage("aggregate", "done"), _stage("filter", "skipped")],
        snapshot=_pin(),
        bench_spec_path=tmp_path / "spec.md",
        out_root=tmp_path,
    )
    assert "**Status**: OK" in md_path.read_text(encoding="utf-8")


def test_md_status_failed_when_any_stage_failed(tmp_path: Path):
    _, md_path = write_report(
        run_id="r1", window_id="w1", view_id="v1",
        started_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        finished_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        stages=[_stage("aggregate", "done"), _stage("filter", "failed")],
        snapshot=None,
        bench_spec_path=None,
        out_root=tmp_path,
    )
    assert "**Status**: FAILED" in md_path.read_text(encoding="utf-8")


def test_md_handles_no_snapshot(tmp_path: Path):
    """Run that crashed before snapshot stage."""
    _, md_path = write_report(
        run_id="r1", window_id="w1", view_id="v1",
        started_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        finished_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        stages=[_stage("aggregate", "failed")],
        snapshot=None,
        bench_spec_path=None,
        out_root=tmp_path,
    )
    md = md_path.read_text(encoding="utf-8")
    assert "_(not produced — run did not reach the snapshot stage)_" in md
    assert "_(not produced — run did not reach the bench-spec stage)_" in md


def test_default_next_steps_when_snapshot_present(tmp_path: Path):
    _, md_path = write_report(
        run_id="r1", window_id="w1", view_id="v1",
        started_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        finished_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        stages=[_stage("snapshot", "done")],
        snapshot=_pin(),
        bench_spec_path=tmp_path / "spec.md",
        out_root=tmp_path,
    )
    md = md_path.read_text(encoding="utf-8")
    # Default next-steps include the bench-module handoff
    assert "Hand the bench spec" in md
    assert "repo-bench match" in md


def test_render_md_pure(tmp_path: Path):
    """Same input → same output."""
    json_path, _ = write_report(
        run_id="r1", window_id="w1", view_id="v1",
        started_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        finished_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        stages=[_stage("filter", "done", n_input=10, n_kept=5)],
        snapshot=_pin(),
        bench_spec_path=tmp_path / "spec.md",
        out_root=tmp_path,
    )
    report = RunReport.model_validate(
        json.loads(json_path.read_text(encoding="utf-8"))
    )
    a = render_md(report)
    b = render_md(report)
    assert a == b


def test_report_dir_under_runs_root(tmp_path: Path):
    """Layout: <runs_root>/<run_id>/{run_report.json,run_report.md}.

    `out_root` here overrides the runs-root, not the data-root — runs
    are output, not cached input.
    """
    json_path, md_path = write_report(
        run_id="some-run-id",
        window_id="w1", view_id="v1",
        started_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        finished_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        stages=[_stage("filter", "done")],
        snapshot=_pin(),
        bench_spec_path=tmp_path / "spec.md",
        out_root=tmp_path,
    )
    assert json_path.parent == tmp_path / "some-run-id"
    assert md_path.parent == tmp_path / "some-run-id"
    assert json_path.name == "run_report.json"
    assert md_path.name == "run_report.md"
