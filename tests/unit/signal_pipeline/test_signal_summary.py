"""Tests for the signal-pipeline rollup (`signal_pipeline.signal_summary`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.signal_pipeline.signal_summary import emit_signal_summary
from spotlights_engine.signal_pipeline.layout import RunDirLayout


pytestmark = pytest.mark.no_stub_stages


def _candidate(id_: str, **overrides) -> dict:
    base = {
        "id": id_,
        "symbol": "Foo.bar",
        "kind": "method",
        "file": "src/foo.py",
        "line_start": 10,
        "line_end": 30,
        "estimated_impact": "high",
        "description": "Reads cache. Heuristic only.",
    }
    base.update(overrides)
    return base


def _change(id_: str, **overrides) -> dict:
    base = {
        "candidate_ref": id_,
        "change_id": f"chg-{id_}",
        "change_type": "replace",
        "mechanism": "swap A for B",
        "required_changes": "edit foo.py:10-30",
        "expected_effect": "p99 down 50%",
        "evaluation_metric": "decode latency",
    }
    base.update(overrides)
    return base


@pytest.fixture
def layout(tmp_path: Path) -> RunDirLayout:
    layout = RunDirLayout(tmp_path / "run")
    layout.root.mkdir(parents=True, exist_ok=True)
    return layout


@pytest.fixture
def output_folder(layout: RunDirLayout) -> Path:
    """Default rollup target — sibling of the run dir, mirroring runner default."""
    return layout.root / "report"


def test_no_op_when_candidates_missing(
    layout: RunDirLayout, output_folder: Path
) -> None:
    """If stage 03 didn't run, rollup is a silent no-op (no files written)."""
    emit_signal_summary(layout, output_folder)
    assert not (output_folder / "signal_summary.json").exists()
    assert not (output_folder / "signal_summary.md").exists()


def test_emits_summary_with_candidates_only(
    layout: RunDirLayout, output_folder: Path
) -> None:
    """Stage 03 ran but stage 04 didn't — change column is None."""
    candidates = [_candidate("cand-0001"), _candidate("cand-0002")]
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": candidates}), encoding="utf-8"
    )

    emit_signal_summary(layout, output_folder)

    entries = json.loads((output_folder / "signal_summary.json").read_text(encoding="utf-8"))
    assert len(entries) == 2
    assert entries[0]["candidate"]["id"] == "cand-0001"
    assert entries[0]["change"] is None
    assert entries[1]["change"] is None

    md = (output_folder / "signal_summary.md").read_text(encoding="utf-8")
    assert "cand-0001" in md
    assert "cand-0002" in md
    # Both candidates should announce missing change spec
    assert md.count("_(stage 04 not run for this candidate)_") == 2


def test_joins_candidates_and_changes(
    layout: RunDirLayout, output_folder: Path
) -> None:
    """Each candidate joined with its matching cand-XXXX.json change spec."""
    candidates = [_candidate("cand-0001"), _candidate("cand-0002")]
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": candidates}), encoding="utf-8"
    )
    changes_dir = layout.root / "04_changes"
    changes_dir.mkdir()
    (changes_dir / "cand-0001.json").write_text(
        json.dumps(_change("cand-0001", mechanism="UNIQUE-MECH-1")),
        encoding="utf-8",
    )
    (changes_dir / "cand-0002.json").write_text(
        json.dumps(_change("cand-0002")), encoding="utf-8"
    )

    emit_signal_summary(layout, output_folder)

    entries = json.loads((output_folder / "signal_summary.json").read_text(encoding="utf-8"))
    assert entries[0]["change"]["mechanism"] == "UNIQUE-MECH-1"
    assert entries[1]["change"]["candidate_ref"] == "cand-0002"

    md = (output_folder / "signal_summary.md").read_text(encoding="utf-8")
    assert "UNIQUE-MECH-1" in md


def test_partial_changes_some_missing(
    layout: RunDirLayout, output_folder: Path
) -> None:
    """Mixing: cand-0001 has a change spec, cand-0002 doesn't."""
    candidates = [_candidate("cand-0001"), _candidate("cand-0002")]
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": candidates}), encoding="utf-8"
    )
    changes_dir = layout.root / "04_changes"
    changes_dir.mkdir()
    (changes_dir / "cand-0001.json").write_text(
        json.dumps(_change("cand-0001")), encoding="utf-8"
    )

    emit_signal_summary(layout, output_folder)

    entries = json.loads((output_folder / "signal_summary.json").read_text(encoding="utf-8"))
    assert entries[0]["change"] is not None
    assert entries[1]["change"] is None

    md = (output_folder / "signal_summary.md").read_text(encoding="utf-8")
    assert md.count("_(stage 04 not run for this candidate)_") == 1


def test_idempotent_overwrite(
    layout: RunDirLayout, output_folder: Path
) -> None:
    """Calling twice with different stage 03 contents overwrites cleanly."""
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": [_candidate("cand-0001")]}), encoding="utf-8"
    )
    emit_signal_summary(layout, output_folder)
    first_md = (output_folder / "signal_summary.md").read_text(encoding="utf-8")
    assert "cand-0001" in first_md
    assert "cand-0002" not in first_md

    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": [_candidate("cand-0002")]}), encoding="utf-8"
    )
    emit_signal_summary(layout, output_folder)
    second_md = (output_folder / "signal_summary.md").read_text(encoding="utf-8")
    assert "cand-0002" in second_md
    assert "cand-0001" not in second_md  # no leftover from first run


def test_handles_bare_array_candidates(
    layout: RunDirLayout, output_folder: Path
) -> None:
    """`03_candidates.json` may be `[...]` directly, not `{candidates: [...]}`."""
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps([_candidate("cand-0001")]), encoding="utf-8"
    )
    emit_signal_summary(layout, output_folder)
    entries = json.loads((output_folder / "signal_summary.json").read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["candidate"]["id"] == "cand-0001"


def test_summary_table_renders_one_row_per_candidate(
    layout: RunDirLayout, output_folder: Path
) -> None:
    candidates = [
        _candidate("cand-0001", estimated_impact="high"),
        _candidate("cand-0002", estimated_impact="medium"),
        _candidate("cand-0003", estimated_impact="low"),
    ]
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": candidates}), encoding="utf-8"
    )
    emit_signal_summary(layout, output_folder)
    md = (output_folder / "signal_summary.md").read_text(encoding="utf-8")
    # Summary section is a single 4-row Markdown table (header + sep + 3 entries)
    summary_lines = [
        line for line in md.splitlines()
        if line.startswith("| cand-")
    ]
    assert len(summary_lines) == 3
    assert "high" in summary_lines[0]
    assert "medium" in summary_lines[1]
    assert "low" in summary_lines[2]


def test_writes_to_external_folder(
    layout: RunDirLayout, tmp_path: Path
) -> None:
    """`output_folder` can point outside the run dir; folder is created if missing."""
    candidates = [_candidate("cand-0001")]
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": candidates}), encoding="utf-8"
    )
    external = tmp_path / "published" / "report"
    assert not external.exists()
    emit_signal_summary(layout, external)
    assert (external / "signal_summary.json").exists()
    assert (external / "signal_summary.md").exists()
    # Nothing written into the run dir itself.
    assert not (layout.root / "signal_summary.json").exists()
    assert not (layout.root / "signal_summary.md").exists()


def test_markdown_stamps_source_run_and_render_time(
    layout: RunDirLayout, output_folder: Path
) -> None:
    """The rendered markdown stamps the source run path + a UTC timestamp."""
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": [_candidate("cand-0001")]}), encoding="utf-8"
    )
    emit_signal_summary(layout, output_folder)
    md = (output_folder / "signal_summary.md").read_text(encoding="utf-8")
    assert "_Source run:_" in md
    assert str(layout.root) in md
    assert "_Rendered:_" in md
    # ISO-8601 UTC timestamp ends in `+00:00`
    assert "+00:00" in md
