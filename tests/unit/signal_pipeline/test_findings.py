"""Tests for the findings rollup (`signal_pipeline.findings`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.signal_pipeline.findings import emit_findings
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
    layout = RunDirLayout(tmp_path)
    layout.root.mkdir(parents=True, exist_ok=True)
    return layout


def test_no_op_when_candidates_missing(layout: RunDirLayout) -> None:
    """If stage 03 didn't run, rollup is a silent no-op (no files written)."""
    emit_findings(layout)
    assert not (layout.root / "findings.json").exists()
    assert not (layout.root / "findings.md").exists()


def test_emits_findings_with_candidates_only(layout: RunDirLayout) -> None:
    """Stage 03 ran but stage 04 didn't — change column is None."""
    candidates = [_candidate("cand-0001"), _candidate("cand-0002")]
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": candidates}), encoding="utf-8"
    )

    emit_findings(layout)

    findings = json.loads((layout.root / "findings.json").read_text(encoding="utf-8"))
    assert len(findings) == 2
    assert findings[0]["candidate"]["id"] == "cand-0001"
    assert findings[0]["change"] is None
    assert findings[1]["change"] is None

    md = (layout.root / "findings.md").read_text(encoding="utf-8")
    assert "cand-0001" in md
    assert "cand-0002" in md
    # Both candidates should announce missing change spec
    assert md.count("_(stage 04 not run for this candidate)_") == 2


def test_joins_candidates_and_changes(layout: RunDirLayout) -> None:
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

    emit_findings(layout)

    findings = json.loads((layout.root / "findings.json").read_text(encoding="utf-8"))
    assert findings[0]["change"]["mechanism"] == "UNIQUE-MECH-1"
    assert findings[1]["change"]["candidate_ref"] == "cand-0002"

    md = (layout.root / "findings.md").read_text(encoding="utf-8")
    assert "UNIQUE-MECH-1" in md


def test_partial_changes_some_missing(layout: RunDirLayout) -> None:
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

    emit_findings(layout)

    findings = json.loads((layout.root / "findings.json").read_text(encoding="utf-8"))
    assert findings[0]["change"] is not None
    assert findings[1]["change"] is None

    md = (layout.root / "findings.md").read_text(encoding="utf-8")
    assert md.count("_(stage 04 not run for this candidate)_") == 1


def test_idempotent_overwrite(layout: RunDirLayout) -> None:
    """Calling twice with different stage 03 contents overwrites cleanly."""
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": [_candidate("cand-0001")]}), encoding="utf-8"
    )
    emit_findings(layout)
    first_md = (layout.root / "findings.md").read_text(encoding="utf-8")
    assert "cand-0001" in first_md
    assert "cand-0002" not in first_md

    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": [_candidate("cand-0002")]}), encoding="utf-8"
    )
    emit_findings(layout)
    second_md = (layout.root / "findings.md").read_text(encoding="utf-8")
    assert "cand-0002" in second_md
    assert "cand-0001" not in second_md  # no leftover from first run


def test_handles_bare_array_candidates(layout: RunDirLayout) -> None:
    """`03_candidates.json` may be `[...]` directly, not `{candidates: [...]}`."""
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps([_candidate("cand-0001")]), encoding="utf-8"
    )
    emit_findings(layout)
    findings = json.loads((layout.root / "findings.json").read_text(encoding="utf-8"))
    assert len(findings) == 1
    assert findings[0]["candidate"]["id"] == "cand-0001"


def test_summary_table_renders_one_row_per_candidate(layout: RunDirLayout) -> None:
    candidates = [
        _candidate("cand-0001", estimated_impact="high"),
        _candidate("cand-0002", estimated_impact="medium"),
        _candidate("cand-0003", estimated_impact="low"),
    ]
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps({"candidates": candidates}), encoding="utf-8"
    )
    emit_findings(layout)
    md = (layout.root / "findings.md").read_text(encoding="utf-8")
    # Summary section is a single 4-row Markdown table (header + sep + 3 entries)
    summary_lines = [
        line for line in md.splitlines()
        if line.startswith("| cand-")
    ]
    assert len(summary_lines) == 3
    assert "high" in summary_lines[0]
    assert "medium" in summary_lines[1]
    assert "low" in summary_lines[2]
