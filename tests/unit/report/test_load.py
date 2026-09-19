from __future__ import annotations

import pytest

from spotlights_engine.report.render import load

from ._fixtures import write_run


def test_missing_result_json_raises_file_not_found(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "run")


def test_loads_with_every_optional_overlay_absent(tmp_path):
    """The graceful-degradation path: only result.json on disk."""
    d = load(write_run(tmp_path))

    assert d["run_id"] == "run-min-0001"
    assert d["objective"] == "cut median TTFT"
    assert d["ranked"] is False
    # no run_manifest.json
    assert d["models_used"] == []
    # no evolve/ or apply/ trees
    assert d["evolved"] == set()
    assert d["applied"] == set()
    # no findings in this fixture, so the whole research layer is absent
    assert d["research"] is None
    # unranked rows fall back to impact order
    assert [r["id"] for r in d["rows"]] == ["cand-demo-0001", "cand-demo-0002"]
    assert [r["impact"] for r in d["rows"]] == ["high", "low"]
    assert d["rows"][0]["rank"] is None
    assert d["rows"][0]["score"] is None
    assert d["method"] == ""


def test_module_aggregates_come_from_the_final_candidate_set(tmp_path):
    d = load(write_run(tmp_path))

    assert d["status_counts"] == {
        "SUCCEEDED": 1,
        "DEGRADED": 0,
        "FAILED": 0,
        "SKIPPED": 0,
    }
    assert d["modules"] == [{"name": "demo/mod", "status": "SUCCEEDED", "total": 2, "high": 1}]


def test_ranking_overlay_supplies_rank_and_reorders_rows(tmp_path):
    d = load(write_run(tmp_path, ranked=True))

    assert d["ranked"] is True
    assert d["method"] == "weighted-impact-v1"
    # the overlay wins over impact order: by impact these rows would come back
    # ["cand-demo-0001", "cand-demo-0002"], so this pins the rank sort itself
    assert [r["id"] for r in d["rows"]] == ["cand-demo-0002", "cand-demo-0001"]
    assert [r["impact"] for r in d["rows"]] == ["low", "medium"]
    assert d["rows"][0]["rank"] == 1
    assert d["rows"][0]["score"] == 0.91
    assert d["rows"][0]["symbol"] == "lookup_table"
