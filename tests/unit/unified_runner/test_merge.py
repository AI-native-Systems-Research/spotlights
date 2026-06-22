"""Tests for `unified_runner.merge.merge_reports`."""

from __future__ import annotations

import pytest

from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.unified_runner.errors import MergeIdCollisionError
from spotlights_engine.unified_runner.merge import merge_reports


def test_merge_both_pipelines_concatenates_lists(signal_report, dr_report):
    merged = merge_reports(signal=signal_report, dr=dr_report, run_id="run-uni")
    assert {c.id for c in merged.candidates} == {"cand-signal-0001", "cand-core-0001"}
    assert [f.finding_id for f in merged.findings] == ["find-core-0001"]
    assert [a.anomaly_id for a in merged.anomalies] == ["anom-signal-0001"]


def test_merge_sets_pipeline_unified(signal_report, dr_report):
    merged = merge_reports(signal=signal_report, dr=dr_report, run_id="run-uni")
    assert merged.run.pipeline == "unified"
    assert merged.run.run_id == "run-uni"


def test_merge_sums_costs(signal_report, dr_report):
    merged = merge_reports(signal=signal_report, dr=dr_report, run_id="run-uni")
    assert merged.run.cost_usd == pytest.approx(3.0)


def test_merge_cost_none_when_all_none(make_signal_report, make_dr_report):
    sig = make_signal_report(cost=None)
    dr = make_dr_report(cost=None)
    merged = merge_reports(signal=sig, dr=dr, run_id="run-uni")
    assert merged.run.cost_usd is None


def test_merge_cost_partial_none_treated_as_zero(make_signal_report, make_dr_report):
    sig = make_signal_report(cost=None)
    dr = make_dr_report(cost=2.5)
    merged = merge_reports(signal=sig, dr=dr, run_id="run-uni")
    assert merged.run.cost_usd == pytest.approx(2.5)


def test_merge_started_at_is_earliest(make_signal_report, make_dr_report):
    sig = make_signal_report(started_at="2026-06-22T10:00:00+00:00")
    dr = make_dr_report(started_at="2026-06-22T09:00:00+00:00")
    merged = merge_reports(signal=sig, dr=dr, run_id="run-uni")
    assert merged.run.started_at == "2026-06-22T09:00:00+00:00"


def test_merge_finished_at_is_latest(make_signal_report, make_dr_report):
    sig = make_signal_report(finished_at="2026-06-22T10:30:00+00:00")
    dr = make_dr_report(finished_at="2026-06-22T11:00:00+00:00")
    merged = merge_reports(signal=sig, dr=dr, run_id="run-uni")
    assert merged.run.finished_at == "2026-06-22T11:00:00+00:00"


def test_merge_signal_only_passes_through(signal_report):
    merged = merge_reports(signal=signal_report, dr=None, run_id="run-uni")
    assert {c.id for c in merged.candidates} == {"cand-signal-0001"}
    assert merged.findings == []
    assert merged.run.pipeline == "unified"
    assert merged.run.cost_usd == pytest.approx(1.0)


def test_merge_dr_only_passes_through(dr_report):
    merged = merge_reports(signal=None, dr=dr_report, run_id="run-uni")
    assert [c.id for c in merged.candidates] == ["cand-core-0001"]
    assert [f.finding_id for f in merged.findings] == ["find-core-0001"]
    assert merged.run.pipeline == "unified"


def test_merge_requires_at_least_one_report():
    with pytest.raises(ValueError, match="at least one"):
        merge_reports(signal=None, dr=None, run_id="run-uni")


def test_merge_raises_on_candidate_id_collision(make_signal_report, make_dr_report, make_candidate):
    # Force the DR side to use a `signal` segment so it collides with the signal pipeline.
    dr = make_dr_report(candidates=[make_candidate(id="cand-signal-0001")])
    sig = make_signal_report(candidates=[make_candidate(id="cand-signal-0001", origin="telemetry_anomaly")])
    with pytest.raises(MergeIdCollisionError, match="duplicate candidate id"):
        merge_reports(signal=sig, dr=dr, run_id="run-uni")


def test_merge_raises_on_anomaly_id_collision(make_signal_report, make_dr_report, make_anomaly):
    sig = make_signal_report(anomalies=[make_anomaly(id="anom-x-0001")])
    dr = make_dr_report()
    # Manually inject a colliding anomaly into DR (which normally has none).
    dr.anomalies.append(make_anomaly(id="anom-x-0001"))
    with pytest.raises(MergeIdCollisionError, match="duplicate anomaly id"):
        merge_reports(signal=sig, dr=dr, run_id="run-uni")


def test_merge_rejects_divergent_context(make_signal_report, make_dr_report):
    sig = make_signal_report(context=SpotlightContext(objective="reduce p99"))
    dr = make_dr_report(context=SpotlightContext(objective="something else"))
    with pytest.raises(ValueError, match="context differs"):
        merge_reports(signal=sig, dr=dr, run_id="run-uni")


def test_merge_picks_dr_project_tree(signal_report, dr_report):
    merged = merge_reports(signal=signal_report, dr=dr_report, run_id="run-uni")
    # By construction equal, so this just sanity-checks the pick.
    assert merged.project_tree == dr_report.project_tree


def test_merge_validates_via_spotlight_report_schema(signal_report, dr_report):
    """The merged object should pass `SpotlightReport.model_validate_json`."""
    merged = merge_reports(signal=signal_report, dr=dr_report, run_id="run-uni")
    raw = merged.model_dump_json()
    from spotlights_engine.schemas.pipeline import SpotlightReport

    SpotlightReport.model_validate_json(raw)
