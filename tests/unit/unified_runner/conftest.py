"""Shared fixtures for unified_runner tests."""

from __future__ import annotations

import pytest

from spotlights_engine.schemas.anomaly import Anomaly
from spotlights_engine.schemas.candidate import (
    Candidate,
    CodeLocation,
    CodeSpan,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import RunInfo, SpotlightReport
from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.schemas.proposal import Proposal


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(
            name="demo", summary="x", source_root="", external_dependencies=[]
        ),
        modules=[Module(name="core", path="src/core")],
    )


def _ctx() -> SpotlightContext:
    return SpotlightContext(objective="reduce p99 latency", workload_hints=["batch=8"])


def _candidate(*, id: str, origin: str = "code_agent") -> Candidate:
    return Candidate(
        id=id,
        module_qualified_name="core",
        origin=origin,  # type: ignore[arg-type]
        locations=[
            CodeLocation(
                file="src/core/x.py",
                spans=[
                    CodeSpan(
                        line_start=1, line_end=10, symbol="f", kind="function"
                    )
                ],
            )
        ],
        description="d",
        current_approach="ca",
        evolve_rationale="er",
        estimated_impact="high",
        estimated_impact_explanation="x",
        proposals=[],
    )


def _proposal(*, id: str, source: str = "research_finding") -> Proposal:
    return Proposal(
        id=id,
        source=source,  # type: ignore[arg-type]
        title="t",
        description="d",
        rationale="r",
    )


def _finding(*, id: str = "find-core-0001") -> Finding:
    return Finding(
        finding_id=id,
        title="t",
        url="https://example.com",
        source_type="paper",
        technique_summary="ts",
    )


def _anomaly(*, id: str = "anom-signal-0001") -> Anomaly:
    return Anomaly(anomaly_id=id, type="latency_spike")


def _signal_report(
    *,
    candidates=None,
    anomalies=None,
    cost: float | None = 1.0,
    started_at: str = "2026-06-22T10:00:00+00:00",
    finished_at: str = "2026-06-22T10:30:00+00:00",
    context: SpotlightContext | None = None,
) -> SpotlightReport:
    ctx = context if context is not None else _ctx()
    return SpotlightReport(
        project_tree=_tree(),
        context=ctx,
        candidates=candidates if candidates is not None else [_candidate(id="cand-signal-0001", origin="telemetry_anomaly")],
        findings=[],
        anomalies=anomalies if anomalies is not None else [_anomaly(id="anom-signal-0001")],
        run=RunInfo(
            pipeline="signal",
            run_id="run-sig",
            started_at=started_at,
            finished_at=finished_at,
            cost_usd=cost,
        ),
        issues=[],
    )


def _dr_report(
    *,
    candidates=None,
    findings=None,
    cost: float | None = 2.0,
    started_at: str = "2026-06-22T09:00:00+00:00",
    finished_at: str = "2026-06-22T11:00:00+00:00",
    context: SpotlightContext | None = None,
) -> SpotlightReport:
    ctx = context if context is not None else _ctx()
    return SpotlightReport(
        project_tree=_tree(),
        context=ctx,
        candidates=candidates if candidates is not None else [_candidate(id="cand-core-0001")],
        findings=findings if findings is not None else [_finding(id="find-core-0001")],
        anomalies=[],
        run=RunInfo(
            pipeline="deep_research",
            run_id="run-dr",
            started_at=started_at,
            finished_at=finished_at,
            cost_usd=cost,
        ),
        issues=[],
    )


@pytest.fixture
def signal_report():
    return _signal_report()


@pytest.fixture
def dr_report():
    return _dr_report()


@pytest.fixture
def make_signal_report():
    return _signal_report


@pytest.fixture
def make_dr_report():
    return _dr_report


@pytest.fixture
def project_tree():
    return _tree()


@pytest.fixture
def context():
    return _ctx()


@pytest.fixture
def make_candidate():
    return _candidate


@pytest.fixture
def make_proposal():
    return _proposal


@pytest.fixture
def make_finding():
    return _finding


@pytest.fixture
def make_anomaly():
    return _anomaly
