"""Schema-level tests for the unified `SpotlightReport` and its parts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.candidate import (
    Candidate,
    CandidateOrigin,
    CodeKind,
    CodeLocation,
    CodeSpan,
    EstimatedImpact,
)
from spotlights_engine.schemas.common import SpotlightContext, StepIssue
from spotlights_engine.schemas.pipeline import RunInfo, SpotlightReport
from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.schemas.proposal import Proposal, ProposalSource

FIXTURE = Path(__file__).parent / "fixtures" / "spotlight_report_v1.json"


def _span(**overrides) -> CodeSpan:
    payload = {"line_start": 1, "line_end": 2, "symbol": "f", "kind": "function"}
    payload.update(overrides)
    return CodeSpan.model_validate(payload)


def _location(**overrides) -> CodeLocation:
    payload = {"file": "src/x.py", "spans": [_span().model_dump()]}
    payload.update(overrides)
    return CodeLocation.model_validate(payload)


def _candidate(**overrides) -> Candidate:
    payload = {
        "id": "cand-0001",
        "module_qualified_name": "core",
        "origin": "code_agent",
        "locations": [_location().model_dump()],
        "description": "d",
        "current_approach": "ca",
        "evolve_rationale": "er",
        "estimated_impact": "high",
        "estimated_impact_explanation": "x",
    }
    payload.update(overrides)
    return Candidate.model_validate(payload)


def _proposal(**overrides) -> Proposal:
    payload = {
        "id": "prop-0001",
        "source": "research_finding",
        "finding_ref_id": "find-0001",
        "title": "t",
        "description": "d",
        "rationale": "r",
    }
    payload.update(overrides)
    return Proposal.model_validate(payload)


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="demo", summary="x"),
        modules=[Module(name="core", path="src/core")],
    )


def _ctx() -> SpotlightContext:
    return SpotlightContext(objective="o")


def _run() -> RunInfo:
    return RunInfo(pipeline="deep_research", run_id="r1", started_at="2026-06-18T00:00:00Z")


# extra="forbid" rejection ----------------------------------------------------

@pytest.mark.parametrize(
    "model_cls, payload",
    [
        (CodeSpan, {"line_start": 1, "line_end": 2, "symbol": "f", "kind": "function", "extra": "x"}),
        (CodeLocation, {"file": "x.py", "spans": [_span().model_dump()], "extra": "x"}),
        (
            Candidate,
            {
                "id": "cand-0001",
                "origin": "code_agent",
                "locations": [_location().model_dump()],
                "description": "d",
                "current_approach": "ca",
                "evolve_rationale": "er",
                "estimated_impact": "high",
                "estimated_impact_explanation": "x",
                "extra": "x",
            },
        ),
        (
            Proposal,
            {
                "id": "prop-0001",
                "source": "research_finding",
                "title": "t",
                "description": "d",
                "rationale": "r",
                "extra": "x",
            },
        ),
        (
            RunInfo,
            {"pipeline": "signal", "run_id": "r", "started_at": "t", "extra": "x"},
        ),
        (
            SpotlightReport,
            {
                "project_tree": _tree().model_dump(),
                "context": _ctx().model_dump(),
                "run": _run().model_dump(),
                "extra": "x",
            },
        ),
    ],
)
def test_extra_forbid_rejects_unknown_fields(model_cls, payload) -> None:
    with pytest.raises(ValidationError):
        model_cls.model_validate(payload)


# id pattern validation -------------------------------------------------------

@pytest.mark.parametrize("bad_id", ["cand-1", "cand-00001", "Cand-0001", "cand_0001", "candidate-0001"])
def test_candidate_id_pattern_rejects_bad_ids(bad_id) -> None:
    with pytest.raises(ValidationError):
        _candidate(id=bad_id)


def test_candidate_id_pattern_accepts_four_digits() -> None:
    assert _candidate(id="cand-0042").id == "cand-0042"


@pytest.mark.parametrize("bad_id", ["prop-1", "prop-00001", "Prop-0001", "prop_0001", "proposal-0001"])
def test_proposal_id_pattern_rejects_bad_ids(bad_id) -> None:
    with pytest.raises(ValidationError):
        _proposal(id=bad_id)


def test_proposal_id_pattern_accepts_four_digits() -> None:
    assert _proposal(id="prop-0042").id == "prop-0042"


# Literal value sets ----------------------------------------------------------

CODE_KIND_VALUES: tuple[CodeKind, ...] = (
    "function", "method", "loop", "region", "kernel", "config_block", "plugin_seam",
)


@pytest.mark.parametrize("value", CODE_KIND_VALUES)
def test_code_kind_accepts_each_member(value) -> None:
    assert _span(kind=value).kind == value


def test_code_kind_rejects_non_member() -> None:
    with pytest.raises(ValidationError):
        _span(kind="not_a_kind")


CANDIDATE_ORIGIN_VALUES: tuple[CandidateOrigin, ...] = ("telemetry_anomaly", "code_agent")


@pytest.mark.parametrize("value", CANDIDATE_ORIGIN_VALUES)
def test_candidate_origin_accepts_each_member(value) -> None:
    assert _candidate(origin=value).origin == value


def test_candidate_origin_rejects_non_member() -> None:
    with pytest.raises(ValidationError):
        _candidate(origin="github_signal")


ESTIMATED_IMPACT_VALUES: tuple[EstimatedImpact, ...] = ("high", "medium", "low")


@pytest.mark.parametrize("value", ESTIMATED_IMPACT_VALUES)
def test_estimated_impact_accepts_each_member(value) -> None:
    assert _candidate(estimated_impact=value).estimated_impact == value


def test_estimated_impact_rejects_non_member() -> None:
    with pytest.raises(ValidationError):
        _candidate(estimated_impact="critical")


PROPOSAL_SOURCE_VALUES: tuple[ProposalSource, ...] = (
    "research_finding", "agent_knowledge", "telemetry_anomaly",
)


@pytest.mark.parametrize("value", PROPOSAL_SOURCE_VALUES)
def test_proposal_source_accepts_each_member(value) -> None:
    assert _proposal(source=value).source == value


def test_proposal_source_rejects_non_member() -> None:
    with pytest.raises(ValidationError):
        _proposal(source="github_signal")


@pytest.mark.parametrize("value", ["deep_research", "signal"])
def test_run_info_pipeline_accepts_each_member(value) -> None:
    assert RunInfo(pipeline=value, run_id="r", started_at="t").pipeline == value


def test_run_info_pipeline_rejects_non_member() -> None:
    with pytest.raises(ValidationError):
        RunInfo(pipeline="unified", run_id="r", started_at="t")


def test_schema_version_accepts_one() -> None:
    rep = SpotlightReport.model_validate(json.loads(FIXTURE.read_text()))
    assert rep.schema_version == "1"


def test_schema_version_rejects_other_values() -> None:
    payload = json.loads(FIXTURE.read_text())
    payload["schema_version"] = "2"
    with pytest.raises(ValidationError):
        SpotlightReport.model_validate(payload)


# module_qualified_name nullability ------------------------------------------

def test_module_qualified_name_accepts_none() -> None:
    assert _candidate(module_qualified_name=None).module_qualified_name is None


def test_module_qualified_name_accepts_string() -> None:
    assert _candidate(module_qualified_name="core/util").module_qualified_name == "core/util"


# min_length on locations and spans ------------------------------------------

def test_candidate_rejects_empty_locations() -> None:
    with pytest.raises(ValidationError):
        _candidate(locations=[])


def test_code_location_rejects_empty_spans() -> None:
    with pytest.raises(ValidationError):
        _location(spans=[])


# JSON round-trip backstop (frozen fixture) ----------------------------------

def test_fixture_round_trips() -> None:
    raw = FIXTURE.read_text()
    rep = SpotlightReport.model_validate_json(raw)
    re_dumped = rep.model_dump_json()
    assert SpotlightReport.model_validate_json(re_dumped) == rep


def test_fixture_dump_matches_canonical_payload() -> None:
    payload = json.loads(FIXTURE.read_text())
    rep = SpotlightReport.model_validate(payload)
    assert json.loads(rep.model_dump_json()) == payload


# forward-reference resolution ------------------------------------------------

def test_candidate_proposals_field_resolves_to_list_of_proposal() -> None:
    annotation = Candidate.model_fields["proposals"].annotation
    assert annotation == list[Proposal]


# StepIssue still flows through report.issues --------------------------------

def test_report_issues_accepts_step_issue() -> None:
    issue = StepIssue(step="candidate_discovery", severity="warning", message="m")
    rep = SpotlightReport(
        project_tree=_tree(),
        context=_ctx(),
        candidates=[],
        run=_run(),
        issues=[issue],
    )
    assert rep.issues[0].step == "candidate_discovery"
