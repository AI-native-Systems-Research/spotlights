"""Unit tests for `schemas.spotlight_report`."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.spotlight_report import (
    CodeSpan,
    Location,
    RunInfo,
    SpotlightAnomaly,
    SpotlightCandidate,
    SpotlightFinding,
    SpotlightProposal,
    SpotlightReport,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _location() -> Location:
    return Location(
        file="vllm/v1/kv_offload/cpu/manager.py",
        spans=[CodeSpan(line_start=19, line_end=22, symbol="_CACHE_POLICIES")],
    )


def _candidate(**overrides: object) -> SpotlightCandidate:
    base: dict[str, object] = dict(
        id="cand-0001",
        module_qualified_name="v1.kv_offload",
        origin="code_agent",
        locations=[_location()],
        kind="plugin_seam",
        description="d",
        current_approach="ca",
        evolve_rationale="er",
        estimated_impact="high",
        estimated_impact_explanation="eie",
    )
    base.update(overrides)
    return SpotlightCandidate(**base)  # type: ignore[arg-type]


def _proposal(**overrides: object) -> SpotlightProposal:
    base: dict[str, object] = dict(
        id="prop-0001",
        source="research_finding",
        title="t",
        description="d",
        rationale="r",
    )
    base.update(overrides)
    return SpotlightProposal(**base)  # type: ignore[arg-type]


def _finding(**overrides: object) -> SpotlightFinding:
    base: dict[str, object] = dict(
        finding_id="find-0001",
        module_qualified_name="v1.kv_offload",
        title="t",
        url="https://example.com/x",
        source_type="paper",
        technique_summary="ts",
    )
    base.update(overrides)
    return SpotlightFinding(**base)  # type: ignore[arg-type]


def _report(**overrides: object) -> SpotlightReport:
    base: dict[str, object] = dict(
        project_tree={"repository": {"name": "vllm", "summary": "s"}, "modules": []},
        context={"objective": "reduce TTFT"},
        run=RunInfo(
            pipeline="deep_research",
            run_id="run-1",
            started_at="2026-06-14T00:00:00Z",
        ),
    )
    base.update(overrides)
    return SpotlightReport(**base)  # type: ignore[arg-type]


# ── extra="forbid" ───────────────────────────────────────────────────────────


def test_extra_fields_rejected_on_top_level() -> None:
    with pytest.raises(ValidationError):
        SpotlightReport.model_validate(
            {
                "schema_version": "1",
                "project_tree": {
                    "repository": {"name": "v", "summary": "s"},
                    "modules": [],
                },
                "context": {"objective": "o"},
                "run": {
                    "pipeline": "deep_research",
                    "run_id": "r",
                    "started_at": "2026-06-14T00:00:00Z",
                },
                "extra_field": "nope",
            }
        )


def test_extra_fields_rejected_on_candidate() -> None:
    with pytest.raises(ValidationError):
        SpotlightCandidate.model_validate(
            {
                **_candidate().model_dump(),
                "extra_field": "nope",
            }
        )


def test_extra_fields_rejected_on_proposal() -> None:
    with pytest.raises(ValidationError):
        SpotlightProposal.model_validate(
            {**_proposal().model_dump(), "extra_field": "nope"}
        )


def test_extra_fields_allowed_on_anomaly() -> None:
    # SpotlightAnomaly intentionally uses extra="allow" so future signal-side
    # fields don't require a schema bump.
    a = SpotlightAnomaly.model_validate(
        {"anomaly_id": "a-1", "type": "hot_path", "extra": "kept"}
    )
    assert getattr(a, "extra", None) == "kept"


# ── id patterns ──────────────────────────────────────────────────────────────


def test_candidate_id_pattern() -> None:
    _candidate(id="cand-0042")
    with pytest.raises(ValidationError):
        _candidate(id="cand-1")
    with pytest.raises(ValidationError):
        _candidate(id="candidate-0001")


def test_proposal_id_pattern() -> None:
    _proposal(id="prop-9999")
    with pytest.raises(ValidationError):
        _proposal(id="prop-1")
    with pytest.raises(ValidationError):
        _proposal(id="proposal-0001")


def test_finding_id_pattern() -> None:
    _finding(finding_id="find-0008")
    with pytest.raises(ValidationError):
        _finding(finding_id="find-1")


# ── Literal acceptance ──────────────────────────────────────────────────────


@pytest.mark.parametrize("origin", ["telemetry_anomaly", "code_agent"])
def test_candidate_origin_accepts_each_value(origin: str) -> None:
    _candidate(origin=origin)


def test_candidate_origin_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        _candidate(origin="github_signal")


@pytest.mark.parametrize(
    "source", ["research_finding", "agent_knowledge", "telemetry_anomaly"]
)
def test_proposal_source_accepts_each_value(source: str) -> None:
    _proposal(source=source)


def test_proposal_source_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        _proposal(source="hallucination")


@pytest.mark.parametrize(
    "ptype",
    ["prefetch", "reorder", "replace", "tune", "add_cache", "fuse", "other"],
)
def test_proposal_type_accepts_each_value(ptype: str) -> None:
    _proposal(proposal_type=ptype)


def test_proposal_type_accepts_none() -> None:
    _proposal(proposal_type=None)


def test_proposal_type_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        _proposal(proposal_type="rewrite")


@pytest.mark.parametrize("pipeline", ["deep_research", "signal"])
def test_run_pipeline_accepts_each_value(pipeline: str) -> None:
    RunInfo(pipeline=pipeline, run_id="r", started_at="2026-06-14T00:00:00Z")  # type: ignore[arg-type]


def test_run_pipeline_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        RunInfo(pipeline="other", run_id="r", started_at="2026-06-14T00:00:00Z")  # type: ignore[arg-type]


def test_schema_version_accepts_only_1() -> None:
    _report()
    with pytest.raises(ValidationError):
        _report(schema_version="2")


# ── module_qualified_name str | None ────────────────────────────────────────


def test_candidate_module_qualified_name_optional() -> None:
    c = _candidate(module_qualified_name=None)
    assert c.module_qualified_name is None
    c = _candidate(module_qualified_name="v1.kv_offload")
    assert c.module_qualified_name == "v1.kv_offload"


def test_finding_module_qualified_name_required_nonempty() -> None:
    with pytest.raises(ValidationError):
        _finding(module_qualified_name=None)
    with pytest.raises(ValidationError):
        _finding(module_qualified_name="")


# ── min_length=1 on locations / spans ───────────────────────────────────────


def test_candidate_locations_min_length_1() -> None:
    with pytest.raises(ValidationError):
        _candidate(locations=[])


def test_location_spans_min_length_1() -> None:
    with pytest.raises(ValidationError):
        Location(file="x.py", spans=[])


def test_codespan_symbol_min_length_1() -> None:
    with pytest.raises(ValidationError):
        CodeSpan(line_start=1, line_end=1, symbol="")


# ── canonical-shape dict round-trips cleanly ────────────────────────────────


def test_canonical_shape_validates() -> None:
    payload: dict[str, object] = {
        "schema_version": "1",
        "project_tree": {
            "repository": {"name": "vllm", "summary": "s"},
            "modules": [],
        },
        "context": {"objective": "reduce TTFT"},
        "candidates": [
            {
                "id": "cand-0001",
                "module_qualified_name": "v1.kv_offload",
                "origin": "code_agent",
                "locations": [
                    {
                        "file": "a.py",
                        "spans": [
                            {"line_start": 1, "line_end": 5, "symbol": "f"}
                        ],
                    }
                ],
                "kind": "function",
                "description": "d",
                "current_approach": "ca",
                "evolve_rationale": "er",
                "estimated_impact": "low",
                "estimated_impact_explanation": "eie",
                "state": "DISCOVERED",
                "proposals": [
                    {
                        "id": "prop-0001",
                        "source": "agent_knowledge",
                        "source_refs": [],
                        "author": "claude",
                        "title": "t",
                        "description": "d",
                        "rationale": "r",
                        "proposal_type": None,
                        "mechanism": None,
                        "required_changes": None,
                        "expected_effect": None,
                        "evaluation_metric": None,
                    }
                ],
            }
        ],
        "findings": [
            {
                "finding_id": "find-0001",
                "module_qualified_name": "v1.kv_offload",
                "title": "t",
                "url": "https://example.com/x",
                "source_type": "paper",
                "technique_summary": "ts",
                "supporting_evidence": "",
            }
        ],
        "anomalies": [],
        "run": {
            "pipeline": "deep_research",
            "run_id": "run-1",
            "started_at": "2026-06-14T00:00:00Z",
            "finished_at": None,
            "model": None,
            "cost_usd": None,
            "duration_s": None,
            "parameters": {},
        },
        "issues": [],
    }
    SpotlightReport.model_validate(payload)


# ── frozen-fixture round-trip (backstop for §4.8 schema-version policy) ─────


def _frozen_payload() -> dict[str, object]:
    return {
        "schema_version": "1",
        "project_tree": {
            "repository": {"name": "vllm", "summary": "vLLM subset for testing"},
            "modules": [
                {
                    "name": "v1",
                    "path": "vllm/v1",
                    "description": "",
                    "depends_on": [],
                    "main_files": [],
                    "submodules": [],
                }
            ],
        },
        "context": {
            "objective": "reduce the media TTFT and median TPOT",
            "workload_hints": ["multi-turn agentic workload"],
            "validation_plan": [],
        },
        "candidates": [
            {
                "id": "cand-0001",
                "module_qualified_name": "v1.kv_offload",
                "origin": "code_agent",
                "locations": [
                    {
                        "file": "vllm/v1/kv_offload/cpu/manager.py",
                        "spans": [
                            {
                                "line_start": 19,
                                "line_end": 22,
                                "symbol": "_CACHE_POLICIES",
                            }
                        ],
                    }
                ],
                "kind": "plugin_seam",
                "description": "Registration table mapping eviction-policy strings.",
                "current_approach": "Exposes lru -> LRUCachePolicy.",
                "evolve_rationale": "A replacement policy is a contained extension.",
                "estimated_impact": "high",
                "estimated_impact_explanation": "Eviction policy controls hit rate.",
                "state": "AGENT_PROPOSALS_CREATED",
                "proposals": [
                    {
                        "id": "prop-0001",
                        "source": "research_finding",
                        "source_refs": ["find-0001"],
                        "author": None,
                        "title": "Add S3-FIFO CachePolicy",
                        "description": "Implement S3FIFOCachePolicy.",
                        "rationale": "The candidate is a plugin seam.",
                        "proposal_type": None,
                        "mechanism": None,
                        "required_changes": None,
                        "expected_effect": None,
                        "evaluation_metric": None,
                    },
                    {
                        "id": "prop-0002",
                        "source": "agent_knowledge",
                        "source_refs": [],
                        "author": "claude",
                        "title": "Prefix-chain-aware policy",
                        "description": "Implement PrefixChainCachePolicy.",
                        "rationale": "Novel: not covered by literature.",
                        "proposal_type": None,
                        "mechanism": None,
                        "required_changes": None,
                        "expected_effect": None,
                        "evaluation_metric": None,
                    },
                ],
            }
        ],
        "findings": [
            {
                "finding_id": "find-0001",
                "module_qualified_name": "v1.kv_offload",
                "title": "S3-FIFO",
                "url": "https://example.com/s3fifo",
                "source_type": "paper",
                "technique_summary": "Probationary FIFO admission.",
                "supporting_evidence": "",
            }
        ],
        "anomalies": [],
        "run": {
            "pipeline": "deep_research",
            "run_id": "vllm_subset_2026-06-14",
            "started_at": "2026-06-14T00:00:00Z",
            "finished_at": "2026-06-14T00:05:00Z",
            "model": "claude-opus-4-7[1m]",
            "cost_usd": 1.5969,
            "duration_s": 291.59,
            "parameters": {
                "max_findings_per_module": 30,
                "continue_on_module_failure": True,
            },
        },
        "issues": [],
    }


def test_frozen_fixture_json_round_trip() -> None:
    """Backstop for the schema-version policy in spec §4.8.

    A dict that validates today must validate after re-serialization to JSON
    and re-loading. Failures here flag accidental breaking changes to v1
    that should have come with a schema_version bump.
    """
    payload = _frozen_payload()
    report = SpotlightReport.model_validate(payload)
    serialized = report.model_dump_json()
    reloaded = SpotlightReport.model_validate(json.loads(serialized))
    assert reloaded == report
    assert reloaded.model_dump(mode="json") == report.model_dump(mode="json")
