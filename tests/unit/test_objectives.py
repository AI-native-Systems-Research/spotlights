"""Unit tests for the objective setting module (Bundle G)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from spotlights_engine.objectives import (
    assemble_proposal,
    build_intent,
    finalize_objective,
)
from spotlights_engine.objectives.schemas import (
    Objective,
    ObjectiveIntent,
    ObjectiveProposal,
)


class TestBuildIntent:
    def test_valid_intent(self):
        intent = build_intent(
            target_metric="request_latency_p99",
            target_direction="minimize",
            workload_classes=["agentic"],
        )
        assert intent.target_metric == "request_latency_p99"
        assert intent.target_direction == "minimize"
        assert intent.workload_classes == ["agentic"]
        assert intent.priorities == []
        assert intent.notes == ""

    def test_all_fields(self):
        intent = build_intent(
            target_metric="throughput_rps",
            target_direction="maximize",
            workload_classes=["batch-inference", "mixed"],
            priorities=["memory_efficiency", "correctness_coverage"],
            notes="Q3 goal from leadership",
        )
        assert intent.target_direction == "maximize"
        assert len(intent.workload_classes) == 2
        assert intent.priorities == ["memory_efficiency", "correctness_coverage"]
        assert intent.notes == "Q3 goal from leadership"

    def test_empty_metric_rejected(self):
        with pytest.raises(ValidationError):
            build_intent(
                target_metric="",
                target_direction="minimize",
                workload_classes=["agentic"],
            )

    def test_empty_workload_classes_rejected(self):
        with pytest.raises(ValidationError):
            build_intent(
                target_metric="latency",
                target_direction="minimize",
                workload_classes=[],
            )

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValidationError):
            build_intent(
                target_metric="latency",
                target_direction="sideways",
                workload_classes=["agentic"],
            )


class TestAssembleProposal:
    def test_produces_proposal(self):
        intent = build_intent(
            target_metric="request_latency_p99",
            target_direction="minimize",
            workload_classes=["agentic"],
        )
        proposal = assemble_proposal(intent)
        assert isinstance(proposal, ObjectiveProposal)
        assert proposal.intent is intent
        assert len(proposal.rationale) > 0
        assert "request_latency_p99" in proposal.rationale
        assert proposal.requires_approval is True

    def test_rationale_mentions_direction(self):
        intent = build_intent(
            target_metric="throughput_rps",
            target_direction="maximize",
            workload_classes=["batch-inference"],
        )
        proposal = assemble_proposal(intent)
        assert "increasing" in proposal.rationale

    def test_priorities_produce_coverage_note(self):
        intent = build_intent(
            target_metric="latency",
            target_direction="minimize",
            workload_classes=["agentic"],
            priorities=["memory_efficiency"],
        )
        proposal = assemble_proposal(intent)
        assert len(proposal.coverage_notes) > 0
        assert "memory_efficiency" in proposal.coverage_notes[0]


class TestFinalizeObjective:
    def test_produces_frozen_objective(self):
        intent = build_intent(
            target_metric="request_latency_p99",
            target_direction="minimize",
            workload_classes=["agentic"],
        )
        proposal = assemble_proposal(intent)
        objective = finalize_objective(
            proposal, session_id="sess-001", approved_by="alice"
        )
        assert isinstance(objective, Objective)
        assert objective.frozen is True
        assert objective.session_id == "sess-001"
        assert objective.approved_by == "alice"
        assert objective.intent == intent
        assert objective.rationale == proposal.rationale
        assert len(objective.objective_id) > 0
        assert objective.created_at is not None

    def test_objective_is_immutable(self):
        intent = build_intent(
            target_metric="latency",
            target_direction="minimize",
            workload_classes=["agentic"],
        )
        proposal = assemble_proposal(intent)
        objective = finalize_objective(
            proposal, session_id="sess-002", approved_by="bob"
        )
        with pytest.raises(ValidationError):
            objective.approved_by = "mallory"  # type: ignore[misc]


class TestRoleAndComponents:
    def test_role_defaults_to_pm(self):
        intent = build_intent(
            target_metric="latency",
            target_direction="minimize",
            workload_classes=["agentic"],
        )
        assert intent.role == "pm"

    def test_developer_role(self):
        intent = build_intent(
            target_metric="kv_cache_hit_rate",
            target_direction="maximize",
            workload_classes=["long-context"],
            target_components=["kv_cache", "memory_allocator"],
            role="developer",
        )
        assert intent.role == "developer"
        assert intent.target_components == ["kv_cache", "memory_allocator"]

    def test_old_json_without_role_defaults_to_pm(self):
        old_json = (
            '{"target_metric": "latency", "target_direction": "minimize",'
            ' "workload_classes": ["agentic"], "priorities": [], "notes": ""}'
        )
        intent = ObjectiveIntent.model_validate_json(old_json)
        assert intent.role == "pm"
        assert intent.target_components == []

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            build_intent(
                target_metric="latency",
                target_direction="minimize",
                workload_classes=["agentic"],
                role="wizard",
            )

    def test_role_preserved_in_objective(self):
        intent = build_intent(
            target_metric="throughput",
            target_direction="maximize",
            workload_classes=["batch-inference"],
            target_components=["scheduler"],
            role="developer",
        )
        proposal = assemble_proposal(intent)
        objective = finalize_objective(proposal, session_id="sess-dev", approved_by="dev1")
        assert objective.intent.role == "developer"
        assert objective.intent.target_components == ["scheduler"]
        parsed = json.loads(objective.model_dump_json())
        assert parsed["intent"]["role"] == "developer"
        assert parsed["intent"]["target_components"] == ["scheduler"]

    def test_target_components_default_empty(self):
        intent = build_intent(
            target_metric="latency",
            target_direction="minimize",
            workload_classes=["mixed"],
        )
        assert intent.target_components == []


class TestJsonRoundtrip:
    def test_objective_serializes_to_json(self):
        intent = build_intent(
            target_metric="gpu_memory_peak",
            target_direction="minimize",
            workload_classes=["long-context"],
            priorities=["correctness_coverage"],
            notes="Memory OOMs on 128k inputs",
        )
        proposal = assemble_proposal(intent)
        objective = finalize_objective(
            proposal, session_id="sess-003", approved_by="carol"
        )
        json_str = objective.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["intent"]["target_metric"] == "gpu_memory_peak"
        assert parsed["frozen"] is True

        restored = Objective.model_validate_json(json_str)
        assert restored == objective
