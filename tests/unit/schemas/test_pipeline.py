"""Unit tests for the per-step pipeline I/O schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import SpotlightContext, StepIssue
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import (
    CandidateDeepResearchInput,
    CandidateDiscoveryInput,
    ModuleDeepResearchInput,
    ModuleDeepResearchOutput,
    ModuleRun,
    ProposalFromFindingCreatorInput,
    SpotlightsManagerInput,
)
from spotlights_engine.schemas.project import File, Module, ProjectTree, Repository


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="demo", summary="Demo repository.", source_root="src"),
        modules=[Module(name="core", path="src/core")],
    )


def _research_tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="demo", summary="Demo repository.", source_root="src"),
        modules=[
            Module(
                name="inference",
                path="src/inference",
                submodules=[
                    Module(
                        name="attention",
                        path="src/inference/attention",
                        description="Attention kernels and scheduling.",
                        main_files=[File(path="attention.py", role="core implementation")],
                    )
                ],
            )
        ],
    )


def _ctx() -> SpotlightContext:
    return SpotlightContext(
        objective="reduce decode latency",
        workload_hints=["batch=1-8"],
        validation_plan=["bench tokens/sec"],
    )


def _candidate(**overrides) -> Candidate:
    payload = {
        "id": "cand-core-0001",
        "module_qualified_name": "core",
        "origin": "code_agent",
        "locations": [
            {
                "file": "src/core/x.py",
                "spans": [
                    {
                        "line_start": 1,
                        "line_end": 2,
                        "symbol": "core.x.run",
                        "kind": "function",
                    }
                ],
            }
        ],
        "description": "d",
        "current_approach": "ca",
        "evolve_rationale": "er",
        "estimated_impact": "high",
        "estimated_impact_explanation": "x",
    }
    payload.update(overrides)
    return Candidate.model_validate(payload)


def _candidates() -> Candidates:
    return Candidates(module_qualified_name="core", candidates=[_candidate()])


def _finding(idx: int = 1) -> Finding:
    return Finding(
        finding_id=f"find-mod-{idx:04d}",
        title="t",
        url="https://x",
        source_type="paper",
        technique_summary="ts",
    )


def test_candidate_discovery_input_round_trips() -> None:
    cdi = CandidateDiscoveryInput(
        project_tree=_tree(), module_qualified_name="core", context=_ctx()
    )
    again = CandidateDiscoveryInput.model_validate_json(cdi.model_dump_json())
    assert again == cdi


def test_candidate_discovery_input_rejects_empty_qualified_name() -> None:
    with pytest.raises(ValidationError):
        CandidateDiscoveryInput(project_tree=_tree(), module_qualified_name="", context=_ctx())


def test_module_deep_research_input_accepts_project_tree_and_context() -> None:
    request = ModuleDeepResearchInput(
        project_tree=_research_tree(),
        module_qualified_name="inference/attention",
        context=SpotlightContext(
            objective="reduce decode latency",
            workload_hints=["batch size 1-8"],
            validation_plan=["compare tokens/sec"],
        ),
        repo_path=Path("/tmp/example-repo"),
    )

    assert request.project_tree.repository.name == "demo"
    assert request.num_search_runs == 3
    assert request.search_consensus_threshold is None
    assert request.repo_path == Path("/tmp/example-repo")


def test_module_deep_research_output_allows_empty_findings() -> None:
    output = ModuleDeepResearchOutput()

    assert output.findings == []
    assert output.issues == []


def test_module_deep_research_output_list_defaults_do_not_share_state() -> None:
    first = ModuleDeepResearchOutput()
    second = ModuleDeepResearchOutput()

    first.issues.append(
        StepIssue(step="module_deep_research", severity="warning", message="network timeout")
    )

    assert len(first.issues) == 1
    assert second.issues == []


def test_module_deep_research_output_schema_round_trips_through_json() -> None:
    schema = ModuleDeepResearchOutput.model_json_schema()
    assert "findings" in schema["properties"]
    assert "issues" in schema["properties"]
    assert json.loads(json.dumps(schema)) == schema


def test_proposal_from_finding_creator_requires_findings_field() -> None:
    with pytest.raises(ValidationError):
        ProposalFromFindingCreatorInput(candidates=_candidates(), context=_ctx())


def test_module_run_status_constraints() -> None:
    mr = ModuleRun(module_qualified_name="core", status="SKIPPED", candidates=None)
    assert mr.status == "SKIPPED"
    assert mr.candidates is None
    assert mr.issues == []

    mr_failed = ModuleRun(
        module_qualified_name="core",
        status="FAILED",
        issues=[
            StepIssue(
                step="candidate_discovery",
                severity="error",
                message="boom",
                recoverable=False,
            )
        ],
    )
    assert mr_failed.issues[0].recoverable is False


def test_spotlights_manager_input_defaults() -> None:
    inp = SpotlightsManagerInput(repo_path=Path("/tmp/repo"), context=_ctx())
    assert inp.num_search_runs == 3
    assert inp.search_consensus_threshold is None
    assert inp.continue_on_module_failure is True
    # Additive: module mode stays the default so existing callers are unaffected.
    assert inp.deep_research_mode == "module"


def test_spotlights_manager_input_rejects_unknown_deep_research_mode() -> None:
    with pytest.raises(ValidationError):
        SpotlightsManagerInput(
            repo_path=Path("/tmp/repo"),
            context=_ctx(),
            deep_research_mode="hybrid",  # type: ignore[arg-type]
        )


# CandidateDeepResearchInput (step 3, candidate mode) -------------------------


def test_candidate_deep_research_input_defaults() -> None:
    request = CandidateDeepResearchInput(
        project_tree=_research_tree(),
        module_qualified_name="inference/attention",
        context=_ctx(),
        repo_path=Path("/tmp/example-repo"),
    )

    assert request.candidates == []
    assert request.num_search_runs == 3
    assert request.search_consensus_threshold is None
    assert request.enable_claude_search is False


def test_candidate_deep_research_input_forbids_module_only_knobs() -> None:
    # The module-mode knobs are structurally absent, not just ignored.
    base: dict[str, Any] = dict(
        project_tree=_research_tree(),
        module_qualified_name="inference/attention",
        context=_ctx(),
        repo_path=Path("/tmp/example-repo"),
    )
    for extra in ("max_findings_per_module", "include_candidate_hotspots"):
        with pytest.raises(ValidationError):
            CandidateDeepResearchInput.model_validate({**base, extra: 5})


def test_candidate_deep_research_input_round_trips_through_json() -> None:
    request = CandidateDeepResearchInput(
        project_tree=_research_tree(),
        module_qualified_name="inference/attention",
        context=_ctx(),
        repo_path=Path("/tmp/example-repo"),
        candidates=list(_candidates().candidates),
        num_search_runs=4,
        search_consensus_threshold=2,
    )

    again = CandidateDeepResearchInput.model_validate_json(request.model_dump_json())

    assert again.num_search_runs == 4
    assert again.search_consensus_threshold == 2
    assert [c.id for c in again.candidates] == [c.id for c in request.candidates]


def test_module_run_round_trips_through_json() -> None:
    mr = ModuleRun(module_qualified_name="core", status="SUCCEEDED", candidates=_candidates())

    again = ModuleRun.model_validate_json(mr.model_dump_json())
    assert again.module_qualified_name == "core"
    assert again.status == "SUCCEEDED"
