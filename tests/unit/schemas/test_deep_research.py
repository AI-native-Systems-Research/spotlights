"""Unit tests for deep-research path schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.deep_research import (
    Finding,
    ModuleDeepResearchInput,
    ModuleDeepResearchOutput,
    ProjectModules,
    SpotlightContext,
    StepIssue,
)
from spotlights_engine.schemas.modules import File, Module, ProjectTree, Repository


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="demo", summary="Demo repository."),
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


def test_module_deep_research_input_accepts_project_tree_and_context() -> None:
    request = ModuleDeepResearchInput(
        project_tree=_tree(),
        module_qualified_name="inference.attention",
        context=SpotlightContext(
            objective="reduce decode latency",
            workload_hints=["batch size 1-8"],
            validation_plan=["compare tokens/sec"],
        ),
        repo_path=Path("/tmp/example-repo"),
    )

    assert request.project_tree.repository.name == "demo"
    assert request.max_findings_per_module == 10
    assert request.repo_path == Path("/tmp/example-repo")


def test_project_modules_alias_reuses_project_tree_contract() -> None:
    assert ProjectModules is ProjectTree


def test_module_deep_research_output_allows_empty_findings() -> None:
    output = ModuleDeepResearchOutput()

    assert output.findings == []
    assert output.issues == []


def test_output_list_defaults_do_not_share_state() -> None:
    first = ModuleDeepResearchOutput()
    second = ModuleDeepResearchOutput()

    first.issues.append(
        StepIssue(step="module_deep_research", severity="warning", message="network timeout")
    )

    assert len(first.issues) == 1
    assert second.issues == []


def test_finding_id_pattern_is_strict() -> None:
    Finding(
        finding_id="find-0001",
        title="Paged attention",
        url="https://example.com/paged-attention",
        source_type="paper",
        technique_summary="Use paged KV allocation to reduce fragmentation.",
    )

    with pytest.raises(ValidationError):
        Finding(
            finding_id="finding-1",
            title="Paged attention",
            url="https://example.com/paged-attention",
            source_type="paper",
            technique_summary="Use paged KV allocation to reduce fragmentation.",
        )


def test_context_objective_is_required_and_non_empty() -> None:
    with pytest.raises(ValidationError):
        SpotlightContext(objective="")


def test_module_deep_research_output_schema_round_trips_through_json() -> None:
    schema = ModuleDeepResearchOutput.model_json_schema()
    assert "findings" in schema["properties"]
    assert "issues" in schema["properties"]
    assert json.loads(json.dumps(schema)) == schema
