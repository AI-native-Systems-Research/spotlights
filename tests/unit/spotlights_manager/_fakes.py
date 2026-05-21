"""Shared fakes/builders for spotlights_manager unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spotlights_engine.candidate_discovery.api import (
    DiscoveryResult,
    IterationTelemetry,
)
from spotlights_engine.modules_extractor.agent import ExtractionInvocation
from spotlights_engine.modules_extractor.extractor import ExtractorResult
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import SpotlightContext, StepIssue
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import (
    ModuleDeepResearchOutput,
    SpotlightsManagerInput,
)
from spotlights_engine.schemas.project import (
    File,
    Module,
    ProjectTree,
    Repository,
)


def make_tree() -> ProjectTree:
    """Two-leaf tree: v1.kv_offload, v1.attention.paged_kv."""
    return ProjectTree(
        repository=Repository(name="demo", summary="fixture"),
        modules=[
            Module(
                name="v1",
                path="src/v1",
                description="v1 namespace",
                submodules=[
                    Module(
                        name="kv_offload",
                        path="src/v1/kv_offload",
                        description="kv offload",
                        main_files=[File(path="src/v1/kv_offload/core.py", role="entry")],
                    ),
                    Module(
                        name="attention",
                        path="src/v1/attention",
                        description="attention namespace",
                        submodules=[
                            Module(
                                name="paged_kv",
                                path="src/v1/attention/paged_kv",
                                description="paged kv",
                                main_files=[
                                    File(
                                        path="src/v1/attention/paged_kv/core.py",
                                        role="entry",
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def make_candidate(idx: int = 0) -> Candidate:
    n = idx + 1
    return Candidate(
        id=f"cand-{n:04d}",
        file="src/v1/kv_offload/core.py",
        line_start=1,
        line_end=10,
        symbol=f"hot_{n}",
        kind="function",
        description="x",
        current_approach="x",
        evolve_rationale="x",
        estimated_impact="medium",
        estimated_impact_explanation="x",
    )


def make_candidates(qn: str, n: int = 1) -> Candidates:
    return Candidates(
        module_qualified_name=qn,
        candidates=[make_candidate(i) for i in range(n)],
    )


def make_finding(idx: int = 0) -> Finding:
    n = idx + 1
    return Finding(
        finding_id=f"find-{n:04d}",
        title="t",
        url="https://example.com",
        source_type="paper",
        technique_summary="s",
    )


def make_research_output(
    n_findings: int = 1, issues: list[StepIssue] | None = None
) -> ModuleDeepResearchOutput:
    return ModuleDeepResearchOutput(
        findings=[make_finding(i) for i in range(n_findings)],
        issues=list(issues or []),
    )


def make_input(repo_path: Path) -> SpotlightsManagerInput:
    return SpotlightsManagerInput(
        repo_path=repo_path,
        context=SpotlightContext(objective="reduce latency"),
    )


def make_extractor_result(tree: ProjectTree) -> ExtractorResult:
    return ExtractorResult(
        project_tree=tree,
        invocation=ExtractionInvocation(
            session_id="session",
            duration_s=1.0,
            cost_usd=None,
            input_tokens=None,
            output_tokens=None,
        ),
    )


def make_iteration_telemetry(n: int = 0) -> IterationTelemetry:
    return IterationTelemetry(
        n=n,
        agent="claude_code",
        duration_s=0.1,
        candidate_count=1,
    )


def make_discovery_result(qn: str, n_candidates: int = 1) -> DiscoveryResult:
    return DiscoveryResult(
        candidates=make_candidates(qn, n=n_candidates),
        iterations=[make_iteration_telemetry()],
        total_duration_s=0.1,
        total_cost_usd=None,
    )


__all__ = [
    "make_candidate",
    "make_candidates",
    "make_discovery_result",
    "make_extractor_result",
    "make_finding",
    "make_input",
    "make_iteration_telemetry",
    "make_research_output",
    "make_tree",
]
