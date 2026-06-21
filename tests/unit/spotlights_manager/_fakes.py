"""Shared fakes/builders for spotlights_manager unit tests."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.agent_proposals.api import AgentProposalsResult
from spotlights_engine.candidate_discovery.api import (
    DiscoveryResult,
    IterationTelemetry,
)
from spotlights_engine.modules_extractor.agent import ExtractionInvocation
from spotlights_engine.modules_extractor.extractor import ExtractorResult
from spotlights_engine.proposal_from_finding_creator.api import (
    ProposalFromFindingCreatorResult,
)
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import SpotlightContext, StepIssue
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import (
    AgentProposalsOutput,
    ModuleDeepResearchOutput,
    ProposalFromFindingCreatorOutput,
    SpotlightsManagerInput,
)
from spotlights_engine.schemas.project import (
    File,
    Module,
    ProjectTree,
    Repository,
)
from spotlights_engine.utils.id_helpers import slug_for
from spotlights_engine.utils.schema_compat import make_location


def make_tree() -> ProjectTree:
    """Two-leaf tree: v1/kv_offload, v1/attention/paged_kv."""
    return ProjectTree(
        repository=Repository(name="demo", summary="fixture", source_root="src"),
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


def make_candidate(idx: int = 0, *, segment: str = "v1_kv_offload") -> Candidate:
    n = idx + 1
    return Candidate(
        id=f"cand-{segment}-{n:04d}",
        origin="code_agent",
        locations=[
            make_location(
                file="src/v1/kv_offload/core.py",
                line_start=1,
                line_end=10,
                symbol=f"hot_{n}",
                kind="function",
            )
        ],
        description="x",
        current_approach="x",
        evolve_rationale="x",
        estimated_impact="medium",
        estimated_impact_explanation="x",
        proposals=[],
    )


def make_candidates(qn: str, n: int = 1) -> Candidates:
    # Discovery now mints ids prefixed with the producing module's slug (D3), so
    # a fake `discover` that returns these must do the same to mirror reality.
    segment = slug_for(qn)
    return Candidates(
        module_qualified_name=qn,
        candidates=[make_candidate(i, segment=segment) for i in range(n)],
    )


def make_finding(idx: int = 0, *, segment: str = "v1_kv_offload") -> Finding:
    n = idx + 1
    return Finding(
        finding_id=f"find-{segment}-{n:04d}",
        title="t",
        url="https://example.com",
        source_type="paper",
        technique_summary="s",
    )


def make_research_output(
    n_findings: int = 1,
    issues: list[StepIssue] | None = None,
    *,
    segment: str = "v1_kv_offload",
) -> ModuleDeepResearchOutput:
    return ModuleDeepResearchOutput(
        findings=[make_finding(i, segment=segment) for i in range(n_findings)],
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


def _advance_candidate(c: Candidate) -> Candidate:
    """Round-trip a step-2 candidate forward to the post-step-4 shape with
    no synthesized proposals.

    The new `Candidate` carries no `state`; pipeline progress is tracked in the
    manager's parallel state map (decision D2). A step-4 run that mints no
    research-backed proposals therefore returns the candidate unchanged."""
    return c.model_copy(deep=True)


def make_proposal_from_finding_result(
    qn: str,
    n_candidates: int = 1,
    *,
    issues: list[StepIssue] | None = None,
) -> ProposalFromFindingCreatorResult:
    """Default step-4 fake: every input candidate advances to
    `FINDING_PROPOSALS_CREATED` with no proposals and no issues.

    Mirrors the synthetic zero-findings shape the orchestrator already
    produces; tests that need richer outputs should construct their own."""
    cands = make_candidates(qn, n=n_candidates)
    advanced = Candidates(
        module_qualified_name=cands.module_qualified_name,
        candidates=[_advance_candidate(c) for c in cands.candidates],
    )
    output = ProposalFromFindingCreatorOutput(
        candidates=advanced,
        issues=list(issues or []),
    )
    return ProposalFromFindingCreatorResult(
        output=output,
        per_pair_durations_s={},
        total_duration_s=0.0,
    )


def _advance_candidate_to_agent_proposals(c: Candidate) -> Candidate:
    """Advance a step-4-output candidate to the post-step-5 shape with no
    agent proposals.

    As with `_advance_candidate`, the schema `Candidate` has no `state` and a
    step-5 run that mints no agent-knowledge proposals returns it unchanged."""
    return c.model_copy(deep=True)


def make_agent_proposals_result(
    candidates: Candidates,
    *,
    issues: list[StepIssue] | None = None,
    total_duration_s: float = 0.0,
    per_candidate_durations_s: dict[str, dict[str, float]] | None = None,
) -> AgentProposalsResult:
    advanced = Candidates(
        module_qualified_name=candidates.module_qualified_name,
        candidates=[
            _advance_candidate_to_agent_proposals(c) for c in candidates.candidates
        ],
    )
    output = AgentProposalsOutput(
        candidates=advanced,
        issues=list(issues or []),
    )
    return AgentProposalsResult(
        output=output,
        per_candidate_durations_s=dict(per_candidate_durations_s or {}),
        total_duration_s=total_duration_s,
    )


def patch_agent_proposals(monkeypatch, orch_module) -> None:
    """Stub `create_agent_proposals_with_telemetry` on the orchestrator with a
    synchronous fake that advances every candidate to `AGENT_PROPOSALS_CREATED`
    with no proposals and no issues."""

    def _fake(
        inp,
        *,
        config,
        claude_runner=None,
        codex_runner=None,
        candidate_states=None,
        proposal_id_start=1,
        segment=None,
    ):
        return make_agent_proposals_result(inp.candidates)

    monkeypatch.setattr(
        orch_module, "create_agent_proposals_with_telemetry", _fake
    )


def patch_proposal_from_finding(monkeypatch, orch_module) -> None:
    """Stub `create_proposals_with_telemetry` on the orchestrator with a
    synchronous fake that mirrors the input shape — used by every existing
    manager test that wants to drive step 4 without invoking Claude."""

    def _fake(
        inp,
        *,
        config,
        runner=None,
        candidate_states=None,
        proposal_id_start=1,
        segment=None,
    ):
        cands = inp.candidates
        advanced = Candidates(
            module_qualified_name=cands.module_qualified_name,
            candidates=[_advance_candidate(c) for c in cands.candidates],
        )
        output = ProposalFromFindingCreatorOutput(
            candidates=advanced,
            issues=[],
        )
        return ProposalFromFindingCreatorResult(
            output=output,
            per_pair_durations_s={},
            total_duration_s=0.0,
        )

    monkeypatch.setattr(orch_module, "create_proposals_with_telemetry", _fake)


__all__ = [
    "make_candidate",
    "make_candidates",
    "make_discovery_result",
    "make_agent_proposals_result",
    "make_extractor_result",
    "make_finding",
    "make_input",
    "make_iteration_telemetry",
    "make_proposal_from_finding_result",
    "make_research_output",
    "make_tree",
    "patch_agent_proposals",
    "patch_proposal_from_finding",
]
