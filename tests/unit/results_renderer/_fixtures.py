"""Fixture builders for results_renderer tests.

These fabricate a `<artifacts_dir>/spotlights_manager/` tree on disk that
mirrors what the orchestrator would have produced — using only the
persistence helpers, never the orchestrator itself, so the tests stay
focused on the renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spotlights_engine.costing.manifest import (
    ModelUsed,
    RunManifest,
    RunManifestCost,
    RunManifestOutputs,
    RunManifestSpotlights,
    RunManifestTarget,
    RunManifestTiming,
    UsageTotals,
)
from spotlights_engine.modules_extractor.agent import ExtractionInvocation
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import SpotlightContext, StepIssue
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import (
    AgentProposalsOutput,
    ModuleDeepResearchOutput,
    ProposalFromFindingCreatorOutput,
)
from spotlights_engine.schemas.project import (
    File,
    Module,
    ProjectTree,
    Repository,
)
from spotlights_engine.schemas.proposal import Proposal
from spotlights_engine.spotlights_manager import persistence as P
from spotlights_engine.spotlights_manager.persistence import (
    ManagerPaths,
    ModuleCheckpoint,
)
from spotlights_engine.utils.schema_compat import make_location


def make_tree() -> ProjectTree:
    """Two-leaf tree: v1/kv_offload (impact varied), kernels (one cand)."""
    return ProjectTree(
        repository=Repository(
            name="demo",
            summary="Demo repo for renderer tests.",
            source_root="src",
            external_dependencies=["torch", "numpy"],
        ),
        modules=[
            Module(
                name="kernels",
                path="src/kernels",
                description="custom kernels",
                main_files=[File(path="src/kernels/main.py", role="entry")],
            ),
            Module(
                name="v1",
                path="src/v1",
                description="v1 namespace",
                submodules=[
                    Module(
                        name="kv_offload",
                        path="src/v1/kv_offload",
                        description="kv offload",
                        depends_on=["kernels"],
                        main_files=[
                            File(path="src/v1/kv_offload/core.py", role="entry"),
                            File(path="src/v1/kv_offload/util.py", role="util"),
                        ],
                    ),
                ],
            ),
        ],
    )


def make_candidate(
    n: int,
    *,
    impact: str = "medium",
    file: str = "src/v1/kv_offload/core.py",
    deep_proposals: list[Proposal] | None = None,
    agent_proposals: list[Proposal] | None = None,
) -> Candidate:
    return Candidate(
        id=f"cand-mod-{n:04d}",
        origin="code_agent",
        locations=[
            make_location(
                file=file,
                line_start=10,
                line_end=20,
                symbol=f"hot_{n}",
                kind="function",
            )
        ],
        description=f"description {n}",
        current_approach=f"current approach {n}",
        evolve_rationale=f"evolve rationale {n}",
        estimated_impact=impact,  # type: ignore[arg-type]
        estimated_impact_explanation=f"impact explanation {n}",
        proposals=[*(deep_proposals or []), *(agent_proposals or [])],
    )


def make_finding(n: int) -> Finding:
    return Finding(
        finding_id=f"find-mod-{n:04d}",
        title=f"Finding {n}",
        url=f"https://example.com/f/{n}",
        source_type="paper",
        technique_summary=f"technique {n}",
        supporting_evidence=f"evidence {n}",
    )


def make_deep_proposal(finding_id: str, *, prop_id: str = "prop-mod-0001") -> Proposal:
    return Proposal(
        id=prop_id,
        source="research_finding",
        finding_ref_id=finding_id,
        author="claude_code",
        title=f"Deep proposal for {finding_id}",
        description="detailed description",
        rationale="proposal rationale",
    )


def make_agent_proposal(*, prop_id: str = "prop-mod-0002") -> Proposal:
    return Proposal(
        id=prop_id,
        source="agent_knowledge",
        author="claude_code",
        title="Agent proposal",
        description="detailed agent description",
        rationale="novelty rationale",
    )


def write_manifest_with_modules(
    paths: ManagerPaths,
    *,
    context: SpotlightContext | None,
    repo_path: str,
    module_statuses: dict[str, str],
    extractor_duration_s: float = 1.5,
    status: str = "COMPLETE",
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.modules_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": "2026-05-23T12:00:00+00:00",
        "updated_at": "2026-05-23T12:00:00+00:00",
        "status": status,
        "input_fingerprint": {"repo_path": repo_path},
        "config_fingerprint": {},
        "extractor": {"completed": True, "duration_s": extractor_duration_s},
        "modules": {qn: {"status": s, "last_step": None} for qn, s in module_statuses.items()},
    }
    if context is not None:
        manifest["context"] = context.model_dump(mode="json")
    P.write_manifest(paths, manifest)


def write_extractor_outputs(
    paths: ManagerPaths,
    tree: ProjectTree,
    invocation: ExtractionInvocation | None = None,
) -> None:
    inv = invocation or ExtractionInvocation(
        session_id="sess-1",
        duration_s=1.5,
        cost_usd=None,
        input_tokens=None,
        output_tokens=None,
    )
    P.write_extractor_outputs(paths, tree, inv)


def write_module(
    paths: ManagerPaths,
    qn: str,
    *,
    candidates: Candidates | None,
    findings: list[Finding] | None = None,
    proposal_output: ProposalFromFindingCreatorOutput | None = None,
    agent_output: AgentProposalsOutput | None = None,
    status: str = "SUCCEEDED",
    last_step: str | None = "agent_proposals",
    issues: list[StepIssue] | None = None,
) -> None:
    mp = paths.for_module(qn)
    mp.dir.mkdir(parents=True, exist_ok=True)
    cp = ModuleCheckpoint(
        module_qualified_name=qn,
        status=status,  # type: ignore[arg-type]
        last_step=last_step,  # type: ignore[arg-type]
        failed_step=None,
        error=None,
        retryable=False,
        issues=list(issues or []),
        started_at="2026-05-23T12:00:00+00:00",
        updated_at="2026-05-23T12:00:00+00:00",
    )
    P.write_checkpoint(mp, cp)
    if candidates is not None:
        P.write_candidates(mp, candidates)
    if findings is not None:
        P.write_deep_research(
            mp,
            ModuleDeepResearchOutput(findings=findings, issues=[]),
            duration_s=0.0,
        )
    if proposal_output is not None:
        P.write_proposal_from_finding(
            mp, proposal_output, duration_s=0.0, per_pair_durations_s={}
        )
    if agent_output is not None:
        P.write_agent_proposals(
            mp, agent_output, duration_s=0.0, per_candidate_durations_s={}
        )


def make_run_manifest(
    *,
    wall_clock_s: float = 12.0,
    accumulated_duration_s: float = 340.0,
    rate_note: str = "",
    notes: str = "",
    with_models: bool = True,
) -> RunManifest:
    """A representative public run manifest for renderer-section tests."""
    models = (
        [
            ModelUsed(
                model="claude-opus-4-8",
                provider="anthropic",
                role="deep_research",
                usage=UsageTotals(
                    input=1000, output=2000, cache_read=300, cache_create=50
                ),
            ),
            ModelUsed(
                model="gpt-5-codex",
                provider="openai",
                role="agent_proposals",
                usage=UsageTotals(input=500, output=800),
            ),
        ]
        if with_models
        else []
    )
    return RunManifest(
        run_id="run-abc123",
        date="2026-05-23T12:00:00+00:00",
        target=RunManifestTarget(
            repo_url="https://github.com/acme/demo",
            commit_sha="deadbeef",
            objective="reduce latency",
        ),
        spotlights=RunManifestSpotlights(
            commit_sha="cafef00d", pipeline="deep-research"
        ),
        models_used=models,
        total_tokens=4650,
        cost=RunManifestCost(
            amount_usd=0.1234, source="contracted-rate-table", rate_note=rate_note
        ),
        timing=RunManifestTiming(
            wall_clock_s=wall_clock_s,
            accumulated_duration_s=accumulated_duration_s,
            api_time_s=8.5,
        ),
        outputs=RunManifestOutputs(
            candidates_path="output/index.md",
            num_candidates=4,
            module_status={"SUCCEEDED": 1, "DEGRADED": 1},
        ),
        notes=notes,
    )


def write_run_manifest(paths: ManagerPaths, manifest: RunManifest) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    P.write_run_manifest(paths, manifest)


def make_full_run(
    artifacts_dir: Path,
    *,
    repo_path: str = "/repo",
    context: SpotlightContext | None = None,
) -> tuple[ManagerPaths, ProjectTree]:
    """Standard 2-module fixture: kv_offload SUCCEEDED with proposals,
    kernels DEGRADED with one candidate."""
    paths = ManagerPaths(artifacts_dir)
    tree = make_tree()

    ctx = context or SpotlightContext(
        objective="reduce latency",
        workload_hints=["short prompts", "long generations"],
        validation_plan=["microbench", "e2e"],
    )

    write_extractor_outputs(paths, tree)

    findings_kv = [make_finding(1), make_finding(2)]
    deep_props = [
        make_deep_proposal("find-mod-0001", prop_id="prop-mod-0001"),
        # dangling — should not be counted
        make_deep_proposal("find-mod-0099", prop_id="prop-mod-0002"),
    ]
    cand_kv_high = make_candidate(
        1,
        impact="high",
        deep_proposals=deep_props,
        agent_proposals=[make_agent_proposal(prop_id="prop-mod-0003")],
    )
    cand_kv_med = make_candidate(2, impact="medium")
    cand_kv_low = make_candidate(3, impact="low")
    cands_kv = Candidates(
        module_qualified_name="v1/kv_offload",
        candidates=[cand_kv_high, cand_kv_med, cand_kv_low],
    )
    write_module(
        paths,
        "v1/kv_offload",
        candidates=cands_kv,
        findings=findings_kv,
        proposal_output=ProposalFromFindingCreatorOutput(
            candidates=cands_kv, issues=[]
        ),
        agent_output=AgentProposalsOutput(candidates=cands_kv, issues=[]),
        status="SUCCEEDED",
    )

    cand_kernels = make_candidate(
        4, impact="high", file="src/kernels/main.py"
    )
    cands_kernels = Candidates(
        module_qualified_name="kernels",
        candidates=[cand_kernels],
    )
    write_module(
        paths,
        "kernels",
        candidates=cands_kernels,
        findings=[make_finding(3)],
        proposal_output=ProposalFromFindingCreatorOutput(
            candidates=cands_kernels, issues=[]
        ),
        agent_output=AgentProposalsOutput(candidates=cands_kernels, issues=[]),
        status="DEGRADED",
        issues=[
            StepIssue(
                step="module_deep_research",
                severity="warning",
                message="rate limited briefly",
                recoverable=True,
            )
        ],
    )

    write_manifest_with_modules(
        paths,
        context=ctx,
        repo_path=repo_path,
        module_statuses={"v1/kv_offload": "SUCCEEDED", "kernels": "DEGRADED"},
    )

    return paths, tree


__all__ = [
    "make_agent_proposal",
    "make_candidate",
    "make_deep_proposal",
    "make_finding",
    "make_full_run",
    "make_run_manifest",
    "make_tree",
    "write_extractor_outputs",
    "write_manifest_with_modules",
    "write_module",
    "write_run_manifest",
]
