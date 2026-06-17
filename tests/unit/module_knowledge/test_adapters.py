"""Tests for adapters from existing Spotlights outputs to module knowledge."""

from __future__ import annotations

from spotlights_engine.module_knowledge import (
    records_from_candidates,
    records_from_module_deep_research,
    records_from_module_run,
    records_from_project_tree,
    records_from_spotlights_result,
)
from spotlights_engine.schemas.legacy.candidate import Candidate, Candidates
from spotlights_engine.schemas.legacy.common import SpotlightContext
from spotlights_engine.schemas.legacy.finding import Finding
from spotlights_engine.schemas.legacy.pipeline import ModuleDeepResearchOutput, ModuleRun, SpotlightsResult
from spotlights_engine.schemas.legacy.project import File, Module, ProjectTree, Repository
from spotlights_engine.schemas.legacy.proposals import AgentProposal, DeepResearchProposal


def test_records_from_module_deep_research_preserve_source_and_module_context() -> None:
    output = ModuleDeepResearchOutput(
        findings=[
            Finding(
                finding_id="find-0001",
                title="Paged attention",
                url="https://example.com/paper",
                source_type="paper",
                technique_summary="Paged KV allocation reduces fragmentation.",
                supporting_evidence="The paper reports lower memory waste.",
            )
        ]
    )

    records = records_from_module_deep_research(output, module_qualified_name="engine/cache")

    assert len(records) == 1
    record = records[0]
    assert record.source_type == "finding"
    assert record.title == "Paged attention"
    assert record.source.url == "https://example.com/paper"
    assert record.provenance.locator == "module_deep_research:engine/cache:find-0001"
    assert record.metadata["module_qualified_name"] == "engine/cache"
    assert "engine/cache" in record.tags


def test_records_from_project_tree_create_queryable_module_map_records() -> None:
    tree = ProjectTree(
        repository=Repository(name="vllm", summary="LLM serving engine"),
        modules=[
            Module(
                name="engine",
                path="vllm/engine",
                description="Request scheduling and execution.",
                main_files=[File(path="vllm/engine/scheduler.py", role="scheduler")],
            )
        ],
    )

    records = records_from_project_tree(tree, artifact_path="artifacts/ProjectTree.json")

    assert len(records) == 1
    record = records[0]
    assert record.record_id == "module_map:vllm:engine"
    assert record.source_type == "module_map"
    assert "Request scheduling" in record.text
    assert record.provenance.artifact_path == "artifacts/ProjectTree.json"
    assert record.metadata["module_path"] == "vllm/engine"


def _candidate_with_proposals() -> Candidate:
    return Candidate(
        id="cand-0001",
        file="vllm/engine/cache.py",
        line_start=10,
        line_end=20,
        symbol="evict_blocks",
        kind="method",
        description="Evicts KV cache blocks under memory pressure.",
        current_approach="Greedy eviction by age.",
        evolve_rationale="Tail latency may improve with pressure-aware eviction.",
        estimated_impact="high",
        estimated_impact_explanation="Decode stalls are dominated by cache pressure.",
        state="AGENT_PROPOSALS_CREATED",
        deep_research_proposals=[
            DeepResearchProposal(
                title="Pressure-aware KV eviction",
                detailed_description="Use pressure and reuse distance to pick eviction victims.",
                finding_id="find-0001",
                proposal_rationale="The finding reports lower fragmentation.",
                created_by="proposal_from_finding_creator",
            )
        ],
        agent_proposals=[
            AgentProposal(
                title="Add eviction counters",
                detailed_description="Expose counters before changing eviction policy.",
                agent_name="codex",
                novelty_rationale="Instrumentation was not covered by research findings.",
            )
        ],
    )


def test_records_from_candidates_include_candidate_and_attached_proposals() -> None:
    candidates = Candidates(
        module_qualified_name="engine/cache",
        candidates=[_candidate_with_proposals()],
    )

    records = records_from_candidates(candidates)

    assert [record.source_type for record in records] == ["candidate", "proposal", "proposal"]
    assert records[0].record_id == "candidate:engine:cache:cand-0001"
    assert records[0].provenance.locator == "candidate_discovery:engine/cache:cand-0001"
    assert records[0].metadata["file"] == "vllm/engine/cache.py"
    assert records[1].provenance.extractor == "proposal_from_finding_creator"
    assert records[1].metadata["finding_id"] == "find-0001"
    assert records[2].provenance.extractor == "agent_proposals"
    assert records[2].metadata["agent_name"] == "codex"


def test_records_from_module_run_and_spotlights_result_collect_memory_records() -> None:
    candidate = _candidate_with_proposals()
    finding = Finding(
        finding_id="find-0001",
        title="KV reuse distance",
        url="https://example.com/kv-reuse",
        source_type="paper",
        technique_summary="Reuse-distance-aware caching reduces churn.",
    )
    module_run = ModuleRun(
        module_qualified_name="engine/cache",
        status="SUCCEEDED",
        candidates=Candidates(module_qualified_name="engine/cache", candidates=[candidate]),
        findings=[finding],
    )
    result = SpotlightsResult(
        project_tree=ProjectTree(repository=Repository(name="vllm", summary="serving")),
        context=SpotlightContext(objective="Improve decode latency"),
        module_runs={"engine/cache": module_run},
    )

    module_records = records_from_module_run(module_run)
    result_records = records_from_spotlights_result(result)

    assert [record.source_type for record in module_records] == [
        "finding",
        "candidate",
        "proposal",
        "proposal",
    ]
    assert [record.record_id for record in result_records] == [
        record.record_id for record in module_records
    ]
