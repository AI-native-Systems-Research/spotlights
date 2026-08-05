"""Adapters from existing Spotlights outputs into module-knowledge records."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from spotlights_engine.module_knowledge.schemas import KnowledgeRecord, Provenance, SourceRef
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput, ModuleRun
from spotlights_engine.schemas.project import Module, ProjectTree
from spotlights_engine.schemas.proposal import Proposal
from spotlights_engine.utils.schema_compat import primary_file, primary_span, proposals_from

if TYPE_CHECKING:
    from spotlights_engine.spotlights_manager.api import SpotlightsManagerResult


def records_from_findings(
    findings: list[Finding],
    *,
    module_qualified_name: str,
    source_id_prefix: str = "deep_research",
) -> list[KnowledgeRecord]:
    """Convert source-backed findings into archive records."""
    return [
        record_from_finding(
            finding,
            module_qualified_name=module_qualified_name,
            source_id_prefix=source_id_prefix,
        )
        for finding in findings
    ]


def records_from_module_deep_research(
    output: ModuleDeepResearchOutput,
    *,
    module_qualified_name: str,
    source_id_prefix: str = "deep_research",
) -> list[KnowledgeRecord]:
    """Convert deep-research findings into archive records."""
    return records_from_findings(
        output.findings,
        module_qualified_name=module_qualified_name,
        source_id_prefix=source_id_prefix,
    )


def record_from_finding(
    finding: Finding,
    *,
    module_qualified_name: str,
    source_id_prefix: str = "deep_research",
) -> KnowledgeRecord:
    """Convert one source-backed finding to a knowledge record."""
    record_id = _safe_record_id(f"finding:{module_qualified_name}:{finding.finding_id}")
    text = "\n\n".join(
        part for part in [finding.technique_summary, finding.supporting_evidence] if part.strip()
    )
    return KnowledgeRecord(
        record_id=record_id,
        source_type="finding",
        title=finding.title,
        text=text,
        source=SourceRef(
            source_id=_safe_record_id(f"{source_id_prefix}:{finding.finding_id}"),
            title=finding.title,
            url=finding.url,
            trust_tier="credible" if finding.source_type in {"paper", "docs"} else "unknown",
        ),
        provenance=Provenance(
            locator=f"module_deep_research:{module_qualified_name}:{finding.finding_id}",
            extractor="module_deep_research",
        ),
        tags=[
            module_qualified_name,
            finding.source_type,
            *([finding.candidate_id] if finding.candidate_id else []),
        ],
        metadata={
            "module_qualified_name": module_qualified_name,
            "finding_id": finding.finding_id,
            "candidate_id": finding.candidate_id,
        },
    )


def records_from_project_tree(
    project_tree: ProjectTree,
    *,
    artifact_path: str | None = None,
) -> list[KnowledgeRecord]:
    """Convert module/project-tree structure to queryable module-map records."""
    return [
        record_from_module(
            project_tree.repository.name,
            qualified_name,
            module,
            artifact_path=artifact_path,
        )
        for qualified_name, module in project_tree.walk()
    ]


def record_from_module(
    repository_name: str,
    qualified_name: str,
    module: Module,
    *,
    artifact_path: str | None = None,
) -> KnowledgeRecord:
    """Convert one `ProjectTree` module into a module-map record."""
    files = ", ".join(f"{file.path} ({file.role})" for file in module.main_files) or "none"
    dependencies = ", ".join(module.depends_on) or "none"
    text = (
        f"Module {qualified_name} in repository {repository_name}.\n"
        f"Path: {module.path}.\n"
        f"Description: {module.description or 'No description provided.'}\n"
        f"Depends on: {dependencies}.\n"
        f"Main files: {files}."
    )
    return KnowledgeRecord(
        record_id=_safe_record_id(f"module_map:{repository_name}:{qualified_name}"),
        source_type="module_map",
        title=f"{repository_name}: {qualified_name}",
        text=text,
        source=SourceRef(
            source_id=_safe_record_id(f"project_tree:{repository_name}"),
            title=f"Project tree for {repository_name}",
            trust_tier="internal",
        ),
        provenance=Provenance(
            locator=f"project_tree:{qualified_name}",
            extractor="modules_extractor",
            artifact_path=artifact_path,
        ),
        tags=[repository_name, qualified_name, module.name, "module-map"],
        metadata={
            "repository": repository_name,
            "module_qualified_name": qualified_name,
            "module_path": module.path,
        },
    )


def records_from_candidates(candidates: Candidates) -> list[KnowledgeRecord]:
    """Convert a `Candidates` bundle into candidate and proposal records."""
    records: list[KnowledgeRecord] = []
    for candidate in candidates.candidates:
        records.append(
            record_from_candidate(
                candidate,
                module_qualified_name=candidates.module_qualified_name,
            )
        )
        records.extend(
            records_from_candidate_proposals(
                candidate,
                module_qualified_name=candidates.module_qualified_name,
            )
        )
    return records


def record_from_candidate(
    candidate: Candidate,
    *,
    module_qualified_name: str,
) -> KnowledgeRecord:
    """Convert one discovered candidate into a queryable knowledge record."""
    span = primary_span(candidate)
    file = primary_file(candidate)
    location = f"{file}:{span.line_start}-{span.line_end}"
    text = "\n".join(
        [
            f"Candidate {candidate.id} for module {module_qualified_name}.",
            f"Symbol: {span.symbol} ({span.kind}).",
            f"Location: {location}.",
            f"Description: {candidate.description}",
            f"Current approach: {candidate.current_approach}",
            f"Evolution rationale: {candidate.evolve_rationale}",
            "Estimated impact: "
            f"{candidate.estimated_impact} — {candidate.estimated_impact_explanation}",
        ]
    )
    return KnowledgeRecord(
        record_id=_safe_record_id(f"candidate:{module_qualified_name}:{candidate.id}"),
        source_type="candidate",
        title=f"{module_qualified_name}: {span.symbol}",
        text=text,
        source=SourceRef(
            source_id=_safe_record_id(f"candidate_discovery:{module_qualified_name}"),
            title=f"Candidate discovery for {module_qualified_name}",
            trust_tier="internal",
        ),
        provenance=Provenance(
            locator=f"candidate_discovery:{module_qualified_name}:{candidate.id}",
            extractor="candidate_discovery",
        ),
        tags=[
            module_qualified_name,
            span.kind,
            candidate.estimated_impact,
        ],
        metadata={
            "module_qualified_name": module_qualified_name,
            "candidate_id": candidate.id,
            "file": file,
            "line_start": span.line_start,
            "line_end": span.line_end,
            "symbol": span.symbol,
            "kind": span.kind,
        },
    )


def records_from_candidate_proposals(
    candidate: Candidate,
    *,
    module_qualified_name: str,
) -> list[KnowledgeRecord]:
    """Convert proposals attached to one candidate into archive records."""
    records: list[KnowledgeRecord] = []
    for proposal in proposals_from(candidate, "research_finding"):
        records.append(
            record_from_deep_research_proposal(
                proposal,
                candidate=candidate,
                module_qualified_name=module_qualified_name,
            )
        )
    for proposal in proposals_from(candidate, "agent_knowledge"):
        records.append(
            record_from_agent_proposal(
                proposal,
                candidate=candidate,
                module_qualified_name=module_qualified_name,
            )
        )
    return records


def record_from_deep_research_proposal(
    proposal: Proposal,
    *,
    candidate: Candidate,
    module_qualified_name: str,
) -> KnowledgeRecord:
    """Convert a research-backed proposal into an archive record."""
    span = primary_span(candidate)
    text_parts = [
        proposal.description,
        f"Rationale: {proposal.rationale}",
        f"Candidate: {candidate.id} ({span.symbol}).",
    ]
    if proposal.finding_ref_id:
        text_parts.append(f"Finding: {proposal.finding_ref_id}.")
    for label, value in (
        ("Mechanism", proposal.mechanism),
        ("Required changes", proposal.required_changes),
        ("Expected effect", proposal.expected_effect),
        ("Evaluation metric", proposal.evaluation_metric),
    ):
        if value:
            text_parts.append(f"{label}: {value}")
    text = "\n".join(
        text_parts
    )
    return KnowledgeRecord(
        record_id=_safe_record_id(
            f"proposal:{module_qualified_name}:{candidate.id}:{proposal.id}"
        ),
        source_type="proposal",
        title=proposal.title,
        text=text,
        source=SourceRef(
            source_id=_safe_record_id(f"proposal_from_finding:{module_qualified_name}"),
            title=f"Research-backed proposals for {module_qualified_name}",
            trust_tier="internal",
        ),
        provenance=Provenance(
            locator=(
                "proposal_from_finding_creator:"
                f"{module_qualified_name}:{candidate.id}:{proposal.id}"
            ),
            extractor="proposal_from_finding_creator",
        ),
        tags=[module_qualified_name, candidate.id, span.kind, "deep-research-proposal"],
        metadata={
            "module_qualified_name": module_qualified_name,
            "candidate_id": candidate.id,
            "proposal_id": proposal.id,
            "finding_id": proposal.finding_ref_id,
            "created_by": proposal.author,
            "source": proposal.source,
            "mechanism": proposal.mechanism,
            "required_changes": proposal.required_changes,
            "expected_effect": proposal.expected_effect,
            "evaluation_metric": proposal.evaluation_metric,
        },
    )


def record_from_agent_proposal(
    proposal: Proposal,
    *,
    candidate: Candidate,
    module_qualified_name: str,
) -> KnowledgeRecord:
    """Convert an agent-generated proposal into an archive record."""
    span = primary_span(candidate)
    text = "\n".join(
        [
            proposal.description,
            f"Rationale: {proposal.rationale}",
            f"Candidate: {candidate.id} ({span.symbol}).",
            f"Agent: {proposal.author or '(unknown)'}.",
        ]
    )
    return KnowledgeRecord(
        record_id=_safe_record_id(
            f"proposal:{module_qualified_name}:{candidate.id}:{proposal.id}"
        ),
        source_type="proposal",
        title=proposal.title,
        text=text,
        source=SourceRef(
            source_id=_safe_record_id(f"agent_proposals:{module_qualified_name}"),
            title=f"Agent proposals for {module_qualified_name}",
            trust_tier="internal",
        ),
        provenance=Provenance(
            locator=f"agent_proposals:{module_qualified_name}:{candidate.id}:{proposal.id}",
            extractor="agent_proposals",
        ),
        tags=[module_qualified_name, candidate.id, span.kind, "agent-proposal"],
        metadata={
            "module_qualified_name": module_qualified_name,
            "candidate_id": candidate.id,
            "proposal_id": proposal.id,
            "agent_name": proposal.author,
            "source": proposal.source,
        },
    )


def records_from_module_run(module_run: ModuleRun) -> list[KnowledgeRecord]:
    """Convert a manager `ModuleRun` into knowledge records.

    The adapter archives findings, candidates, and proposals when present. It
    does not archive issues; those remain operational status rather than
    evidence-bearing memory.
    """
    records = records_from_findings(
        module_run.findings,
        module_qualified_name=module_run.module_qualified_name,
    )
    if module_run.candidates is not None:
        records.extend(records_from_candidates(module_run.candidates))
    return records


def records_from_spotlights_result(
    result: SpotlightsManagerResult,
) -> list[KnowledgeRecord]:
    """Convert all module runs in a `SpotlightsManagerResult` into records."""
    records: list[KnowledgeRecord] = []
    for module_run in result.module_runs.values():
        records.extend(records_from_module_run(module_run))
    return records


def _safe_record_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_.:-" else ":" for ch in value.strip())
    collapsed = ":".join(part for part in cleaned.split(":") if part)
    if collapsed:
        return collapsed
    return "record:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "record_from_agent_proposal",
    "record_from_candidate",
    "record_from_deep_research_proposal",
    "records_from_candidate_proposals",
    "records_from_candidates",
    "records_from_findings",
    "records_from_module_run",
    "records_from_spotlights_result",
    "record_from_finding",
    "record_from_module",
    "records_from_module_deep_research",
    "records_from_project_tree",
]
