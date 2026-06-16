"""Deep-research → SpotlightReport adapter.

Pure function: walks a `SpotlightsResult`, renumbers ids globally, and
emits a `SpotlightReport`. Determinism comes from sorting `module_runs`
alphabetically by `module_qualified_name`; within a module, candidates
in stored order; proposals per candidate in `deep_research_proposals`
then `agent_proposals` order.

Spec: docs/_review-notes/2026-06-15_spotlight_report_schema.md §2.1.
"""

from __future__ import annotations

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.pipeline import SpotlightsResult
from spotlights_engine.schemas.spotlight_report import (
    CodeSpan,
    Location,
    RunInfo,
    SpotlightCandidate,
    SpotlightFinding,
    SpotlightProposal,
    SpotlightReport,
)


def to_spotlight_report(
    result: SpotlightsResult,
    *,
    run: RunInfo,
) -> SpotlightReport:
    """Convert a `SpotlightsResult` into a `SpotlightReport`.

    Pure: same input → same output (no timestamps, no random ids).
    `run` carries the run-level info (started_at, model, parameters, …)
    populated by the caller from existing manager state.
    """
    findings_out: list[SpotlightFinding] = []
    candidates_out: list[SpotlightCandidate] = []

    # `(module_qualified_name, original_finding_id) -> new_finding_id`.
    finding_id_map: dict[tuple[str, str], str] = {}

    finding_seq = 0
    candidate_seq = 0
    proposal_seq = 0

    issues_out = []

    for qn in sorted(result.module_runs.keys()):
        run_record = result.module_runs[qn]
        issues_out.extend(run_record.issues)

        for finding in run_record.findings:
            finding_seq += 1
            new_finding_id = f"find-{finding_seq:04d}"
            finding_id_map[(qn, finding.finding_id)] = new_finding_id
            findings_out.append(
                SpotlightFinding(
                    finding_id=new_finding_id,
                    module_qualified_name=qn,
                    title=finding.title,
                    url=finding.url,
                    source_type=finding.source_type,
                    technique_summary=finding.technique_summary,
                    supporting_evidence=finding.supporting_evidence,
                )
            )

        if run_record.candidates is None:
            continue

        for cand in run_record.candidates.candidates:
            candidate_seq += 1
            new_cand_id = f"cand-{candidate_seq:04d}"
            proposals_out: list[SpotlightProposal] = []

            for drp in cand.deep_research_proposals:
                proposal_seq += 1
                remapped = finding_id_map.get((qn, drp.finding_id))
                proposals_out.append(
                    SpotlightProposal(
                        id=f"prop-{proposal_seq:04d}",
                        source="research_finding",
                        source_refs=[remapped] if remapped is not None else [],
                        title=drp.title,
                        description=drp.detailed_description,
                        rationale=drp.proposal_rationale,
                    )
                )

            for ap in cand.agent_proposals:
                proposal_seq += 1
                proposals_out.append(
                    SpotlightProposal(
                        id=f"prop-{proposal_seq:04d}",
                        source="agent_knowledge",
                        source_refs=[],
                        author=ap.agent_name,
                        title=ap.title,
                        description=ap.detailed_description,
                        rationale=(
                            f"Novel (not covered by literature): {ap.novelty_rationale}"
                            f"\n\n{ap.detailed_description}"
                        ),
                    )
                )

            candidates_out.append(
                SpotlightCandidate(
                    id=new_cand_id,
                    module_qualified_name=qn,
                    origin="code_agent",
                    locations=[_location_for(cand)],
                    kind=cand.kind,
                    description=cand.description,
                    current_approach=cand.current_approach,
                    evolve_rationale=cand.evolve_rationale,
                    estimated_impact=cand.estimated_impact,
                    estimated_impact_explanation=cand.estimated_impact_explanation,
                    state=cand.state,
                    proposals=proposals_out,
                )
            )

    return SpotlightReport(
        project_tree=result.project_tree,
        context=result.context,
        candidates=candidates_out,
        findings=findings_out,
        anomalies=[],
        run=run,
        issues=issues_out,
    )


def _location_for(cand: Candidate) -> Location:
    return Location(
        file=cand.file,
        spans=[
            CodeSpan(
                line_start=cand.line_start,
                line_end=cand.line_end,
                symbol=cand.symbol,
            )
        ],
    )


__all__ = ["to_spotlight_report"]
