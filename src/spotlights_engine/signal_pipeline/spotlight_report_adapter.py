"""Signal-pipeline → SpotlightReport adapter.

Pure function: takes the parsed signal-pipeline artifacts and emits a
`SpotlightReport`. The runner is responsible for reading and parsing
the on-disk artifacts; this adapter performs no I/O.

Spec: docs/_review-notes/2026-06-15_spotlight_report_schema.md §2.2.
"""

from __future__ import annotations

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.project import ProjectTree
from spotlights_engine.schemas.spotlight_report import (
    CodeSpan,
    Location,
    RunInfo,
    SpotlightAnomaly,
    SpotlightCandidate,
    SpotlightProposal,
    SpotlightReport,
)
from spotlights_engine.signal_pipeline.schemas import Change, Signals


def to_spotlight_report(
    *,
    signals: Signals,
    project_tree: ProjectTree,
    candidates: list[Candidate],
    changes: dict[str, Change],
    context: SpotlightContext,
    run: RunInfo,
) -> SpotlightReport:
    """Convert signal-pipeline artifacts into a `SpotlightReport`.

    `changes` keys are `Candidate.id` values (the fan-out ids written by
    stage 04). A candidate without a corresponding change yields a
    candidate with an empty `proposals` list.

    Pure: same input → same output.
    """
    anomalies_out = [
        SpotlightAnomaly(
            anomaly_id=a.anomaly_id,
            type=a.type,
            description=a.description,
        )
        for a in signals.anomalies
    ]

    module_index = _module_index(project_tree)

    candidates_out: list[SpotlightCandidate] = []
    proposal_seq = 0

    for idx, cand in enumerate(candidates, start=1):
        new_cand_id = f"cand-{idx:04d}"
        proposals_out: list[SpotlightProposal] = []

        change = changes.get(cand.id)
        if change is not None:
            proposal_seq += 1
            proposals_out.append(
                SpotlightProposal(
                    id=f"prop-{proposal_seq:04d}",
                    source="telemetry_anomaly",
                    source_refs=list(cand.anomaly_refs),
                    title=(
                        f"{change.change_type.replace('_', ' ').title()} "
                        f"in {cand.symbol}"
                    ),
                    description=(
                        f"{change.mechanism}\n\nExpected: {change.expected_effect}"
                    ),
                    rationale=cand.evolve_rationale,
                    proposal_type=change.change_type,
                    mechanism=change.mechanism,
                    required_changes=change.required_changes,
                    expected_effect=change.expected_effect,
                    evaluation_metric=change.evaluation_metric,
                )
            )

        candidates_out.append(
            SpotlightCandidate(
                id=new_cand_id,
                module_qualified_name=_assign_module(cand.file, module_index),
                origin="telemetry_anomaly",
                locations=[
                    Location(
                        file=cand.file,
                        spans=[
                            CodeSpan(
                                line_start=cand.line_start,
                                line_end=cand.line_end,
                                symbol=cand.symbol,
                            )
                        ],
                    )
                ],
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
        project_tree=project_tree,
        context=context,
        candidates=candidates_out,
        findings=[],
        anomalies=anomalies_out,
        run=run,
        issues=[],
    )


def _module_index(tree: ProjectTree) -> list[tuple[str, str]]:
    """Flatten the tree to `[(qualified_name, normalized_path), …]`,
    sorted by path-segment depth descending so the deepest match wins
    in `_assign_module`.
    """
    out: list[tuple[str, str]] = []
    for qn, m in tree.walk():
        out.append((_qn_dot(qn), _normalize(m.path)))
    out.sort(key=lambda x: -_depth(x[1]))
    return out


def _qn_dot(qn: str) -> str:
    return qn.replace("/", ".")


def _normalize(p: str) -> str:
    return p.strip("/").rstrip("/")


def _depth(p: str) -> int:
    return 0 if not p else p.count("/") + 1


def _assign_module(file: str, module_index: list[tuple[str, str]]) -> str | None:
    """Match `file` against module paths; return the deepest matching
    module's qualified name, or None if no module prefixes the file."""
    norm_file = _normalize(file)
    for qn, mod_path in module_index:
        if not mod_path:
            # An empty path means "the whole repo" — match-all fallback,
            # but only when no deeper match exists. Since the index is
            # sorted by depth descending, an empty path is last.
            return qn
        if norm_file == mod_path or norm_file.startswith(mod_path + "/"):
            return qn
    return None


__all__ = ["to_spotlight_report"]
