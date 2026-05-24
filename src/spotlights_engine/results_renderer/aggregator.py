"""View-models built from `LoadedRun` for the writer to render."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.project import Module

from spotlights_engine.results_renderer.api import RendererConfig
from spotlights_engine.results_renderer.loader import LoadedRun


class IndexRow(BaseModel):
    """One row of the index table — one per module."""

    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str
    module_page_path: str
    status: str
    n_candidates: int
    n_high_impact_candidates: int
    n_relevant_findings: int


class CandidateRow(BaseModel):
    """One row of the per-module candidates table."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    symbol: str
    candidate_page_path: str
    estimated_impact: str
    n_deep_research_proposals: int


class ModulePageView(BaseModel):
    """View-model for a single module page."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    qualified_name: str
    module: Module | None
    status: str
    candidates_sorted: list[Candidate] = Field(default_factory=list)
    candidate_rows: list[CandidateRow] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    issues: list[StepIssue] = Field(default_factory=list)


def aggregate(
    loaded: LoadedRun, config: RendererConfig
) -> tuple[list[IndexRow], dict[str, ModulePageView], list[str], list[str]]:
    """Build (index_rows, per-module views, skipped_modules, warnings)."""
    rows: list[IndexRow] = []
    views: dict[str, ModulePageView] = {}
    skipped: list[str] = []
    warnings: list[str] = list(loaded.warnings)

    manifest_modules = (
        loaded.manifest.get("modules") if isinstance(loaded.manifest, dict) else None
    ) or {}

    for qn, state in loaded.modules.items():
        manifest_status = _module_status(qn, manifest_modules, state)

        if not _should_include(manifest_status, config):
            skipped.append(qn)
            continue

        module = loaded.project_tree.resolve(qn.replace(".", "/"))
        if module is None:
            warnings.append(
                f"module {qn!r} is in the manifest but missing from project_tree"
            )

        candidates_source = _pick_candidates(state)
        if candidates_source is None:
            candidates: list[Candidate] = []
            if not _is_terminal_skip(manifest_status):
                warnings.append(
                    f"module {qn!r} has no candidates sidecar"
                )
        else:
            candidates = list(candidates_source.candidates)

        findings = (
            list(state.deep_research.findings)
            if state.deep_research is not None
            else []
        )

        candidates_sorted = sorted(
            candidates,
            key=lambda c: (-len(c.deep_research_proposals), c.id),
        )

        n_high = sum(1 for c in candidates if c.estimated_impact == "high")
        n_relevant = _count_relevant_findings(candidates, findings)

        from_slug = _module_page_filename(qn)
        module_slug = from_slug[:-3]  # strip ".md"
        rows.append(
            IndexRow(
                module_qualified_name=qn,
                module_page_path=f"modules/{from_slug}",
                status=str(manifest_status),
                n_candidates=len(candidates),
                n_high_impact_candidates=n_high,
                n_relevant_findings=n_relevant,
            )
        )

        candidate_rows = [
            CandidateRow(
                candidate_id=c.id,
                symbol=c.symbol,
                candidate_page_path=f"{module_slug}/{_candidate_page_filename(c)}",
                estimated_impact=str(c.estimated_impact),
                n_deep_research_proposals=len(c.deep_research_proposals),
            )
            for c in candidates_sorted
        ]

        issues = _aggregate_issues(state)
        views[qn] = ModulePageView(
            qualified_name=qn,
            module=module,
            status=str(manifest_status),
            candidates_sorted=candidates_sorted,
            candidate_rows=candidate_rows,
            findings=findings,
            issues=issues,
        )

    rows.sort(
        key=lambda r: (
            -r.n_relevant_findings,
            r.module_qualified_name,
        )
    )
    return rows, views, skipped, warnings


def _module_page_filename(qn: str) -> str:
    """Page filename for a module's qn, mirroring the on-disk slug."""
    from spotlights_engine.spotlights_manager.persistence import slug_for

    return f"{slug_for(qn)}.md"


def _candidate_page_filename(candidate: Candidate) -> str:
    """Page filename for a candidate. `<symbol-slug>__<id>.md`.

    The `__{id}` suffix prevents collisions across candidates that happen to
    slugify to the same base (e.g. overloaded names, unicode collapsing)."""
    from spotlights_engine.spotlights_manager.persistence import slug_for

    return f"{slug_for(candidate.symbol)}__{candidate.id}.md"


def _module_status(
    qn: str, manifest_modules: dict[str, Any], state: Any
) -> str:
    entry = manifest_modules.get(qn) if isinstance(manifest_modules, dict) else None
    if isinstance(entry, dict) and "status" in entry:
        return str(entry["status"])
    if state.checkpoint is not None:
        return str(state.checkpoint.status)
    return "UNKNOWN"


def _should_include(status: str, config: RendererConfig) -> bool:
    if status == "FAILED" and not config.include_failed_modules:
        return False
    if status == "SKIPPED" and not config.include_skipped_modules:
        return False
    return True


def _is_terminal_skip(status: str) -> bool:
    return status in {"SKIPPED", "FAILED"}


def _pick_candidates(state: Any) -> Any:
    """Latest-known candidates view for a module, mirroring orchestrator
    logic: prefer post-step-5, then post-step-4, then step-2."""
    if state.agent_proposals is not None:
        return state.agent_proposals.candidates
    if state.proposal_from_finding is not None:
        return state.proposal_from_finding.candidates
    return state.candidates


def _count_relevant_findings(
    candidates: list[Candidate], findings: list[Finding]
) -> int:
    referenced = {
        p.finding_id for c in candidates for p in c.deep_research_proposals
    }
    if not referenced:
        return 0
    have = {f.finding_id for f in findings}
    return len(referenced & have)


def _aggregate_issues(state: Any) -> list[StepIssue]:
    issues: list[StepIssue] = []
    if state.checkpoint is not None:
        issues.extend(state.checkpoint.issues)
    if state.deep_research is not None:
        issues.extend(state.deep_research.issues)
    if state.proposal_from_finding is not None:
        issues.extend(state.proposal_from_finding.issues)
    if state.agent_proposals is not None:
        issues.extend(state.agent_proposals.issues)
    seen: set[tuple[str, str, str]] = set()
    deduped: list[StepIssue] = []
    for iss in issues:
        key = (str(iss.step), str(iss.severity), iss.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(iss)
    deduped.sort(key=lambda iss: (str(iss.step), str(iss.severity), iss.message))
    return deduped


__all__ = [
    "CandidateRow",
    "IndexRow",
    "ModulePageView",
    "aggregate",
]
