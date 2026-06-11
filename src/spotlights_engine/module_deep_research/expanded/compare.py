"""Baseline-vs-expanded comparison helpers."""

from __future__ import annotations

from spotlights_engine.module_deep_research.expanded.dedup import normalize_title
from spotlights_engine.module_deep_research.expanded.models import (
    BeforeAfterComparison,
    CoverageReport,
    MergedPaper,
)
from spotlights_engine.schemas.finding import Finding


def compare_findings(
    *, baseline_findings: list[Finding], expanded_findings: list[Finding]
) -> BeforeAfterComparison:
    """Compare vanilla Codex-only findings to expanded findings by title."""
    baseline = {normalize_title(f.title): f.title for f in baseline_findings}
    expanded = {normalize_title(f.title): f.title for f in expanded_findings}
    baseline_keys = set(baseline)
    expanded_keys = set(expanded)
    return BeforeAfterComparison(
        baseline_finding_titles=[baseline[key] for key in sorted(baseline_keys)],
        expanded_finding_titles=[expanded[key] for key in sorted(expanded_keys)],
        added_titles=[expanded[key] for key in sorted(expanded_keys - baseline_keys)],
        baseline_only_titles=[baseline[key] for key in sorted(baseline_keys - expanded_keys)],
        overlap_titles=[expanded[key] for key in sorted(expanded_keys & baseline_keys)],
    )


def build_coverage_report(
    *, module_qualified_name: str, merged_papers: list[MergedPaper]
) -> CoverageReport:
    """Build Venn/matrix-friendly coverage sets from merged papers."""
    agents = sorted({agent for paper in merged_papers for agent in paper.seen_by_agents})
    sets = {
        agent: sorted(
            paper.canonical_id for paper in merged_papers if agent in paper.seen_by_agents
        )
        for agent in agents
    }
    all_sets = [set(values) for values in sets.values()]
    common = sorted(set.intersection(*all_sets)) if all_sets else []
    only_by = {
        agent: sorted(set(values) - set().union(*(set(v) for k, v in sets.items() if k != agent)))
        for agent, values in sets.items()
    }
    return CoverageReport(
        module_qualified_name=module_qualified_name,
        agents=agents,  # type: ignore[arg-type]
        sets=sets,
        only_by=only_by,
        all_agents=common,
    )
