"""Reduce expanded paper records back to Spotlight `Finding` objects."""

from __future__ import annotations

from spotlights_engine.module_deep_research.expanded.models import MergedPaper
from spotlights_engine.schemas.finding import Finding


def reduce_to_findings(*, merged_papers: list[MergedPaper], max_findings: int) -> list[Finding]:
    """Convert top-ranked merged papers into the public Spotlight finding contract."""
    findings: list[Finding] = []
    for idx, merged in enumerate(merged_papers[:max_findings], start=1):
        paper = merged.paper
        evidence = paper.evidence[0].quote_or_note if paper.evidence else paper.why_relevant
        source_url = paper.source_url or paper.pdf_url or "unknown"
        findings.append(
            Finding(
                finding_id=f"find-{idx:04d}",
                title=paper.title,
                url=source_url,
                source_type="paper",
                technique_summary=paper.transferable_idea,
                supporting_evidence=evidence,
            )
        )
    return findings
