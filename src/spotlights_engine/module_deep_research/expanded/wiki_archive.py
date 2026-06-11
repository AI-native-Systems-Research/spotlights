"""Archive expanded deep-search results into the module-knowledge wiki."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from spotlights_engine.module_deep_research.expanded.identifiers import normalize_arxiv_id
from spotlights_engine.module_deep_research.expanded.models import (
    ExpandedResearchReport,
    MergedPaper,
    ResearchPaper,
)
from spotlights_engine.module_knowledge import KnowledgeBase, KnowledgeRecord, Provenance, SourceRef
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput


def archive_expanded_research_to_wiki(
    *,
    report: ExpandedResearchReport,
    output: ModuleDeepResearchOutput,
    artifacts_dir: Path | None,
    knowledge_root: Path | None,
) -> Path | None:
    """Archive a human-readable wiki view, returning the wiki directory if written."""
    root = _resolve_knowledge_root(artifacts_dir=artifacts_dir, knowledge_root=knowledge_root)
    if root is None:
        return None

    kb = KnowledgeBase.open(root)
    kb.archive_many(
        _records_from_report(
            report=report,
            output=output,
            artifacts_dir=artifacts_dir,
            knowledge_root=root,
        )
    )
    return Path(kb.render_wiki().wiki_dir)


def _resolve_knowledge_root(
    *, artifacts_dir: Path | None, knowledge_root: Path | None
) -> Path | None:
    if knowledge_root is not None:
        return knowledge_root
    if artifacts_dir is not None:
        return artifacts_dir / "knowledge"
    return None


def _records_from_report(
    *,
    report: ExpandedResearchReport,
    output: ModuleDeepResearchOutput,
    artifacts_dir: Path | None,
    knowledge_root: Path,
) -> list[KnowledgeRecord]:
    module_name = report.packet.module_qualified_name
    artifact_path = _artifact_path(
        artifacts_dir=artifacts_dir,
        knowledge_root=knowledge_root,
        name="expanded_research_report.json",
    )
    records = [
        _summary_record(
            report=report,
            output=output,
            module_name=module_name,
            artifact_path=artifact_path,
        )
    ]
    records.extend(
        _paper_record(
            paper=merged,
            module_name=module_name,
            artifact_path=artifact_path,
        )
        for merged in report.merged_papers
    )
    return records


def _summary_record(
    *,
    report: ExpandedResearchReport,
    output: ModuleDeepResearchOutput,
    module_name: str,
    artifact_path: str | None,
) -> KnowledgeRecord:
    agents = sorted({agent for paper in report.merged_papers for agent in paper.seen_by_agents})
    selected = report.prompt_evolution.selected_variant_ids
    text = "\n".join(
        [
            "## Expanded deep-search summary",
            "",
            f"- Module: `{module_name}`",
            f"- Module path: `{report.packet.module_path}`",
            f"- Objective: {report.packet.objective}",
            f"- Agents: {', '.join(f'`{agent}`' for agent in agents) or '(none)'}",
            f"- Vanilla baseline findings: {len(report.comparison.baseline_finding_titles)}",
            f"- Expanded findings returned: {len(output.findings)}",
            f"- Merged papers discovered: {len(report.merged_papers)}",
            f"- Added vs vanilla: {len(report.comparison.added_titles)}",
            f"- Overlap with vanilla: {len(report.comparison.overlap_titles)}",
            f"- Baseline-only: {len(report.comparison.baseline_only_titles)}",
            "",
            "## Selected prompt variants",
            "",
            *[
                f"- `{agent}`: {', '.join(f'`{variant}`' for variant in variants)}"
                for agent, variants in sorted(selected.items())
            ],
            "",
            "## Added vs vanilla",
            "",
            *(_bullet_lines(report.comparison.added_titles) or ["- (none)"]),
            "",
            "## Coverage",
            "",
            *[
                f"- `{agent}`: {len(ids)} papers; unique: "
                f"{len(report.coverage.only_by.get(agent, []))}"
                for agent, ids in sorted(report.coverage.sets.items())
            ],
        ]
    )
    return KnowledgeRecord(
        record_id=_record_id(f"expanded_research:{module_name}:summary"),
        source_type="experiment",
        title=f"Expanded deep research summary: {module_name}",
        text=text,
        source=SourceRef(
            source_id=_record_id(f"expanded_research:{module_name}"),
            title=f"Expanded deep research for {module_name}",
            trust_tier="internal",
        ),
        provenance=Provenance(
            locator=f"expanded_module_deep_research:{module_name}:summary",
            extractor="expanded_module_deep_research",
            artifact_path=artifact_path,
        ),
        tags=[module_name, "expanded-deep-research", "summary"],
        metadata={
            "module_qualified_name": module_name,
            "agents": agents,
            "merged_papers": len(report.merged_papers),
            "expanded_findings": len(output.findings),
            "added_vs_vanilla": len(report.comparison.added_titles),
        },
    )


def _paper_record(
    *,
    paper: MergedPaper,
    module_name: str,
    artifact_path: str | None,
) -> KnowledgeRecord:
    source = paper.paper.source_url or paper.paper.pdf_url
    prompt_variants = ", ".join(
        f"`{variant}`" for variant in paper.seen_by_prompt_variants
    )
    canonical_id = _canonical_archive_id(paper)
    duplicate_keys = _valid_duplicate_keys(paper)
    text = "\n".join(
        [
            "## Paper",
            "",
            f"- Canonical ID: `{canonical_id}`",
            f"- Verification: `{paper.verification_status}`",
            f"- Seen by: {', '.join(f'`{agent}`' for agent in paper.seen_by_agents)}",
            f"- Prompt variants: {prompt_variants}",
            f"- Duplicate count: {paper.duplicate_count}",
            "",
            "## Why it matters",
            "",
            paper.paper.why_relevant,
            "",
            "## Transferable idea",
            "",
            paper.paper.transferable_idea,
            "",
            "## Evidence",
            "",
            *(_evidence_lines(paper.paper) or ["- (none)"]),
        ]
    )
    return KnowledgeRecord(
        record_id=_record_id(f"expanded_research:{module_name}:{canonical_id}"),
        source_type="paper",
        title=paper.paper.title,
        text=text,
        source=SourceRef(
            source_id=_record_id(canonical_id),
            title=paper.paper.title,
            url=source,
            authors=paper.paper.authors,
            published_at=paper.paper.date or str(paper.paper.year or ""),
            trust_tier="credible",
        ),
        provenance=Provenance(
            locator=f"expanded_module_deep_research:{module_name}:{canonical_id}",
            extractor="expanded_module_deep_research",
            artifact_path=artifact_path,
        ),
        tags=[
            module_name,
            "expanded-deep-research",
            "paper",
            *paper.seen_by_agents,
            paper.verification_status,
        ],
        metadata={
            "module_qualified_name": module_name,
            "canonical_id": canonical_id,
            "seen_by_agents": list(paper.seen_by_agents),
            "seen_by_prompt_variants": list(paper.seen_by_prompt_variants),
            "duplicate_keys": duplicate_keys,
            "duplicate_count": paper.duplicate_count,
            "verification_status": paper.verification_status,
            "identifiers": _identifiers(paper.paper),
        },
    )


def _canonical_archive_id(paper: MergedPaper) -> str:
    keys = _valid_duplicate_keys(paper)
    return keys[0] if keys else _title_key(paper.paper)


def _valid_duplicate_keys(paper: MergedPaper) -> list[str]:
    keys: list[str] = []
    for key in paper.duplicate_keys:
        if key.startswith("arxiv:"):
            arxiv_id = normalize_arxiv_id(key.removeprefix("arxiv:"))
            if arxiv_id is not None:
                keys.append(f"arxiv:{arxiv_id}")
            continue
        keys.append(key)
    return sorted(set(keys))


def _title_key(paper: ResearchPaper) -> str:
    return "title:" + re.sub(r"[^a-z0-9]+", "-", paper.title.casefold()).strip("-")


def _identifiers(paper: ResearchPaper) -> dict[str, Any]:
    return {
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "openreview_id": paper.openreview_id,
        "source_url": paper.source_url,
        "pdf_url": paper.pdf_url,
    }


def _evidence_lines(paper: ResearchPaper) -> list[str]:
    return [
        "- "
        + "; ".join(
            part
            for part in [
                f"`{item.agent_name}`",
                f"`{item.prompt_variant_id}`",
                item.quote_or_note,
                item.source_url,
            ]
            if part
        )
        for item in paper.evidence
    ]


def _bullet_lines(values: list[str]) -> list[str]:
    return [f"- {value}" for value in values]


def _artifact_path(*, artifacts_dir: Path | None, knowledge_root: Path, name: str) -> str | None:
    if artifacts_dir is None:
        return None
    artifact = artifacts_dir / name
    return os.path.relpath(artifact, knowledge_root).replace(os.sep, "/")


def _record_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", ":", value.strip())
    return ":".join(part for part in cleaned.split(":") if part) or "expanded_research:unknown"


__all__ = ["archive_expanded_research_to_wiki"]
