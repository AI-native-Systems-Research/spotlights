"""Paper deduplication for expanded module deep research."""

from __future__ import annotations

import re

from spotlights_engine.module_deep_research.expanded.identifiers import (
    extract_arxiv_id,
    extract_doi,
)
from spotlights_engine.module_deep_research.expanded.models import MergedPaper, ResearchPaper


def normalize_title(title: str) -> str:
    """Return a lowercase alphanumeric title key."""
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def dedup_keys(paper: ResearchPaper) -> list[str]:
    """Build stable deduplication keys ordered from strongest to weakest."""
    keys: list[str] = []
    if paper.doi:
        keys.append(f"doi:{paper.doi.lower().rstrip('.')}")
    if paper.arxiv_id:
        keys.append(f"arxiv:{extract_arxiv_id(paper.arxiv_id) or paper.arxiv_id}")
    if paper.openreview_id:
        keys.append(f"openreview:{paper.openreview_id}")
    for value in (paper.source_url, paper.pdf_url):
        arxiv_id = extract_arxiv_id(value)
        if arxiv_id:
            keys.append(f"arxiv:{arxiv_id}")
        doi = extract_doi(value)
        if doi:
            keys.append(f"doi:{doi}")
    if not keys:
        author = paper.authors[0].casefold().strip() if paper.authors else "unknown"
        year = str(paper.year or paper.date or "unknown")[:4]
        keys.append(f"title:{normalize_title(paper.title)}:{author}:{year}")
    return sorted(set(keys))


def merge_papers(papers_by_agent: dict[str, list[ResearchPaper]]) -> list[MergedPaper]:
    """Merge agent paper records into canonical groups."""
    groups: list[_MergeGroup] = []
    key_to_group: dict[str, int] = {}

    for agent_name, papers in papers_by_agent.items():
        for paper in papers:
            keys = dedup_keys(paper)
            matches = sorted({key_to_group[key] for key in keys if key in key_to_group})
            if not matches:
                index = len(groups)
                groups.append(
                    _MergeGroup(paper=paper, agents={agent_name}, keys=set(keys), count=1)
                )
                for key in keys:
                    key_to_group[key] = index
                continue

            primary_index = matches[0]
            primary = groups[primary_index]
            primary.add(agent_name, paper, keys)
            for duplicate_index in reversed(matches[1:]):
                duplicate = groups.pop(duplicate_index)
                primary.absorb(duplicate)
                key_to_group = _reindex_after_pop(key_to_group, duplicate_index)
            for key in primary.keys:
                key_to_group[key] = primary_index

    return [group.to_merged() for group in sorted(groups, key=_group_sort_key, reverse=True)]


class _MergeGroup:
    def __init__(
        self, *, paper: ResearchPaper, agents: set[str], keys: set[str], count: int
    ) -> None:
        self.paper = paper
        self.agents = set(agents)
        self.keys = set(keys)
        self.count = count
        self.prompt_variants = {ev.prompt_variant_id for ev in paper.evidence}

    def add(self, agent_name: str, paper: ResearchPaper, keys: list[str]) -> None:
        self.paper = _choose_better(self.paper, paper)
        self.agents.add(agent_name)
        self.keys.update(keys)
        self.prompt_variants.update(ev.prompt_variant_id for ev in paper.evidence)
        self.count += 1

    def absorb(self, other: _MergeGroup) -> None:
        self.paper = _choose_better(self.paper, other.paper)
        self.agents.update(other.agents)
        self.keys.update(other.keys)
        self.prompt_variants.update(other.prompt_variants)
        self.count += other.count

    def to_merged(self) -> MergedPaper:
        status = "cross_checked" if len(self.agents) > 1 else "single_source"
        return MergedPaper(
            canonical_id=sorted(self.keys)[0],
            paper=self.paper,
            seen_by_agents=sorted(self.agents),  # type: ignore[arg-type]
            seen_by_prompt_variants=sorted(self.prompt_variants),
            duplicate_keys=sorted(self.keys),
            duplicate_count=self.count,
            verification_status=status,  # type: ignore[arg-type]
        )


def _choose_better(current: ResearchPaper, candidate: ResearchPaper) -> ResearchPaper:
    current_score = (
        current.relevance_score,
        len(current.abstract or ""),
        len(current.evidence),
        len(current.authors),
    )
    candidate_score = (
        candidate.relevance_score,
        len(candidate.abstract or ""),
        len(candidate.evidence),
        len(candidate.authors),
    )
    return candidate if candidate_score > current_score else current


def _group_sort_key(group: _MergeGroup) -> tuple[float, int, str]:
    return (group.paper.relevance_score, len(group.agents), group.paper.title.casefold())


def _reindex_after_pop(mapping: dict[str, int], removed_index: int) -> dict[str, int]:
    return {
        key: index - 1 if index > removed_index else index
        for key, index in mapping.items()
        if index != removed_index
    }
