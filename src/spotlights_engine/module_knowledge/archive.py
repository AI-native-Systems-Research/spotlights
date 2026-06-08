"""Public Archive/Retrieve API for module knowledge."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from spotlights_engine.module_knowledge.layout import KnowledgeLayout
from spotlights_engine.module_knowledge.retrieve import retrieve_records
from spotlights_engine.module_knowledge.schemas import (
    ArchiveResult,
    KnowledgeRecord,
    RetrievedItem,
    RetrieveRequest,
    WikiRenderResult,
    WikiVerificationReport,
)
from spotlights_engine.module_knowledge.store import read_records, upsert_records, with_content_hash
from spotlights_engine.module_knowledge.wiki import WikiRenderer


class KnowledgeBase:
    """Small local knowledge base: Archive API, Retrieve API, generated wiki."""

    def __init__(self, layout: KnowledgeLayout) -> None:
        self.layout = layout

    @classmethod
    def open(cls, root: str | Path) -> KnowledgeBase:
        layout = KnowledgeLayout(root)
        layout.ensure()
        return cls(layout)

    def archive(self, record: KnowledgeRecord) -> ArchiveResult:
        """Insert or update one knowledge record."""
        return self.archive_many([record])

    def archive_many(self, records: Iterable[KnowledgeRecord]) -> ArchiveResult:
        """Insert or update knowledge records by `record_id`."""
        self.layout.ensure()
        prepared = [with_content_hash(record) for record in records]
        total, inserted, updated = upsert_records(prepared, self.layout.records_jsonl)
        return ArchiveResult(
            path=str(self.layout.records_jsonl),
            records_written=total,
            inserted=inserted,
            updated=updated,
        )

    def records(self) -> list[KnowledgeRecord]:
        """Read archived records in deterministic order."""
        return read_records(self.layout.records_jsonl)

    def retrieve(
        self,
        query: str | RetrieveRequest,
        *,
        top_k: int | None = None,
    ) -> list[RetrievedItem]:
        """Retrieve ranked local records from the archive."""
        return retrieve_records(self.records(), query, top_k=top_k)

    def render_wiki(self) -> WikiRenderResult:
        """Render a generated Markdown wiki from archived JSONL records."""
        return WikiRenderer(self.layout).render()

    def render_retrieval_wiki(
        self,
        query: str,
        results: list[RetrievedItem],
        *,
        query_id: str | None = None,
    ) -> Path:
        """Render one query-result page into the generated wiki."""
        return WikiRenderer(self.layout).render_retrieval(query, results, query_id=query_id)

    def verify_wiki(self, *, strict: bool = True) -> WikiVerificationReport:
        """Verify generated wiki metadata, source hashes, and local links."""
        return WikiRenderer(self.layout).verify(strict=strict)


__all__ = ["KnowledgeBase"]
