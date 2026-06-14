"""Filesystem layout for the module-knowledge archive."""

from __future__ import annotations

from pathlib import Path


class KnowledgeLayout:
    """Resolved paths for one local knowledge base."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    @property
    def archive_dir(self) -> Path:
        return self.root / "archive"

    @property
    def records_jsonl(self) -> Path:
        return self.archive_dir / "records.jsonl"

    @property
    def wiki_dir(self) -> Path:
        return self.root / "wiki"

    @property
    def records_wiki_dir(self) -> Path:
        return self.wiki_dir / "records"

    @property
    def queries_wiki_dir(self) -> Path:
        return self.wiki_dir / "queries"

    def ensure(self) -> None:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.records_wiki_dir.mkdir(parents=True, exist_ok=True)
        self.queries_wiki_dir.mkdir(parents=True, exist_ok=True)


__all__ = ["KnowledgeLayout"]
