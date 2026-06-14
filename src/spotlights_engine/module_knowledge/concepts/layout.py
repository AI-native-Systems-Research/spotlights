"""Filesystem layout for the concepts layer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from spotlights_engine.module_knowledge.concepts.schemas import ConceptKind


class ConceptsLayout:
    """Resolved paths for the concepts subtree under one knowledge root."""

    def __init__(self, knowledge_root: str | Path) -> None:
        self.knowledge_root = Path(knowledge_root).expanduser().resolve()

    @property
    def concepts_dir(self) -> Path:
        return self.knowledge_root / "wiki" / "concepts"

    @property
    def techniques_dir(self) -> Path:
        return self.concepts_dir / "techniques"

    @property
    def entities_dir(self) -> Path:
        return self.concepts_dir / "entities"

    @property
    def wiki_state_path(self) -> Path:
        return self.knowledge_root / "wiki_state.json"

    @property
    def ingest_log_dir(self) -> Path:
        return self.knowledge_root / "wiki" / "log"

    @property
    def wiki_index_path(self) -> Path:
        return self.knowledge_root / "wiki" / "index.md"

    @property
    def records_wiki_dir(self) -> Path:
        return self.knowledge_root / "wiki" / "records"

    def page_path(self, slug: str, kind: ConceptKind) -> Path:
        """Return the filesystem path for a concept page by slug and kind.

        Entity slugs are subject-system-namespaced: ``vllm/v1.kv_offload``
        maps to ``entities/vllm/v1.kv_offload.md``.
        """
        if kind == "technique":
            return self.techniques_dir / f"{slug}.md"
        return self.entities_dir / f"{slug}.md"

    def ensure(self) -> None:
        self.techniques_dir.mkdir(parents=True, exist_ok=True)
        self.entities_dir.mkdir(parents=True, exist_ok=True)
        self.ingest_log_dir.mkdir(parents=True, exist_ok=True)

    def iter_pages(self) -> Iterator[Path]:
        if not self.concepts_dir.exists():
            return
        yield from sorted(self.concepts_dir.rglob("*.md"))


__all__ = ["ConceptsLayout"]
