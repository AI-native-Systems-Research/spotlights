"""Concepts wiki: open/query/ingest/verify for the LLM-curated concept layer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from spotlights_engine.module_knowledge.concepts.layout import ConceptsLayout
from spotlights_engine.module_knowledge.concepts.query import query_concepts
from spotlights_engine.module_knowledge.concepts.schemas import (
    ConceptKind,
    ConceptLinks,
    ConceptPage,
    ConceptVerifyIssue,
    ConceptVerifyReport,
    IngestReport,
    IngestRequest,
    LightSourceRef,
    QueryResult,
    RecordSourceRef,
)

if TYPE_CHECKING:
    pass

_WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


class Wiki:
    """The concepts wiki: one instance per knowledge root."""

    def __init__(self, layout: ConceptsLayout) -> None:
        self._layout = layout

    @classmethod
    def open(cls, root: str | Path) -> "Wiki":
        """Open or create the concepts wiki at ``<root>/wiki/concepts/``. Idempotent."""
        layout = ConceptsLayout(root)
        layout.ensure()
        return cls(layout)

    # ------------------------------------------------------------------ query

    def query(
        self,
        q: str,
        *,
        top_k: int = 5,
        kinds: list[str] | None = None,
    ) -> list[QueryResult]:
        """Lexical ranking over concept pages."""
        return query_concepts(self._load_pages(), q, top_k=top_k, kinds=kinds)

    # ----------------------------------------------------------------- ingest

    async def ingest(self, request: IngestRequest) -> IngestReport:
        """Run an LLM ingest pass that folds the given sources into concept pages."""
        from spotlights_engine.module_knowledge.concepts.ingest import run_ingest

        return await run_ingest(self, request)

    # --------------------------------------------------------- resume support

    def read_resume_marker(self, run_id: str) -> int:
        """Return ``last_ingested_wave_idx`` for run_id, or -1 if no match."""
        state = self._read_wiki_state()
        if state is None or state.get("run_id") != run_id:
            return -1
        return int(state.get("last_ingested_wave_idx", -1))

    def bump_resume_marker(
        self, run_id: str, wave_idx: int, ingested_qns: list[str]
    ) -> None:
        """Atomically update wiki_state.json after a successful ingest."""
        existing = self._read_wiki_state() or {}
        if existing.get("run_id") != run_id:
            existing = {
                "schema_version": "wiki_state.v1",
                "run_id": run_id,
                "ingested_module_qns_in_current_run": [],
            }
        prev_qns: list[str] = existing.get("ingested_module_qns_in_current_run", [])
        merged_qns = sorted(set(prev_qns) | set(ingested_qns))
        new_state = {
            "schema_version": "wiki_state.v1",
            "run_id": run_id,
            "last_ingested_wave_idx": wave_idx,
            "ingested_module_qns_in_current_run": merged_qns,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        _atomic_write_json(self._layout.wiki_state_path, new_state)

    def filter_uningested(self, qns: list[str], run_id: str) -> list[str]:
        """Drop QNs already in wiki_state.ingested_module_qns_in_current_run."""
        state = self._read_wiki_state()
        if state is None or state.get("run_id") != run_id:
            return list(qns)
        already: set[str] = set(state.get("ingested_module_qns_in_current_run", []))
        return [qn for qn in qns if qn not in already]

    # ------------------------------------------------------------------ index

    def render_index(self) -> None:
        """Regenerate the cross-cutting wiki/index.md. Deterministic, no LLM."""
        concept_pages = self._load_pages()
        techniques = [p for p in concept_pages if p.kind == "technique"]
        entities = [p for p in concept_pages if p.kind == "entity"]

        records_dir = self._layout.records_wiki_dir
        record_pages = sorted(records_dir.rglob("*.md")) if records_dir.exists() else []

        index = self._layout.wiki_index_path
        lines = [
            "# Knowledge Wiki",
            "",
            f"Generated index. {len(concept_pages)} concept page(s), "
            f"{len(record_pages)} record page(s).",
            "",
            "## Technique pages",
            "",
        ]
        if techniques:
            for p in sorted(techniques, key=lambda x: x.slug):
                rel = os.path.relpath(
                    self._layout.page_path(p.slug, "technique"), index.parent
                ).replace(os.sep, "/")
                lines.append(f"- [{p.title}]({rel})")
        else:
            lines.append("(none)")
        lines += ["", "## Entity pages", ""]
        if entities:
            for p in sorted(entities, key=lambda x: x.slug):
                rel = os.path.relpath(
                    self._layout.page_path(p.slug, "entity"), index.parent
                ).replace(os.sep, "/")
                lines.append(f"- [{p.title}]({rel})")
        else:
            lines.append("(none)")
        lines += ["", "## Record pages", ""]
        if record_pages:
            for rp in record_pages:
                rel = os.path.relpath(rp, index.parent).replace(os.sep, "/")
                lines.append(f"- [{rp.stem}]({rel})")
        else:
            lines.append("(none)")
        lines.append("")

        index.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(index, "\n".join(lines))

    # ----------------------------------------------------------------- verify

    def verify(self, *, records_jsonl: Path | None = None) -> ConceptVerifyReport:
        """Check content_hash matches body and every [[link]] resolves.

        When records_jsonl is provided (records-and-concepts mode), also
        verifies that every record_id in sources exists in the JSONL.
        """
        issues: list[ConceptVerifyIssue] = []
        pages = self._load_pages_with_errors(issues)
        if not pages and not issues:
            return ConceptVerifyReport(ok=True, checked_pages=0)

        all_slugs = {p.slug for p in pages}

        record_ids_in_archive: set[str] | None = None
        if records_jsonl is not None and records_jsonl.exists():
            from spotlights_engine.module_knowledge.records.store import read_records

            record_ids_in_archive = {r.record_id for r in read_records(records_jsonl)}

        for page in pages:
            path = self._layout.page_path(page.slug, page.kind)
            # Content-hash check
            expected = body_hash(page.body)
            if page.content_hash and page.content_hash != expected:
                issues.append(
                    ConceptVerifyIssue(
                        page_path=str(path),
                        message=f"content_hash mismatch: expected {expected}",
                    )
                )
            # [[link]] resolution
            for match in _WIKI_LINK_RE.finditer(page.body):
                link = match.group(1).strip()
                slug = link.removeprefix("entities/").removeprefix("techniques/")
                if slug not in all_slugs and link not in all_slugs:
                    issues.append(
                        ConceptVerifyIssue(
                            page_path=str(path),
                            severity="warning",
                            message=f"unresolved [[link]]: {link}",
                        )
                    )
            # Record-id referential integrity
            if record_ids_in_archive is not None:
                for src in page.sources:
                    if isinstance(src, RecordSourceRef):
                        if src.record_id not in record_ids_in_archive:
                            issues.append(
                                ConceptVerifyIssue(
                                    page_path=str(path),
                                    message=f"sources cites unknown record_id: {src.record_id}",
                                )
                            )

        return ConceptVerifyReport(
            ok=not issues,
            checked_pages=len(pages),
            issues=issues,
        )

    # --------------------------------------------------------- package-internal

    def _load_pages(self) -> list[ConceptPage]:
        issues: list[ConceptVerifyIssue] = []
        return self._load_pages_with_errors(issues)

    def _load_pages_with_errors(
        self, issues: list[ConceptVerifyIssue]
    ) -> list[ConceptPage]:
        pages: list[ConceptPage] = []
        for path in self._layout.iter_pages():
            try:
                text = path.read_text(encoding="utf-8")
                pages.append(parse_page(text, path))
            except Exception as exc:
                issues.append(
                    ConceptVerifyIssue(
                        page_path=str(path),
                        message=f"failed to parse page: {exc}",
                    )
                )
        return pages

    def _write_page(self, page: ConceptPage) -> None:
        path = self._layout.page_path(page.slug, page.kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, serialize_page(page))

    def _read_wiki_state(self) -> dict | None:
        path = self._layout.wiki_state_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None


# ------------------------------------------------------------------ page I/O


def body_hash(body: str) -> str:
    return "sha256:" + hashlib.sha256((body or "").encode("utf-8")).hexdigest()


def parse_page(text: str, path: Path | None = None) -> ConceptPage:
    """Parse a concept page from raw markdown text (frontmatter + body)."""
    label = str(path) if path else "<string>"
    if not text.startswith("---"):
        raise ValueError(f"missing frontmatter in {label}")
    rest = text[3:]
    # Allow both \n---\n and \n--- (end of file)
    end = rest.find("\n---\n")
    skip = 4
    if end == -1:
        end = rest.find("\n---")
        skip = 4
    if end == -1:
        raise ValueError(f"unclosed frontmatter in {label}")

    fm_text = rest[:end]
    body_text = rest[end + skip :].lstrip("\n")

    fm = yaml.safe_load(fm_text)
    if not isinstance(fm, dict):
        raise ValueError(f"frontmatter is not a mapping in {label}")

    sources: list[RecordSourceRef | LightSourceRef] = []
    for src in fm.get("sources") or []:
        if not isinstance(src, dict):
            continue
        if "record_id" in src:
            sources.append(RecordSourceRef(record_id=src["record_id"], kind=src.get("kind", "")))
        else:
            sources.append(
                LightSourceRef(
                    kind=src.get("kind", ""),
                    id=src.get("id", ""),
                    title=src.get("title", ""),
                    url=src.get("url"),
                )
            )

    raw_links = fm.get("links") or {}
    links = ConceptLinks(
        techniques=list(raw_links.get("techniques") or []),
        entities=list(raw_links.get("entities") or []),
    )

    return ConceptPage(
        schema_version=fm.get("schema_version", "concepts.v1"),
        kind=fm["kind"],
        slug=fm["slug"],
        title=fm["title"],
        as_of_run=fm["as_of_run"],
        subject_system=fm.get("subject_system"),
        subject_systems_seen=list(fm.get("subject_systems_seen") or []),
        objectives_seen=list(fm.get("objectives_seen") or []),
        sources=sources,
        links=links,
        content_hash=fm.get("content_hash"),
        body=body_text,
    )


def serialize_page(page: ConceptPage) -> str:
    """Serialize a ConceptPage to raw markdown text with YAML frontmatter."""
    fm: dict = {
        "schema_version": page.schema_version,
        "kind": page.kind,
        "slug": page.slug,
        "title": page.title,
        "as_of_run": page.as_of_run,
    }
    if page.kind == "technique" and page.subject_systems_seen:
        fm["subject_systems_seen"] = page.subject_systems_seen
    if page.kind == "entity":
        if page.subject_system:
            fm["subject_system"] = page.subject_system
        if page.objectives_seen:
            fm["objectives_seen"] = page.objectives_seen

    if page.sources:
        fm["sources"] = [src.model_dump(exclude_none=True) for src in page.sources]

    if page.links.techniques or page.links.entities:
        fm["links"] = {}
        if page.links.techniques:
            fm["links"]["techniques"] = page.links.techniques
        if page.links.entities:
            fm["links"]["entities"] = page.links.entities

    if page.content_hash:
        fm["content_hash"] = page.content_hash

    fm_yaml = yaml.dump(fm, allow_unicode=True, sort_keys=False, default_flow_style=False)
    body = page.body or ""
    return f"---\n{fm_yaml}---\n\n{body}"


# ----------------------------------------------------------------- file utils


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        Path(tmp).replace(path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, data: dict) -> None:
    _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


__all__ = [
    "Wiki",
    "body_hash",
    "parse_page",
    "serialize_page",
]
