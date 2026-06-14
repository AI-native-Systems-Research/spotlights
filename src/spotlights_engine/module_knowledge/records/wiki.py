"""Generated Markdown wiki for module knowledge archives."""

from __future__ import annotations

import os
import re
from pathlib import Path

from spotlights_engine.module_knowledge.records.layout import KnowledgeLayout
from spotlights_engine.module_knowledge.records.schemas import (
    KnowledgeRecord,
    RetrievedItem,
    WikiRenderResult,
    WikiVerificationIssue,
    WikiVerificationReport,
)
from spotlights_engine.module_knowledge.records.store import read_records, sha256_file, write_records

_GENERATED_FROM_RE = re.compile(r"<!-- generated_from: (?P<path>.*?) -->")
_SOURCE_HASH_RE = re.compile(r"<!-- source_hash: (?P<hash>sha256:[a-f0-9]{64}) -->")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\((?P<link>[^)]+)\)")


class WikiRenderer:
    """Render and verify Markdown pages derived from `archive/records.jsonl`."""

    def __init__(self, layout: KnowledgeLayout) -> None:
        self.layout = layout

    def render(self) -> WikiRenderResult:
        self.layout.ensure()
        if not self.layout.records_jsonl.exists():
            write_records([], self.layout.records_jsonl)
        records = read_records(self.layout.records_jsonl)
        source_hash = sha256_file(self.layout.records_jsonl)
        pages = [self._render_index(records, source_hash)]
        pages.extend(self._render_record(record, source_hash) for record in records)
        return WikiRenderResult(
            wiki_dir=str(self.layout.wiki_dir),
            pages_written=len(pages),
            source_hash=source_hash,
        )

    def render_retrieval(
        self,
        query: str,
        results: list[RetrievedItem],
        *,
        query_id: str | None = None,
    ) -> Path:
        self.layout.ensure()
        if not self.layout.records_jsonl.exists():
            write_records([], self.layout.records_jsonl)
        records = read_records(self.layout.records_jsonl)
        source_hash = sha256_file(self.layout.records_jsonl)
        self._render_index(records, source_hash)
        for item in results:
            self._render_record(item.record, source_hash)
        safe_query_id = _safe_id(query_id or query)[:80]
        page = self.layout.queries_wiki_dir / f"{safe_query_id}.md"
        lines = [
            f"<!-- generated_from: {_rel(page, self.layout.records_jsonl)} -->",
            f"<!-- source_hash: {source_hash} -->",
            "",
            f"# Retrieval: {query}",
            "",
            f"- [Knowledge home]({_rel(page, self.layout.wiki_dir / 'index.md')})",
            "",
            "## Results",
            "",
        ]
        if not results:
            lines.extend(["No results.", ""])
        for idx, item in enumerate(results, start=1):
            record_page = self.layout.records_wiki_dir / f"{_safe_id(item.record.record_id)}.md"
            lines.extend(
                [
                    f"### {idx}. [{item.record.title}]({_rel(page, record_page)})",
                    "",
                    f"- Type: `{item.record.source_type}`",
                    f"- Score: `{item.score:.4f}`",
                    f"- Matched terms: {', '.join(item.matched_terms) or '(none)'}",
                    "",
                    item.record.text,
                    "",
                ]
            )
        page.write_text("\n".join(lines), encoding="utf-8")
        return page

    def verify(self, *, strict: bool = True) -> WikiVerificationReport:
        return verify_wiki(self.layout, strict=strict)

    def _render_index(self, records: list[KnowledgeRecord], source_hash: str) -> Path:
        page = self.layout.wiki_dir / "index.md"
        by_type: dict[str, int] = {}
        for record in records:
            by_type[record.source_type] = by_type.get(record.source_type, 0) + 1
        lines = [
            f"<!-- generated_from: {_rel(page, self.layout.records_jsonl)} -->",
            f"<!-- source_hash: {source_hash} -->",
            "",
            "# Module Knowledge",
            "",
            "Generated view over `archive/records.jsonl`. JSONL is the source of truth.",
            "",
            "## Counts",
            "",
        ]
        if by_type:
            lines.extend(
                f"- `{source_type}`: {count}" for source_type, count in sorted(by_type.items())
            )
        else:
            lines.append("No records archived yet.")
        lines.extend(["", "## Records", ""])
        for record in sorted(
            records, key=lambda item: (item.source_type, item.title, item.record_id)
        ):
            record_page = self.layout.records_wiki_dir / f"{_safe_id(record.record_id)}.md"
            lines.append(f"- [{record.title}]({_rel(page, record_page)}) — `{record.source_type}`")
        page.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return page

    def _render_record(self, record: KnowledgeRecord, source_hash: str) -> Path:
        page = self.layout.records_wiki_dir / f"{_safe_id(record.record_id)}.md"
        lines = [
            f"<!-- generated_from: {_rel(page, self.layout.records_jsonl)} -->",
            f"<!-- source_hash: {source_hash} -->",
            "",
            f"# {record.title}",
            "",
            f"- Record ID: `{record.record_id}`",
            f"- Type: `{record.source_type}`",
            f"- Source: `{record.source.source_id}` — {record.source.title}",
            f"- Trust: `{record.source.trust_tier}`",
            f"- Locator: `{record.provenance.locator}`",
            f"- Content hash: `{record.provenance.content_hash or ''}`",
            f"- Tags: {', '.join(f'`{tag}`' for tag in record.tags) or '(none)'}",
        ]
        if record.source.url:
            lines.append(f"- Original URL: [{record.source.url}]({record.source.url})")
        artifact_link = _artifact_link(self.layout, page, record)
        if artifact_link:
            lines.append(f"- Artifact: {artifact_link}")
        lines.extend(["", "## Evidence", "", record.text, ""])
        page.write_text("\n".join(lines), encoding="utf-8")
        return page


def verify_wiki(layout: KnowledgeLayout, *, strict: bool = True) -> WikiVerificationReport:
    issues: list[WikiVerificationIssue] = []
    checked = 0
    if not layout.wiki_dir.exists():
        return WikiVerificationReport(
            ok=False,
            checked_pages=0,
            issues=[
                WikiVerificationIssue(
                    page_path=str(layout.wiki_dir),
                    message="wiki directory does not exist",
                )
            ],
        )

    for page in sorted(layout.wiki_dir.rglob("*.md")):
        checked += 1
        text = page.read_text(encoding="utf-8")
        _verify_generated_metadata(page, text, issues)
        _verify_local_links(page, text, issues)

    if strict:
        records = read_records(layout.records_jsonl)
        index = layout.wiki_dir / "index.md"
        if not index.exists():
            issues.append(WikiVerificationIssue(page_path=str(index), message="missing wiki index"))
        for record in records:
            record_page = layout.records_wiki_dir / f"{_safe_id(record.record_id)}.md"
            if not record_page.exists():
                issues.append(
                    WikiVerificationIssue(
                        page_path=str(record_page),
                        message=f"missing record page for {record.record_id}",
                    )
                )
    return WikiVerificationReport(ok=not issues, checked_pages=checked, issues=issues)


def _verify_generated_metadata(
    page: Path,
    text: str,
    issues: list[WikiVerificationIssue],
) -> None:
    generated = _GENERATED_FROM_RE.search(text)
    source_hash = _SOURCE_HASH_RE.search(text)
    if not generated or not source_hash:
        issues.append(
            WikiVerificationIssue(
                page_path=str(page),
                message="generated page is missing generated_from or source_hash metadata",
            )
        )
        return
    source = (page.parent / generated.group("path")).resolve()
    if not source.exists():
        issues.append(
            WikiVerificationIssue(
                page_path=str(page),
                message=f"generated source does not exist: {source}",
            )
        )
        return
    expected = source_hash.group("hash")
    actual = sha256_file(source)
    if actual != expected:
        issues.append(
            WikiVerificationIssue(
                page_path=str(page),
                message=f"source hash mismatch: expected {expected}, got {actual}",
            )
        )


def _verify_local_links(page: Path, text: str, issues: list[WikiVerificationIssue]) -> None:
    for match in _MARKDOWN_LINK_RE.finditer(text):
        link = match.group("link").split("#", 1)[0]
        if not link or link.startswith(("http://", "https://", "mailto:")):
            continue
        target = (page.parent / link).resolve()
        if not target.exists():
            issues.append(
                WikiVerificationIssue(
                    page_path=str(page),
                    message=f"broken local link: {link}",
                )
            )


def _artifact_link(layout: KnowledgeLayout, page: Path, record: KnowledgeRecord) -> str | None:
    if not record.provenance.artifact_path:
        return None
    artifact = layout.root / record.provenance.artifact_path
    if not artifact.exists():
        return None
    return f"[`{record.provenance.artifact_path}`]({_rel(page, artifact)})"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _rel(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(Path(to_path).resolve(), from_path.parent.resolve()).replace(os.sep, "/")


__all__ = ["WikiRenderer", "verify_wiki"]
