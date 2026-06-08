"""Tests for generated module-knowledge wiki rendering and verification."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.module_knowledge import KnowledgeBase, KnowledgeRecord, Provenance, SourceRef


def _record(record_id: str = "rec:kv") -> KnowledgeRecord:
    return KnowledgeRecord(
        record_id=record_id,
        source_type="paper",
        title="Paged KV cache",
        text="Paged KV cache improves decode memory locality.",
        source=SourceRef(
            source_id="src:paged-kv",
            title="Paged KV paper",
            url="https://example.com/paged-kv",
            trust_tier="primary",
        ),
        provenance=Provenance(
            locator="paper:abstract",
            extractor="fixture",
            artifact_path="artifacts/paged-kv.json",
        ),
        tags=["kv-cache", "decode"],
    )


def test_render_empty_wiki_creates_verifiable_empty_archive(tmp_path: Path) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")

    render = kb.render_wiki()
    report = kb.verify_wiki(strict=True)

    assert render.pages_written == 1
    assert kb.layout.records_jsonl.exists()
    assert "No records archived yet." in (kb.layout.wiki_dir / "index.md").read_text(
        encoding="utf-8"
    )
    assert report.ok, report.issues


def test_render_wiki_creates_verifiable_generated_pages_and_artifact_links(
    tmp_path: Path,
) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")
    artifact = kb.layout.root / "artifacts" / "paged-kv.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    kb.archive(_record())

    render = kb.render_wiki()
    report = kb.verify_wiki(strict=True)

    index = kb.layout.wiki_dir / "index.md"
    record_page = kb.layout.records_wiki_dir / "rec_kv.md"
    assert render.pages_written == 2
    assert index.exists()
    assert record_page.exists()
    assert "generated_from" in index.read_text(encoding="utf-8")
    assert "[`artifacts/paged-kv.json`](../../artifacts/paged-kv.json)" in record_page.read_text(
        encoding="utf-8"
    )
    assert report.ok, report.issues


def test_verify_wiki_detects_stale_pages_after_archive_changes(tmp_path: Path) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")
    kb.archive(_record())
    kb.render_wiki()

    kb.archive(_record("rec:new"))
    report = kb.verify_wiki(strict=True)

    assert not report.ok
    assert any("source hash mismatch" in issue.message for issue in report.issues)


def test_render_retrieval_wiki_writes_query_page(tmp_path: Path) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")
    kb.archive(_record())
    kb.render_wiki()
    results = kb.retrieve("kv cache decode", top_k=1)

    query_page = kb.render_retrieval_wiki("kv cache decode", results, query_id="kv-cache")
    report = kb.verify_wiki(strict=True)

    assert query_page.exists()
    text = query_page.read_text(encoding="utf-8")
    assert "# Retrieval: kv cache decode" in text
    assert "[Paged KV cache](../records/rec_kv.md)" in text
    assert report.ok, report.issues


def test_render_retrieval_wiki_sanitizes_query_id_and_handles_empty_archive(
    tmp_path: Path,
) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")

    query_page = kb.render_retrieval_wiki("empty query", [], query_id="../bad id")
    report = kb.verify_wiki(strict=True)

    assert query_page.name == "bad_id.md"
    assert query_page.exists()
    assert kb.layout.records_jsonl.exists()
    assert report.ok, report.issues
