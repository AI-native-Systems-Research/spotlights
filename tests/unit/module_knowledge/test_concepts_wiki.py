"""Tests for the concepts-layer wiki: parse/serialize, verify, query, resume markers."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.module_knowledge.concepts.schemas import (
    ConceptLinks,
    ConceptPage,
    LightSourceRef,
    RecordSourceRef,
)
from spotlights_engine.module_knowledge.concepts.wiki import (
    Wiki,
    body_hash,
    parse_page,
    serialize_page,
)


# ------------------------------------------------------------------ helpers


def _technique(
    slug: str = "paged-attention",
    title: str = "Paged Attention",
    body: str = "Paged attention improves KV cache memory efficiency.",
) -> ConceptPage:
    return ConceptPage(
        kind="technique",
        slug=slug,
        title=title,
        as_of_run="run-001",
        body=body,
        content_hash=body_hash(body),
    )


def _entity(
    slug: str = "vllm/v1.kv_offload",
    title: str = "vLLM v1 KV Offload",
    body: str = "The KV offload module manages GPU-to-CPU offloading.",
) -> ConceptPage:
    return ConceptPage(
        kind="entity",
        slug=slug,
        title=title,
        as_of_run="run-001",
        subject_system="vllm",
        body=body,
        content_hash=body_hash(body),
    )


# ----------------------------------------------------------- parse / serialize


def test_serialize_parse_roundtrip_technique() -> None:
    page = _technique()
    parsed = parse_page(serialize_page(page))
    assert parsed.kind == "technique"
    assert parsed.slug == page.slug
    assert parsed.title == page.title
    assert parsed.body == page.body
    assert parsed.content_hash == page.content_hash


def test_serialize_parse_roundtrip_entity() -> None:
    page = _entity()
    parsed = parse_page(serialize_page(page))
    assert parsed.kind == "entity"
    assert parsed.slug == page.slug
    assert parsed.subject_system == "vllm"
    assert parsed.body == page.body


def test_serialize_parse_record_source_ref() -> None:
    page = _technique().model_copy(
        update={"sources": [RecordSourceRef(record_id="rec:abc", kind="finding")]}
    )
    parsed = parse_page(serialize_page(page))
    assert len(parsed.sources) == 1
    src = parsed.sources[0]
    assert isinstance(src, RecordSourceRef)
    assert src.record_id == "rec:abc"


def test_serialize_parse_light_source_ref() -> None:
    page = _technique().model_copy(
        update={
            "sources": [
                LightSourceRef(kind="paper", id="arxiv:1234", title="KV Paper", url=None)
            ]
        }
    )
    parsed = parse_page(serialize_page(page))
    assert len(parsed.sources) == 1
    src = parsed.sources[0]
    assert isinstance(src, LightSourceRef)
    assert src.id == "arxiv:1234"


def test_serialize_parse_links() -> None:
    page = _entity().model_copy(
        update={"links": ConceptLinks(techniques=["paged-attention"], entities=[])}
    )
    parsed = parse_page(serialize_page(page))
    assert parsed.links.techniques == ["paged-attention"]


def test_parse_page_raises_on_missing_frontmatter() -> None:
    with pytest.raises(ValueError, match="missing frontmatter"):
        parse_page("No frontmatter here.\n")


def test_parse_page_raises_on_unclosed_frontmatter() -> None:
    with pytest.raises(ValueError, match="unclosed frontmatter"):
        parse_page("---\nkind: technique\nslug: foo\n")


# ----------------------------------------------------------------- body_hash


def test_body_hash_is_stable() -> None:
    assert body_hash("hello world") == body_hash("hello world")


def test_body_hash_starts_with_sha256_prefix() -> None:
    assert body_hash("x").startswith("sha256:")


def test_body_hash_differs_for_different_bodies() -> None:
    assert body_hash("abc") != body_hash("xyz")


def test_body_hash_empty_string() -> None:
    assert body_hash("").startswith("sha256:")


# ----------------------------------------------------------------- Wiki.open


def test_wiki_open_creates_directories(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    assert wiki._layout.techniques_dir.exists()
    assert wiki._layout.entities_dir.exists()


# --------------------------------------------------------------- Wiki.verify


def test_verify_empty_wiki_is_ok(tmp_path: Path) -> None:
    report = Wiki.open(tmp_path / "kb").verify()
    assert report.ok
    assert report.checked_pages == 0


def test_verify_page_with_correct_hash_is_ok(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki._write_page(_technique())
    report = wiki.verify()
    assert report.ok, report.issues
    assert report.checked_pages == 1


def test_verify_detects_content_hash_mismatch(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki._write_page(_technique().model_copy(update={"content_hash": "sha256:deadbeef"}))
    report = wiki.verify()
    assert not report.ok
    assert any("content_hash mismatch" in i.message for i in report.issues)


def test_verify_missing_hash_is_not_an_error(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki._write_page(_technique().model_copy(update={"content_hash": None}))
    assert wiki.verify().ok


def test_verify_resolved_wiki_link_is_ok(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    target_body = "Target body."
    wiki._write_page(
        ConceptPage(
            kind="technique",
            slug="chunked-prefill",
            title="Chunked Prefill",
            as_of_run="run-001",
            body=target_body,
            content_hash=body_hash(target_body),
        )
    )
    ref_body = "See [[chunked-prefill]] for details."
    wiki._write_page(
        ConceptPage(
            kind="technique",
            slug="paged-attention",
            title="Paged Attention",
            as_of_run="run-001",
            body=ref_body,
            content_hash=body_hash(ref_body),
        )
    )
    assert wiki.verify().ok


def test_verify_unresolved_wiki_link_emits_warning(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    body = "See [[does-not-exist]] for details."
    wiki._write_page(
        ConceptPage(
            kind="technique",
            slug="paged-attention",
            title="Paged Attention",
            as_of_run="run-001",
            body=body,
            content_hash=body_hash(body),
        )
    )
    report = wiki.verify()
    assert not report.ok
    warnings = [i for i in report.issues if i.severity == "warning"]
    assert any("does-not-exist" in i.message for i in warnings)


def test_verify_with_records_jsonl_known_record_id_is_ok(tmp_path: Path) -> None:
    from spotlights_engine.module_knowledge.records.schemas import (
        KnowledgeRecord,
        Provenance,
        SourceRef,
    )
    from spotlights_engine.module_knowledge.records.store import write_records

    jsonl = tmp_path / "records.jsonl"
    write_records(
        [
            KnowledgeRecord(
                record_id="rec:kv",
                source_type="paper",
                title="KV Cache Paper",
                text="body",
                source=SourceRef(
                    source_id="src:kv",
                    title="KV paper",
                    url="https://example.com",
                    trust_tier="primary",
                ),
                provenance=Provenance(
                    locator="abstract",
                    extractor="fixture",
                    artifact_path="a.json",
                ),
            )
        ],
        jsonl,
    )

    wiki = Wiki.open(tmp_path / "kb")
    wiki._write_page(
        _technique().model_copy(
            update={"sources": [RecordSourceRef(record_id="rec:kv", kind="finding")]}
        )
    )
    assert wiki.verify(records_jsonl=jsonl).ok


def test_verify_with_records_jsonl_unknown_record_id_is_error(tmp_path: Path) -> None:
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text("", encoding="utf-8")

    wiki = Wiki.open(tmp_path / "kb")
    wiki._write_page(
        _technique().model_copy(
            update={"sources": [RecordSourceRef(record_id="rec:missing", kind="finding")]}
        )
    )
    report = wiki.verify(records_jsonl=jsonl)
    assert not report.ok
    assert any("unknown record_id" in i.message for i in report.issues)


# ---------------------------------------------------------------- Wiki.query


def test_query_returns_best_matching_page(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki._write_page(_technique(slug="paged-attention", title="Paged Attention"))
    wiki._write_page(_technique(slug="chunked-prefill", title="Chunked Prefill"))

    results = wiki.query("paged attention kv cache", top_k=5)

    assert len(results) >= 1
    assert results[0].page.slug == "paged-attention"


def test_query_empty_wiki_returns_empty(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    assert wiki.query("anything", top_k=5) == []


def test_query_kind_filter_excludes_wrong_kind(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki._write_page(_technique(slug="paged-attention", title="Paged Attention"))
    wiki._write_page(_entity(slug="vllm/scheduler", title="vLLM Scheduler"))

    results = wiki.query("paged attention scheduler", top_k=10, kinds=["entity"])

    slugs = [r.page.slug for r in results]
    assert "vllm/scheduler" in slugs
    assert "paged-attention" not in slugs


# ----------------------------------------------------------- Wiki.render_index


def test_render_index_contains_technique_and_entity_sections(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki._write_page(_technique())
    wiki._write_page(_entity())
    wiki.render_index()

    text = wiki._layout.wiki_index_path.read_text(encoding="utf-8")
    assert "## Technique pages" in text
    assert "## Entity pages" in text
    assert "Paged Attention" in text
    assert "vLLM v1 KV Offload" in text


def test_render_index_empty_wiki(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki.render_index()

    text = wiki._layout.wiki_index_path.read_text(encoding="utf-8")
    assert "0 concept page(s)" in text


# ----------------------------------------------- resume marker round-trip


def test_read_resume_marker_absent_returns_minus_one(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    assert wiki.read_resume_marker("run-001") == -1


def test_bump_and_read_resume_marker(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki.bump_resume_marker("run-001", wave_idx=2, ingested_qns=["a", "b"])
    assert wiki.read_resume_marker("run-001") == 2


def test_read_resume_marker_different_run_returns_minus_one(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki.bump_resume_marker("run-001", wave_idx=3, ingested_qns=["a"])
    assert wiki.read_resume_marker("run-002") == -1


def test_bump_resume_marker_accumulates_qns_across_waves(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki.bump_resume_marker("run-001", wave_idx=0, ingested_qns=["a", "b"])
    wiki.bump_resume_marker("run-001", wave_idx=1, ingested_qns=["c"])

    state = wiki._read_wiki_state()
    assert set(state["ingested_module_qns_in_current_run"]) == {"a", "b", "c"}
    assert state["last_ingested_wave_idx"] == 1


def test_bump_resume_marker_resets_on_new_run_id(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki.bump_resume_marker("run-001", wave_idx=5, ingested_qns=["a", "b"])
    wiki.bump_resume_marker("run-002", wave_idx=0, ingested_qns=["x"])

    state = wiki._read_wiki_state()
    assert state["run_id"] == "run-002"
    assert state["ingested_module_qns_in_current_run"] == ["x"]


# ------------------------------------------------- Wiki.filter_uningested


def test_filter_uningested_no_state_returns_all(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    assert wiki.filter_uningested(["a", "b", "c"], "run-001") == ["a", "b", "c"]


def test_filter_uningested_drops_already_done(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki.bump_resume_marker("run-001", wave_idx=0, ingested_qns=["a", "b"])
    assert wiki.filter_uningested(["a", "b", "c"], "run-001") == ["c"]


def test_filter_uningested_different_run_returns_all(tmp_path: Path) -> None:
    wiki = Wiki.open(tmp_path / "kb")
    wiki.bump_resume_marker("run-001", wave_idx=0, ingested_qns=["a", "b"])
    assert wiki.filter_uningested(["a", "b", "c"], "run-002") == ["a", "b", "c"]
