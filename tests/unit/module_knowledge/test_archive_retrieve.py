"""Tests for the module-knowledge Archive and Retrieve APIs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spotlights_engine.module_knowledge import (
    KnowledgeBase,
    KnowledgeRecord,
    KnowledgeSourceType,
    Provenance,
    RetrieveRequest,
    SourceRef,
)


def _record(
    record_id: str,
    *,
    title: str,
    text: str,
    source_type: KnowledgeSourceType = "paper",
    tags: list[str] | None = None,
) -> KnowledgeRecord:
    return KnowledgeRecord(
        record_id=record_id,
        source_type=source_type,
        title=title,
        text=text,
        source=SourceRef(source_id=f"src:{record_id}", title=title, trust_tier="credible"),
        provenance=Provenance(locator=f"fixture:{record_id}", extractor="test"),
        tags=tags or [],
    )


def test_archive_upserts_records_and_fills_content_hash(tmp_path: Path) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")

    first = kb.archive(_record("rec:one", title="KV cache", text="Paged KV cache evidence"))
    second = kb.archive(_record("rec:one", title="KV cache v2", text="Updated evidence"))

    records = kb.records()
    assert first.inserted == 1
    assert second.updated == 1
    assert second.records_written == 1
    assert len(records) == 1
    assert records[0].title == "KV cache v2"
    assert records[0].provenance.content_hash is not None
    assert records[0].provenance.content_hash.startswith("sha256:")


def test_archive_recomputes_content_hash_from_record_content(tmp_path: Path) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")
    stale = _record("rec:one", title="KV cache", text="Paged KV cache evidence").model_copy(
        update={
            "provenance": Provenance(
                locator="fixture:rec:one",
                extractor="test",
                content_hash="sha256:" + ("0" * 64),
            )
        }
    )

    kb.archive(stale)

    [record] = kb.records()
    assert record.provenance.content_hash != stale.provenance.content_hash
    assert record.provenance.content_hash is not None
    assert record.provenance.content_hash.startswith("sha256:")


def test_retrieve_ranks_by_local_lexical_evidence_and_filters(tmp_path: Path) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")
    kb.archive_many(
        [
            _record(
                "rec:kv",
                title="Paged KV cache eviction",
                text="KV cache eviction reduces decode memory pressure.",
                source_type="paper",
                tags=["kv-cache", "decode"],
            ),
            _record(
                "rec:router",
                title="Router batching",
                text="Request routing and batching notes.",
                source_type="docs",
                tags=["routing"],
            ),
        ]
    )

    results = kb.retrieve(
        RetrieveRequest(
            query="kv cache decode eviction",
            top_k=5,
            source_types=["paper"],
            tags=["kv-cache"],
        )
    )

    assert [item.record.record_id for item in results] == ["rec:kv"]
    assert {"kv", "cache", "decode", "eviction"}.issubset(set(results[0].matched_terms))
    assert results[0].score > 0


def test_retrieve_strips_sentence_punctuation(tmp_path: Path) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")
    kb.archive(_record("rec:one", title="Memory note", text="Decode memory fragmentation."))

    results = kb.retrieve("fragmentation", top_k=1)

    assert [item.record.record_id for item in results] == ["rec:one"]
    assert results[0].matched_terms == ["fragmentation"]


def test_retrieve_top_k_argument_overrides_request_and_is_validated(tmp_path: Path) -> None:
    kb = KnowledgeBase.open(tmp_path / "kb")
    kb.archive_many(
        [
            _record("rec:one", title="KV cache one", text="KV cache evidence."),
            _record("rec:two", title="KV cache two", text="KV cache evidence."),
        ]
    )

    results = kb.retrieve(RetrieveRequest(query="kv cache", top_k=10), top_k=1)

    assert len(results) == 1
    with pytest.raises(ValidationError):
        kb.retrieve("kv cache", top_k=0)
