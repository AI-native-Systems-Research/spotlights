"""CLI coverage for `spotlights-engine knowledge`."""

from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine import cli
from spotlights_engine.module_knowledge import KnowledgeRecord, Provenance, SourceRef


def _record_payload() -> dict:
    record = KnowledgeRecord(
        record_id="paper:kv-cache",
        source_type="paper",
        title="Paged KV cache",
        text="Paged KV cache reduces decode memory fragmentation.",
        source=SourceRef(
            source_id="src:paged-kv",
            title="Paged KV cache paper",
            url="https://example.com/paged-kv",
            trust_tier="credible",
        ),
        provenance=Provenance(locator="paper:abstract", extractor="fixture"),
        tags=["kv-cache", "decode"],
    )
    return record.model_dump(mode="json")


def test_knowledge_cli_archive_retrieve_and_render_wiki(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "knowledge"
    record_json = tmp_path / "record.json"
    record_json.write_text(json.dumps(_record_payload()), encoding="utf-8")

    archive_rc = cli.main(
        [
            "knowledge",
            "--root",
            str(root),
            "archive",
            "--record-json",
            str(record_json),
            "--render-wiki",
        ]
    )
    archive_out = capsys.readouterr().out

    retrieve_rc = cli.main(
        [
            "knowledge",
            "--root",
            str(root),
            "retrieve",
            "kv cache decode",
            "--source-type",
            "paper",
            "--tag",
            "kv-cache",
            "--render-query-page",
            "--query-id",
            "kv-cache",
        ]
    )
    retrieve_out = capsys.readouterr().out

    verify_rc = cli.main(["knowledge", "--root", str(root), "wiki", "verify"])
    verify_out = capsys.readouterr().out

    assert archive_rc == 0
    assert "inserted: 1" in archive_out
    assert retrieve_rc == 0
    assert "results: 1" in retrieve_out
    assert "Paged KV cache" in retrieve_out
    assert verify_rc == 0
    assert "checked_pages:" in verify_out
    assert (root / "wiki" / "queries" / "kv-cache.md").exists()


def test_knowledge_cli_retrieve_json_output(tmp_path: Path, capsys) -> None:
    root = tmp_path / "knowledge"
    record_json = tmp_path / "record.json"
    record_json.write_text(json.dumps([_record_payload()]), encoding="utf-8")
    assert (
        cli.main(["knowledge", "--root", str(root), "archive", "--record-json", str(record_json)])
        == 0
    )
    capsys.readouterr()

    rc = cli.main(
        [
            "knowledge",
            "--root",
            str(root),
            "retrieve",
            "fragmentation",
            "--json",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out)
    assert payload[0]["record"]["record_id"] == "paper:kv-cache"


def test_knowledge_cli_invalid_record_returns_error(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"record_id": "missing-fields"}', encoding="utf-8")

    rc = cli.main(
        [
            "knowledge",
            "--root",
            str(tmp_path / "knowledge"),
            "archive",
            "--record-json",
            str(bad),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "invalid knowledge record JSON" in captured.err
