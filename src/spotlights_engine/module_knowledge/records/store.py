"""JSONL storage helpers for module knowledge records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from spotlights_engine.module_knowledge.records.schemas import KnowledgeRecord


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def read_records(path: str | Path) -> list[KnowledgeRecord]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[KnowledgeRecord] = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(KnowledgeRecord.model_validate_json(stripped))
        except ValueError as exc:
            raise ValueError(f"invalid knowledge JSONL at {source}:{line_no}: {exc}") from exc
    return records


def write_records(records: Iterable[KnowledgeRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record.record_id)
    payload = "".join(record.model_dump_json() + "\n" for record in ordered)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        Path(tmp_name).replace(target)
    except OSError:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def upsert_records(records: Iterable[KnowledgeRecord], path: str | Path) -> tuple[int, int, int]:
    """Upsert records by `record_id`.

    Returns `(total_written, inserted, updated)` after the write.
    """
    existing = {record.record_id: record for record in read_records(path)}
    inserted = 0
    updated = 0
    for record in records:
        if record.record_id in existing:
            updated += 1
        else:
            inserted += 1
        existing[record.record_id] = record
    write_records(existing.values(), path)
    return len(existing), inserted, updated


def canonical_record_hash(record: KnowledgeRecord) -> str:
    payload = record.model_dump(mode="json", exclude={"provenance": {"content_hash"}})
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def with_content_hash(record: KnowledgeRecord) -> KnowledgeRecord:
    """Return `record` with a deterministic archive content hash."""
    provenance = record.provenance.model_copy(
        update={"content_hash": canonical_record_hash(record)}
    )
    return record.model_copy(update={"provenance": provenance})


__all__ = [
    "canonical_record_hash",
    "read_records",
    "sha256_file",
    "sha256_text",
    "upsert_records",
    "with_content_hash",
    "write_records",
]
