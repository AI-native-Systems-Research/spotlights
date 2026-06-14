"""LLM-driven ingest pass for the concepts layer."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from spotlights_engine.module_knowledge.concepts.schemas import (
    ConceptLinks,
    ConceptPage,
    IngestReport,
    IngestRequest,
    LightSourceRef,
    RecordSourceRef,
)
from spotlights_engine.module_knowledge.concepts.wiki import body_hash

if TYPE_CHECKING:
    from spotlights_engine.module_knowledge.concepts.wiki import Wiki

_ELIGIBLE_TYPES = frozenset({"finding", "module_map"})


async def run_ingest(wiki: "Wiki", request: IngestRequest) -> IngestReport:
    """Run one ingest pass; return a report of pages created/updated."""
    start = time.monotonic()

    sources = _load_sources(wiki, request)
    if not sources:
        return IngestReport(duration_s=time.monotonic() - start)

    query_str = _build_query(sources, request)
    existing_results = wiki.query(query_str, top_k=8)

    all_pages = wiki._load_pages()
    tech_slugs = [p.slug for p in all_pages if p.kind == "technique"]
    entity_slugs = [p.slug for p in all_pages if p.kind == "entity"]

    from spotlights_engine.module_knowledge.concepts.prompts import (
        INGEST_JSON_SCHEMA,
        render_ingest_prompt,
    )

    prompt = render_ingest_prompt(
        sources=sources,
        existing_pages=[r.page for r in existing_results],
        tech_slugs=tech_slugs,
        entity_slugs=entity_slugs,
        subject_system=request.subject_system,
        objective=request.objective,
        run_id=request.run_id,
        mode=request.mode,
    )

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = wiki._layout.ingest_log_dir / f"ingest-{ts}"

    from spotlights_engine.signal_pipeline.claude_subprocess import run_claude

    fn = partial(
        run_claude,
        prompt=prompt,
        log_dir=log_dir,
        json_schema=json.dumps(INGEST_JSON_SCHEMA),
        cwd=wiki._layout.knowledge_root,
        max_turns=10,
        timeout_s=300,
        permission_mode="plan",
        allowed_tools=[],
    )
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, fn)

    llm_calls = 1
    cost_usd = float(result.cost_usd or 0.0)

    if result.error or result.structured_output is None:
        return IngestReport(
            llm_calls=llm_calls,
            cost_usd=cost_usd,
            duration_s=time.monotonic() - start,
        )

    edits = _parse_edits(result.structured_output)
    existing_by_slug = {p.slug: p for p in all_pages}
    source_refs = _build_source_refs(sources, request)

    pages_updated: list[str] = []
    pages_created: list[str] = []

    for edit in edits:
        slug = str(edit.get("slug", "")).strip()
        kind = str(edit.get("kind", "")).strip()
        action = str(edit.get("action", "")).strip()
        title = str(edit.get("title", "")).strip()
        body = str(edit.get("body", "")).strip()

        if not slug or kind not in ("technique", "entity") or not title or not body:
            continue

        links = ConceptLinks(
            techniques=list(edit.get("links_techniques") or []),
            entities=list(edit.get("links_entities") or []),
        )

        if slug in existing_by_slug:
            page = _merge_page(
                existing_by_slug[slug],
                body=body,
                title=title,
                links=links,
                source_refs=source_refs,
                run_id=request.run_id,
                subject_system=request.subject_system,
                objective=request.objective,
            )
            pages_updated.append(slug)
        else:
            page = _create_page(
                slug=slug,
                kind=kind,
                title=title,
                body=body,
                links=links,
                source_refs=source_refs,
                run_id=request.run_id,
                subject_system=request.subject_system if kind == "entity" else None,
                objective=request.objective,
            )
            pages_created.append(slug)

        wiki._write_page(page)

    return IngestReport(
        pages_updated=pages_updated,
        pages_created=pages_created,
        llm_calls=llm_calls,
        cost_usd=cost_usd,
        duration_s=time.monotonic() - start,
    )


def _load_sources(wiki: "Wiki", request: IngestRequest) -> list[dict]:
    """Return normalized source dicts for the prompt."""
    if request.mode == "records-and-concepts":
        from spotlights_engine.module_knowledge.records.store import read_records

        records_jsonl = wiki._layout.knowledge_root / "archive" / "records.jsonl"
        record_map = {r.record_id: r for r in read_records(records_jsonl)}
        result: list[dict] = []
        for rid in request.record_ids:
            r = record_map.get(rid)
            if r is None or r.source_type not in _ELIGIBLE_TYPES:
                continue
            result.append(
                {
                    "id": r.record_id,
                    "kind": r.source_type,
                    "title": r.title,
                    "body": r.text,
                    "url": r.source.url,
                    "module_qn": r.provenance.locator,
                }
            )
        return result
    else:
        return [
            {
                "id": s.id,
                "kind": s.kind,
                "title": s.title,
                "body": s.body,
                "url": s.url,
                "module_qn": s.module_qn,
            }
            for s in request.sources
        ]


def _build_query(sources: list[dict], request: IngestRequest) -> str:
    parts = [request.subject_system]
    if request.objective:
        parts.append(request.objective)
    for src in sources[:3]:
        parts.append(src.get("title", ""))
        if src.get("module_qn"):
            parts.append(src["module_qn"])
    return " ".join(p for p in parts if p)


def _build_source_refs(sources: list[dict], request: IngestRequest) -> list:
    if request.mode == "records-and-concepts":
        return [RecordSourceRef(record_id=s["id"], kind=s["kind"]) for s in sources]
    return [
        LightSourceRef(
            kind=s["kind"],
            id=s["id"],
            title=s["title"],
            url=s.get("url"),
        )
        for s in sources
    ]


def _parse_edits(structured_output: dict | list) -> list[dict]:
    if isinstance(structured_output, dict):
        edits = structured_output.get("edits", [])
    elif isinstance(structured_output, list):
        edits = structured_output
    else:
        return []
    return [e for e in edits if isinstance(e, dict)]


def _merge_page(
    existing: ConceptPage,
    *,
    body: str,
    title: str,
    links: ConceptLinks,
    source_refs: list,
    run_id: str,
    subject_system: str | None,
    objective: str | None,
) -> ConceptPage:
    existing_source_ids: set[str] = set()
    for s in existing.sources:
        existing_source_ids.add(s.record_id if isinstance(s, RecordSourceRef) else s.id)

    new_refs = [
        s
        for s in source_refs
        if (s.record_id if isinstance(s, RecordSourceRef) else s.id) not in existing_source_ids
    ]

    sss = list(existing.subject_systems_seen)
    if existing.kind == "technique" and subject_system and subject_system not in sss:
        sss.append(subject_system)

    objs = list(existing.objectives_seen)
    if existing.kind == "entity" and objective and objective not in objs:
        objs.append(objective)

    merged_links = ConceptLinks(
        techniques=sorted(set(existing.links.techniques) | set(links.techniques)),
        entities=sorted(set(existing.links.entities) | set(links.entities)),
    )

    return existing.model_copy(
        update={
            "title": title,
            "body": body,
            "as_of_run": run_id,
            "sources": list(existing.sources) + new_refs,
            "links": merged_links,
            "subject_systems_seen": sss,
            "objectives_seen": objs,
            "content_hash": body_hash(body),
        }
    )


def _create_page(
    *,
    slug: str,
    kind: str,
    title: str,
    body: str,
    links: ConceptLinks,
    source_refs: list,
    run_id: str,
    subject_system: str | None,
    objective: str | None,
) -> ConceptPage:
    return ConceptPage(
        kind=kind,
        slug=slug,
        title=title,
        body=body,
        as_of_run=run_id,
        subject_system=subject_system if kind == "entity" else None,
        subject_systems_seen=[subject_system] if kind == "technique" and subject_system else [],
        objectives_seen=[objective] if kind == "entity" and objective else [],
        sources=source_refs,
        links=links,
        content_hash=body_hash(body),
    )


__all__ = ["run_ingest"]
