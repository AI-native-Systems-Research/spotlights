"""Ingest prompt templates and JSON schema for the concepts layer."""

from __future__ import annotations

from spotlights_engine.module_knowledge.concepts.schemas import ConceptPage

INGEST_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "kebab-case page slug"},
                    "kind": {"type": "string", "enum": ["technique", "entity"]},
                    "action": {"type": "string", "enum": ["create", "update"]},
                    "title": {"type": "string"},
                    "body": {"type": "string", "description": "full markdown body"},
                    "links_techniques": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "slugs of technique pages this page links to",
                    },
                    "links_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "slugs of entity pages this page links to",
                    },
                },
                "required": ["slug", "kind", "action", "title", "body"],
            },
        }
    },
    "required": ["edits"],
}


def render_ingest_prompt(
    *,
    sources: list[dict],
    existing_pages: list[ConceptPage],
    tech_slugs: list[str],
    entity_slugs: list[str],
    subject_system: str,
    objective: str | None,
    run_id: str,
    mode: str,
) -> str:
    """Build the LLM ingest prompt from loaded sources and retrieved pages."""
    lines: list[str] = []

    lines += [
        "# Concepts wiki ingest",
        "",
        "You are updating a persistent concepts wiki that accumulates knowledge across",
        "analysis runs of a software project.",
        "",
        "## Task",
        "",
        "Given the new sources below, update or create concept pages in the wiki.",
        "Return a structured list of page edits (slug, kind, action, title, body).",
        "",
        f"- **Subject system:** {subject_system}",
        f"- **Run ID:** {run_id}",
        f"- **Objective:** {objective or '(none)'}",
        f"- **Mode:** {mode}",
        "",
    ]

    # Slug inventory
    lines += [
        "## Existing page slugs",
        "",
        "Before creating a new page, check this list. If an existing slug names the",
        "same concept (even under a synonym), update that page instead.",
        "",
        "**Technique pages:**",
    ]
    if tech_slugs:
        lines += [f"  - {s}" for s in sorted(tech_slugs)]
    else:
        lines.append("  (none yet)")
    lines += [
        "",
        "**Entity pages** (format: `<subject_system>/<module>`):",
    ]
    if entity_slugs:
        lines += [f"  - {s}" for s in sorted(entity_slugs)]
    else:
        lines.append("  (none yet)")
    lines.append("")

    # Existing retrieved pages
    if existing_pages:
        lines += [
            "## Retrieved concept pages (existing wiki context)",
            "",
            "These pages already exist and are most relevant to the new sources.",
            "Update them rather than duplicating their content.",
            "",
        ]
        for page in existing_pages:
            lines += [
                f"### [{page.kind}] {page.slug} — {page.title}",
                f"*(as_of_run: {page.as_of_run})*",
                "",
                page.body or "(no body)",
                "",
                "---",
                "",
            ]

    # New sources
    lines += [
        "## New sources to incorporate",
        "",
        "Synthesize these into concept pages. Each technique should get (or update) a",
        "technique page. Each module/region should get (or update) an entity page.",
        "",
    ]
    for i, src in enumerate(sources, start=1):
        lines += [
            f"### Source {i}: [{src['kind']}] {src['title']}",
            f"- id: `{src['id']}`",
        ]
        if src.get("url"):
            lines.append(f"- url: {src['url']}")
        if src.get("module_qn"):
            lines.append(f"- module: `{src['module_qn']}`")
        lines += ["", src.get("body", ""), "", "---", ""]

    # Instructions
    lines += [
        "## Instructions",
        "",
        "1. **Technique pages** (`kind: technique`): slug is kebab-case concept name",
        "   (e.g. `speculative-decoding`). One page per concept across all subject systems.",
        "   Sections: `## Synthesis`, `## Where it has been considered in subject systems`,",
        "   `## Known caveats`.",
        "",
        "2. **Entity pages** (`kind: entity`): slug is `<subject_system>/<module-qualified-name>`",
        "   (e.g. `vllm/v1.kv_offload`). Sections: `## What this region does`,",
        "   `## Observations across runs`, `## Techniques considered here`, `## Open questions`.",
        "",
        "3. Cross-link pages with `[[slug]]` syntax (e.g. `[[speculative-decoding]]`,",
        "   `[[vllm/v1.kv_offload]]`).",
        "",
        "4. Treat the wiki as **cumulative**: do not delete prior observations when updating.",
        "   Append new observations; revise synthesis to incorporate new evidence.",
        "",
        "5. **Do not fabricate** information beyond what the sources contain.",
        "",
        "Return JSON with an `edits` array. Each edit must have: `slug`, `kind`, `action`",
        "(`create` or `update`), `title`, `body` (full markdown body for the page).",
        "Optionally include `links_techniques` and `links_entities` listing slugs this",
        "page links to.",
    ]

    return "\n".join(lines)


__all__ = ["INGEST_JSON_SCHEMA", "render_ingest_prompt"]
