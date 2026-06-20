"""Markdown emission for the results renderer.

All writes are atomic temp-file + `os.replace`. Per-candidate pages are
written first, then the module page that links to them, then `index.md`
last — so a crash mid-render never leaves a parent page that points at
a missing child. When `config.overwrite=True` the writer wipes
`output_folder/modules/` before emitting so re-renders after a manifest
with a removed module don't leave orphan pages.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from spotlights_engine.results_renderer.aggregator import (
    IndexRow,
    ModulePageView,
    aggregate,
)
from spotlights_engine.results_renderer.api import (
    RendererConfig,
    RendererInput,
    RendererResult,
)
from spotlights_engine.results_renderer.errors import RendererSetupError
from spotlights_engine.results_renderer.loader import LoadedRun
from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import SpotlightContext, StepIssue
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.project import Repository
from spotlights_engine.utils.schema_compat import (
    primary_file,
    primary_span,
    proposals_from,
)


def emit(
    *,
    input: RendererInput,
    loaded: LoadedRun,
    config: RendererConfig,
) -> RendererResult:
    """Render the loaded run into Markdown under `input.output_folder`."""
    output = input.output_folder
    _prepare_output_folder(output, config)

    modules_dir = output / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    rows, views, skipped, warnings = aggregate(loaded, config)

    module_pages: dict[str, Path] = {}
    candidate_pages: dict[str, dict[str, Path]] = {}
    for qn, view in views.items():
        module_page_filename = _page_filename_for_view(view)
        page_path = modules_dir / module_page_filename
        findings_by_id = {f.finding_id: f for f in view.findings}

        candidates_by_id = {cand.id: cand for cand in view.candidates_sorted}
        per_module: dict[str, Path] = {}
        for row in view.candidate_rows:
            cand = candidates_by_id[row.candidate_id]
            cand_path = modules_dir / row.candidate_page_path
            cand_text = _render_candidate_page(view, cand, config, findings_by_id)
            _atomic_write_text(cand_path, cand_text)
            per_module[row.candidate_id] = cand_path
        candidate_pages[qn] = per_module

        page_text = _render_module_page(view)
        _atomic_write_text(page_path, page_text)
        module_pages[qn] = page_path

    index_path = output / "index.md"
    index_text = _render_index(
        repository=loaded.project_tree.repository,
        manifest=loaded.manifest,
        context=loaded.context,
        rows=rows,
        warnings=warnings,
    )
    _atomic_write_text(index_path, index_text)

    return RendererResult(
        index_path=index_path,
        module_pages=module_pages,
        candidate_pages=candidate_pages,
        skipped_modules=skipped,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Output folder lifecycle
# ---------------------------------------------------------------------------


def _prepare_output_folder(output: Path, config: RendererConfig) -> None:
    if output.exists() and not output.is_dir():
        raise RendererSetupError(
            f"output_folder exists and is not a directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    if not config.overwrite:
        if any(output.iterdir()):
            raise RendererSetupError(
                f"output_folder is not empty and overwrite=False: {output}"
            )
        return
    modules_dir = output / "modules"
    if modules_dir.exists():
        shutil.rmtree(modules_dir)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _page_filename_for_view(view: ModulePageView) -> str:
    from spotlights_engine.spotlights_manager.persistence import slug_for

    return f"{slug_for(view.qualified_name)}.md"


# ---------------------------------------------------------------------------
# index.md
# ---------------------------------------------------------------------------


def _render_index(
    *,
    repository: Repository,
    manifest: dict[str, Any],
    context: SpotlightContext | None,
    rows: list[IndexRow],
    warnings: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"# Spotlights Run — {repository.name}")
    lines.append("")

    lines.append("## Repository")
    lines.append(f"- **Name:** {repository.name}")
    lines.append(f"- **Summary:** {repository.summary}")
    lines.append(
        "- **External dependencies:** "
        + (
            ", ".join(repository.external_dependencies)
            if repository.external_dependencies
            else "_(none)_"
        )
    )
    input_fp = manifest.get("input_fingerprint") if isinstance(manifest, dict) else None
    repo_path = (
        input_fp.get("repo_path") if isinstance(input_fp, dict) else None
    ) or "_(unknown)_"
    extractor = manifest.get("extractor") if isinstance(manifest, dict) else None
    extractor_duration = (
        extractor.get("duration_s") if isinstance(extractor, dict) else None
    )
    lines.append(f"- **Repo path:** {repo_path}")
    lines.append(
        f"- **Run created:** {manifest.get('created_at', '_(unknown)_')}"
    )
    lines.append(f"- **Run status:** {manifest.get('status', '_(unknown)_')}")
    lines.append(
        "- **Extractor duration (s):** "
        + (
            f"{float(extractor_duration):.1f}"
            if isinstance(extractor_duration, (int, float))
            else "_(unknown)_"
        )
    )
    lines.append("")

    lines.append("## Context")
    if context is None:
        lines.append("_(context unavailable for pre-existing run)_")
    else:
        lines.append(f"- **Objective:** {context.objective}")
        lines.append("- **Workload hints:**")
        if context.workload_hints:
            for hint in context.workload_hints:
                lines.append(f"  - {hint}")
        else:
            lines.append("  - _(none)_")
        lines.append("- **Validation plan:**")
        if context.validation_plan:
            for step in context.validation_plan:
                lines.append(f"  - {step}")
        else:
            lines.append("  - _(none)_")
    lines.append("")

    lines.append("## Modules")
    lines.append("")
    if not rows:
        lines.append("_(no modules to display)_")
    else:
        lines.append(
            "| Module | Candidates | High-impact | Relevant findings |"
        )
        lines.append("|---|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| [{row.module_qualified_name}]({row.module_page_path}) "
                f"| {row.n_candidates} "
                f"| {row.n_high_impact_candidates} "
                f"| {row.n_relevant_findings} |"
            )
    lines.append("")

    if warnings:
        lines.append("## Renderer warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# modules/<slug>.md
# ---------------------------------------------------------------------------


def _render_module_page(view: ModulePageView) -> str:
    lines: list[str] = []
    lines.append(f"# {view.qualified_name}")
    lines.append("")
    lines.append("[← All modules](../index.md)")
    lines.append("")

    lines.append("## Module")
    module = view.module
    if module is None:
        lines.append(
            "_(module not present in project_tree — only checkpoint state is available)_"
        )
    else:
        lines.append(f"- **Path:** `{module.path}`")
        if module.description:
            lines.append(f"- **Description:** {module.description}")
        lines.append(
            "- **Depends on:** "
            + (", ".join(module.depends_on) if module.depends_on else "_(none)_")
        )
        lines.append("- **Main files:**")
        if module.main_files:
            for f in module.main_files:
                lines.append(f"  - `{f.path}` — {f.role}")
        else:
            lines.append("  - _(none)_")

    lines.append(f"- **Run status:** {view.status}")
    lines.append(f"- **Findings:** {len(view.findings)}")
    lines.append(f"- **Issues:** {len(view.issues)}")
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    if not view.candidate_rows:
        lines.append("_No candidates._")
        lines.append("")
    else:
        lines.append("| Candidate | Impact | Deep research proposals |")
        lines.append("|---|---|---:|")
        for row in view.candidate_rows:
            cell_symbol = _escape_table_cell(row.symbol)
            lines.append(
                f"| [`{cell_symbol}`]({row.candidate_page_path}) "
                f"| {row.estimated_impact} "
                f"| {row.n_deep_research_proposals} |"
            )
        lines.append("")

    lines.append("## Findings (full list)")
    lines.append("")
    if not view.findings:
        lines.append("_No findings._")
        lines.append("")
    else:
        for idx, f in enumerate(view.findings, start=1):
            lines.append(f"{idx}. **{f.title}**")
            lines.append(f"   - Source type: {f.source_type}")
            lines.append(f"   - URL: <{f.url}>")
            lines.append(f"   - Technique: {f.technique_summary}")
            if f.supporting_evidence:
                lines.append(f"   - Evidence: {f.supporting_evidence}")
        lines.append("")

    lines.append("## Issues")
    lines.append("")
    if not view.issues:
        lines.append("_No issues._")
        lines.append("")
    else:
        lines.extend(_render_issues_grouped(view.issues))

    return "\n".join(lines).rstrip() + "\n"


def _render_candidate_page(
    view: ModulePageView,
    cand: Candidate,
    config: RendererConfig,
    findings_by_id: dict[str, Finding],
) -> str:
    span = primary_span(cand)
    lines: list[str] = []
    lines.append(f"# {span.symbol}")
    lines.append("")
    module_page = _page_filename_for_view(view)
    lines.append(f"[← {view.qualified_name}](../{module_page})")
    lines.append("")

    file_link = _file_link(primary_file(cand), config)
    lines.append(
        f"- **File:** {file_link} (lines {span.line_start}–{span.line_end})"
    )
    lines.append(f"- **Symbol:** `{span.symbol}`")
    lines.append(f"- **Kind:** {span.kind}")
    lines.append(f"- **Estimated impact:** {cand.estimated_impact}")
    lines.append(f"- **Id:** `{cand.id}`")
    lines.append("")

    lines.append("## Description")
    lines.append(cand.description)
    lines.append("")

    lines.append("## Current approach")
    lines.append(cand.current_approach)
    lines.append("")

    lines.append("## Estimated impact explanation")
    lines.append(cand.estimated_impact_explanation)
    lines.append("")

    lines.append("## Evolve rationale")
    lines.append(cand.evolve_rationale)
    lines.append("")

    research_proposals = proposals_from(cand, "research_finding")
    agent_knowledge_proposals = proposals_from(cand, "agent_knowledge")

    lines.append("## Deep research proposals")
    lines.append("")
    if not research_proposals:
        lines.append("_No proposals._")
        lines.append("")
    else:
        for n, p in enumerate(research_proposals, start=1):
            lines.append(f"### {n}. {p.title}")
            finding = (
                findings_by_id.get(p.finding_ref_id)
                if p.finding_ref_id is not None
                else None
            )
            if finding is not None:
                lines.append(
                    f"- **Finding:** `{p.finding_ref_id}` — *{finding.title}*"
                )
                lines.append(f"- **Source URL:** <{finding.url}>")
            elif p.finding_ref_id is not None:
                lines.append(f"- **Finding:** `{p.finding_ref_id}`")
            if p.author:
                lines.append(f"- **Created by:** {p.author}")
            lines.append("")
            lines.append("**Detailed description.**")
            lines.append("")
            lines.append(p.description)
            lines.append("")
            lines.append("**Proposal rationale.**")
            lines.append("")
            lines.append(p.rationale)
            lines.append("")
            lines.append("---")
            lines.append("")

    lines.append("## Agent proposals")
    lines.append("")
    if not agent_knowledge_proposals:
        lines.append("_No proposals._")
        lines.append("")
    else:
        for n, p in enumerate(agent_knowledge_proposals, start=1):
            lines.append(f"### {n}. {p.title}")
            lines.append(f"- **Agent:** {p.author or '(unknown)'}")
            lines.append("")
            lines.append("**Detailed description.**")
            lines.append("")
            lines.append(p.description)
            lines.append("")
            lines.append("**Novelty rationale.**")
            lines.append("")
            lines.append(p.rationale)
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


_TABLE_CELL_WS_RE = re.compile(r"\s+")


def _escape_table_cell(text: str) -> str:
    """Escape `|` and collapse whitespace for safe inclusion in a markdown
    table cell. Whitespace runs (including newlines) become a single space."""
    return _TABLE_CELL_WS_RE.sub(" ", text.replace("|", r"\|")).strip()


def _file_link(file_path: str, config: RendererConfig) -> str:
    if config.relative_repo_links:
        return f"[`{file_path}`]({file_path})"
    return f"`{file_path}`"


def _render_issues_grouped(issues: list[StepIssue]) -> list[str]:
    lines: list[str] = []
    by_step: dict[str, list[StepIssue]] = {}
    for iss in issues:
        by_step.setdefault(str(iss.step), []).append(iss)
    for step in sorted(by_step.keys()):
        lines.append(f"### {step}")
        for iss in by_step[step]:
            recoverable = "recoverable" if iss.recoverable else "unrecoverable"
            lines.append(
                f"- **[{iss.severity}, {recoverable}]** {iss.message}"
            )
        lines.append("")
    return lines


# Re-exports for tests
__all__ = [
    "emit",
]
