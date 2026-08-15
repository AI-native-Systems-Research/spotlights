"""Per-directory decision report for assignment-based module extraction.

The report is a deterministic join over persisted skeleton, assignment,
resolved-owner, metadata, coverage, lint, configuration, and Stage-1 artifacts.
It performs no LLM calls and no repository walk.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spotlights_engine.modules_extractor.assignments import (
    AssignmentCoverageReport,
    AssignmentIssue,
    nearest_module_ancestor,
)
from spotlights_engine.modules_extractor.constants import MERGE_THRESHOLD_DEFAULT
from spotlights_engine.modules_extractor.derive import AssignmentLint
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentTree,
    ModuleInfo,
    ModuleMetadataMap,
    ResolvedAssignmentTree,
    Skeleton,
    SkeletonNode,
    SourceRootDecision,
)

SCHEMA_VERSION_V2: Final[Literal["tree_decisions.v2"]] = "tree_decisions.v2"
REPORT_JSON_NAME = "tree_decisions.json"
REPORT_MD_NAME = "tree_decisions.md"
SOURCE_ARTIFACTS_V2 = [
    "02_skeleton/skeleton.json",
    "assignment_tree.json",
    "resolved_assignments.json",
    "module_metadata.json",
    "coverage.json",
    "metadata_coverage.json",
    "assignment_lints.json",
    "extractor_config.json",
    "source_root_decision.json",
]

_EXCLUDED_SYMBOL = "⊘"
_MAX_REASON_CHARS = 110


def _norm(path: str) -> str:
    return path.strip().strip("/")


def _node_key(node: SkeletonNode) -> str:
    return _norm(node.path)


class ExcludedPathEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    reason: str
    explanation: str


class PrunedInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ignored_dir_names: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)
    skipped_symlinks: list[str] = Field(default_factory=list)


class GeneratedFrom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skeleton_fingerprint: str
    artifacts: list[str] = Field(default_factory=list)


def _inline_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def _one_line(text: str, limit: int = _MAX_REASON_CHARS) -> str:
    collapsed = " ".join(text.split()).replace("|", "\\|")
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "…"
    return collapsed


def _excluded_section(report: TreeDecisionReportV2) -> list[str]:
    if not report.excluded_paths:
        return []
    out = [f"## {_EXCLUDED_SYMBOL} Excluded by stage 1", ""]
    out.append("| path | reason | explanation |")
    out.append("|---|---|---|")
    for entry in report.excluded_paths:
        out.append(
            f"| `{entry.path}` | {entry.reason} | {_one_line(entry.explanation)} |"
        )
    out.append("")
    return out


def _pruned_section(report: TreeDecisionReportV2) -> list[str]:
    pruned = report.pruned
    out = ["## Pruned (not nodes)", ""]
    out.append(f"- ignored directory names: {_inline_list(pruned.ignored_dir_names)}")
    out.append(
        "- excluded paths (stage 1, applied to the walk): "
        + _inline_list(pruned.excluded_paths)
    )
    out.append(f"- skipped symlinks: {_inline_list(pruned.skipped_symlinks)}")
    out.append("")
    return out


# ── Report model and rendering ─────────────────────────────────────────────

DecisionV2 = Literal["module_parent", "module_leaf", "part", "missing"]

_SYMBOLS_V2: dict[str, str] = {
    "module_parent": "●",
    "module_leaf": "◐",
    "part": "○",
    "missing": "✗",
}

_TAGS_V2: dict[str, str] = {
    "module_parent": "MODULE parent",
    "module_leaf": "MODULE leaf",
    "part": "PART",
    "missing": "MISSING",
}


class DecisionCountsV2(BaseModel):
    """Per-decision tallies over a set of v2 nodes. `invalid_nodes` and
    `linted_nodes` are orthogonal to the decision buckets, so
    `module_parent + module_leaf + part + missing == total_nodes` always."""

    model_config = ConfigDict(extra="forbid")

    total_nodes: int = 0
    module_parent: int = 0
    module_leaf: int = 0
    part: int = 0
    missing: int = 0
    invalid_nodes: int = 0
    linted_nodes: int = 0

    @property
    def modules(self) -> int:
        return self.module_parent + self.module_leaf


class ReportSummaryV2(DecisionCountsV2):
    """Whole-run tallies. `excluded` is *not* part of `total_nodes`."""

    excluded: int = 0


class TopLevelSummaryV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    decision: DecisionV2
    emitted_name: str | None = None
    counts: DecisionCountsV2


class RootIssue(BaseModel):
    """A report-level problem that has no skeleton node to hang on."""

    model_config = ConfigDict(extra="forbid")

    code: str
    path: str | None = None
    detail: str


class TreeDecisionNodeV2(BaseModel):
    """One source-bearing directory and the single v2 decision made about it.

    `decision` reflects the raw parseable label, not its validity; validity
    problems are the orthogonal `invalid_codes`. `owner` is set for `part`
    nodes (the nearest `MODULE`-labeled strict ancestor, when one exists);
    `origin`/`keep_reason` come from the resolved artifact when available.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    decision: DecisionV2
    owner: str | None = None
    origin: str | None = None
    keep_reason: str | None = None
    required: bool = False
    required_reasons: list[str] = Field(default_factory=list)
    organizational_only: bool = False
    direct_source_file_count: int = Field(default=0, ge=0)
    subtree_source_file_count: int = Field(default=0, ge=0)
    source_child_count: int = Field(default=0, ge=0)
    territory_source_file_count: int | None = None
    emitted_name: str | None = None
    description: str | None = None
    lint_codes: list[str] = Field(default_factory=list)
    invalid_codes: list[str] = Field(default_factory=list)
    children: list[TreeDecisionNodeV2] = Field(default_factory=list)


class TreeDecisionReportV2(BaseModel):
    """The canonical v2 `tree_decisions.json` artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["tree_decisions.v2"] = SCHEMA_VERSION_V2
    contract: Literal["assignments"] = "assignments"
    source_root: str
    merge_threshold: int = Field(ge=0)
    generated_from: GeneratedFrom
    summary: ReportSummaryV2
    top_level: list[TopLevelSummaryV2] = Field(default_factory=list)
    nodes: list[TreeDecisionNodeV2] = Field(default_factory=list)
    excluded_paths: list[ExcludedPathEntry] = Field(default_factory=list)
    pruned: PrunedInfo = Field(default_factory=PrunedInfo)
    extra_assignments: list[str] = Field(default_factory=list)
    root_issues: list[RootIssue] = Field(default_factory=list)
    lints: list[AssignmentLint] = Field(default_factory=list)

    def iter_nodes(self):
        def _walk(nodes: list[TreeDecisionNodeV2]):
            for n in nodes:
                yield n
                yield from _walk(n.children)

        yield from _walk(self.nodes)


TreeDecisionNodeV2.model_rebuild()


def build_tree_decision_report_v2(
    skeleton: Skeleton,
    *,
    assignments: AssignmentTree | None,
    resolved: ResolvedAssignmentTree | None = None,
    metadata: Mapping[str, ModuleInfo] | None = None,
    coverage: AssignmentCoverageReport | None = None,
    lints: list[AssignmentLint] | None = None,
    issues: list[AssignmentIssue] | None = None,
    decision: SourceRootDecision | None = None,
    merge_threshold: int,
) -> TreeDecisionReportV2:
    """Join the assignment-contract artifacts into one per-node report.

    Best-effort by design: every input except the skeleton and threshold is
    optional, so a failed run — exhausted repairs, partial assignments, no
    metadata — still gets its report. `decision` per skeleton path, first
    match wins on the raw parseable label: `MODULE` (parent vs leaf via the
    nearest-module-ancestor rule over the labeled module set) → `PART` →
    `missing`. Validity is orthogonal (`invalid_codes` from `issues` +
    `coverage.invalid_assignments`); lints are orthogonal warnings.
    """
    labels: dict[str, str] = dict(assignments.assignments) if assignments else {}
    decisions_map = dict(assignments.module_decisions) if assignments else {}
    module_set = {p for p, label in labels.items() if label == "MODULE"}

    # Parent modules: those owning at least one direct child module.
    module_parents: set[str] = set()
    for m in module_set:
        parent = nearest_module_ancestor(m, module_set)
        if parent is not None:
            module_parents.add(parent)

    all_issues: list[AssignmentIssue] = list(issues or [])
    if coverage is not None:
        all_issues.extend(coverage.invalid_assignments)
    issue_codes: dict[str, list[str]] = {}
    root_issues: list[RootIssue] = []
    for issue in all_issues:
        if issue.path is not None and issue.path in skeleton.all_paths():
            codes = issue_codes.setdefault(issue.path, [])
            if issue.code not in codes:
                codes.append(issue.code)
        else:
            root_issues.append(
                RootIssue(code=issue.code, path=issue.path, detail=issue.detail)
            )

    lint_codes: dict[str, list[str]] = {}
    for lint in lints or []:
        codes = lint_codes.setdefault(lint.path, [])
        if lint.code not in codes:
            codes.append(lint.code)

    territory_counts: dict[str, int] | None = None
    if resolved is not None:
        from spotlights_engine.modules_extractor.derive import (
            territory_source_file_counts,
        )

        territory_counts = territory_source_file_counts(resolved, skeleton)

    organizational = set(skeleton.organizational_only)
    inventory = skeleton.all_paths()
    extra_assignments = sorted(set(labels) - inventory)

    def _build(node: SkeletonNode) -> TreeDecisionNodeV2:
        path = _norm(node.path)
        label = labels.get(path)
        if label == "MODULE":
            decision_value: DecisionV2 = (
                "module_parent" if path in module_parents else "module_leaf"
            )
        elif label == "PART":
            decision_value = "part"
        else:
            decision_value = "missing"

        owner: str | None = None
        origin: str | None = None
        keep_reason: str | None = None
        if resolved is not None and path in resolved.owners:
            owner_path = resolved.owners[path]
            owner = owner_path if owner_path != path else None
            origin = resolved.origins.get(path)
        elif label == "PART":
            owner = nearest_module_ancestor(path, module_set)
        if label == "MODULE":
            entry = decisions_map.get(path)
            keep_reason = entry.keep_reason if entry is not None else None

        info = (metadata or {}).get(path) if label == "MODULE" else None
        return TreeDecisionNodeV2(
            path=path,
            decision=decision_value,
            owner=owner,
            origin=origin,
            keep_reason=keep_reason,
            required=node.required,
            required_reasons=list(node.required_reasons),
            organizational_only=path in organizational,
            direct_source_file_count=node.direct_source_file_count,
            subtree_source_file_count=node.subtree_source_file_count,
            source_child_count=node.source_child_count,
            territory_source_file_count=(
                territory_counts.get(path)
                if territory_counts is not None and label == "MODULE"
                else None
            ),
            emitted_name=(
                _normalize_segment_for_report(path) if label == "MODULE" else None
            ),
            description=info.description if info is not None else None,
            lint_codes=lint_codes.get(path, []),
            invalid_codes=issue_codes.get(path, []),
            children=[_build(c) for c in sorted(node.children, key=_node_key)],
        )

    nodes = [_build(n) for n in sorted(skeleton.nodes, key=_node_key)]

    summary = ReportSummaryV2(
        **_subtree_counts_v2(nodes).model_dump(),
        excluded=(
            len(decision.excluded_source_paths) if decision is not None else 0
        ),
    )
    top_level = [
        TopLevelSummaryV2(
            path=n.path,
            decision=n.decision,
            emitted_name=n.emitted_name,
            counts=_subtree_counts_v2([n]),
        )
        for n in nodes
    ]
    excluded_paths = [
        ExcludedPathEntry(path=e.path, reason=e.reason, explanation=e.explanation)
        for e in sorted(
            decision.excluded_source_paths if decision is not None else [],
            key=lambda e: e.path,
        )
    ]

    return TreeDecisionReportV2(
        source_root=skeleton.source_root,
        merge_threshold=merge_threshold,
        generated_from=GeneratedFrom(
            skeleton_fingerprint=skeleton.inventory_fingerprint,
            artifacts=list(SOURCE_ARTIFACTS_V2),
        ),
        summary=summary,
        top_level=top_level,
        nodes=nodes,
        excluded_paths=excluded_paths,
        pruned=PrunedInfo(
            ignored_dir_names=sorted(skeleton.ignored),
            excluded_paths=sorted(skeleton.excluded),
            skipped_symlinks=sorted(skeleton.skipped_symlinks),
        ),
        extra_assignments=extra_assignments,
        root_issues=root_issues,
        lints=sorted(lints or [], key=lambda lint: (lint.path, lint.code)),
    )


def _normalize_segment_for_report(path: str) -> str:
    from spotlights_engine.schemas.project import _normalize_module_segment

    return _normalize_module_segment(path.rsplit("/", 1)[-1])


def _subtree_counts_v2(nodes: list[TreeDecisionNodeV2]) -> DecisionCountsV2:
    counts = DecisionCountsV2()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        counts.total_nodes += 1
        setattr(counts, node.decision, getattr(counts, node.decision) + 1)
        if node.invalid_codes:
            counts.invalid_nodes += 1
        if node.lint_codes:
            counts.linted_nodes += 1
        stack.extend(node.children)
    return counts


# ── v2 Markdown rendering ─────────────────────────────────────────────────


def render_markdown_v2(
    report: TreeDecisionReportV2,
    *,
    under: str | None = None,
    max_depth: int | None = None,
) -> str:
    """Render the deterministic full-depth assignment report.

    Optional filters are used only for CLI views.
    """
    sections = _sections_v2(report, under)
    rollups = _rollup_index_v2(report)

    out: list[str] = []
    out.append("# Module extractor — tree decisions (assignments)")
    out.append("")
    out.append(f"- **source root**: `{report.source_root or '<repo root>'}`")
    out.append(
        f"- **skeleton fingerprint**: `{report.generated_from.skeleton_fingerprint}`"
    )
    out.append(f"- **schema**: `{report.schema_version}`")
    out.append(f"- **contract**: `{report.contract}`")
    out.append(f"- **merge threshold**: {report.merge_threshold}")
    if under is not None:
        out.append(f"- **filtered to**: `{_norm(under)}`")
    if max_depth is not None:
        out.append(f"- **max depth**: {max_depth}")
    out.append("")

    out.extend(_summary_section_v2(report))
    out.extend(
        [
            "Legend: `●` module parent  `◐` module leaf  `○` part  "
            "`✗` MISSING (no label)  `⊘` excluded (stage 1).",
            "",
            "Deterministically pruned directories (`node_modules`, "
            "`__pycache__`, hidden dirs, …) and skipped symlinks are "
            "content-free and unbounded; they are listed under **Pruned**, "
            "never as nodes.",
            "",
        ]
    )
    out.extend(_problems_section_v2(report))
    out.extend(_lints_section_v2(report))

    for node in sections:
        out.append(f"## `{node.path}`")
        out.append("")
        out.append(_mini_summary_v2(rollups[node.path]))
        out.append("")
        out.append("```text")
        out.extend(_tree_lines_v2(node, rollups, max_depth))
        out.append("```")
        out.append("")

    if not sections:
        out.append("_No skeleton nodes to show._")
        out.append("")

    out.extend(_excluded_section(report))
    out.extend(_pruned_section(report))

    return "\n".join(out).rstrip("\n") + "\n"


def _sections_v2(
    report: TreeDecisionReportV2, under: str | None
) -> list[TreeDecisionNodeV2]:
    if under is None:
        return list(report.nodes)
    target = _norm(under)
    for node in report.iter_nodes():
        if node.path == target:
            return [node]
    return []


def _rollup_index_v2(report: TreeDecisionReportV2) -> dict[str, DecisionCountsV2]:
    index: dict[str, DecisionCountsV2] = {}

    def _visit(node: TreeDecisionNodeV2) -> DecisionCountsV2:
        counts = DecisionCountsV2(total_nodes=1)
        setattr(counts, node.decision, 1)
        if node.invalid_codes:
            counts.invalid_nodes = 1
        if node.lint_codes:
            counts.linted_nodes = 1
        for child in node.children:
            child_counts = _visit(child)
            counts.total_nodes += child_counts.total_nodes
            counts.module_parent += child_counts.module_parent
            counts.module_leaf += child_counts.module_leaf
            counts.part += child_counts.part
            counts.missing += child_counts.missing
            counts.invalid_nodes += child_counts.invalid_nodes
            counts.linted_nodes += child_counts.linted_nodes
        index[node.path] = counts
        return counts

    for node in report.nodes:
        _visit(node)
    return index


def _counts_row_v2(label: str, counts: DecisionCountsV2) -> str:
    return (
        f"| {label} | {counts.total_nodes} | "
        f"{counts.modules} ({counts.module_parent} / {counts.module_leaf}) | "
        f"{counts.part} | {counts.missing} |"
    )


def _summary_section_v2(report: TreeDecisionReportV2) -> list[str]:
    out = ["## Summary", ""]
    out.append("| scope | nodes | modules (parent / leaf) | parts | missing |")
    out.append("|---|---:|---:|---:|---:|")
    out.append(_counts_row_v2("**whole run**", report.summary))
    for top in report.top_level:
        out.append(_counts_row_v2(f"`{top.path}`", top.counts))
    out.append("")
    out.append(
        f"Stage-1 semantic exclusions: {report.summary.excluded} path(s). "
        f"Invalid nodes: {report.summary.invalid_nodes}. "
        f"Linted nodes: {report.summary.linted_nodes}."
    )
    out.append("")
    return out


def _mini_summary_v2(counts: DecisionCountsV2) -> str:
    text = (
        f"{counts.total_nodes} node(s) — {counts.modules} module(s) "
        f"({counts.module_parent} parent / {counts.module_leaf} leaf), "
        f"{counts.part} part(s), {counts.missing} missing"
    )
    if counts.invalid_nodes:
        text += f", {counts.invalid_nodes} invalid"
    if counts.linted_nodes:
        text += f", {counts.linted_nodes} linted"
    return text + "."


def _problems_section_v2(report: TreeDecisionReportV2) -> list[str]:
    missing = [n for n in report.iter_nodes() if n.decision == "missing"]
    invalid = [n for n in report.iter_nodes() if n.invalid_codes]
    if (
        not missing
        and not invalid
        and not report.root_issues
        and not report.extra_assignments
    ):
        return []
    out = ["## Problems", ""]
    if missing:
        out.append(f"### Missing labels ({len(missing)})")
        out.append("")
        for node in sorted(missing, key=lambda n: n.path):
            req = "required" if node.required else "optional"
            out.append(f"- `{node.path}` — {req}, no assignment")
        out.append("")
    if invalid:
        out.append(f"### Invalid nodes ({len(invalid)})")
        out.append("")
        for node in sorted(invalid, key=lambda n: n.path):
            out.append(f"- `{node.path}` — {', '.join(node.invalid_codes)}")
        out.append("")
    if report.root_issues:
        out.append(f"### Report-level issues ({len(report.root_issues)})")
        out.append("")
        for issue in report.root_issues:
            where = f" `{issue.path}`" if issue.path else ""
            out.append(f"- {issue.code}{where}: {_one_line(issue.detail)}")
        out.append("")
    if report.extra_assignments:
        out.append(
            f"### Extra assignment keys outside the inventory "
            f"({len(report.extra_assignments)})"
        )
        out.append("")
        for path in report.extra_assignments:
            out.append(f"- `{path}`")
        out.append("")
    return out


def _lints_section_v2(report: TreeDecisionReportV2) -> list[str]:
    if not report.lints:
        return []
    out = [f"## Lints ({len(report.lints)} — warnings only)", ""]
    for lint in report.lints:
        out.append(f"- `{lint.path}` — {lint.code}: {_one_line(lint.message)}")
    out.append("")
    return out


def _tree_lines_v2(
    root: TreeDecisionNodeV2,
    rollups: dict[str, DecisionCountsV2],
    max_depth: int | None,
) -> list[str]:
    entries: list[tuple[str, str, str]] = []

    def _emit(
        node: TreeDecisionNodeV2, prefix: str, connector: str, depth: int
    ) -> None:
        label = node.path if depth == 0 else node.path.rsplit("/", 1)[-1]
        label = f"{label} ({node.direct_source_file_count} files)"
        left = f"{prefix}{connector}{_SYMBOLS_V2[node.decision]} {label}"
        entries.append((left, _tag_of_v2(node), _detail_of_v2(node, rollups)))

        child_prefix = prefix if depth == 0 else prefix + (
            "    " if connector.startswith("└") else "│   "
        )
        if max_depth is not None and depth >= max_depth:
            hidden = rollups[node.path].total_nodes - 1
            if hidden > 0:
                entries.append(
                    (f"{child_prefix}… {hidden} deeper node(s) hidden", "", "")
                )
            return
        last = len(node.children) - 1
        for i, child in enumerate(node.children):
            _emit(
                child,
                child_prefix,
                "└── " if i == last else "├── ",
                depth + 1,
            )

    _emit(root, "", "", 0)

    width = max((len(left) for left, _, _ in entries), default=0)
    lines: list[str] = []
    for left, tag, detail in entries:
        line = left if not tag else f"{left.ljust(width)}  {tag}"
        if detail:
            line = f"{line}  {detail}"
        lines.append(line.rstrip())
    return lines


def _tag_of_v2(node: TreeDecisionNodeV2) -> str:
    tag = _TAGS_V2[node.decision]
    if node.decision == "part" and node.owner:
        tag = f"{tag} → {node.owner}"
    if node.invalid_codes:
        tag = f"{tag} !INVALID"
    return tag


def _detail_of_v2(
    node: TreeDecisionNodeV2, rollups: dict[str, DecisionCountsV2]
) -> str:
    parts: list[str] = []
    if node.emitted_name:
        parts.append(f"[name: {node.emitted_name}]")
    if node.origin:
        parts.append(f"[origin: {node.origin}]")
    if node.keep_reason:
        parts.append(f"[keep: {_one_line(node.keep_reason)}]")
    if node.territory_source_file_count is not None:
        parts.append(f"[territory: {node.territory_source_file_count} files]")
    if node.lint_codes:
        parts.append(f"[lint: {', '.join(node.lint_codes)}]")
    if node.decision == "module_parent":
        counts = rollups[node.path]
        parts.append(
            f"(subtree: {counts.total_nodes} nodes — {counts.modules} modules, "
            f"{counts.part} parts"
            + (f", {counts.missing} missing" if counts.missing else "")
            + ")"
        )
    return "  ".join(parts)


# ── Offline CLI ───────────────────────────────────────────────────────────


class ArtifactsNotFoundError(FileNotFoundError):
    """A required input artifact is absent from the run directory."""


def _resolve_run_dir(run_dir: Path) -> Path:
    if (run_dir / "02_skeleton" / "skeleton.json").exists():
        return run_dir
    if (run_dir / "skeleton.json").exists():
        return run_dir
    nested = run_dir / "modules_extractor"
    if nested.is_dir():
        return nested
    return run_dir


def _load_json(run_dir: Path, candidates: list[str], *, required: bool):
    for rel in candidates:
        path = run_dir / rel
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    if required:
        raise ArtifactsNotFoundError(
            f"none of {candidates} found under {run_dir}"
        )
    return None


def load_report_from_run_dir(run_dir: Path) -> TreeDecisionReportV2:
    """Rebuild the assignment report from an assignment-contract run directory.

    Only the skeleton is hard-required; everything else is best-effort so a
    failed run still renders.
    """
    base = _resolve_run_dir(run_dir)
    skeleton = Skeleton.model_validate(
        _load_json(
            base, ["02_skeleton/skeleton.json", "skeleton.json"], required=True
        )
    )

    assignment_data = _load_json(
        base,
        [
            "assignment_tree.json",
            "03_enrich/assignment_tree.json",
            "03_enrich/merged/assignment_tree.json",
        ],
        required=False,
    )
    resolved_data = _load_json(
        base,
        ["resolved_assignments.json", "03_enrich/resolved_assignments.json"],
        required=False,
    )
    metadata_data = _load_json(
        base,
        ["module_metadata.json", "03_enrich/module_metadata.json"],
        required=False,
    )
    coverage_data = _load_json(base, ["coverage.json"], required=False)
    lints_data = _load_json(base, ["assignment_lints.json"], required=False)
    config_data = _load_json(base, ["extractor_config.json"], required=False)
    decision_data = _load_json(
        base,
        ["source_root_decision.json", "01_source_root/decision.json"],
        required=False,
    )

    root_issues: list[RootIssue] = []

    resolved: ResolvedAssignmentTree | None = None
    if resolved_data is not None:
        resolved = ResolvedAssignmentTree.model_validate(resolved_data)
        # Persisted derived maps are audit data, never trusted input.
        from spotlights_engine.modules_extractor.coverage import CrossArtifactError
        from spotlights_engine.modules_extractor.derive import (
            verify_resolved_assignments,
        )

        try:
            verify_resolved_assignments(resolved, skeleton)
        except CrossArtifactError as exc:
            root_issues.append(
                RootIssue(
                    code="resolved_artifact_mismatch",
                    detail=f"resolved_assignments.json failed recomputation: {exc}",
                )
            )
            resolved = None

    assignments: AssignmentTree | None = None
    if assignment_data is not None:
        assignments = AssignmentTree.model_validate(assignment_data)
    elif resolved is not None:
        assignments = AssignmentTree(
            assignments=dict(resolved.assignments),
            module_decisions=dict(resolved.module_decisions),
        )

    metadata = None
    if metadata_data is not None:
        metadata = ModuleMetadataMap.model_validate(metadata_data).modules

    coverage: AssignmentCoverageReport | None = None
    if (
        isinstance(coverage_data, dict)
        and coverage_data.get("schema_version") == "coverage.v2"
    ):
        coverage = AssignmentCoverageReport.model_validate(coverage_data)

    lints = (
        [AssignmentLint.model_validate(x) for x in lints_data]
        if isinstance(lints_data, list)
        else None
    )

    merge_threshold = MERGE_THRESHOLD_DEFAULT
    if resolved is not None:
        merge_threshold = resolved.merge_threshold
    elif isinstance(config_data, dict) and isinstance(
        config_data.get("merge_threshold"), int
    ):
        merge_threshold = config_data["merge_threshold"]

    decision = (
        SourceRootDecision.model_validate(decision_data)
        if decision_data is not None
        else None
    )

    report = build_tree_decision_report_v2(
        skeleton,
        assignments=assignments,
        resolved=resolved,
        metadata=metadata,
        coverage=coverage,
        lints=lints,
        issues=None,
        decision=decision,
        merge_threshold=merge_threshold,
    )
    if root_issues:
        report.root_issues = [*root_issues, *report.root_issues]
    return report




def load_any_report_from_run_dir(run_dir: Path) -> TreeDecisionReportV2:
    base = _resolve_run_dir(run_dir)
    existing = _load_json(base, [REPORT_JSON_NAME], required=False)
    if isinstance(existing, dict) and existing.get("schema_version") == SCHEMA_VERSION_V2:
        try:
            return TreeDecisionReportV2.model_validate(existing)
        except ValidationError:
            pass
    return load_report_from_run_dir(run_dir)


def render_any_markdown(
    report: TreeDecisionReportV2,
    *,
    under: str | None = None,
    max_depth: int | None = None,
) -> str:
    return render_markdown_v2(report, under=under, max_depth=max_depth)


def report_json_text(report: TreeDecisionReportV2) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spotlights_engine.modules_extractor.tree_report",
        description="Rebuild the per-directory assignment report for a modules-extractor run.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print Markdown instead of writing both report files",
    )
    parser.add_argument("--under", default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args(argv)

    if args.max_depth is not None and args.max_depth < 0:
        parser.error("--max-depth must be >= 0")
    if (args.under is not None or args.max_depth is not None) and not args.stdout:
        parser.error(
            "--under/--max-depth render a partial view; pass --stdout. The "
            "files written into a run directory are always full-depth."
        )

    try:
        report = load_any_report_from_run_dir(args.run_dir)
    except ArtifactsNotFoundError as exc:
        print(f"tree_report: {exc}", file=sys.stderr)
        return 2

    markdown = render_any_markdown(
        report, under=args.under, max_depth=args.max_depth
    )
    if args.under is not None and not _sections_v2(report, args.under):
        print(
            f"tree_report: no node matches --under {args.under!r}",
            file=sys.stderr,
        )
        return 2

    if args.stdout:
        sys.stdout.write(markdown)
        return 0

    base = _resolve_run_dir(args.run_dir)
    (base / REPORT_JSON_NAME).write_text(report_json_text(report), encoding="utf-8")
    (base / REPORT_MD_NAME).write_text(markdown, encoding="utf-8")
    print(f"tree_report: wrote {base / REPORT_JSON_NAME} and {base / REPORT_MD_NAME}")
    return 0


__all__ = [
    "SCHEMA_VERSION_V2",
    "DecisionCountsV2",
    "ExcludedPathEntry",
    "PrunedInfo",
    "ReportSummaryV2",
    "RootIssue",
    "TopLevelSummaryV2",
    "TreeDecisionNodeV2",
    "TreeDecisionReportV2",
    "build_tree_decision_report_v2",
    "load_any_report_from_run_dir",
    "load_report_from_run_dir",
    "render_any_markdown",
    "render_markdown_v2",
    "report_json_text",
]


if __name__ == "__main__":
    raise SystemExit(main())
