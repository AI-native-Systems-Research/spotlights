"""Derived per-directory decision report for a modules-extractor run.

See `design/module_extractor_visualization.md`. This module answers one
question for every source-bearing directory the pipeline saw: **what did the
run decide about it?** It is a *pure join* over already-persisted artifacts:
no LLM, no filesystem walking, no new instrumentation.

Two report versions coexist while the `ExtractorConfig.contract` migration
flag exists:

- `tree_decisions.v1` — the tree contract: emitted (parent/leaf), folded into
  an emitted ancestor, optional-unaccounted, or missing; joined from
  `skeleton.json`, `enriched_tree.json`, v1 `coverage.json`, and
  `source_root_decision.json`.
- `tree_decisions.v2` — the assignment contract: `module_parent`,
  `module_leaf`, `part` (with its resolved owner), or `missing`; joined from
  the skeleton, raw or resolved assignments, module metadata when available,
  v2 coverage, structured issues, lints, and the effective `merge_threshold`.
  Invalid-node codes and lints are orthogonal to the decision buckets, so the
  summary always reconciles with the inventory.

Two outputs, both written at the run-dir root next to `project_tree.json`:
`tree_decisions.json` (canonical) and `tree_decisions.md` (human rendering).
Both are deterministic: identical inputs produce byte-identical bytes (no
timestamps, no absolute paths, no set iteration order).

Stage-1 semantic exclusions never entered the skeleton, so they are grafted in
as a flat `excluded_paths` ledger rather than as tree nodes; deterministic
prunes (`node_modules`, `__pycache__`, symlinks, …) are unbounded and
content-free, so they are summarized once under `pruned`.

Offline use, for any past run (the artifacts have always been persisted)::

    python -m spotlights_engine.modules_extractor.tree_report <run_dir> \
        [--stdout] [--under PATH] [--max-depth N]

The CLI dispatches on an existing `tree_decisions.json.schema_version` when
present, otherwise on which contract's artifacts exist, and rebuilds the
matching report version. `--under`/`--max-depth` render a partial *view* and
so require `--stdout`: the two files in a run directory are always the
full-depth report.
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
from spotlights_engine.modules_extractor.coverage import CoverageReport
from spotlights_engine.modules_extractor.derive import AssignmentLint
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentTree,
    EnrichedTree,
    ModuleInfo,
    ModuleMetadataMap,
    ResolvedAssignmentTree,
    Skeleton,
    SkeletonNode,
    SourceRootDecision,
)

SCHEMA_VERSION: Final[Literal["tree_decisions.v1"]] = "tree_decisions.v1"
SCHEMA_VERSION_V2: Final[Literal["tree_decisions.v2"]] = "tree_decisions.v2"

REPORT_JSON_NAME = "tree_decisions.json"
REPORT_MD_NAME = "tree_decisions.md"

# Per-version source-artifact lists (recorded in `generated_from`).
SOURCE_ARTIFACTS = [
    "02_skeleton/skeleton.json",
    "enriched_tree.json",
    "coverage.json",
    "source_root_decision.json",
]
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

# One decision per skeleton node. `excluded` (Stage-1 semantic exclusion) is
# deliberately *not* in this enum: those paths are absent from the skeleton, so
# they carry no node metadata and live in `TreeDecisionReport.excluded_paths`.
Decision = Literal[
    "emitted_parent",
    "emitted_leaf",
    "folded",
    "missing",
    "optional_unaccounted",
]

_SYMBOLS: dict[str, str] = {
    "emitted_parent": "●",  # ●
    "emitted_leaf": "◐",  # ◐
    "folded": "○",  # ○
    "optional_unaccounted": "✕",  # ✕
    "missing": "✗",  # ✗
}
_EXCLUDED_SYMBOL = "⊘"  # ⊘

_TAGS: dict[str, str] = {
    "emitted_parent": "EMITTED parent",
    "emitted_leaf": "EMITTED leaf",
    "folded": "FOLDED",
    "optional_unaccounted": "UNACCOUNTED (optional)",
    "missing": "MISSING (required!)",
}

# A fold reason is a free-text model string; the tree view keeps one line per
# node, so it is collapsed and bounded here. The full text is in the JSON.
_MAX_REASON_CHARS = 110


def _norm(path: str) -> str:
    """Same normalization `coverage.py` applies before set membership."""
    return path.strip().strip("/")


# ── Models ────────────────────────────────────────────────────────────────


class DecisionCounts(BaseModel):
    """Per-decision tallies over a set of skeleton nodes."""

    model_config = ConfigDict(extra="forbid")

    total_nodes: int = 0
    emitted_parent: int = 0
    emitted_leaf: int = 0
    folded: int = 0
    missing: int = 0
    optional_unaccounted: int = 0
    invalid_folds: int = 0

    @property
    def emitted(self) -> int:
        return self.emitted_parent + self.emitted_leaf


class ReportSummary(DecisionCounts):
    """Whole-run tallies. `excluded` is *not* part of `total_nodes`: Stage-1
    exclusions never entered the skeleton."""

    excluded: int = 0


class TopLevelSummary(BaseModel):
    """One row per top-level skeleton node — the ten-line shape of the run."""

    model_config = ConfigDict(extra="forbid")

    path: str
    decision: Decision
    emitted_name: str | None = None
    counts: DecisionCounts


class ExcludedPathEntry(BaseModel):
    """A Stage-1 semantic exclusion, grafted in from `source_root_decision`."""

    model_config = ConfigDict(extra="forbid")

    path: str
    reason: str
    explanation: str


class PrunedInfo(BaseModel):
    """Deterministic, content-free prunes — listed once, never as nodes."""

    model_config = ConfigDict(extra="forbid")

    ignored_dir_names: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)
    skipped_symlinks: list[str] = Field(default_factory=list)


class GeneratedFrom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skeleton_fingerprint: str
    artifacts: list[str] = Field(default_factory=list)


class TreeDecisionNode(BaseModel):
    """One source-bearing directory and the single decision made about it.

    `children` mirrors the skeleton's nesting so a consumer never re-derives
    hierarchy from path prefixes.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    decision: Decision
    required: bool
    required_reasons: list[str] = Field(default_factory=list)
    organizational_only: bool = False
    direct_source_file_count: int = Field(default=0, ge=0)
    source_child_count: int = Field(default=0, ge=0)
    folded_into: str | None = None
    fold_reason: str | None = None
    evidence_files: list[str] = Field(default_factory=list)
    fold_invalid: bool = False
    emitted_name: str | None = None
    top_level_module: bool = False
    children: list[TreeDecisionNode] = Field(default_factory=list)


class TreeDecisionReport(BaseModel):
    """The canonical `tree_decisions.json` artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["tree_decisions.v1"] = SCHEMA_VERSION
    source_root: str
    generated_from: GeneratedFrom
    summary: ReportSummary
    top_level: list[TopLevelSummary] = Field(default_factory=list)
    nodes: list[TreeDecisionNode] = Field(default_factory=list)
    excluded_paths: list[ExcludedPathEntry] = Field(default_factory=list)
    pruned: PrunedInfo = Field(default_factory=PrunedInfo)

    def iter_nodes(self):
        """Preorder traversal over every `TreeDecisionNode`."""

        def _walk(nodes: list[TreeDecisionNode]):
            for n in nodes:
                yield n
                yield from _walk(n.children)

        yield from _walk(self.nodes)


TreeDecisionNode.model_rebuild()


# ── Build ─────────────────────────────────────────────────────────────────


def build_tree_decision_report(
    skeleton: Skeleton,
    enriched: EnrichedTree,
    coverage: CoverageReport,
    decision: SourceRootDecision | None = None,
) -> TreeDecisionReport:
    """Join the four stage artifacts into one per-node decision report.

    A single preorder walk with pre-built dict/set lookups — `O(nodes)`, with
    recursion depth equal to directory depth (tens, not thousands).

    Decision resolution per skeleton path, first match wins:
    `emitted` (parent vs leaf via the emitted module's `submodules`) →
    `folded` → `missing` (iff `required`) → `optional_unaccounted`.
    `invalid_folds` tags the node with `fold_invalid` rather than forming a
    separate decision, so every skeleton node still lands in exactly one
    decision bucket and the summary reconciles with `coverage.json`.

    `decision` is optional so the report can still be produced on a failure
    path that never reached (or never persisted) Stage 1's output; without it
    `excluded_paths` is simply empty.
    """
    emitted_by_path = {
        _norm(m.path): m for m in enriched.iter_all_modules()
    }
    folds_by_path = {_norm(f.path): f for f in enriched.folds}
    # `coverage` is authoritative for set membership; the enriched tree is the
    # only source of the *names* and fold metadata. Union them so a report is
    # still coherent if it is built on a failure path from a coverage report
    # computed against a different (earlier) tree.
    emitted_paths = set(coverage.emitted) | set(emitted_by_path)
    folded_paths = set(coverage.folded) | set(folds_by_path)
    invalid_folds = set(coverage.invalid_folds)
    organizational = set(skeleton.organizational_only)
    top_module_paths = {_norm(m.path) for m in enriched.modules}

    def _build(node: SkeletonNode) -> TreeDecisionNode:
        path = _norm(node.path)
        module = emitted_by_path.get(path)
        fold = folds_by_path.get(path)

        if path in emitted_paths:
            has_subs = bool(module.submodules) if module is not None else False
            decision_value: Decision = (
                "emitted_parent" if has_subs else "emitted_leaf"
            )
        elif path in folded_paths:
            decision_value = "folded"
        elif node.required:
            decision_value = "missing"
        else:
            decision_value = "optional_unaccounted"

        return TreeDecisionNode(
            path=path,
            decision=decision_value,
            required=node.required,
            required_reasons=list(node.required_reasons),
            organizational_only=path in organizational,
            direct_source_file_count=node.direct_source_file_count,
            source_child_count=node.source_child_count,
            folded_into=_norm(fold.into) if fold is not None else None,
            fold_reason=fold.reason if fold is not None else None,
            evidence_files=(
                sorted(fold.evidence_files) if fold is not None else []
            ),
            fold_invalid=path in invalid_folds,
            emitted_name=module.name if module is not None else None,
            top_level_module=path in top_module_paths,
            children=[_build(c) for c in sorted(node.children, key=_node_key)],
        )

    nodes = [_build(n) for n in sorted(skeleton.nodes, key=_node_key)]

    summary = ReportSummary(
        **subtree_counts_of(nodes).model_dump(),
        excluded=(
            len(decision.excluded_source_paths) if decision is not None else 0
        ),
    )
    top_level = [
        TopLevelSummary(
            path=n.path,
            decision=n.decision,
            emitted_name=n.emitted_name,
            counts=subtree_counts_of([n]),
        )
        for n in nodes
    ]
    excluded_paths = [
        ExcludedPathEntry(
            path=e.path, reason=e.reason, explanation=e.explanation
        )
        for e in sorted(
            decision.excluded_source_paths if decision is not None else [],
            key=lambda e: e.path,
        )
    ]

    return TreeDecisionReport(
        source_root=skeleton.source_root,
        generated_from=GeneratedFrom(
            skeleton_fingerprint=skeleton.inventory_fingerprint,
            artifacts=list(SOURCE_ARTIFACTS),
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
    )


def _node_key(node: SkeletonNode) -> str:
    return _norm(node.path)


def subtree_counts_of(nodes: list[TreeDecisionNode]) -> DecisionCounts:
    """Tally `nodes` and everything beneath them (each node counted once)."""
    counts = DecisionCounts()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        counts.total_nodes += 1
        setattr(counts, node.decision, getattr(counts, node.decision) + 1)
        if node.fold_invalid:
            counts.invalid_folds += 1
        stack.extend(node.children)
    return counts


# ── Markdown rendering ────────────────────────────────────────────────────


def render_markdown(
    report: TreeDecisionReport,
    *,
    under: str | None = None,
    max_depth: int | None = None,
) -> str:
    """Render `report` as deterministic Markdown.

    Structured for monorepo scale (the design's reference point is vLLM:
    ~345 required nodes, 500–700 directories): a global summary table with one
    row per top-level node first, the Problems section second, then one `##`
    section per top-level skeleton node — headings give free navigation in any
    editor or on GitHub with no tooling. Nothing is ever omitted from the files
    the pipeline writes; `under`/`max_depth` are CLI-only view filters.
    """
    sections = _sections(report, under)
    rollups = _rollup_index(report)

    out: list[str] = []
    out.append("# Module extractor — tree decisions")
    out.append("")
    out.append(f"- **source root**: `{report.source_root or '<repo root>'}`")
    out.append(
        f"- **skeleton fingerprint**: `{report.generated_from.skeleton_fingerprint}`"
    )
    out.append(f"- **schema**: `{report.schema_version}`")
    if under is not None:
        out.append(f"- **filtered to**: `{_norm(under)}`")
    if max_depth is not None:
        out.append(f"- **max depth**: {max_depth}")
    out.append("")

    out.extend(_summary_section(report))
    out.extend(_legend_lines())
    out.extend(_problems_section(report))

    for node in sections:
        out.append(f"## `{node.path}`")
        out.append("")
        out.append(_mini_summary(rollups[node.path]))
        out.append("")
        out.append("```text")
        out.extend(_tree_lines(node, rollups, max_depth))
        out.append("```")
        out.append("")

    if not sections:
        out.append("_No skeleton nodes to show._")
        out.append("")

    out.extend(_excluded_section(report))
    out.extend(_pruned_section(report))

    return "\n".join(out).rstrip("\n") + "\n"


def _sections(
    report: TreeDecisionReport, under: str | None
) -> list[TreeDecisionNode]:
    if under is None:
        return list(report.nodes)
    target = _norm(under)
    for node in report.iter_nodes():
        if node.path == target:
            return [node]
    return []


def _rollup_index(report: TreeDecisionReport) -> dict[str, DecisionCounts]:
    """Subtree counts for every node, computed bottom-up in one pass."""
    index: dict[str, DecisionCounts] = {}

    def _visit(node: TreeDecisionNode) -> DecisionCounts:
        counts = DecisionCounts(total_nodes=1)
        setattr(counts, node.decision, 1)
        if node.fold_invalid:
            counts.invalid_folds = 1
        for child in node.children:
            child_counts = _visit(child)
            counts.total_nodes += child_counts.total_nodes
            counts.emitted_parent += child_counts.emitted_parent
            counts.emitted_leaf += child_counts.emitted_leaf
            counts.folded += child_counts.folded
            counts.missing += child_counts.missing
            counts.optional_unaccounted += child_counts.optional_unaccounted
            counts.invalid_folds += child_counts.invalid_folds
        index[node.path] = counts
        return counts

    for node in report.nodes:
        _visit(node)
    return index


def _counts_row(label: str, counts: DecisionCounts) -> str:
    return (
        f"| {label} | {counts.total_nodes} | "
        f"{counts.emitted} ({counts.emitted_parent} / {counts.emitted_leaf}) | "
        f"{counts.folded} | {counts.optional_unaccounted} | {counts.missing} |"
    )


def _summary_section(report: TreeDecisionReport) -> list[str]:
    out = ["## Summary", ""]
    out.append("| scope | nodes | emitted (parent / leaf) | folded | unaccounted | missing |")
    out.append("|---|---:|---:|---:|---:|---:|")
    out.append(_counts_row("**whole run**", report.summary))
    for top in report.top_level:
        out.append(_counts_row(f"`{top.path}`", top.counts))
    out.append("")
    out.append(
        f"Stage-1 semantic exclusions: {report.summary.excluded} "
        f"path(s). Invalid folds: {report.summary.invalid_folds}."
    )
    out.append("")
    return out


def _legend_lines() -> list[str]:
    return [
        "Legend: `●` emitted parent  `◐` emitted leaf  `○` folded  "
        "`✕` optional-unaccounted  `✗` MISSING (required!)  "
        "`⊘` excluded (stage 1).",
        "",
        "Deterministically pruned directories (`node_modules`, `__pycache__`, "
        "hidden dirs, …) and skipped symlinks are content-free and unbounded; "
        "they are listed under **Pruned**, never as nodes.",
        "",
    ]


def _mini_summary(counts: DecisionCounts) -> str:
    text = (
        f"{counts.total_nodes} node(s) — {counts.emitted} emitted "
        f"({counts.emitted_parent} parent / {counts.emitted_leaf} leaf), "
        f"{counts.folded} folded, {counts.optional_unaccounted} unaccounted, "
        f"{counts.missing} missing"
    )
    if counts.invalid_folds:
        text += f", {counts.invalid_folds} invalid fold(s)"
    return text + "."


def _problems_section(report: TreeDecisionReport) -> list[str]:
    missing = [n for n in report.iter_nodes() if n.decision == "missing"]
    invalid = [n for n in report.iter_nodes() if n.fold_invalid]
    if not missing and not invalid:
        return []
    out = ["## Problems", ""]
    if missing:
        out.append(f"### Missing required paths ({len(missing)})")
        out.append("")
        for node in sorted(missing, key=lambda n: n.path):
            reasons = ", ".join(node.required_reasons) or "required"
            out.append(f"- `{node.path}` — {reasons}")
        out.append("")
    if invalid:
        out.append(f"### Invalid folds ({len(invalid)})")
        out.append("")
        for node in sorted(invalid, key=lambda n: n.path):
            into = node.folded_into or "?"
            out.append(f"- `{node.path}` → `{into}`")
        out.append("")
    return out


def _excluded_section(
    report: TreeDecisionReport | TreeDecisionReportV2,
) -> list[str]:
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


def _pruned_section(
    report: TreeDecisionReport | TreeDecisionReportV2,
) -> list[str]:
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


def _inline_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{v}`" for v in values)


def _one_line(text: str, limit: int = _MAX_REASON_CHARS) -> str:
    collapsed = " ".join(text.split()).replace("|", "\\|")
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "…"
    return collapsed


def _tree_lines(
    root: TreeDecisionNode,
    rollups: dict[str, DecisionCounts],
    max_depth: int | None,
) -> list[str]:
    """Box-drawing tree for one section, with the label column aligned."""
    entries: list[tuple[str, str, str]] = []

    def _emit(node: TreeDecisionNode, prefix: str, connector: str, depth: int) -> None:
        label = node.path if depth == 0 else node.path.rsplit("/", 1)[-1]
        label = f"{label} ({node.direct_source_file_count} files)"
        left = f"{prefix}{connector}{_SYMBOLS[node.decision]} {label}"
        entries.append((left, _tag_of(node), _detail_of(node, rollups)))

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


def _tag_of(node: TreeDecisionNode) -> str:
    tag = _TAGS[node.decision]
    if node.decision == "folded":
        tag = f"{tag} → {node.folded_into or '?'}"
    if node.fold_invalid:
        tag = f"{tag} !INVALID-FOLD"
    return tag


def _detail_of(
    node: TreeDecisionNode, rollups: dict[str, DecisionCounts]
) -> str:
    parts: list[str] = []
    if node.emitted_name:
        parts.append(f"[name: {node.emitted_name}]")
    if node.required:
        reasons = "; ".join(node.required_reasons) or "required"
        parts.append(f"[required: {_one_line(reasons)}]")
    else:
        parts.append("[optional]")
    if node.organizational_only:
        parts.append("[organizational-only]")
    if node.fold_reason:
        parts.append(f"({_one_line(node.fold_reason)})")
    if node.decision == "emitted_parent":
        counts = rollups[node.path]
        parts.append(
            f"(subtree: {counts.total_nodes} nodes — {counts.emitted} emitted, "
            f"{counts.folded} folded, {counts.optional_unaccounted} unaccounted"
            + (f", {counts.missing} missing" if counts.missing else "")
            + ")"
        )
    return "  ".join(parts)


# ── v2 (assignment contract) ──────────────────────────────────────────────

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
    """Deterministic Markdown for a v2 report — same full-depth structure,
    problems section, exclusions/pruned sections, and `--under`/`--max-depth`
    stdout-only filtering as v1."""
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
    """Accept either the `modules_extractor/` run dir or its parent."""
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


def load_report_from_run_dir(run_dir: Path) -> TreeDecisionReport:
    """Rebuild the v1 report from an existing run directory's artifacts."""
    base = _resolve_run_dir(run_dir)
    skeleton = Skeleton.model_validate(
        _load_json(
            base, ["02_skeleton/skeleton.json", "skeleton.json"], required=True
        )
    )
    enriched = EnrichedTree.model_validate(
        _load_json(
            base,
            [
                "enriched_tree.json",
                "03_enrich/enriched_tree.json",
                "03_enrich/merged/enriched_tree.json",
            ],
            required=True,
        )
    )
    coverage = CoverageReport.model_validate(
        _load_json(
            base,
            ["coverage.json", "03_enrich/merged/coverage.json"],
            required=True,
        )
    )
    decision_data = _load_json(
        base,
        ["source_root_decision.json", "01_source_root/decision.json"],
        required=False,
    )
    decision = (
        SourceRootDecision.model_validate(decision_data)
        if decision_data is not None
        else None
    )
    return build_tree_decision_report(skeleton, enriched, coverage, decision)


def load_report_v2_from_run_dir(run_dir: Path) -> TreeDecisionReportV2:
    """Rebuild the v2 report from an assignment-contract run directory.

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


def detect_report_version(run_dir: Path) -> str:
    """Which report version this run directory should get.

    Dispatch on an existing `tree_decisions.json.schema_version` when present,
    otherwise on which contract's artifacts exist.
    """
    base = _resolve_run_dir(run_dir)
    existing = _load_json(base, [REPORT_JSON_NAME], required=False)
    if isinstance(existing, dict) and existing.get("schema_version") in (
        SCHEMA_VERSION,
        SCHEMA_VERSION_V2,
    ):
        return existing["schema_version"]
    for rel in (
        "resolved_assignments.json",
        "assignment_tree.json",
        "03_enrich/assignment_tree.json",
        "03_enrich/merged/assignment_tree.json",
    ):
        if (base / rel).exists():
            return SCHEMA_VERSION_V2
    return SCHEMA_VERSION


def load_any_report_from_run_dir(
    run_dir: Path,
) -> TreeDecisionReport | TreeDecisionReportV2:
    base = _resolve_run_dir(run_dir)
    existing = _load_json(base, [REPORT_JSON_NAME], required=False)
    if isinstance(existing, dict):
        version = existing.get("schema_version")
        try:
            if version == SCHEMA_VERSION_V2:
                return TreeDecisionReportV2.model_validate(existing)
            if version == SCHEMA_VERSION:
                return TreeDecisionReport.model_validate(existing)
        except ValidationError:
            # A truncated/old report still supplies the dispatch version; fall
            # through and rebuild it from the contract artifacts below.
            pass

    if detect_report_version(base) == SCHEMA_VERSION_V2:
        return load_report_v2_from_run_dir(run_dir)
    return load_report_from_run_dir(run_dir)


def render_any_markdown(
    report: TreeDecisionReport | TreeDecisionReportV2,
    *,
    under: str | None = None,
    max_depth: int | None = None,
) -> str:
    if isinstance(report, TreeDecisionReportV2):
        return render_markdown_v2(report, under=under, max_depth=max_depth)
    return render_markdown(report, under=under, max_depth=max_depth)


def report_json_text(report: TreeDecisionReport | TreeDecisionReportV2) -> str:
    """The exact bytes `atomic_write_json` would write for this report."""
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spotlights_engine.modules_extractor.tree_report",
        description=(
            "Rebuild the per-directory decision report for a past "
            "modules-extractor run."
        ),
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="a modules_extractor/ run directory (or its parent)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the Markdown to stdout instead of writing both files",
    )
    parser.add_argument(
        "--under",
        default=None,
        help=(
            "render only the subtree rooted at this repo-relative path "
            "(a partial view; requires --stdout)"
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help=(
            "truncate the rendered tree at this depth, 0 = roots only "
            "(a partial view; requires --stdout)"
        ),
    )
    args = parser.parse_args(argv)

    if args.max_depth is not None and args.max_depth < 0:
        parser.error("--max-depth must be >= 0")

    # `--under`/`--max-depth` are *view* filters. Writing one into the run
    # directory would replace the canonical full-depth `tree_decisions.md` with
    # a partial rendering and leave it disagreeing with the (always full)
    # `tree_decisions.json` beside it, so a filtered view only ever goes to
    # stdout — the files in a run directory stay full-depth.
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
    if args.under is not None:
        matched = (
            _sections_v2(report, args.under)
            if isinstance(report, TreeDecisionReportV2)
            else _sections(report, args.under)
        )
        if not matched:
            print(
                f"tree_report: no node matches --under {args.under!r}",
                file=sys.stderr,
            )
            return 2

    if args.stdout:
        sys.stdout.write(markdown)
        return 0

    base = _resolve_run_dir(args.run_dir)
    (base / REPORT_JSON_NAME).write_text(
        report_json_text(report), encoding="utf-8"
    )
    (base / REPORT_MD_NAME).write_text(markdown, encoding="utf-8")
    print(f"tree_report: wrote {base / REPORT_JSON_NAME} and {base / REPORT_MD_NAME}")
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V2",
    "DecisionCounts",
    "DecisionCountsV2",
    "ExcludedPathEntry",
    "PrunedInfo",
    "ReportSummary",
    "ReportSummaryV2",
    "RootIssue",
    "TopLevelSummary",
    "TopLevelSummaryV2",
    "TreeDecisionNode",
    "TreeDecisionNodeV2",
    "TreeDecisionReport",
    "TreeDecisionReportV2",
    "build_tree_decision_report",
    "build_tree_decision_report_v2",
    "detect_report_version",
    "load_any_report_from_run_dir",
    "load_report_from_run_dir",
    "load_report_v2_from_run_dir",
    "render_any_markdown",
    "render_markdown",
    "render_markdown_v2",
    "report_json_text",
]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
