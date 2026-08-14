"""Derived per-directory decision report for a modules-extractor run.

See `design/module_extractor_visualization.md`. This module answers one
question for every source-bearing directory the pipeline saw: **what did the
run decide about it?** — emitted as a module (parent or leaf), folded into an
emitted ancestor, left unaccounted, or missing.

It is a *pure join* over four already-persisted artifacts (`skeleton.json`,
`enriched_tree.json`, `coverage.json`, `source_root_decision.json`): no LLM,
no filesystem walking, no new instrumentation, and — per the design's
non-goals — no change to any existing artifact, schema, or stage behavior.

Two outputs, both written at the run-dir root next to `project_tree.json`:

- `tree_decisions.json` — the canonical artifact (`TreeDecisionReport`).
- `tree_decisions.md`   — the human rendering (`render_markdown`).

Both are deterministic: identical inputs produce byte-identical bytes (no
timestamps, no absolute paths, no set iteration order).

Stage-1 semantic exclusions never entered the skeleton, so they are grafted in
as a flat `excluded_paths` ledger rather than as tree nodes; deterministic
prunes (`node_modules`, `__pycache__`, symlinks, …) are unbounded and
content-free, so they are summarized once under `pruned`.

Offline use, for any past run (the artifacts have always been persisted)::

    python -m spotlights_engine.modules_extractor.tree_report <run_dir> \
        [--stdout] [--under PATH] [--max-depth N]

`--under`/`--max-depth` render a partial *view* and so require `--stdout`: the
two files in a run directory are always the full-depth report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.coverage import CoverageReport
from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedTree,
    Skeleton,
    SkeletonNode,
    SourceRootDecision,
)

SCHEMA_VERSION: Final[Literal["tree_decisions.v1"]] = "tree_decisions.v1"

REPORT_JSON_NAME = "tree_decisions.json"
REPORT_MD_NAME = "tree_decisions.md"

SOURCE_ARTIFACTS = [
    "02_skeleton/skeleton.json",
    "enriched_tree.json",
    "coverage.json",
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


def _excluded_section(report: TreeDecisionReport) -> list[str]:
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


def _pruned_section(report: TreeDecisionReport) -> list[str]:
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
    """Rebuild the report from an existing run directory's artifacts."""
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


def report_json_text(report: TreeDecisionReport) -> str:
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
        report = load_report_from_run_dir(args.run_dir)
    except ArtifactsNotFoundError as exc:
        print(f"tree_report: {exc}", file=sys.stderr)
        return 2

    markdown = render_markdown(
        report, under=args.under, max_depth=args.max_depth
    )
    if args.under is not None and not _sections(report, args.under):
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
    "DecisionCounts",
    "ExcludedPathEntry",
    "PrunedInfo",
    "ReportSummary",
    "TopLevelSummary",
    "TreeDecisionNode",
    "TreeDecisionReport",
    "build_tree_decision_report",
    "load_report_from_run_dir",
    "render_markdown",
    "report_json_text",
]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
