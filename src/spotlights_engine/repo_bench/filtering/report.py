"""Independent evaluation of filter rules — per-rule drop/keep statistics.

Unlike the standard `derive()` path (which short-circuits on first
failing predicate), this evaluates every PR against every predicate
independently. The result shows how many PRs each rule would reject
on its own, enabling overlap analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from spotlights_engine.repo_bench.filtering.rules import Rule, RulePredicate, RuleRanker
from spotlights_engine.repo_bench.schemas import RawPR
from spotlights_engine.repo_bench.storage import raw_dir, read_jsonl_lenient


@dataclass(frozen=True)
class RuleStats:
    name: str
    n_dropped: int
    n_kept: int
    pct_dropped: float
    pct_kept: float


@dataclass(frozen=True)
class FilterReport:
    n_input: int
    n_kept_combined: int
    n_dropped_combined: int
    per_rule: list[RuleStats]
    overlap_matrix: dict[str, dict[str, int]] = field(default_factory=dict)


def generate_filter_report(
    window_id: str,
    rules: list[Rule],
    *,
    data_root_override: Path | None = None,
) -> FilterReport:
    """Evaluate each predicate independently over all raw PRs.

    Returns per-rule keep/drop counts plus the combined result.
    The overlap matrix shows how many PRs are dropped by both rule_i
    and rule_j (diagonal = self-drop count).
    """
    predicates = [r for r in rules if isinstance(r, RulePredicate)]
    if not predicates:
        raise ValueError("at least one predicate rule required")

    raw_path = raw_dir(window_id, root=data_root_override) / "prs.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"no raw scrape at {raw_path}; run `repo-bench aggregate` first"
        )

    rows: list[RawPR] = []
    for d in read_jsonl_lenient(raw_path):
        try:
            rows.append(RawPR.model_validate(d))
        except ValidationError:
            pass

    n_input = len(rows)

    dropped_sets: dict[str, set[int]] = {p.NAME: set() for p in predicates}

    for pr in rows:
        for p in predicates:
            if not p.keep(pr):
                dropped_sets[p.NAME].add(pr.pr_number)

    # Combined: a PR is kept only if it passes ALL predicates.
    all_dropped = set()
    for s in dropped_sets.values():
        all_dropped |= s
    n_kept_combined = n_input - len(all_dropped)
    n_dropped_combined = len(all_dropped)

    per_rule: list[RuleStats] = []
    for p in predicates:
        n_dropped = len(dropped_sets[p.NAME])
        n_kept = n_input - n_dropped
        per_rule.append(RuleStats(
            name=p.NAME,
            n_dropped=n_dropped,
            n_kept=n_kept,
            pct_dropped=100.0 * n_dropped / n_input if n_input else 0.0,
            pct_kept=100.0 * n_kept / n_input if n_input else 0.0,
        ))

    # Overlap matrix: overlap[a][b] = PRs dropped by BOTH a and b.
    names = [p.NAME for p in predicates]
    overlap: dict[str, dict[str, int]] = {}
    for a in names:
        overlap[a] = {}
        for b in names:
            overlap[a][b] = len(dropped_sets[a] & dropped_sets[b])

    return FilterReport(
        n_input=n_input,
        n_kept_combined=n_kept_combined,
        n_dropped_combined=n_dropped_combined,
        per_rule=per_rule,
        overlap_matrix=overlap,
    )


_RULE_DESCRIPTIONS: dict[str, str] = {
    "not-bot": "Created by a bot",
    "not-revert": "Revert PRs",
    "not-chore": "Chore (bugfix/CI/test/doc/refactor)",
    "any-perf-signal-or-label": "Not related to performance optimization",
    "any-perf-signal": "Not related to performance optimization",
    "any-strict-perf-claim": "No strict perf claim",
    "title-strict-perf-claim": "No perf claim in title",
    "body-strict-perf-claim": "No perf claim in body",
    "any-loose-perf-claim": "No loose perf claim",
}


def format_report(report: FilterReport) -> str:
    """Format a FilterReport as a human-readable string."""
    lines: list[str] = []
    lines.append(f"Filter Report — {report.n_input} input PRs")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  After all filters: {report.n_kept_combined} PRs kept "
                 f"({100.0 * report.n_kept_combined / report.n_input:.1f}%)")
    lines.append("")
    lines.append("  Per-rule breakdown (independent evaluation):")
    lines.append(f"  {'Category':<45} {'Count':>6} {'%':>7}")
    lines.append(f"  {'-'*45} {'-'*6} {'-'*7}")
    for rs in report.per_rule:
        desc = _RULE_DESCRIPTIONS.get(rs.name, rs.name)
        lines.append(f"  {desc:<45} {rs.n_dropped:>6} {rs.pct_dropped:>6.1f}%")
    lines.append("")

    if len(report.per_rule) > 1:
        lines.append("  Overlap matrix (PRs matching BOTH row and column):")
        names = [rs.name for rs in report.per_rule]
        descs = [_RULE_DESCRIPTIONS.get(n, n) for n in names]
        max_desc = max(len(d) for d in descs)
        header = " " * (max_desc + 4) + "  ".join(f"{d[:8]:>8}" for d in descs)
        lines.append(f"  {header}")
        for i, a in enumerate(names):
            row_vals = "  ".join(f"{report.overlap_matrix[a][b]:>8}" for b in names)
            lines.append(f"  {descs[i]:<{max_desc}}    {row_vals}")
        lines.append("")

    return "\n".join(lines)


def format_report_md(report: FilterReport) -> str:
    """Format a FilterReport as markdown."""
    lines: list[str] = []
    lines.append(f"# Filter Report — {report.n_input} input PRs")
    lines.append("")
    pct_kept = 100.0 * report.n_kept_combined / report.n_input if report.n_input else 0.0
    lines.append(f"**After all filters**: {report.n_kept_combined} PRs kept ({pct_kept:.1f}%)")
    lines.append("")
    lines.append("## Per-rule breakdown (independent evaluation)")
    lines.append("")
    lines.append("| Category | Count | % of total |")
    lines.append("|----------|------:|-----------:|")
    for rs in report.per_rule:
        desc = _RULE_DESCRIPTIONS.get(rs.name, rs.name)
        lines.append(f"| {desc} | {rs.n_dropped} | {rs.pct_dropped:.1f}% |")
    lines.append("")

    if len(report.per_rule) > 1:
        lines.append("## Overlap matrix")
        lines.append("")
        lines.append("PRs matching BOTH row and column category.")
        lines.append("")
        names = [rs.name for rs in report.per_rule]
        descs = [_RULE_DESCRIPTIONS.get(n, n) for n in names]
        header = "| | " + " | ".join(descs) + " |"
        sep = "|---|" + "|".join("---:" for _ in names) + "|"
        lines.append(header)
        lines.append(sep)
        for i, a in enumerate(names):
            row = " | ".join(str(report.overlap_matrix[a][b]) for b in names)
            lines.append(f"| {descs[i]} | {row} |")
        lines.append("")

    return "\n".join(lines)


def report_to_dict(report: FilterReport) -> dict:
    """Serialize a FilterReport to a JSON-friendly dict."""
    return {
        "n_input": report.n_input,
        "n_kept_combined": report.n_kept_combined,
        "n_dropped_combined": report.n_dropped_combined,
        "pct_dropped_combined": round(
            100.0 * report.n_dropped_combined / report.n_input, 2
        ) if report.n_input else 0.0,
        "per_rule": [
            {
                "name": rs.name,
                "n_dropped": rs.n_dropped,
                "n_kept": rs.n_kept,
                "pct_dropped": round(rs.pct_dropped, 2),
                "pct_kept": round(rs.pct_kept, 2),
            }
            for rs in report.per_rule
        ],
        "overlap_matrix": report.overlap_matrix,
    }


__all__ = [
    "FilterReport",
    "RuleStats",
    "format_report",
    "format_report_md",
    "generate_filter_report",
    "report_to_dict",
]
