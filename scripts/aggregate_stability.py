"""Aggregate candidates and proposals across N Spotlights runs to measure
cross-run stability.

Usage:
    python scripts/aggregate_stability.py <run_dir_1> [<run_dir_2> ...]

Each argument is a directory that contains `output/result.json`. The script
groups candidates by (module_qualified_name, symbol) — where `symbol` is the
first span's symbol on the candidate's first location — and reports how many
runs each candidate appeared in, bucketed by frequency.

Proposals are grouped by (candidate_key, proposal_title). Title-based grouping
is a rough MVP; two proposals with different wordings for the same idea will
count as distinct. Fine for a first pass at stability signal.

Output: Markdown to stdout with three sections:
1. Candidate stability table (5/5 → 1/5 buckets)
2. Proposal stability table (per candidate)
3. Summary counts + run manifest
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    """Group key: (module qualified name, first-span symbol)."""
    module = candidate.get("module_qualified_name") or "<unknown>"
    locations = candidate.get("locations") or []
    if locations and locations[0].get("spans"):
        symbol = locations[0]["spans"][0].get("symbol") or "<unknown>"
    else:
        symbol = "<unknown>"
    return (module, symbol)


def _proposal_key(cand_key: tuple[str, str], proposal: dict[str, Any]) -> tuple[str, str, str]:
    """Group key: (module, symbol, first ~90 chars of proposal title)."""
    title = (proposal.get("title") or "").strip()[:90]
    return (cand_key[0], cand_key[1], title)


def _load_run(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "output" / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"missing: {result_path}")
    return json.loads(result_path.read_text())


def _extract_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report.get("candidates") or []


def _bucket(count: int, n_runs: int) -> str:
    return f"{count}/{n_runs}"


def render_report(run_dirs: list[Path]) -> str:
    n_runs = len(run_dirs)
    if n_runs == 0:
        return "# Stability report\n\n(no runs supplied)\n"

    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        payload = _load_run(run_dir)
        runs.append(
            {
                "dir": run_dir.name,
                "path": str(run_dir),
                "report": payload["report"],
            }
        )

    # candidate stability: which candidate keys appeared in how many runs
    cand_runs: dict[tuple[str, str], set[int]] = defaultdict(set)
    # cross-reference each candidate key back to its highest-impact record
    cand_first_hit: dict[tuple[str, str], dict[str, Any]] = {}
    # proposal stability keyed at (cand_key, title)
    prop_runs: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    prop_first_hit: dict[tuple[str, str, str], dict[str, Any]] = {}

    for idx, run in enumerate(runs):
        for cand in _extract_candidates(run["report"]):
            k = _candidate_key(cand)
            cand_runs[k].add(idx)
            cand_first_hit.setdefault(k, cand)
            for prop in cand.get("proposals") or []:
                pk = _proposal_key(k, prop)
                prop_runs[pk].add(idx)
                prop_first_hit.setdefault(pk, prop)

    # bucket candidates by frequency
    cand_by_freq: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for k, run_indices in cand_runs.items():
        cand_by_freq[len(run_indices)].append(k)

    lines: list[str] = []
    lines.append("# Cross-run stability report")
    lines.append("")
    lines.append(f"Runs analyzed: **{n_runs}**")
    lines.append("")
    for i, run in enumerate(runs):
        lines.append(f"- Run {i}: `{run['dir']}` — {len(_extract_candidates(run['report']))} candidate(s)")
    lines.append("")

    lines.append("## Candidate stability")
    lines.append("")
    lines.append("Grouped by `(module, symbol)`. Higher frequency = more robust across runs.")
    lines.append("")
    lines.append("| Freq | Module | Symbol | Impact | Description |")
    lines.append("|---|---|---|---|---|")
    for freq in sorted(cand_by_freq.keys(), reverse=True):
        for k in sorted(cand_by_freq[freq]):
            first = cand_first_hit[k]
            impact = first.get("estimated_impact") or "-"
            desc = (first.get("description") or "").replace("\n", " ")[:100]
            lines.append(
                f"| {_bucket(freq, n_runs)} | `{k[0]}` | `{k[1]}` | {impact} | {desc} |"
            )
    lines.append("")

    # proposal stability grouped per-candidate
    lines.append("## Proposal stability (per candidate)")
    lines.append("")
    lines.append("Proposals grouped by `(candidate, title-prefix)`. Same title → same proposal here (rough).")
    lines.append("")
    # organize proposals under their candidate keys
    per_cand_props: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for pk, run_indices in prop_runs.items():
        cand_key = (pk[0], pk[1])
        per_cand_props[cand_key].append((len(run_indices), pk[2], prop_first_hit[pk]))

    for cand_key in sorted(per_cand_props.keys()):
        cand_freq = len(cand_runs[cand_key])
        first = cand_first_hit[cand_key]
        lines.append(f"### `{cand_key[1]}` (candidate freq {_bucket(cand_freq, n_runs)})")
        lines.append(f"module: `{cand_key[0]}`, impact: {first.get('estimated_impact', '-')}")
        lines.append("")
        lines.append("| Freq | Proposal title | Kind |")
        lines.append("|---|---|---|")
        for freq, title, prop in sorted(per_cand_props[cand_key], key=lambda x: (-x[0], x[1])):
            kind = prop.get("origin") or prop.get("kind") or "-"
            safe_title = title.replace("|", "\\|")
            lines.append(f"| {_bucket(freq, n_runs)} | {safe_title} | {kind} |")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    n_cands = len(cand_runs)
    n_props = len(prop_runs)
    cand_5of5 = sum(1 for k, v in cand_runs.items() if len(v) == n_runs)
    prop_5of5 = sum(1 for k, v in prop_runs.items() if len(v) == n_runs)
    lines.append(f"- Unique candidate keys observed: **{n_cands}**")
    lines.append(f"- Unique proposal keys observed: **{n_props}**")
    lines.append(f"- Candidates appearing in {n_runs}/{n_runs} runs: **{cand_5of5}**")
    lines.append(f"- Proposals appearing in {n_runs}/{n_runs} runs: **{prop_5of5}**")

    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: aggregate_stability.py <run_dir_1> [<run_dir_2> ...]", file=sys.stderr)
        sys.exit(2)
    run_dirs = [Path(a) for a in sys.argv[1:]]
    for d in run_dirs:
        if not d.exists():
            print(f"missing directory: {d}", file=sys.stderr)
            sys.exit(2)
    print(render_report(run_dirs))


if __name__ == "__main__":
    main()
