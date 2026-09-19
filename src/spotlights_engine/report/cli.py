"""CLI for `spotlights-engine report`."""

from __future__ import annotations

import argparse
from pathlib import Path

from spotlights_engine.report.render import load, render


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="spotlights-engine report",
        description="Render a finished Spotlights run directory into one "
        "self-contained experiment page.",
    )
    ap.add_argument("run_dir", type=Path, help="run directory containing result.json")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output path (default: <run-dir>/experiment.html)",
    )
    args = ap.parse_args(argv)

    run = args.run_dir.resolve()
    d = load(run)
    out = args.out or (run / "experiment.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(d), encoding="utf-8")

    high = sum(1 for r in d["rows"] if r["impact"] == "high")
    print(f"run        {d['run_id']}")
    print(f"objective  {d['objective'][:70]}")
    print(f"candidates {len(d['rows'])} ({high} high impact) over {len(d['modules'])} modules")
    R = d.get("research")
    if R:
        print(
            f"research   {len(R['findings'])} findings ({R['works']} distinct "
            f"works) over {len(R['researched'])} modules; "
            f"{R['pairs']} pairs -> {R['grounded']} grounded proposals; "
            f"{len(R['orphans'])} findings unused"
        )
    print(f"written    {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0
