"""CLI for `spotlights-engine prep-evolve`.

Wired into the top-level dispatch in `spotlights_engine.cli.main`. Resolution
and failure ordering follow plan §2: fail before writing anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spotlights_engine.prep_evolve.api import (
    PrepEvolveConfig,
    PrepEvolveInput,
    prep_evolve,
)
from spotlights_engine.prep_evolve.errors import PrepEvolveError


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine prep-evolve",
        description=(
            "Generate a ready-to-run evolve bundle for a selected candidate: "
            "the evolver's native config, a seed/target, the findings digest, "
            "and an evaluator scaffold. Does not run the evolve."
        ),
    )
    p.add_argument("--result", type=Path, required=True, help="Path to result.json.")
    p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Rendered index.md; used only as a --repo fallback.",
    )
    p.add_argument(
        "--module",
        required=True,
        help="Slash-form qualified name, e.g. v1/attention.",
    )
    p.add_argument("--candidate", required=True, help="Candidate id, e.g. cand-0002.")
    p.add_argument(
        "--repo",
        default=None,
        help="Target repo path; wins over --index. One of --repo/--index must resolve.",
    )
    p.add_argument(
        "--evolver",
        required=True,
        help="skydiscover | coral | nous | agentic-strategy-evolution | all.",
    )
    p.add_argument(
        "--out", type=Path, required=True, help="Bundles parent dir, e.g. evolve_bundles/."
    )
    p.add_argument(
        "--scope",
        choices=["candidate", "module-main-files"],
        default="candidate",
        help="candidate (default) | module-main-files (CORAL/Nous only).",
    )
    p.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        default=None,
        help="Override the inferred optimization direction.",
    )
    p.add_argument("--model", default=None, help="Override the default evolver LLM model.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Rewrite only generator-owned files (manifest-guarded).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    inp = PrepEvolveInput(
        result=args.result,
        index=args.index,
        module=args.module,
        candidate=args.candidate,
        repo=args.repo,
        evolver=args.evolver,
        out=args.out,
        scope=args.scope,
        direction=args.direction,
        model=args.model,
        force=args.force,
    )

    try:
        result = prep_evolve(inp, PrepEvolveConfig())
    except PrepEvolveError as exc:
        print(f"prep-evolve: {exc}", file=sys.stderr)
        return 2

    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    for s in result.skipped:
        print(f"skipped {s.evolver}: {s.reason}", file=sys.stderr)

    if not result.bundles:
        print("prep-evolve: no bundles emitted", file=sys.stderr)
        return 1

    for b in result.bundles:
        print(f"{b.evolver}: {b.path} ({len(b.files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
