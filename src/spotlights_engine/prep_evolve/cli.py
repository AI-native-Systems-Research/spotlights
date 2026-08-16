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
    p.add_argument(
        "--result",
        type=Path,
        required=True,
        help=(
            "A finished run's result.json, or the run directory / index.md / "
            "sorted/ dir / sorted_candidates.{json,md} that self-locates it."
        ),
    )
    p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Rendered index.md; used only as a --repo fallback.",
    )
    p.add_argument(
        "--module",
        default=None,
        help="Slash-form qualified name (optional; inferred from --candidate).",
    )
    p.add_argument(
        "--candidate",
        default=None,
        help="Candidate id, e.g. cand-....; omit to process every candidate.",
    )
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
        "--out",
        type=Path,
        default=None,
        help="Bundles base dir (optional; defaults to the run directory).",
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
    p.add_argument(
        "--top-n",
        dest="top_n",
        default="all",
        help=(
            "For a sorted --result: build only the top N ranked candidates "
            "('all' = every ranked candidate, the default)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    raw_top_n = str(args.top_n).strip().lower()
    if raw_top_n == "all":
        top_n: int | None = None
    else:
        try:
            top_n = int(raw_top_n)
        except ValueError:
            print(
                f"prep-evolve: --top-n must be a positive integer or 'all', "
                f"got {args.top_n!r}",
                file=sys.stderr,
            )
            return 2
        if top_n < 1:
            print(
                f"prep-evolve: --top-n must be >= 1, got {top_n}",
                file=sys.stderr,
            )
            return 2

    inp = PrepEvolveInput(
        result=args.result,
        index=args.index,
        module=args.module,
        candidate=args.candidate,
        repo=args.repo,
        evolver=args.evolver,
        out=args.out,
        top_n=top_n,
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

    for b in result.bundles:
        print(f"{b.evolver}: {b.path} ({len(b.files)} files)")

    print(
        f"prep-evolve: {len(result.bundles)} bundle(s) written, "
        f"{len(result.skipped)} skipped",
        file=sys.stderr,
    )
    for s in result.skipped:
        if s.candidate_id and s.module_qualified_name:
            who = f"{s.module_qualified_name}/{s.candidate_id} "
        elif s.candidate_id:
            who = f"{s.candidate_id} "
        else:
            who = ""
        print(f"  skipped {who}({s.evolver}): {s.reason}", file=sys.stderr)

    return 0 if (result.bundles or result.skipped) else 1


if __name__ == "__main__":
    sys.exit(main())
