"""CLI for `spotlights-engine fix`.

Wired into the top-level dispatch in `spotlights_engine.cli.main`. Argument
shape deliberately mirrors `prep-evolve` (`--result`, `--repo`, `--index`,
`--candidate`, `--module`, `--out`, `--top-n`, `--direction`) so the two stages
are interchangeable at the call site.

`--print-prompt` runs resolution, worktree creation, and validation, then
prints the worktree path and the prompt and exits — leaving the worktree in
place for `/spotlights-fix-candidate` to work in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from spotlights_engine.one_shot_fix.api import (
    NOTES_NAME,
    OneShotFixConfig,
    OneShotFixInput,
    one_shot_fix,
)
from spotlights_engine.one_shot_fix.errors import OneShotFixError
from spotlights_engine.prep_evolve.errors import PrepEvolveError


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine fix",
        description=(
            "Implement one candidate with a single Claude Code session in a "
            "throwaway git worktree, and write fix.patch + FIX-NOTES.md. "
            "Runs no tests and no benchmarks; the target repo is never modified."
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
        "--repo",
        default=None,
        help=(
            "Target repo path; wins over --index. Must be a git checkout. "
            "One of --repo/--index must resolve."
        ),
    )
    p.add_argument(
        "--module",
        default=None,
        help="Slash-form qualified name (optional; inferred from --candidate).",
    )
    p.add_argument(
        "--candidate",
        default=None,
        help="Candidate id, e.g. cand-....; omit to fix every candidate.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Artifacts base dir (optional; defaults to the run directory).",
    )
    p.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        default=None,
        help="Override the inferred optimization direction.",
    )
    p.add_argument(
        "--top-n",
        dest="top_n",
        default="all",
        help=(
            "For a sorted --result: fix only the top N ranked candidates "
            "('all' = every ranked candidate, the default)."
        ),
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=40,
        help="Cap on agent turns per candidate (default: 40).",
    )
    p.add_argument(
        "--wallclock",
        dest="wallclock_s",
        type=int,
        default=1800,
        help="Wall-clock cap in seconds per candidate (default: 1800).",
    )
    p.add_argument(
        "--print-prompt",
        action="store_true",
        help=(
            "Resolve, create and validate the worktree, then print the "
            "worktree path and the prompt and exit. Runs no agent, writes no "
            "artifacts, and leaves the worktree in place for the caller."
        ),
    )
    return p


def _parse_top_n(raw: str) -> int | None:
    """`'all'` -> None; a positive int -> that int. Raises ValueError otherwise."""
    text = str(raw).strip().lower()
    if text == "all":
        return None
    n = int(text)  # ValueError propagates
    if n < 1:
        raise ValueError(f"--top-n must be >= 1, got {n}")
    return n


# Maps `OneShotFixInput` field names to the CLI flag that sets them, so a
# pydantic validation error can name what the user actually typed instead of
# the model's internal field name.
_FIELD_TO_FLAG = {
    "wallclock_s": "--wallclock",
    "max_turns": "--max-turns",
    "top_n": "--top-n",
}


def _format_validation_error(exc: ValidationError) -> str:
    """One `fix: --flag: <constraint>, got <value>` line per pydantic error.

    Deliberately drops pydantic's multi-line dump and its
    `https://errors.pydantic.dev/...` URL, and swaps the model field name
    (e.g. `wallclock_s`) for the CLI flag the user typed (`--wallclock`).
    """
    lines = []
    for err in exc.errors():
        field = str(err["loc"][0]) if err["loc"] else "arguments"
        flag = _FIELD_TO_FLAG.get(field, f"--{field.replace('_', '-')}")
        lines.append(f"fix: {flag}: {err['msg']}, got {err.get('input')!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    try:
        top_n = _parse_top_n(args.top_n)
    except ValueError as exc:
        print(
            f"fix: --top-n must be a positive integer or 'all', got "
            f"{args.top_n!r} ({exc})",
            file=sys.stderr,
        )
        return 2

    try:
        inp = OneShotFixInput(
            result=args.result,
            index=args.index,
            repo=args.repo,
            module=args.module,
            candidate=args.candidate,
            out=args.out,
            direction=args.direction,
            top_n=top_n,
            max_turns=args.max_turns,
            wallclock_s=args.wallclock_s,
            print_prompt=args.print_prompt,
        )
    except ValidationError as exc:
        print(_format_validation_error(exc), file=sys.stderr)
        return 2

    try:
        result = one_shot_fix(inp, OneShotFixConfig())
    except (PrepEvolveError, OneShotFixError) as exc:
        print(f"fix: {exc}", file=sys.stderr)
        return 2

    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)

    for preview in result.prompts:
        print(f"CANDIDATE: {preview.candidate_id}")
        print(f"MODULE:    {preview.module_qualified_name}")
        print(f"BASE:      {preview.base_sha}")
        print(f"WORKTREE:  {preview.worktree}")
        print(f"WORKTREE_PARENT:  {preview.worktree_parent}")
        print("PROMPT:")
        print(preview.prompt)

    for fix in result.fixes:
        state = "patch + notes" if fix.patch_produced else "notes only (no patch)"
        print(f"{fix.candidate_id}: {fix.path} ({state})")
        if fix.out_of_scope_files:
            print(
                f"warning: {fix.candidate_id}: patch touches files outside the "
                f"declared scope: {', '.join(fix.out_of_scope_files)} — see "
                f"{fix.path}/{NOTES_NAME}",
                file=sys.stderr,
            )

    if result.fixes:
        print(
            "fix: nothing was verified — no tests and no benchmarks were run. "
            "See FIX-NOTES.md for the recorded oracles.",
            file=sys.stderr,
        )
    for s in result.skipped:
        who = f"{s.module_qualified_name}/{s.candidate_id} " if s.candidate_id else ""
        print(f"  skipped {who}: {s.reason}", file=sys.stderr)

    produced = result.fixes or result.prompts or result.skipped
    return 0 if produced else 1


if __name__ == "__main__":
    sys.exit(main())
