"""CLI for `spotlights-engine apply`.

Wired into the top-level dispatch in `spotlights_engine.cli.main`. Argument
shape deliberately mirrors `prep-evolve` (`--result`, `--repo`, `--index`,
`--candidate`, `--module`, `--out`, `--top-n`, `--direction`) so the two stages
are interchangeable at the call site.

`--print-prompt` runs resolution, worktree creation, and validation, then
prints the worktree path and the prompt and exits — leaving the worktree in
place for `/spotlights-apply-candidate` to work in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from spotlights_engine.model_config import load_model_config
from spotlights_engine.one_shot_apply.api import (
    NOTES_NAME,
    OneShotApplyConfig,
    OneShotApplyInput,
    one_shot_apply,
    render_prompt_block,
)
from spotlights_engine.one_shot_apply.errors import OneShotApplyError
from spotlights_engine.prep_evolve.errors import PrepEvolveError


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine apply",
        description=(
            "Implement one candidate with a single Claude Code session in a "
            "throwaway git worktree, and write apply.patch + apply.prompt.txt "
            "+ APPLY-NOTES.md. "
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
        help="Candidate id, e.g. cand-....; omit to apply every candidate.",
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
            "For a sorted --result: apply only the top N ranked candidates "
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
        "--claude-model",
        default=None,
        metavar="ID",
        help=(
            "Model id passed to `claude --model`. Overrides models.yaml "
            "(or $SPOTLIGHTS_MODELS_FILE) for this run. Omit to use the file."
        ),
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


# Maps `OneShotApplyInput` field names to the CLI flag that sets them, so a
# pydantic validation error can name what the user actually typed instead of
# the model's internal field name.
_FIELD_TO_FLAG = {
    "wallclock_s": "--wallclock",
    "max_turns": "--max-turns",
    "claude_model": "--claude-model",
    "top_n": "--top-n",
}


def _format_validation_error(exc: ValidationError) -> str:
    """One `apply: --flag: <constraint>, got <value>` line per pydantic error.

    Deliberately drops pydantic's multi-line dump and its
    `https://errors.pydantic.dev/...` URL, and swaps the model field name
    (e.g. `wallclock_s`) for the CLI flag the user typed (`--wallclock`).
    """
    lines = []
    for err in exc.errors():
        field = str(err["loc"][0]) if err["loc"] else "arguments"
        flag = _FIELD_TO_FLAG.get(field, f"--{field.replace('_', '-')}")
        lines.append(f"apply: {flag}: {err['msg']}, got {err.get('input')!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    try:
        top_n = _parse_top_n(args.top_n)
    except ValueError as exc:
        print(
            f"apply: --top-n must be a positive integer or 'all', got "
            f"{args.top_n!r} ({exc})",
            file=sys.stderr,
        )
        return 2

    # Resolved before the input is built, and guarded on its own: a malformed
    # models file raises ValueError, which the ValidationError handler below
    # would not catch. An explicitly empty flag means "inherit", same as the
    # main CLI.
    try:
        if args.claude_model is None:
            claude_model = load_model_config().claude
        else:
            claude_model = args.claude_model.strip() or None
    except (OSError, ValueError) as exc:
        print(f"apply: {exc}", file=sys.stderr)
        return 2

    try:
        inp = OneShotApplyInput(
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
            claude_model=claude_model,
            print_prompt=args.print_prompt,
        )
    except ValidationError as exc:
        print(_format_validation_error(exc), file=sys.stderr)
        return 2

    try:
        result = one_shot_apply(inp, OneShotApplyConfig())
    except (PrepEvolveError, OneShotApplyError) as exc:
        print(f"apply: {exc}", file=sys.stderr)
        return 2

    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)

    for preview in result.prompts:
        # Same renderer the run path writes to `apply.prompt.txt`, so a skill
        # that tees this stdout into that filename produces the same bytes the
        # engine would have written itself.
        print(render_prompt_block(preview), end="")

    for artifact in result.patches:
        # File count first, matching `prep-evolve`'s per-bundle line, so a sweep
        # over both arms scans as one column. The count alone cannot carry the
        # three distinct outcomes, though — a failed diff also has no patch, and
        # printing it like the benign "the agent chose not to edit anything"
        # would hide the one case whose edits were lost and needs a re-run. So
        # the two abnormal states keep a suffix; the normal one reads bare.
        detail = f"{len(artifact.files)} files"
        if artifact.collection_error:
            detail += ", patch collection FAILED"
        elif not artifact.patch_produced:
            detail += ", no patch"
        print(f"{artifact.candidate_id}: {artifact.path} ({detail})")
        if artifact.collection_error:
            print(
                f"warning: {artifact.candidate_id}: could not collect the patch: "
                f"{artifact.collection_error} — the agent session ran but its edits "
                f"were not captured and the worktree is gone; re-run this "
                f"candidate. See {Path(artifact.path) / NOTES_NAME}",
                file=sys.stderr,
            )
        if artifact.out_of_scope_files:
            print(
                f"warning: {artifact.candidate_id}: patch touches files outside the "
                f"declared scope: {', '.join(artifact.out_of_scope_files)} — see "
                f"{Path(artifact.path) / NOTES_NAME}",
                file=sys.stderr,
            )

    if result.patches:
        print(
            "apply: nothing was verified — no tests and no benchmarks were run. "
            "See APPLY-NOTES.md for the recorded oracles.",
            file=sys.stderr,
        )
    if result.patches or result.skipped:
        # The same closing tally `prep-evolve` prints, on stderr for the same
        # reason: stdout stays the machine-readable one-line-per-candidate list.
        # Gated because `--print-prompt` writes nothing and skips nothing, and
        # "0 candidate(s) written" under a prompt dump reads as a failure.
        print(
            f"apply: {len(result.patches)} candidate(s) written, "
            f"{len(result.skipped)} skipped",
            file=sys.stderr,
        )
    for s in result.skipped:
        # A skip raised during *selection* — a ranked id that is no longer in
        # result.json — knows the candidate id but not its module, so
        # `module_qualified_name` is None. Joining only the parts that are set
        # keeps that case reading as `skipped cand-xxxx:` rather than
        # interpolating a literal `None/cand-xxxx`.
        who = "/".join(p for p in (s.module_qualified_name, s.candidate_id) if p)
        label = f"  skipped {who}: " if who else "  skipped: "
        print(f"{label}{s.reason}", file=sys.stderr)

    produced = result.patches or result.prompts or result.skipped
    return 0 if produced else 1


if __name__ == "__main__":
    sys.exit(main())
