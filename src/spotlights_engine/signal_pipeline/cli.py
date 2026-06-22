"""CLI entry point for the signal pipeline.

Registered in `pyproject.toml` as the `signal-pipeline` script. Argparse
binds to `SignalPipelineInput` + `StageSelection` + `InjectSpec[]`; the
real work lives in `runner.run_pipeline`.

Examples (from the approved plan):

    signal-pipeline --artifacts-dir runs/skel --repo . --no-resume
    signal-pipeline --artifacts-dir runs/x --repo ../vllm --only-stage 02
    signal-pipeline --artifacts-dir runs/x --repo ../vllm --from-stage 03 \\
        --inject 03=path/to/candidates.json
    signal-pipeline --artifacts-dir runs/x --repo ../vllm \\
        --inject 04/cand-0001=path/to/change.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spotlights_engine.defaults import (
    DEFAULT_ARTIFACTS,
    DEFAULT_OUTPUT,
    DEFAULT_REPO,
)
from spotlights_engine.signal_pipeline.layout import ALL_STAGES, StageId
from spotlights_engine.signal_pipeline.runner import (
    InjectSpec,
    InjectValidationError,
    PipelineLayoutError,
    PipelinePreconditionError,
    SignalPipelineInput,
    StageSelection,
    run_pipeline,
)


def _stage_arg(value: str) -> StageId:
    if value not in ALL_STAGES:
        raise argparse.ArgumentTypeError(
            f"invalid stage {value!r}; valid: {ALL_STAGES}"
        )
    return value  # type: ignore[return-value]


def _inject_arg(raw: str) -> InjectSpec:
    try:
        return InjectSpec.parse(raw)
    except (InjectValidationError, ValueError) as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="signal-pipeline",
        description=(
            "Signal-based discovery pipeline (MVP). Produces (Change, "
            "ExecutionResult) pairs from telemetry + a subject repo. See "
            "docs/signal-based/ for the design."
        ),
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help=(
            "Directory holding this run's stage artifacts (created if missing) "
            f"(default: {DEFAULT_ARTIFACTS})."
        ),
    )
    p.add_argument(
        "--output-folder",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Where the human-readable rollup (signal_summary.json + signal_summary.md) "
            f"is written (default: {DEFAULT_OUTPUT})."
        ),
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO,
        dest="subject_root",
        metavar="REPO",
        help=(
            "Path to the subject system's repo (the system being analyzed) "
            f"(default: {DEFAULT_REPO})."
        ),
    )
    p.add_argument(
        "--telemetry-from",
        type=Path,
        default=None,
        help=(
            "Path to a directory or file holding pre-computed signals "
            "(e.g. data/for_idan/runs/run-N50-kvprobe-tierC-hotpath-v2/). "
            "Real Bundle A is deferred — see step 4 of the plan."
        ),
    )
    p.add_argument(
        "--backend-id",
        default="claude_code",
        help="Execution backend id forwarded to stage 05 (default: claude_code).",
    )

    # Stage selection — three flags map to one StageSelection.
    sel_group = p.add_mutually_exclusive_group()
    sel_group.add_argument(
        "--from-stage",
        type=_stage_arg,
        default=None,
        help="Run stages from this id onward (must come with --to-stage or default to last).",
    )
    sel_group.add_argument(
        "--only-stage",
        type=_stage_arg,
        default=None,
        help="Run only this stage. Implies --no-resume by default.",
    )
    p.add_argument(
        "--to-stage",
        type=_stage_arg,
        default=None,
        help="Stop after this stage (paired with --from-stage). Default: 05.",
    )

    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Re-run all stages in the selection from scratch (default: resume).",
    )
    p.add_argument(
        "--no-projecttree-cache",
        dest="projecttree_cache",
        action="store_false",
        default=True,
        help=(
            "Force stage 02 to re-extract instead of reading the cross-run "
            "ProjectTree cache at ~/.cache/spotlights-engine/projecttree/. "
            "The fresh extraction still updates the cache."
        ),
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Soft cap on the number of candidates stage 03 produces. "
            "When set, the prompt asks the model to keep the top-N by signal "
            "strength × significance. Not enforced as a hard schema cap. "
            "Default: no cap (the model decides)."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Model id forwarded as `--model` to every claude-p invocation in "
            "this run, except for stages that pin their own model in code. "
            "Overrides the project-default constant `signal_pipeline.DEFAULT_MODEL`. "
            "Examples: claude-opus-4-7, claude-sonnet-4-6."
        ),
    )
    p.add_argument(
        "--inject",
        type=_inject_arg,
        action="append",
        default=[],
        metavar="NN[/<id>]=path",
        help=(
            "Substitute a stage's output. Single-artifact: NN=file.json. "
            "Fan-out whole-dir: NN=dir/. Fan-out per-id: NN/<id>=file.json. "
            "Repeatable."
        ),
    )

    return p


def _resolve_selection(args: argparse.Namespace) -> tuple[StageSelection, bool]:
    """Map argparse output to (StageSelection, resume).

    `--only-stage 03` is equivalent to `--from-stage 03 --to-stage 03 --no-resume`
    per the plan's partial-run table.
    """
    resume = args.resume
    if args.only_stage is not None:
        return StageSelection.only(args.only_stage), False  # only-stage flips to no-resume
    from_s: StageId = args.from_stage or "01"
    to_s: StageId = args.to_stage or "05"
    return StageSelection(from_stage=from_s, to_stage=to_s), resume


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sel, resume = _resolve_selection(args)

    sp_input = SignalPipelineInput(
        subject_root=args.subject_root,
        telemetry_from=args.telemetry_from,
        backend_id=args.backend_id,
        projecttree_cache=args.projecttree_cache,
        max_candidates=args.max_candidates,
        model=args.model,
    )

    try:
        result = run_pipeline(
            sp_input,
            run_dir=args.artifacts_dir,
            output_folder=args.output_folder,
            stages=sel,
            resume=resume,
            inject=args.inject,
        )
    except PipelinePreconditionError as e:
        print(f"signal-pipeline: {e}", file=sys.stderr)
        return 2
    except PipelineLayoutError as e:
        print(f"signal-pipeline: layout error: {e}", file=sys.stderr)
        return 3
    except InjectValidationError as e:
        print(f"signal-pipeline: --inject error: {e}", file=sys.stderr)
        return 4

    # Compact summary on stdout — useful for CI / scripting.
    summary = {
        "artifacts_dir": str(result.run_dir),
        "output_folder": str(result.output_folder),
        "completed_stages": result.completed_stages,
        "skipped_stages": result.skipped_stages,
        "issues": result.issues,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
