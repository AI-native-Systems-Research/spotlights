"""CLI for unified-runner invocations.

Two ways the top-level `spotlights-engine` dispatcher (`cli.py:main`) routes
work into this entry:

- **`spotlights-engine telemetry [flags...]`** — single-pipeline shortcut.
  Calls `main(argv, force_pipelines=["telemetry"])`; the `--pipelines` flag
  is rejected on this path (the verb pins the pipeline).
- **`spotlights-engine --pipelines a,b [flags...]`** — multi-pipeline (or
  explicit single).  Calls `main(argv, force_pipelines=None)`; the value of
  `--pipelines` drives `UnifiedInput.pipelines`.

The `deep-research` verb and the no-verb path do *not* route here — they go
to the flat-DR CLI (`_build_argparser()` in `cli.py`) for backwards-compat.
Programmatic callers wanting deep-research through the unified plumbing can
use `--pipelines deep-research`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from spotlights_engine.defaults import (
    DEFAULT_ARTIFACTS,
    DEFAULT_OUTPUT,
    DEFAULT_REPO,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.spotlights_manager import ModuleFilter
from spotlights_engine.unified_runner import (
    PipelineName,
    UnifiedConfig,
    UnifiedInput,
    UnifiedResult,
    run_unified,
)

_DEFAULT_OBJECTIVE = "reduce hot-path latency on common workloads"

# Mapping between user-facing CLI verb / flag values (hyphenated) and the
# schema's Literal values (snake_case).
_CLI_TO_SCHEMA: dict[str, PipelineName] = {
    "deep-research": "deep_research",
    "telemetry": "telemetry",
}


def _parse_pipelines(raw: str) -> list[PipelineName]:
    """Parse a comma-separated `--pipelines deep-research,telemetry` value."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--pipelines requires at least one value")
    mapped: list[PipelineName] = []
    seen: set[PipelineName] = set()
    for part in parts:
        if part not in _CLI_TO_SCHEMA:
            valid = ", ".join(sorted(_CLI_TO_SCHEMA))
            raise argparse.ArgumentTypeError(
                f"unknown pipeline {part!r}; valid: {valid}"
            )
        canonical = _CLI_TO_SCHEMA[part]
        if canonical in seen:
            raise argparse.ArgumentTypeError(
                f"pipeline {part!r} listed twice in --pipelines"
            )
        seen.add(canonical)
        mapped.append(canonical)
    return mapped


def _build_parser(*, allow_pipelines_flag: bool) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine",
        description=(
            "Run one or more spotlights pipelines through the unified runner. "
            "Module extraction happens exactly once; selected pipelines run "
            "concurrently and their outputs are merged into a single "
            "SpotlightReport."
        ),
    )

    if allow_pipelines_flag:
        p.add_argument(
            "--pipelines",
            type=_parse_pipelines,
            required=True,
            metavar="A,B",
            help=(
                "Comma-separated list of pipelines to run. Valid values: "
                "deep-research, telemetry. Single value runs that pipeline "
                "alone; multiple values run them concurrently and merge."
            ),
        )

    # Shared inputs ----------------------------------------------------------
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument(
        "--include",
        action="append",
        nargs="+",
        default=None,
        metavar="QN",
        help=(
            "Restrict the deep-research fan-out to one or more slash-form "
            "qualified names (the telemetry pipeline doesn't filter by "
            "module today)."
        ),
    )
    p.add_argument("--objective", default=_DEFAULT_OBJECTIVE)
    p.add_argument("--hint", action="append", default=[])
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help=(
            "Top-level artifacts root. Each unified run lands at "
            "<artifacts-dir>/<run_id>/ with `_extractor/`, `telemetry/`, "
            "and `deep_research/` sub-dirs."
        ),
    )
    p.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT)

    # Deep-research knobs ----------------------------------------------------
    p.add_argument("--max-findings-per-module", type=int, default=None)
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--max-parallel-pairs", type=int, default=None)
    p.add_argument("--max-parallel-candidates", type=int, default=None)
    p.add_argument("--debug-first-n-pairs", type=int, default=None)
    p.add_argument("--debug-first-n-candidates", type=int, default=None)

    # Telemetry knobs --------------------------------------------------------
    p.add_argument("--telemetry-from", type=Path, default=None)
    p.add_argument("--backend-id", default="claude_code")
    p.add_argument("--max-candidates", type=int, default=None, metavar="N")
    p.add_argument("--model", type=str, default=None, metavar="ID")

    # Resume + verbosity -----------------------------------------------------
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help=(
            "Clear the unified run dir before starting. Implemented at the "
            "unified-runner layer (sub-pipelines always run with resume=True)."
        ),
    )
    p.set_defaults(resume=True)

    verb = p.add_mutually_exclusive_group()
    verb.add_argument("--quiet", "-q", action="store_true")
    verb.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--log-file", type=Path, default=None)

    return p


def _flatten_include(raw: list[list[str]] | None) -> list[str]:
    if not raw:
        return []
    flat: list[str] = []
    for group in raw:
        flat.extend(group)
    return flat


_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
_HANDLER_TAG = "_spotlights_cli_handler"


def _configure_logging(args: argparse.Namespace) -> None:
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    root = logging.getLogger("spotlights_engine")
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, _HANDLER_TAG, False):
            root.removeHandler(h)
            h.close()
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(level)
    setattr(stderr_handler, _HANDLER_TAG, True)
    root.addHandler(stderr_handler)
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        setattr(fh, _HANDLER_TAG, True)
        root.addHandler(fh)


def _build_input(
    args: argparse.Namespace, *, pipelines: list[PipelineName]
) -> UnifiedInput:
    return UnifiedInput(
        repo_path=args.repo,
        context=SpotlightContext(
            objective=args.objective, workload_hints=list(args.hint)
        ),
        pipelines=pipelines,
        telemetry_from=args.telemetry_from,
        backend_id=args.backend_id,
        max_candidates=args.max_candidates,
        model=args.model,
        max_findings_per_module=args.max_findings_per_module,
    )


def _build_config(args: argparse.Namespace) -> UnifiedConfig:
    include = _flatten_include(args.include)
    return UnifiedConfig(
        artifacts_dir=args.artifacts_dir,
        output_folder=args.output_folder,
        resume=args.resume,
        module_filter=ModuleFilter(include=include) if include else None,
        max_parallel_sessions=args.max_parallel,
        max_parallel_pairs=args.max_parallel_pairs,
        max_parallel_candidates=args.max_parallel_candidates,
        debug_first_n_pairs=args.debug_first_n_pairs,
        debug_first_n_candidates=args.debug_first_n_candidates,
    )


def main(
    argv: list[str] | None = None,
    *,
    force_pipelines: list[PipelineName] | None = None,
) -> int:
    """Entry point. `force_pipelines` pins the pipeline list (e.g. the
    `telemetry` verb path); when None, `--pipelines` on the CLI drives it.
    """
    allow_flag = force_pipelines is None
    args = _build_parser(allow_pipelines_flag=allow_flag).parse_args(argv)
    pipelines: list[PipelineName] = (
        force_pipelines if force_pipelines is not None else args.pipelines
    )

    args.repo = args.repo.resolve()
    args.artifacts_dir = args.artifacts_dir.resolve()
    args.output_folder = args.output_folder.resolve()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    _configure_logging(args)

    inp = _build_input(args, pipelines=pipelines)
    cfg = _build_config(args)

    result: UnifiedResult = run_unified(inp, config=cfg)
    summary = {
        "run_dir": str(result.run_dir),
        "report": str(result.run_dir / "spotlight_report.json"),
        "pipelines": list(result.report.run.pipelines),
        "candidates": len(result.report.candidates),
        "findings": len(result.report.findings),
        "anomalies": len(result.report.anomalies),
        "issues": len(result.report.issues),
        "cost_usd": result.report.run.cost_usd,
        "started_at": result.report.run.started_at,
        "finished_at": result.report.run.finished_at,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
