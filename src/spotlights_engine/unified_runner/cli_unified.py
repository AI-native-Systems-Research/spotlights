"""CLI for `spotlights-engine both` — runs DR + signal pipelines from one verb.

Dispatched from `spotlights_engine.cli.main`'s prefix branch.

Flag set is the union of the existing DR + signal CLIs, with shared flags
(`--repo`, `--objective`, `--artifacts-dir`, `--output-folder`, `--include`)
applying to both, and per-pipeline-only flags forwarded to the relevant
sub-pipeline.
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
    UnifiedConfig,
    UnifiedInput,
    UnifiedResult,
    run_unified,
)

_DEFAULT_OBJECTIVE = "reduce hot-path latency on common workloads"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine both",
        description=(
            "Run the deep-research and signal-based discovery pipelines together "
            "with a single shared module-extraction step. Emits a unified "
            "SpotlightReport at <run_dir>/spotlight_report.json."
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
            "Restrict the DR fan-out to one or more slash-form qualified names "
            "(signal pipeline doesn't filter by module today)."
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
            "<artifacts-dir>/<run_id>/ with `_extractor/`, `signal/`, and "
            "`deep_research/` sub-dirs."
        ),
    )
    p.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT)

    # DR knobs ---------------------------------------------------------------
    p.add_argument("--max-findings-per-module", type=int, default=None)
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--max-parallel-pairs", type=int, default=None)
    p.add_argument("--max-parallel-candidates", type=int, default=None)
    p.add_argument("--debug-first-n-pairs", type=int, default=None)
    p.add_argument("--debug-first-n-candidates", type=int, default=None)

    # Signal knobs -----------------------------------------------------------
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


def _build_input(args: argparse.Namespace, *, mode: str) -> UnifiedInput:
    return UnifiedInput(
        repo_path=args.repo,
        context=SpotlightContext(
            objective=args.objective, workload_hints=list(args.hint)
        ),
        mode=mode,  # type: ignore[arg-type]
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


def main(argv: list[str] | None = None, *, mode: str = "both") -> int:
    args = _build_parser().parse_args(argv)
    args.repo = args.repo.resolve()
    args.artifacts_dir = args.artifacts_dir.resolve()
    args.output_folder = args.output_folder.resolve()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    _configure_logging(args)

    inp = _build_input(args, mode=mode)
    cfg = _build_config(args)

    result: UnifiedResult = run_unified(inp, config=cfg)
    summary = {
        "run_dir": str(result.run_dir),
        "report": str(result.run_dir / "spotlight_report.json"),
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
