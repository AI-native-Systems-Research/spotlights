"""Public CLI for `spotlight-engine`.

Thin shim over `spotlights_manager.run_with_telemetry`. Architectural inputs
(`--repo`, `--objective`, `--hint`, `--max-findings-per-module`) bind to
`SpotlightsManagerInput` / `SpotlightContext`; runtime/infra knobs
(`--output-folder`, `--artifacts-dir`, parallelism, debug caps, `--no-resume`)
bind to `SpotlightsManagerConfig`. Library users construct those types
directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import SpotlightsManagerInput
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    SpotlightsManagerResult,
    run_with_telemetry,
)


_DEFAULT_REPO = Path("../vllm")
_DEFAULT_OUTPUT = Path("./spotlight-out")
_DEFAULT_ARTIFACTS = Path("./artifacts")
_DEFAULT_OBJECTIVE = "reduce hot-path latency on common workloads"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlight-engine",
        description=(
            "Propose evidence-backed, high-leverage code changes for a target "
            "repo. Runs structural extraction, candidate discovery, deep "
            "research, and proposal generation; writes Markdown results."
        ),
    )

    p.add_argument(
        "--repo",
        type=Path,
        default=_DEFAULT_REPO,
        help=f"Path to the target repo (default: {_DEFAULT_REPO}).",
    )
    p.add_argument(
        "--include",
        action="append",
        default=None,
        nargs="+",
        metavar="QN",
        help=(
            "Restrict to one or more dot-form leaf qualified names "
            "(e.g. v1.kv_offload). Repeat the flag or pass multiple values "
            "after one flag. Default: all modules."
        ),
    )
    p.add_argument(
        "--objective",
        default=_DEFAULT_OBJECTIVE,
        help=f"Spotlight objective (default: {_DEFAULT_OBJECTIVE!r}).",
    )
    p.add_argument(
        "--hint",
        action="append",
        default=[],
        help=(
            "Workload hint (repeatable). Threaded into "
            "SpotlightContext.workload_hints."
        ),
    )
    p.add_argument(
        "--output-folder",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Where index.md and per-module pages land "
            f"(default: {_DEFAULT_OUTPUT})."
        ),
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_DEFAULT_ARTIFACTS,
        help=(
            "Where checkpoints and raw transcripts land (resume key) "
            f"(default: {_DEFAULT_ARTIFACTS})."
        ),
    )

    p.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Modules processed concurrently (default: 1).",
    )
    p.add_argument(
        "--max-parallel-pairs",
        type=int,
        default=None,
        help=(
            "Within-step parallelism for step 4 (proposal_from_finding_creator). "
            "Default: ProposalFromFindingConfig default (5)."
        ),
    )
    p.add_argument(
        "--max-parallel-candidates",
        type=int,
        default=None,
        help=(
            "Within-step parallelism for step 5 (agent_proposals). "
            "Default: AgentProposalsConfig default (5)."
        ),
    )
    p.add_argument(
        "--max-findings-per-module",
        type=int,
        default=None,
        help=(
            "Cap on findings produced by step 3 per module. "
            "Default: SpotlightsManagerInput default (10)."
        ),
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Refuse to start over an existing run dir.",
    )

    p.add_argument(
        "--debug-first-n-pairs",
        type=int,
        default=None,
        help="Debug-only: cap step 4 to the first N (candidate, finding) pairs.",
    )
    p.add_argument(
        "--debug-first-n-candidates",
        type=int,
        default=None,
        help="Debug-only: cap step 5 to the first N candidates.",
    )

    return p


def _flatten_include(raw: list[list[str]] | None) -> list[str]:
    if not raw:
        return []
    flat: list[str] = []
    for group in raw:
        flat.extend(group)
    return flat


def _build_input(args: argparse.Namespace) -> SpotlightsManagerInput:
    context_kwargs = {
        "objective": args.objective,
        "workload_hints": list(args.hint),
    }
    input_kwargs: dict = {
        "repo_path": args.repo,
        "context": SpotlightContext(**context_kwargs),
    }
    if args.max_findings_per_module is not None:
        input_kwargs["max_findings_per_module"] = args.max_findings_per_module
    return SpotlightsManagerInput(**input_kwargs)


def _build_config(args: argparse.Namespace) -> SpotlightsManagerConfig:
    proposal_cfg: ProposalFromFindingConfig | None = None
    if args.max_parallel_pairs is not None or args.debug_first_n_pairs is not None:
        kwargs: dict = {}
        if args.max_parallel_pairs is not None:
            kwargs["max_parallel_pairs"] = args.max_parallel_pairs
        if args.debug_first_n_pairs is not None:
            kwargs["debug_first_n_pairs"] = args.debug_first_n_pairs
        proposal_cfg = ProposalFromFindingConfig(**kwargs)

    agent_cfg: AgentProposalsConfig | None = None
    if (
        args.max_parallel_candidates is not None
        or args.debug_first_n_candidates is not None
    ):
        kwargs = {}
        if args.max_parallel_candidates is not None:
            kwargs["max_parallel_candidates"] = args.max_parallel_candidates
        if args.debug_first_n_candidates is not None:
            kwargs["debug_first_n_candidates"] = args.debug_first_n_candidates
        agent_cfg = AgentProposalsConfig(**kwargs)

    include = _flatten_include(args.include)
    return SpotlightsManagerConfig(
        artifacts_dir=args.artifacts_dir,
        output_folder=args.output_folder,
        max_parallel_sessions=args.max_parallel,
        module_filter=ModuleFilter(include=include) if include else None,
        proposal_from_finding=proposal_cfg,
        agent_proposals=agent_cfg,
        resume=args.resume,
    )


def _print_summary(result: SpotlightsManagerResult) -> None:
    """Render the §2 stdout shape from a completed run."""
    qns = list(result.module_runs.keys())
    if qns:
        kept_label = f"{len(qns)} module{'s' if len(qns) != 1 else ''} kept"
        kept_label = f"{kept_label} ({', '.join(qns)})"
    else:
        kept_label = "0 modules kept"
    print(f"[1/5] modules_extractor … {kept_label}")

    for qn in qns:
        run = result.module_runs[qn]
        n_cands = len(run.candidates.candidates) if run.candidates else 0
        n_findings = len(run.findings)
        n_proposals = (
            sum(len(c.deep_research_proposals) for c in run.candidates.candidates)
            if run.candidates
            else 0
        )
        n_agent_proposals = (
            sum(len(c.agent_proposals) for c in run.candidates.candidates)
            if run.candidates
            else 0
        )
        print(f"[2/5] candidate_discovery ({qn}) … {n_cands} candidates")
        print(f"[3/5] module_deep_research ({qn}) … {n_findings} findings")
        print(
            f"[4/5] proposal_from_finding_creator ({qn}) … "
            f"{n_proposals} proposals attached"
        )
        print(
            f"[5/5] agent_proposals ({qn}) … "
            f"{n_agent_proposals} agent proposals attached"
        )

    if result.renderer_result is not None:
        print(f"results: {result.renderer_result.index_path}")
    else:
        print("results: (renderer skipped)")
        for issue in result.manager_issues:
            print(f"  manager-issue [{issue.step}/{issue.severity}]: {issue.message}")


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    inp = _build_input(args)
    cfg = _build_config(args)

    result = run_with_telemetry(inp, config=cfg)
    _print_summary(result)

    any_unrecoverable = any(
        any(not iss.recoverable for iss in run.issues)
        for run in result.module_runs.values()
    )
    return 1 if any_unrecoverable else 0


if __name__ == "__main__":
    sys.exit(main())
