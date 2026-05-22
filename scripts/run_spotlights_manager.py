"""Run SpotlightsManager end-to-end against the vllm repo.

Examples:
    uv run --no-sync python scripts/run_spotlights_manager.py \\
        --include v1.kv_offload --max-parallel 1
    uv run --no-sync python scripts/run_spotlights_manager.py \\
        --include v1.kv_offload v1.attention.paged_kv --max-parallel 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import SpotlightsManagerInput
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run_with_telemetry,
)

REPO_PATH = Path("/Users/ophir/PycharmProjects/vllm")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "spotlights_manager"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo",
        type=Path,
        default=REPO_PATH,
        help=f"Path to the target repo (default: {REPO_PATH})",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help=f"Where to write checkpoints (default: {ARTIFACTS_DIR})",
    )
    p.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Restrict to these dot-form leaf qualified names (e.g. v1.kv_offload).",
    )
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument(
        "--max-parallel-pairs",
        type=int,
        default=None,
        help=(
            "Within-step parallelism for step 4 (proposal_from_finding_creator). "
            "Defaults to ProposalFromFindingConfig's default (5)."
        ),
    )
    p.add_argument(
        "--debug-first-n-pairs",
        type=int,
        default=10,
        help=(
            "Debug-only: cap step 4 to the first N (candidate, finding) pairs. "
            "Use only for local iteration."
        ),
    )
    p.add_argument(
        "--objective",
        default="reduce hot-path latency on common workloads",
        help="Spotlight objective handed to candidate_discovery + module_deep_research.",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Refuse to start over an existing run dir.",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    inp = SpotlightsManagerInput(
        repo_path=args.repo,
        context=SpotlightContext(objective=args.objective),
    )
    proposal_cfg: ProposalFromFindingConfig | None = None
    if args.max_parallel_pairs is not None or args.debug_first_n_pairs is not None:
        proposal_kwargs: dict = {}
        if args.max_parallel_pairs is not None:
            proposal_kwargs["max_parallel_pairs"] = args.max_parallel_pairs
        if args.debug_first_n_pairs is not None:
            proposal_kwargs["debug_first_n_pairs"] = args.debug_first_n_pairs
        proposal_cfg = ProposalFromFindingConfig(**proposal_kwargs)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=args.artifacts_dir,
        max_parallel_sessions=args.max_parallel,
        module_filter=ModuleFilter(include=list(args.include)) if args.include else None,
        proposal_from_finding=proposal_cfg,
        resume=args.resume,
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    result = run_with_telemetry(inp, config=cfg)

    print(f"modules: {len(result.module_runs)}")
    for qn, mr in result.module_runs.items():
        n_cands = len(mr.candidates.candidates) if mr.candidates else 0
        n_findings = len(mr.findings)
        n_proposals = (
            sum(len(c.deep_research_proposals) for c in mr.candidates.candidates)
            if mr.candidates
            else 0
        )
        print(
            f"  {qn}: status={mr.status} "
            f"candidates={n_cands} findings={n_findings} "
            f"proposals={n_proposals} "
            f"issues={len(mr.issues)}"
        )
    inv = result.extractor_invocation
    print(f"extractor: duration_s={inv.duration_s:.1f} cost_usd={inv.cost_usd}")
    print(f"artifacts: {args.artifacts_dir}")


if __name__ == "__main__":
    main()
