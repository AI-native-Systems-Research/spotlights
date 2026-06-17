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

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.schemas.legacy.common import SpotlightContext
from spotlights_engine.schemas.legacy.pipeline import SpotlightsManagerInput
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run_with_telemetry,
)

REPO_PATH = Path("/Users/ophir/PycharmProjects/vllm")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "spotlights_manager"
OUTPUT_FOLDER = (
    Path(__file__).resolve().parent.parent / "tmp" / "spotlights_results"
)


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
        "--output-folder",
        type=Path,
        default=OUTPUT_FOLDER,
        help=(
            "Where the renderer writes index.md and modules/*.md "
            f"(default: {OUTPUT_FOLDER})"
        ),
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
        default=None,
        help=(
            "Debug-only: cap step 4 to the first N (candidate, finding) pairs. "
            "Disabled by default; use only for local iteration."
        ),
    )
    p.add_argument(
        "--max-parallel-candidates",
        type=int,
        default=None,
        help=(
            "Within-step parallelism for step 5 (agent_proposals). "
            "Defaults to AgentProposalsConfig's default (5)."
        ),
    )
    p.add_argument(
        "--debug-first-n-candidates",
        type=int,
        default=None,
        help=(
            "Debug-only: cap step 5 to the first N candidates. "
            "Disabled by default; use only for local iteration."
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

    agent_proposals_cfg: AgentProposalsConfig | None = None
    if (
        args.max_parallel_candidates is not None
        or args.debug_first_n_candidates is not None
    ):
        agent_kwargs: dict = {}
        if args.max_parallel_candidates is not None:
            agent_kwargs["max_parallel_candidates"] = args.max_parallel_candidates
        if args.debug_first_n_candidates is not None:
            agent_kwargs["debug_first_n_candidates"] = args.debug_first_n_candidates
        agent_proposals_cfg = AgentProposalsConfig(**agent_kwargs)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=args.artifacts_dir,
        output_folder=args.output_folder,
        max_parallel_sessions=args.max_parallel,
        module_filter=ModuleFilter(include=list(args.include)) if args.include else None,
        proposal_from_finding=proposal_cfg,
        agent_proposals=agent_proposals_cfg,
        resume=args.resume,
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.output_folder.mkdir(parents=True, exist_ok=True)
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
        n_agent_proposals = (
            sum(len(c.agent_proposals) for c in mr.candidates.candidates)
            if mr.candidates
            else 0
        )
        print(
            f"  {qn}: status={mr.status} "
            f"candidates={n_cands} findings={n_findings} "
            f"proposals={n_proposals} "
            f"agent_proposals={n_agent_proposals} "
            f"issues={len(mr.issues)}"
        )
    inv = result.extractor_invocation
    print(f"extractor: duration_s={inv.duration_s:.1f} cost_usd={inv.cost_usd}")
    print(f"artifacts: {args.artifacts_dir}")
    if result.renderer_result is not None:
        print(f"results: {result.renderer_result.index_path}")
    for issue in result.manager_issues:
        print(f"manager-issue [{issue.step}/{issue.severity}]: {issue.message}")


if __name__ == "__main__":
    main()
