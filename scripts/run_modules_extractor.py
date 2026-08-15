"""Run only the modules_extractor (step 1) against a target repo.

Examples:
    uv run --no-sync python scripts/run_modules_extractor.py
    uv run --no-sync python scripts/run_modules_extractor.py \\
        --repo /path/to/repo --artifacts-dir tmp/modules_extractor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spotlights_engine.defaults import DEFAULT_REPO
from spotlights_engine.modules_extractor import (
    ExtractorConfig,
    extract_with_telemetry,
)
from spotlights_engine.schemas.pipeline import ModulesExtractorInput

REPO_PATH = DEFAULT_REPO
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "modules_extractor_only"


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
        help=(
            "Where to persist prompt/schema/raw output. "
            f"Pre-existing run dirs are rejected (default: {ARTIFACTS_DIR})."
        ),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the resulting ProjectTree as JSON.",
    )
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--timeout-s", type=int, default=1800)
    p.add_argument(
        "--claude-bin",
        default="claude",
        help="Claude Code binary to invoke (default: claude).",
    )
    # `default=None` on both so a library-default flip propagates: the values
    # are passed to ExtractorConfig only when explicitly supplied, never
    # silently pinned by an old argparse default.
    p.add_argument(
        "--contract",
        choices=["tree", "assignments"],
        default=None,
        help="Stage-3 contract (default: the library default).",
    )
    p.add_argument(
        "--merge-threshold",
        type=int,
        default=None,
        help=(
            "Assignment-contract size threshold (default: the library "
            "default). Ignored under the tree contract."
        ),
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    inp = ModulesExtractorInput(repo_path=args.repo)
    cfg_kwargs: dict = dict(
        artifacts_dir=args.artifacts_dir,
        claude_bin=args.claude_bin,
        max_turns=args.max_turns,
        timeout_s=args.timeout_s,
    )
    if args.contract is not None:
        cfg_kwargs["contract"] = args.contract
    if args.merge_threshold is not None:
        cfg_kwargs["merge_threshold"] = args.merge_threshold
    cfg = ExtractorConfig(**cfg_kwargs)

    # Persist the effective config at the run level so an offline reader can
    # recover the contract/threshold a run actually used.
    args.artifacts_dir.joinpath("extractor_config.json").write_text(
        json.dumps(
            cfg.model_dump(mode="json", exclude={"artifacts_dir"}),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = extract_with_telemetry(inp, config=cfg)

    tree = result.project_tree
    inv = result.invocation
    walked = list(tree.walk())
    leaves = list(tree.leaves())
    print(f"repository: {tree.repository.name}")
    print(f"modules: {len(walked)} ({len(leaves)} leaves)")
    for qn, _m in walked:
        print(f"  {qn}")
    print(f"extractor: duration_s={inv.duration_s:.1f} cost_usd={inv.cost_usd}")
    print(f"artifacts: {args.artifacts_dir}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(tree.model_dump(mode="json"), indent=2) + "\n"
        )
        print(f"project_tree: {args.output_json}")


if __name__ == "__main__":
    main()
