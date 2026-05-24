"""Render an existing spotlights_manager run dir as Markdown.

Reads `<artifacts_dir>/spotlights_manager/` (written by a previous manager
run) and emits `index.md` and `modules/*.md` under `--output-folder`. Does
NOT invoke the manager pipeline.

Examples:
    uv run --no-sync python scripts/render_spotlights.py \\
        --artifacts-dir tmp/spotlights_manager \\
        --output-folder tmp/spotlights_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spotlights_engine.results_renderer import (
    RendererInput,
    RendererLoadError,
    RendererSetupError,
    render,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Same value passed to SpotlightsManagerConfig (the renderer reads "
        "<artifacts_dir>/spotlights_manager/).",
    )
    p.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Where index.md and modules/*.md are written.",
    )
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    try:
        result = render(
            RendererInput(
                artifacts_dir=args.artifacts_dir,
                output_folder=args.output_folder,
            )
        )
    except (RendererSetupError, RendererLoadError) as exc:
        print(f"render failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"index: {result.index_path}")
    print(f"module pages: {len(result.module_pages)}")
    if result.skipped_modules:
        print(f"skipped: {', '.join(result.skipped_modules)}")
    for w in result.warnings:
        print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
