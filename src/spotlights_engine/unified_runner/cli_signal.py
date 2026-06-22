"""CLI for `spotlights-engine signal` — runs only the signal pipeline through the unified plumbing.

Same arg surface as `cli_unified` (so users can swap between `signal` and `both`
trivially), but `mode="signal"` so the unified runner skips DR.

Note: this is *not* a replacement for the existing `signal-pipeline` console
script.  That standalone CLI still exists for direct stage selection
(`--from-stage`, `--to-stage`, `--inject`, including the opt-in stage 05).
This `signal` verb is the unified-shape entry point: it always extracts once
into the unified `_extractor/` cache and emits a unified `SpotlightReport`
with `pipeline="unified"` (the merge over a single contributor).
"""

from __future__ import annotations

from spotlights_engine.unified_runner.cli_unified import main as _unified_main


def main(argv: list[str] | None = None) -> int:
    return _unified_main(argv, mode="signal")


if __name__ == "__main__":
    raise SystemExit(main())
