"""Unified runner — extract once, run signal + deep-research, merge into one report.

Public entry: `run_unified(input, config) -> UnifiedResult`. CLI surface lives in
`cli_unified` (the `spotlights-engine both` prefix dispatch in
`spotlights_engine.cli`) and `cli_signal` (the `spotlights-engine signal` prefix).

See `docs/specs/spotlight_report.md` and the design at
`docs/_review-notes/2026-06-15_spotlight_report_schema.md` for the schema this
package targets. The signal stage 05 / `repo_guard` collision (signal mutates
the subject; DR fingerprints it) is sidestepped by locking signal's
`--to-stage 04` for `mode=both`; stage 05 is opt-in via the standalone
`signal-pipeline` CLI for now.
"""

from __future__ import annotations

from spotlights_engine.unified_runner.schemas import (
    UnifiedConfig,
    UnifiedInput,
    UnifiedMode,
    UnifiedResult,
    UnifiedRunSummary,
)
from spotlights_engine.unified_runner.errors import (
    MergeIdCollisionError,
    UnifiedResumeMismatchError,
    UnifiedSetupError,
)
from spotlights_engine.unified_runner.merge import merge_reports
from spotlights_engine.unified_runner.runner import run_unified

__all__ = [
    "MergeIdCollisionError",
    "UnifiedConfig",
    "UnifiedInput",
    "UnifiedMode",
    "UnifiedResult",
    "UnifiedResumeMismatchError",
    "UnifiedRunSummary",
    "UnifiedSetupError",
    "merge_reports",
    "run_unified",
]
