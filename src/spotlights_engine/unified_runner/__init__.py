"""Unified runner — extract once, run selected pipelines, merge into one report.

Public entry: `run_unified(input, config) -> UnifiedResult`. CLI surface lives
in `cli_telemetry` (the `spotlights-engine telemetry` prefix) and the
top-level `--pipelines` flag dispatch in `spotlights_engine.cli`.

See `docs/specs/spotlight_report.md` for the schema this package targets. The
telemetry-pipeline stage 05 / `repo_guard` collision (telemetry mutates the
subject; DR fingerprints it) is sidestepped by locking telemetry's
`--to-stage 04` whenever the unified runner invokes it; stage 05 is opt-in
via the standalone `signal-pipeline` CLI for now.
"""

from __future__ import annotations

from spotlights_engine.unified_runner.schemas import (
    PipelineName,
    UnifiedConfig,
    UnifiedInput,
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
    "PipelineName",
    "UnifiedConfig",
    "UnifiedInput",
    "UnifiedResult",
    "UnifiedResumeMismatchError",
    "UnifiedRunSummary",
    "UnifiedSetupError",
    "merge_reports",
    "run_unified",
]
