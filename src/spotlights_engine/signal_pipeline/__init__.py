"""Signal-based discovery pipeline — MVP runner + stages.

See `docs/signal-based/` for the design (flow doc, mvp module APIs, human
overview). The plan that introduced this package is at
`/Users/idanfr/.claude/plans/humble-plotting-cook.md`.

Entry points:
- `run_pipeline(...)` — programmatic
- `signal-pipeline` — CLI (registered in pyproject)

Public surface kept small. Stage implementations live in
`spotlights_engine.signal_pipeline.stages.*` and are *not* re-exported here
so callers don't accidentally bypass the runner's resume/inject logic.
"""

from __future__ import annotations

from spotlights_engine.signal_pipeline.runner import (
    InjectSpec,
    SignalPipelineInput,
    SignalPipelineResult,
    StageId,
    StageSelection,
    run_pipeline,
)

__all__ = [
    "InjectSpec",
    "SignalPipelineInput",
    "SignalPipelineResult",
    "StageId",
    "StageSelection",
    "run_pipeline",
]
