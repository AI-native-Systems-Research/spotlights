"""SpotlightsManager — orchestrates steps 1-3 of the deep-research pipeline.

Public surface:

- `run(input, *, config)` — cross-pipeline entrypoint
  (`SpotlightsManagerInput` -> `SpotlightReport`).
- `run_with_telemetry(input, *, config)` — runtime-rich variant returning a
  `SpotlightsManagerResult` with per-module telemetry and the extractor
  invocation.
- `SpotlightsManagerConfig`, `ModuleTelemetry`, `SpotlightsManagerResult` —
  the manager's runtime types.
- `ModuleFilter` — subset selection for testing/smoke runs.
- `ManagerSetupError`, `ResumeMismatchError` — error types.
"""

from __future__ import annotations

from spotlights_engine.spotlights_manager.api import (
    ModuleTelemetry,
    SpotlightsManagerConfig,
    SpotlightsManagerResult,
    run,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager.errors import (
    ManagerSetupError,
    ResumeMismatchError,
)
from spotlights_engine.spotlights_manager.filters import ModuleFilter

__all__ = [
    "ManagerSetupError",
    "ModuleFilter",
    "ModuleTelemetry",
    "ResumeMismatchError",
    "SpotlightsManagerConfig",
    "SpotlightsManagerResult",
    "run",
    "run_with_telemetry",
]
