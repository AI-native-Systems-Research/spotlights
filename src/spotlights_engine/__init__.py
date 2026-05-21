"""Spotlight Engine: spine for schemas, orchestration, and integration."""

from spotlights_engine.spotlights_manager import (
    ManagerSetupError,
    ModuleFilter,
    ModuleTelemetry,
    ResumeMismatchError,
    SpotlightsManagerConfig,
    SpotlightsManagerResult,
    run,
    run_with_telemetry,
)

__version__ = "0.0.0"

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
