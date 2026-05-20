"""Canonical shared types for the Spotlight Engine.

This module is the only part of `spotlights_engine` that other repos may import.
Everything else in the package is internal and may change without notice.
"""

from __future__ import annotations

_EXPORTS = {
    "Candidate": "spotlights_engine.schemas.candidate",
    "Candidates": "spotlights_engine.schemas.candidate",
    "Change": "spotlights_engine.schemas.change",
    "File": "spotlights_engine.schemas.modules",
    "Finding": "spotlights_engine.schemas.deep_research",
    "Module": "spotlights_engine.schemas.modules",
    "ModuleDeepResearchInput": "spotlights_engine.schemas.deep_research",
    "ModuleDeepResearchOutput": "spotlights_engine.schemas.deep_research",
    "ProjectModules": "spotlights_engine.schemas.deep_research",
    "ProjectTree": "spotlights_engine.schemas.modules",
    "Repository": "spotlights_engine.schemas.modules",
    "SpotlightContext": "spotlights_engine.schemas.deep_research",
    "StepIssue": "spotlights_engine.schemas.deep_research",
    "Anomaly": "spotlights_engine.schemas.signal",
    "TraceSummary": "spotlights_engine.schemas.signal",
    "WorkloadProfile": "spotlights_engine.schemas.signal",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name)
    return getattr(module, name)


__all__ = sorted(_EXPORTS)
