"""Canonical shared types for the Spotlight Engine.

This module is the only part of `spotlights_engine` that other repos may import.
Everything else in the package is internal and may change without notice.
"""

from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.change import Change
from spotlights_engine.schemas.modules import File, Module, ProjectTree, Repository
from spotlights_engine.schemas.signal import Anomaly, TraceSummary, WorkloadProfile

__all__ = [
    "Anomaly",
    "Candidate",
    "Candidates",
    "Change",
    "File",
    "Module",
    "ProjectTree",
    "Repository",
    "TraceSummary",
    "WorkloadProfile",
]
