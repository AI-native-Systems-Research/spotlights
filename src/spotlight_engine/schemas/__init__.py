"""Canonical shared types for the Spotlight Engine.

This module is the only part of `spotlight_engine` that other repos may import.
Everything else in the package is internal and may change without notice.
"""

from spotlight_engine.schemas.candidate import Candidate
from spotlight_engine.schemas.change import Change
from spotlight_engine.schemas.signal import Anomaly, TraceSummary, WorkloadProfile

__all__ = [
    "Anomaly",
    "Candidate",
    "Change",
    "TraceSummary",
    "WorkloadProfile",
]
