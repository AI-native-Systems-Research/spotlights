"""Re-exports of the Signal types from `spotlight_observability`.

Consumers should write `from spotlights_engine.schemas import TraceSummary` so they
do not need to know observability's internal layout. The canonical definitions
live in `spotlight_observability.signals`; this module re-exports them, it does
not re-define them.
"""

from __future__ import annotations

from spotlight_observability.signals import Anomaly, TraceSummary, WorkloadProfile

__all__ = ["Anomaly", "TraceSummary", "WorkloadProfile"]
