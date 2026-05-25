"""Stage 01 — signal extraction (Bundle A).

**Phase 1 stub.** Emits a placeholder `Signals` bundle so downstream stages
can run end-to-end without `spotlight_observability` installed. Step 4 of
the implementation order replaces this with a fixture-loader that reads
`<telemetry_from>/01_signals.json` (the user-supplied or fixture-derived
file). Real Bundle A — heuristic extraction from raw telemetry — comes from
the sibling `spotlight-observability` repo.

See `docs/signal-based/signal_discovery_flow.md` "Bundle A" and
`docs/_review-notes/signal_extraction_run_n50_kvprobe.md` for the fixture
that step 4 will target.
"""

from __future__ import annotations

from typing import Any

from spotlights_engine.signal_pipeline.schemas import (
    AnomalyLite,
    Signals,
    TraceSummaryLite,
    WorkloadProfileLite,
)
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


def parse_artifact(raw: Any) -> Signals:
    return Signals.model_validate(raw)


def run(ctx: StageContext) -> Signals:
    return Signals(
        workload=WorkloadProfileLite(
            workload_id="stub-workload",
            description="Phase 1 stub — replace via --telemetry-from in step 4.",
        ),
        traces=[
            TraceSummaryLite(
                trace_id="stub-trace-1",
                summary="Phase 1 stub trace.",
                raw_trace_pointer=None,
            )
        ],
        anomalies=[
            AnomalyLite(
                anomaly_id="stub-anomaly-1",
                type="stub",
                description="Phase 1 stub anomaly.",
            )
        ],
    )


SPEC = StageSpec(
    stage_id="01",
    name="signal_extraction",
    shape="single",
    upstream=(),
    parse_artifact=parse_artifact,
    run=run,
)


__all__ = ["SPEC"]
