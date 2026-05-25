"""Stage 01 — signal extraction (Bundle A).

Phase 1 was a synthetic stub. This step adds a **fixture loader**: when
the user passes `--telemetry-from <path>`, stage 01 reads a hand-prepared
`Signals` JSON from there instead of fabricating placeholder data.

When `telemetry_from` is None, we still return synthetic data so the
runner state-machine tests can run without a fixture. This is a
deferred-Bundle-A bridge — once `spotlight-observability` lands its real
heuristic extractor, the synthetic branch goes away and this stage
becomes "real Bundle A or hand-craft a JSON, no third option".

`telemetry_from` resolution:

- `<file>.json` — used directly (any filename ending `.json`)
- `<dir>/`     — looks for `01_signals.json` inside; alternative names
  `signals.json` / `signal.json` recognized as a convenience.

Schema is `signal_pipeline.schemas.Signals` (placeholder for the locked
contract in `spotlight_observability.signals` until the sibling lands).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spotlights_engine.signal_pipeline.schemas import (
    AnomalyLite,
    Signals,
    TraceSummaryLite,
    WorkloadProfileLite,
)
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


# Filename candidates the directory-form loader will try, in order.
_DIR_CANDIDATES = ("01_signals.json", "signals.json", "signal.json")


def parse_artifact(raw: Any) -> Signals:
    return Signals.model_validate(raw)


def _resolve_signals_file(target: Path) -> Path:
    """Map `--telemetry-from <target>` to the JSON file to read.

    Raises `FileNotFoundError` with a clear message if the target is a
    directory that doesn't contain any of the recognized filenames, or
    if the path doesn't exist at all.
    """
    if not target.exists():
        raise FileNotFoundError(
            f"--telemetry-from path does not exist: {target}"
        )
    if target.is_file():
        return target
    # Directory case
    for name in _DIR_CANDIDATES:
        candidate = target / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"--telemetry-from {target} is a directory but contains none of "
        f"{list(_DIR_CANDIDATES)}; pass the file directly or place a "
        f"`01_signals.json` inside it"
    )


def _load_signals(telemetry_from: Path) -> Signals:
    """Real loader path. Factored for ease of monkeypatching in tests."""
    import json

    src = _resolve_signals_file(telemetry_from)
    raw = json.loads(src.read_text(encoding="utf-8"))
    return Signals.model_validate(raw)


def _synthetic_signals() -> Signals:
    """Phase-1-bridge synthetic data for runs without `--telemetry-from`.

    Goes away once Bundle A is real."""
    return Signals(
        workload=WorkloadProfileLite(
            workload_id="stub-workload",
            description=(
                "Synthetic Signals — no --telemetry-from provided. Real "
                "Bundle A from spotlight-observability is still deferred."
            ),
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


def run(ctx: StageContext) -> Signals:
    target = ctx.signal_input.telemetry_from
    if target is None:
        return _synthetic_signals()
    return _load_signals(target)


SPEC = StageSpec(
    stage_id="01",
    name="signal_extraction",
    shape="single",
    upstream=(),
    parse_artifact=parse_artifact,
    run=run,
)


__all__ = ["SPEC"]
