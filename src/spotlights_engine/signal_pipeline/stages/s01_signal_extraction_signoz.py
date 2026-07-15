"""Stage 01 — SigNoz/SQL telemetry source.

The `--signoz` path: read a run's OTel data live from SigNoz via read-only
ClickHouse SQL, selected by `run_id`, and produce the same `Signals`
`{workload, traces, anomalies}` as the file path. `s01_signal_extraction`
dispatches here when `ctx.signal_input.signoz` is set; the shared stage
contract (error type, output schema, budgets) lives in `_s01_signal_common`.

`extract(ctx)` is the entry point: resolve the concrete run, then run one
`claude -p` agent over `prompts/signal_extraction_signoz.md`, which authors
its SQL against the `signoz-sql` tool (run-scoping is canonical/tool-owned).
"""

from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine.signal_pipeline.schemas import Signals
from spotlights_engine.signal_pipeline.stages._s01_signal_common import (
    MAX_TURNS,
    OUTPUT_JSON_SCHEMA,
    TIMEOUT_S,
    SignalExtractionError,
)
from spotlights_engine.signal_pipeline.stages._types import StageContext

_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "prompts" / "signal_extraction_signoz.md"
)

# How many of the *other* runs to name in the multi-run auto-select warning
# before truncating (the full list could be long).
_MAX_OTHERS_LISTED = 10

# Project root (…/spotlights). This module lives at
# src/spotlights_engine/signal_pipeline/stages/, so parents[4] is the root.
# The SigNoz agent runs with this as `cwd` so headless `claude -p` discovers
# the project's `.claude/` settings/credentials — it keys auth off the cwd
# and does not walk up the tree. (The file path uses the telemetry dir as
# cwd; the SigNoz path has no telemetry dir, so it would otherwise default to
# the generated log dir, where no `.claude/` exists.)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _warn_multiple_runs(chosen: str, runs: list[str], on_event=None) -> None:
    """Loud, non-blocking banner when `--signoz` auto-selects among several
    runs. Goes to the live progress stream; `max(runs)` is the latest because
    the `YYYYMMDDTHHMMSSZ` id sorts chronologically."""
    if on_event is None:
        return
    others = [r for r in runs if r != chosen]
    shown = others[:_MAX_OTHERS_LISTED]
    lines = [
        "=" * 72,
        f"SigNoz: {len(runs)} runs found — auto-selected the LATEST:",
        f"    {chosen}",
        f"Not analyzed ({len(others)} other run(s)):",
        *(f"    {r}" for r in shown),
    ]
    if len(others) > len(shown):
        lines.append(f"    … and {len(others) - len(shown)} more")
    lines += [
        "To analyze a specific run instead, pass:  --signoz --run-id <id>",
        "=" * 72,
    ]
    on_event("\n".join(lines))


def _resolve_run(run_id: str | None, on_event=None) -> str:
    """Resolve the concrete run to analyze. Explicit `run_id` wins; otherwise
    auto-select the latest run and warn loudly if more than one exists.

    Run selection is canonical/tool-owned — `SignozClient` is lazy-imported
    here, mirroring the `run_claude` import in `_extract_via_signoz`."""
    if run_id:
        return run_id
    from spotlights_engine.signal_pipeline.signoz_tool import SignozClient, SignozError

    # Wrap credential/network failures (missing .env, absent creds, HTTP
    # errors) into SignalExtractionError so this path honours the stage's
    # error-type contract like every other branch.
    try:
        runs = SignozClient.from_env().list_runs()
    except SignozError as e:
        raise SignalExtractionError(f"--signoz: {e}") from e
    if not runs:
        raise SignalExtractionError("--signoz: no runs found in SigNoz")
    chosen = max(runs)
    if len(runs) > 1:
        _warn_multiple_runs(chosen, runs, on_event)
    return chosen


def _extract_via_signoz(
    run_id: str, log_dir: Path, on_event=None, model: str | None = None
) -> Signals:
    """Run one `claude -p` session against SigNoz. Test-monkeypatch seam.

    `Read` + `Bash`, `permission_mode="plan"`, the shared
    `{workload, traces, anomalies}` schema. The agent reaches the run only
    through `signoz-sql` (run-scoping is canonical), so there is no telemetry
    dir — `cwd` is the project root so `claude -p` resolves the project's
    `.claude/` auth (it keys off cwd; see `_PROJECT_ROOT`). The *prompt* is the
    iteration surface, as on the file path."""
    from spotlights_engine.signal_pipeline.claude_subprocess import run_claude

    prompt = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").format(
        run_id=run_id,
    )
    schema_text = json.dumps(OUTPUT_JSON_SCHEMA)

    result = run_claude(
        prompt=prompt,
        log_dir=log_dir,
        json_schema=schema_text,
        cwd=_PROJECT_ROOT,
        max_turns=MAX_TURNS,
        timeout_s=TIMEOUT_S,
        permission_mode="plan",  # read-only; SELECT-only SQL via signoz-sql
        allowed_tools=("Read", "Bash"),
        on_event=on_event,
        model=model,
    )
    if result.error is not None or result.structured_output is None:
        raise SignalExtractionError(
            f"stage 01 (signoz extraction, run {run_id}) failed after "
            f"{result.duration_s:.1f}s: {result.error}"
        )

    return Signals.model_validate(result.structured_output)


def extract(ctx: StageContext) -> Signals:
    """Stage-01 SigNoz entry point: resolve the run, then extract."""
    run_id = _resolve_run(ctx.signal_input.run_id, ctx.on_event)
    return _extract_via_signoz(run_id, ctx.log_dir, ctx.on_event, ctx.model)


__all__ = ["extract"]
