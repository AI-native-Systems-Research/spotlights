"""Stage 01 — signal extraction (Bundle A).

Branches, in order of dispatch:

0. **SigNoz (live telemetry)** — `--signoz` (optionally `--run-id <id>`).
   Resolves the run (explicit id, else latest of `list_runs()` with a loud
   warning when several exist), then spawns one `claude -p` agent that reads
   the run from SigNoz via the `signoz-sql` tool (read-only ClickHouse SQL).
   No telemetry directory — `cwd` is the scratch log dir. Iteration surface
   is `prompts/signal_extraction_signoz.md`. Mutually exclusive with
   `--telemetry-from` (enforced at the CLI).

1. **Pre-cooked Signals JSON** — `--telemetry-from <file.json>` or a
   directory containing `01_signals.json` (alternates: `signals.json`,
   `signal.json`). Cheap, deterministic, used when you want full control.

2. **Agent-extracted from raw OTel directory** — `--telemetry-from <dir>`
   where the dir contains `traces.jsonl`. Spawns one `claude -p` agent
   session with `cwd` = the telemetry dir; the agent has `Bash` + `Read`
   tools to inspect whatever files are present and produces a `Signals`
   JSON. **No hardcoded filename or schema parsing** — telemetry shape
   is allowed to evolve (new headers, renamed metrics, additional files)
   without code changes here. Iteration surface is the prompt template
   (`prompts/signal_extraction.md`), not the parser.

3. **Synthetic fallback** — `telemetry_from is None`. Placeholder so the
   runner state-machine tests don't need a fixture. Goes away when
   `spotlight-observability` ships its real Bundle A.

The output schema is the minimal contract: `{workload, traces,
anomalies}` with `workload.workload_id` required. The lite Pydantic
types in `signal_pipeline.schemas` use `extra="allow"`, so any richer
fields the agent surfaces ride through to disk and become available to
Bundle C in stage 03.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spotlights_engine.signal_pipeline.schemas import (
    AnomalyLite,
    Signals,
    TraceSummaryLite,
    WorkloadProfileLite,
)
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


# Filename candidates the pre-cooked-JSON branch will try, in order.
_DIR_CANDIDATES = ("01_signals.json", "signals.json", "signal.json")

# Discriminator for the agent-extraction branch: an OTel-spans-style file
# in the dir is enough to trigger it. Other filenames are fluid — the
# agent inspects what's there.
_OTEL_DISCRIMINATOR = "traces.jsonl"

_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "prompts" / "signal_extraction.md"
)
_PROMPT_SIGNOZ_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "prompts" / "signal_extraction_signoz.md"
)

# How many of the *other* runs to name in the multi-run auto-select warning
# before truncating (the full list could be long).
_MAX_OTHERS_LISTED = 10

# Project root (…/spotlights). This stage lives at
# src/spotlights_engine/signal_pipeline/stages/, so parents[4] is the root.
# The SigNoz agent runs with this as `cwd` so headless `claude -p` discovers
# the project's `.claude/` settings/credentials — it keys auth off the cwd
# and does not walk up the tree. (The file path uses the telemetry dir as
# cwd; the SigNoz path has no telemetry dir, so it would otherwise default to
# the generated log dir, where no `.claude/` exists.)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Minimal output schema. Top-level keys required, plus `workload_id` —
# everything else is the agent's call. `additionalProperties` is
# *unspecified* so the agent can surface richer fields without violating
# the constraint.
_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "required": ["workload", "traces", "anomalies"],
    "properties": {
        "workload": {
            "type": "object",
            "required": ["workload_id"],
            "properties": {
                "workload_id": {"type": "string", "minLength": 1},
            },
        },
        "traces": {"type": "array", "items": {"type": "object"}},
        "anomalies": {"type": "array", "items": {"type": "object"}},
    },
}

# Generous budget — agent may iterate to grep, jq, sample, then summarize.
# Wall-clock is the binding limit in practice: with a large (~76MB)
# metrics.jsonl, repeated full-file Python/jq passes plus model latency
# can blow past 30min before the agent reaches the summarize step. 60min
# gives headroom; turns (60) are rarely the limiter.
_MAX_TURNS = 60
_TIMEOUT_S = 3600


def parse_artifact(raw: Any) -> Signals:
    return Signals.model_validate(raw)


def _has_pre_cooked_signals(directory: Path) -> bool:
    return any((directory / name).is_file() for name in _DIR_CANDIDATES)


def _has_otel_discriminator(directory: Path) -> bool:
    return (directory / _OTEL_DISCRIMINATOR).is_file()


def _resolve_signals_file(target: Path) -> Path:
    """Pre-cooked-JSON branch: map target to the JSON file to read."""
    if target.is_file():
        return target
    for name in _DIR_CANDIDATES:
        candidate = target / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"--telemetry-from {target} is a directory but contains neither a "
        f"pre-cooked Signals JSON ({list(_DIR_CANDIDATES)}) nor an OTel "
        f"telemetry file ({_OTEL_DISCRIMINATOR}); pass a file directly, "
        f"place `01_signals.json` inside, or include `traces.jsonl` so "
        f"the agent extraction branch can run"
    )


def _load_signals(telemetry_from: Path) -> Signals:
    """Pre-cooked-JSON loader. Factored for ease of monkeypatching."""
    src = _resolve_signals_file(telemetry_from)
    raw = json.loads(src.read_text(encoding="utf-8"))
    return Signals.model_validate(raw)


def _extract_signals_via_claude(
    target_dir: Path, log_dir: Path, on_event=None, model: str | None = None
) -> Signals:
    """Agent-extraction branch. Test-monkeypatch seam.

    Spawns one `claude -p` session with `cwd=target_dir`, hands it
    `Bash` + `Read` so it can inspect whatever files are present, and
    schema-constrains output to `{workload, traces, anomalies}`. The
    *prompt* is the iteration surface — telemetry-format changes get
    addressed by tightening the prompt, not by editing this function.

    `claude_subprocess` is lazy-imported per the safety order in the
    approved plan — keeps `_check_layout()` the canonical fail site
    when origin/main isn't present.
    """
    from spotlights_engine.signal_pipeline.claude_subprocess import run_claude

    prompt = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").format(
        target_dir=str(target_dir),
    )
    schema_text = json.dumps(_OUTPUT_JSON_SCHEMA)

    result = run_claude(
        prompt=prompt,
        log_dir=log_dir,
        json_schema=schema_text,
        cwd=target_dir,
        max_turns=_MAX_TURNS,
        timeout_s=_TIMEOUT_S,
        permission_mode="plan",  # read-only; agent never edits the dir
        allowed_tools=("Read", "Bash"),
        on_event=on_event,
        model=model,
    )
    if result.error is not None or result.structured_output is None:
        raise SignalExtractionError(
            f"stage 01 (agent extraction) failed after {result.duration_s:.1f}s: "
            f"{result.error}"
        )

    # The lite schemas use `extra="allow"`, so any bonus fields the agent
    # surfaced (vllm-specific metrics, richer anomaly metadata, etc.)
    # round-trip through `01_signals.json` and reach Bundle C unchanged.
    return Signals.model_validate(result.structured_output)


class SignalExtractionError(RuntimeError):
    pass


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


def _resolve_signoz_run(run_id: str | None, on_event=None) -> str:
    """Resolve the concrete run to analyze. Explicit `run_id` wins; otherwise
    auto-select the latest run and warn loudly if more than one exists.

    Run selection is canonical/tool-owned — `SignozClient` is lazy-imported
    here, mirroring the `run_claude` import in `_extract_signals_via_claude`."""
    if run_id:
        return run_id
    from spotlights_engine.signal_pipeline.signoz_tool import SignozClient

    runs = SignozClient.from_env().list_runs()
    if not runs:
        raise SignalExtractionError("--signoz: no runs found in SigNoz")
    chosen = max(runs)
    if len(runs) > 1:
        _warn_multiple_runs(chosen, runs, on_event)
    return chosen


def _extract_signals_via_signoz(
    run_id: str, log_dir: Path, on_event=None, model: str | None = None
) -> Signals:
    """SigNoz-extraction branch. Test-monkeypatch seam.

    Mirrors `_extract_signals_via_claude`: one `claude -p` session, `Read` +
    `Bash`, `permission_mode="plan"`, the same `{workload, traces, anomalies}`
    schema. The agent reaches the run only through `signoz-sql` (run-scoping is
    canonical), so there is no telemetry dir — `cwd` is the project root so
    `claude -p` resolves the project's `.claude/` auth (it keys off cwd; see
    `_PROJECT_ROOT`). The *prompt* is the iteration surface, as on the file path."""
    from spotlights_engine.signal_pipeline.claude_subprocess import run_claude

    prompt = _PROMPT_SIGNOZ_TEMPLATE_PATH.read_text(encoding="utf-8").format(
        run_id=run_id,
    )
    schema_text = json.dumps(_OUTPUT_JSON_SCHEMA)

    result = run_claude(
        prompt=prompt,
        log_dir=log_dir,
        json_schema=schema_text,
        cwd=_PROJECT_ROOT,
        max_turns=_MAX_TURNS,
        timeout_s=_TIMEOUT_S,
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


def _synthetic_signals() -> Signals:
    """Bridge for runs without `--telemetry-from`. Goes away when
    Bundle A is real or the user always supplies real telemetry."""
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
    if ctx.signal_input.signoz:
        run_id = _resolve_signoz_run(ctx.signal_input.run_id, ctx.on_event)
        return _extract_signals_via_signoz(
            run_id, ctx.log_dir, ctx.on_event, ctx.model
        )
    target = ctx.signal_input.telemetry_from
    if target is None:
        return _synthetic_signals()
    if not target.exists():
        raise FileNotFoundError(
            f"--telemetry-from path does not exist: {target}"
        )
    if target.is_dir():
        # Pre-cooked Signals JSON wins if present (cheaper, deterministic).
        # Else: if the dir looks like an OTel capture, dispatch to the
        # agent. Else: clear error from `_resolve_signals_file`.
        if not _has_pre_cooked_signals(target) and _has_otel_discriminator(target):
            return _extract_signals_via_claude(
                target, ctx.log_dir, ctx.on_event, ctx.model
            )
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
