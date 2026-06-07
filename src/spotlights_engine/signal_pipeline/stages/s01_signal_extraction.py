"""Stage 01 — signal extraction (Bundle A).

Three branches, in order of preference:

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
_MAX_TURNS = 60
_TIMEOUT_S = 1800


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
