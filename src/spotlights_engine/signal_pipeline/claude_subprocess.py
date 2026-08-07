"""Generic `claude -p` subprocess wrapper for stages 01/03/04/05.

Parameterised over `permission_mode`, `cwd`, `json_schema`, and
`allowed_tools`. This is now a thin adapter over the centralized
`llm_session` package: `ClaudeSession` owns argv building, env scrubbing (the
`ANTHROPIC_AUTH_TOKEN` auth-leak fix now lives in `llm_session.env.clean_env`),
streaming, and stream-json parsing; `run_with_retry` wraps each spawn in the
process-wide concurrency limiter and transport-level backoff-retry.

Write-mode stages (s05 runs `bypassPermissions`) are single-attempt by
construction: `run_with_retry` derives a `max_attempts=1` policy for
`acceptEdits`/`bypassPermissions` so a partially-applied edit is never
re-applied.

Persists the assembled prompt + raw stdout/stderr under `log_dir` for
debugging, regardless of success / failure. The runner's per-stage `log_dir`
is `_logs/<NN>_<name>/<timestamp>/`, so each invocation gets its own directory
and old runs aren't clobbered.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spotlights_engine.llm_session import (
    ClaudeSession,
    ClaudeSessionOptions,
    CLIResolutionError,
    run_with_retry,
)
from spotlights_engine.llm_session.resolve import resolve_cli
from spotlights_engine.llm_session.transport import default_on_event


class ClaudeNotAvailableError(RuntimeError):
    pass


class ClaudeSubprocessError(RuntimeError):
    """Wraps any failure mode that produces no usable structured_output."""


@dataclass
class ClaudeRunResult:
    """Outcome of one `run_claude` invocation.

    On success: `structured_output` is the parsed dict/list from the
    final `result` event in the stream-json stdout.
    On failure: caller can inspect `error` plus the persisted log_dir
    for raw streams.

    `model` comes from the stream's `system/init` event; `cost_usd` and
    `num_turns` come from the terminal `result` event. Either may be
    None on older `claude` versions or on early failure.
    """

    structured_output: dict | list | None
    duration_s: float
    error: str | None = None
    model: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None


def ensure_claude_available(claude_bin: str = "claude") -> None:
    """Verify `claude` resolves to a usable (non-shim) binary on this platform.

    Raises `ClaudeNotAvailableError` if the resolved path is missing or
    would force-fall-back to the buffering `claude.CMD` shim on Windows.
    """
    try:
        resolve_cli("claude", claude_bin)
    except CLIResolutionError as e:
        raise ClaudeNotAvailableError(str(e)) from e


def run_claude(
    *,
    prompt: str,
    log_dir: Path,
    json_schema: str | None = None,
    cwd: Path | None = None,
    max_turns: int = 60,
    timeout_s: int = 1800,
    permission_mode: Literal["plan", "default", "acceptEdits", "bypassPermissions"] = "plan",
    allowed_tools: Iterable[str] | None = None,
    claude_bin: str = "claude",
    on_event: Callable[[str], None] | None = default_on_event,
    model: str | None = None,
) -> ClaudeRunResult:
    """Spawn one `claude -p` session and return its parsed structured_output.

    The prompt is fed on stdin; output is `stream-json`. Streams + the
    prompt itself are persisted to `log_dir` for post-mortem. `on_event` is
    invoked with a one-line summary of each stream-json event as it arrives;
    pass `None` to silence. `model`, when set, is forwarded as
    `--model <id>` to the subprocess; otherwise the CLI's default applies.
    """
    try:
        resolve_cli("claude", claude_bin)
    except CLIResolutionError as e:
        raise ClaudeNotAvailableError(str(e)) from e

    session = ClaudeSession(
        ClaudeSessionOptions(
            model=model,
            permission_mode=permission_mode,
            allowed_tools=tuple(allowed_tools) if allowed_tools is not None else None,
            max_turns=max_turns,
            json_schema=json_schema,
            output_format="stream-json",
            stream_events=True,
            timeout_s=timeout_s,
            claude_bin=claude_bin,
        )
    )

    # The session persists prompt.md / schema.json / raw_stdout.log /
    # raw_stderr.log into log_dir. Write-mode (acceptEdits/bypassPermissions)
    # gets a single attempt automatically inside run_with_retry.
    result = run_with_retry(
        session,
        prompt,
        cwd=cwd,
        on_event=on_event,
        log_dir=log_dir,
        label=f"signal_pipeline.{permission_mode}",
    )

    resolved_model = result.model

    if result.error is not None:
        error = result.error
        if not result.timed_out and result.returncode != 0:
            error = f"{error} (see {log_dir / 'raw_stderr.log'})"
        return ClaudeRunResult(
            structured_output=None,
            duration_s=result.duration_s,
            error=error,
            model=resolved_model,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
        )

    _write_meta(
        log_dir,
        model=resolved_model,
        cost_usd=result.cost_usd,
        duration_s=result.duration_s,
        num_turns=result.num_turns,
    )

    if isinstance(result.structured_output, (dict, list)):
        return ClaudeRunResult(
            structured_output=result.structured_output,
            duration_s=result.duration_s,
            model=resolved_model,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
        )

    return ClaudeRunResult(
        structured_output=None,
        duration_s=result.duration_s,
        error="claude produced no structured_output and no usable fallback",
        model=resolved_model,
        cost_usd=result.cost_usd,
        num_turns=result.num_turns,
    )


def _write_meta(
    log_dir: Path,
    *,
    model: str | None,
    cost_usd: float | None,
    duration_s: float,
    num_turns: int | None,
) -> None:
    """Persist a small per-invocation meta.json next to raw_stdout.log.

    The runner reads these to populate `status.json` with model + cost
    info per stage (summing across fan-out invocations)."""
    meta = {
        "model": model,
        "cost_usd": cost_usd,
        "duration_s": round(duration_s, 3),
        "num_turns": num_turns,
    }
    (log_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "ClaudeNotAvailableError",
    "ClaudeRunResult",
    "ClaudeSubprocessError",
    "ensure_claude_available",
    "run_claude",
]
