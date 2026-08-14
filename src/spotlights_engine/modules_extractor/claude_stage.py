"""Shared structured-Claude mechanics for the modules extractor.

Stages 1 and 3 of the two-phase extractor need the same plumbing the legacy
`agent.run_extraction` already has: resolve the Claude executable, clean the
environment, hand the CLI a `--json-schema`, stream events, capture raw
artifacts, extract the terminal `result` event, parse `structured_output`, and
collect usage. Rather than make `two_phase.py` reach into a pile of
underscore-prefixed helpers in `agent.py`, the provider-neutral pieces live here
and both the legacy runner and the new orchestrator import them.

`run_structured_claude_stage` performs **one** attempt (no retry loop): the
caller owns the repair budget, per the plan. Hard subprocess failures (missing
executable, timeout, nonzero exit, malformed stream, absent terminal event,
empty message) are translated into the existing extractor error taxonomy and
raised. A Pydantic validation failure is *not* raised — it is returned on the
result object with the raw payload preserved on disk, so the caller can append
the exact errors to one repair prompt.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from spotlights_engine.costing.usage import AgentUsage, claude_usage_from_stream
from spotlights_engine.modules_extractor.errors import (
    ExtractorAgentError,
    ExtractorSetupError,
)
from spotlights_engine.signal_pipeline._subprocess_util import (
    ClaudeResolutionError,
    StreamingTimeout,
    resolve_claude_argv0,
    run_streaming_claude,
)

# Environment keys stripped before launching Claude. Shared with the legacy
# runner so both paths sanitize identically.
_DROP_EXACT = frozenset(
    {
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "VIRTUAL_ENV",
    }
)
_DROP_PREFIX = ("VSCODE_", "OPTQUEST_", "SPOTLIGHTS_")

# Linux MAX_ARG_STRLEN is 131_072 bytes per argument. `claude --json-schema`
# inlines the schema as one argv string; leave headroom for environment growth.
MAX_SCHEMA_BYTES = 120_000

TModel = TypeVar("TModel", bound=BaseModel)


# ── Atomic artifact writes ────────────────────────────────────────────────


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` via a temp file + `os.replace` (atomic on POSIX
    and Windows). The parent directory must already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


# ── Provider-neutral helpers (shared with agent.py) ───────────────────────


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in _DROP_EXACT or key.startswith(_DROP_PREFIX):
            env.pop(key)
    return env


def extract_result_event(stdout: bytes) -> dict | None:
    """Return the terminal `result` event, or None when the last JSON event is
    not a result.

    Non-JSON and non-object lines are skipped rather than raised on: a stray
    runtime warning on stdout, or a line truncated by a kill, must not abort an
    otherwise complete run — a genuinely broken stream still fails through the
    "no terminal result event" path in the caller.
    """
    last_event: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        last_event = obj
    if last_event is None or last_event.get("type") != "result":
        return None
    return last_event


def final_message_text(result_event: dict) -> str:
    structured = result_event.get("structured_output")
    if isinstance(structured, (dict, list)):
        return json.dumps(structured)
    result = result_event.get("result")
    if isinstance(result, str) and result:
        return result
    message = result_event.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "".join(chunks)
    return ""


def duration_seconds(event: dict) -> float | None:
    for key in ("duration_s", "elapsed_s"):
        value = as_float(event.get(key))
        if value is not None:
            return value
    for key in ("duration_ms", "elapsed_ms"):
        value = as_float(event.get(key))
        if value is not None:
            return value / 1000.0
    return None


def as_int(v: object) -> int | None:
    if v is None:
        return None
    try:
        return int(v)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def as_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def add_optional_float(current: float | None, value: float | None) -> float | None:
    if current is None and value is None:
        return None
    return (current or 0.0) + (value or 0.0)


def add_optional_int(current: int | None, value: int | None) -> int | None:
    if current is None and value is None:
        return None
    return (current or 0) + (value or 0)


def notify(on_event: Callable[[str], None] | None, message: str) -> None:
    if on_event is None:
        return
    try:
        on_event(message)
    except Exception:  # noqa: BLE001 - UI callbacks must not abort extraction
        pass


# ── Single-attempt structured stage ───────────────────────────────────────


@dataclass
class StageTelemetry:
    """Per-attempt telemetry from one Claude invocation."""

    session_id: str | None
    duration_s: float
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    usage: AgentUsage | None


@dataclass
class ClaudeStageResult(Generic[TModel]):
    """Outcome of one structured Claude attempt.

    `parsed` is set only when Pydantic validation succeeded; otherwise
    `validation_error` carries the failure and the raw payload is on disk for
    the caller to fold into a repair prompt. Subprocess-level failures never
    reach here — they raise from `run_structured_claude_stage`.
    """

    parsed: TModel | None
    validation_error: ValidationError | None
    result_event: dict
    raw_payload: str
    telemetry: StageTelemetry


def resolve_and_check(
    *, claude_bin: str, repo_path: Path
) -> list[str]:
    """Resolve the Claude argv prefix and validate `repo_path`.

    Raised errors mirror the legacy runner's taxonomy.
    """
    try:
        argv0 = resolve_claude_argv0(claude_bin)
    except ClaudeResolutionError as e:
        raise ExtractorSetupError(str(e), executable=claude_bin) from e
    if not repo_path.exists() or not repo_path.is_dir():
        raise ExtractorSetupError(
            f"repo_path does not exist or is not a directory: {repo_path}",
            repo_path=str(repo_path),
        )
    return argv0


def build_schema_text(output_type: type[BaseModel]) -> str:
    """JSON-schema text for `output_type`, enforcing the argv size guard."""
    schema_text = json.dumps(output_type.model_json_schema(), indent=2)
    schema_bytes = len(schema_text.encode("utf-8"))
    if schema_bytes >= MAX_SCHEMA_BYTES:
        raise ExtractorSetupError(
            f"{output_type.__name__} schema ({schema_bytes} bytes) exceeds "
            f"claude --json-schema argv ceiling ({MAX_SCHEMA_BYTES} bytes)",
            schema_bytes=schema_bytes,
            limit=MAX_SCHEMA_BYTES,
        )
    return schema_text


def run_structured_claude_stage(
    *,
    output_type: type[TModel],
    repo_path: Path,
    prompt: str,
    stage_name: str,
    attempt_dir: Path | None,
    claude_bin: str = "claude",
    max_turns: int,
    timeout_s: int,
    on_event: Callable[[str], None] | None = None,
) -> ClaudeStageResult[TModel]:
    """Run one structured-output Claude attempt over `repo_path`.

    Writes the rendered prompt and schema before launch, then raw stdout/stderr,
    the terminal result event, and the extracted final message before Pydantic
    parsing — so a parse/schema failure still leaves the raw evidence on disk.
    """
    argv0 = resolve_and_check(claude_bin=claude_bin, repo_path=repo_path)
    schema_text = build_schema_text(output_type)

    if attempt_dir is not None:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(attempt_dir / "prompt.md", prompt)
        atomic_write_text(attempt_dir / "schema.json", schema_text)

    argv = argv0 + [
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--json-schema",
        schema_text,
        "--permission-mode",
        "plan",
        "--max-turns",
        str(max_turns),
    ]

    try:
        result = run_streaming_claude(
            argv=argv,
            prompt=prompt,
            env=clean_env(),
            cwd=repo_path,
            timeout_s=timeout_s,
            on_event=on_event,
        )
    except StreamingTimeout as exc:
        if attempt_dir is not None:
            atomic_write_bytes(attempt_dir / "stream.jsonl", exc.stdout or b"")
            atomic_write_bytes(attempt_dir / "stderr.log", exc.stderr or b"")
        raise ExtractorAgentError(
            f"claude ({stage_name}) timed out after {exc.duration_s:.1f}s",
            timeout_s=timeout_s,
            stage=stage_name,
        ) from exc

    if attempt_dir is not None:
        atomic_write_bytes(attempt_dir / "stream.jsonl", result.stdout or b"")
        atomic_write_bytes(attempt_dir / "stderr.log", result.stderr or b"")

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:].decode("utf-8", "replace")
        stdout_tail = result.stdout[-500:].decode("utf-8", "replace")
        raise ExtractorAgentError(
            f"claude ({stage_name}) exit={result.returncode}",
            returncode=result.returncode,
            stderr_tail=stderr_tail,
            stdout_tail=stdout_tail,
            stage=stage_name,
        )

    result_event = extract_result_event(result.stdout)
    if result_event is None:
        raise ExtractorAgentError(
            f"claude ({stage_name}) stream-json had no terminal result event",
            stage=stage_name,
        )
    if attempt_dir is not None:
        atomic_write_json(attempt_dir / "result_event.json", result_event)

    usage = claude_usage_from_stream(result.stdout)
    reported_duration = duration_seconds(result_event)
    telemetry = StageTelemetry(
        session_id=result_event.get("session_id"),
        duration_s=(
            reported_duration if reported_duration is not None else result.duration_s
        ),
        cost_usd=as_float(result_event.get("total_cost_usd")),
        input_tokens=usage.input if usage is not None else None,
        output_tokens=usage.output if usage is not None else None,
        usage=usage,
    )

    payload = final_message_text(result_event)
    if not payload.strip():
        raise ExtractorAgentError(
            f"claude ({stage_name}) returned an empty final message",
            stage=stage_name,
        )
    if attempt_dir is not None:
        atomic_write_text(attempt_dir / "last_message.json", payload)

    try:
        parsed = output_type.model_validate_json(payload)
    except ValidationError as exc:
        if attempt_dir is not None:
            atomic_write_text(
                attempt_dir / "validation_error.txt", str(exc)
            )
        return ClaudeStageResult(
            parsed=None,
            validation_error=exc,
            result_event=result_event,
            raw_payload=payload,
            telemetry=telemetry,
        )

    if attempt_dir is not None:
        atomic_write_text(
            attempt_dir / "parsed_model.json",
            parsed.model_dump_json(indent=2),
        )

    return ClaudeStageResult(
        parsed=parsed,
        validation_error=None,
        result_event=result_event,
        raw_payload=payload,
        telemetry=telemetry,
    )


__all__ = [
    "MAX_SCHEMA_BYTES",
    "ClaudeStageResult",
    "StageTelemetry",
    "add_optional_float",
    "add_optional_int",
    "as_float",
    "as_int",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "build_schema_text",
    "clean_env",
    "duration_seconds",
    "extract_result_event",
    "final_message_text",
    "notify",
    "resolve_and_check",
    "run_structured_claude_stage",
]
