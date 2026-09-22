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
import re
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


_API_STATUS_RE = re.compile(r"\((\d{3})\)")
_RATE_LIMIT_RE = re.compile(r"rate.?limit", re.IGNORECASE)


def api_failure_reason(stdout: bytes) -> str | None:
    """Classify a failed CLI run as an API-level failure, from its stream.

    Returns a short reason (``"rate_limit (429)"``, ``"request_timeout"``,
    ``"overloaded"``, ``"api_error (<status>)"``) when the stream shows the run
    died on the API transport — a rate limit, request timeout, or server
    error — rather than on anything repo- or prompt-specific. Returns None for
    every other failure so callers keep failing fast.

    Only the *last* stream event decides: an early transient ``api_retry`` the
    CLI recovered from must not mark a later unrelated failure as retryable.
    The two shapes seen in practice:

    - the CLI exhausts its internal retries and emits a terminal ``result``
      with ``is_error`` and an ``API Error: ... (429)`` / ``Request timed
      out`` text;
    - the stream is cut off (killed / extractor timeout) while the last event
      is still a ``system``/``api_retry`` or the CLI's terminal ``assistant``
      event carrying a top-level transport ``error``.
    """
    last: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            last = obj
    if last is None:
        return None
    if last.get("type") == "system" and last.get("subtype") == "api_retry":
        status = last.get("error_status")
        name = last.get("error") or "api_error"
        return f"{name} ({status})" if status is not None else str(name)
    if last.get("type") == "assistant":
        error = last.get("error")
        if isinstance(error, str):
            lowered = error.lower()
            if _RATE_LIMIT_RE.search(error) is not None:
                return "rate_limit"
            if "timeout" in lowered:
                return "request_timeout"
            if "overload" in lowered:
                return "overloaded"
            if "server" in lowered:
                return "server_error"
    if last.get("type") == "result" and last.get("is_error"):
        text = last.get("result")
        if not isinstance(text, str):
            return None
        lowered = text.lower()
        if "request timed out" in lowered:
            return "request_timeout"
        if _RATE_LIMIT_RE.search(text) is not None or "429" in text:
            return "rate_limit (429)" if "429" in text else "rate_limit"
        if "overloaded" in lowered:
            return "overloaded"
        # Proxy failing to reach the upstream surfaces as "All connection
        # attempts failed" (sometimes wrapped in a 401). It is a transport
        # blip, not a bad credential — a token that just authenticated other
        # stages does not go stale mid-run — so keep it retryable rather than
        # fast-failing the whole extractor on one flaky shard.
        if "all connection attempts failed" in lowered:
            return "connection_failed"
        match = _API_STATUS_RE.search(text)
        if match is not None:
            status = int(match.group(1))
            if status == 408 or status >= 500:
                return f"api_error ({status})"
    return None


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


def _iter_json_objects(text: str):
    """Yield every top-level ``{...}`` region in `text` that parses as a JSON
    object.

    Brace-matched, string-aware (ignores braces inside JSON strings). Only the
    outermost object at each start position is yielded, so an inner payload
    like ``StructuredOutput({...})`` yields the ``{...}`` — exactly the shape a
    chatty local model emits when it types the tool call as prose instead of
    invoking the tool.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        matched = False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        yield obj
                    i = j + 1
                    matched = True
                    break
            j += 1
        if not matched:
            break


def salvage_structured_payload(
    stdout: bytes, output_type: type[BaseModel]
) -> str | None:
    """Reconstruct a structured payload from a run that never called the tool.

    A chatty local model sometimes types the answer as assistant text —
    ``StructuredOutput({...})`` or bare JSON — instead of invoking the
    StructuredOutput tool. Left alone it would loop against the Stop hook until
    it burns `max_turns`/timeout; `_SalvageWatcher` cuts that short by killing
    the child once the full payload has streamed. Either way the terminal event
    carries no `structured_output`, so this reconstructs the answer from text.

    This scans every assistant text event for embedded JSON objects and, for
    each *required* field of `output_type`, keeps the last value seen. Local
    models split the required fields across separate emissions (e.g.
    ``module_decisions`` in one turn, ``assignments`` in the next); merging the
    latest of each reassembles a complete payload. Returns the merged JSON text
    only when every required field was found, else None.
    """
    schema = output_type.model_json_schema()
    required = set(schema.get("required") or schema.get("properties") or {})
    if not required:
        return None
    latest: dict[str, object] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict) or ev.get("type") != "assistant":
            continue
        message = ev.get("message") or {}
        for chunk in message.get("content") or []:
            if not isinstance(chunk, dict) or chunk.get("type") != "text":
                continue
            for obj in _iter_json_objects(chunk.get("text") or ""):
                for key in required:
                    if key in obj:
                        latest[key] = obj[key]
    if not required.issubset(latest):
        return None
    return json.dumps({key: latest[key] for key in latest})


class _SalvageWatcher:
    """Incremental sibling of `salvage_structured_payload` for live streams.

    Fed one stdout event at a time, it tracks which *required* fields of
    `output_type` have appeared inside assistant **text** (the JSON a chatty
    local model types instead of calling the tool). `__call__` returns True the
    moment every required field has been seen, so `run_streaming_claude` can
    kill the child right after the answer streams instead of letting the CLI
    re-nudge the model to `max_turns`.

    Inert for a well-behaved model: one that emits a real `tool_use` never puts
    the payload in text, so the watcher never fires and the run proceeds
    normally.
    """

    def __init__(self, output_type: type[BaseModel]) -> None:
        schema = output_type.model_json_schema()
        self._required = set(schema.get("required") or schema.get("properties") or {})
        self._seen: set[str] = set()

    def __call__(self, ev: dict) -> bool:
        if not self._required:
            return False
        if ev.get("type") != "assistant":
            return False
        message = ev.get("message") or {}
        for chunk in message.get("content") or []:
            if not isinstance(chunk, dict) or chunk.get("type") != "text":
                continue
            for obj in _iter_json_objects(chunk.get("text") or ""):
                self._seen.update(k for k in self._required if k in obj)
        return self._required.issubset(self._seen)


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


def _stage_telemetry_from_stream(
    stdout: bytes, *, fallback_duration_s: float
) -> StageTelemetry:
    """Build telemetry for both successful and terminal-error CLI results."""
    result_event = extract_result_event(stdout) or {}
    usage = claude_usage_from_stream(stdout)
    reported_duration = duration_seconds(result_event)
    return StageTelemetry(
        session_id=result_event.get("session_id"),
        duration_s=(
            reported_duration
            if reported_duration is not None
            else fallback_duration_s
        ),
        cost_usd=as_float(result_event.get("total_cost_usd")),
        input_tokens=usage.input if usage is not None else None,
        output_tokens=usage.output if usage is not None else None,
        usage=usage,
    )


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
    claude_model: str | None = None,
    max_turns: int,
    timeout_s: int,
    on_event: Callable[[str], None] | None = None,
    salvage: bool = False,
) -> ClaudeStageResult[TModel]:
    """Run one structured-output Claude attempt over `repo_path`.

    Writes the rendered prompt and schema before launch, then raw stdout/stderr,
    the terminal result event, and the extracted final message before Pydantic
    parsing — so a parse/schema failure still leaves the raw evidence on disk.

    When `salvage` is set, a subprocess-level failure (timeout, nonzero exit,
    absent terminal event, terminal `is_error`) does not immediately raise: the
    raw stream is scanned for a structured payload the model typed as text but
    never emitted through the tool (see `salvage_structured_payload`). A
    reconstructed payload is validated and returned like a normal attempt; only
    when nothing salvageable is found does the original error propagate.
    """
    argv0 = resolve_and_check(claude_bin=claude_bin, repo_path=repo_path)
    schema_text = build_schema_text(output_type)

    def _salvage_result(
        stdout: bytes, *, fallback_duration_s: float, why: str
    ) -> ClaudeStageResult[TModel] | None:
        if not salvage:
            return None
        payload = salvage_structured_payload(stdout, output_type)
        if not payload or not payload.strip():
            return None
        telem = _stage_telemetry_from_stream(
            stdout, fallback_duration_s=fallback_duration_s
        )
        result_event = extract_result_event(stdout) or {}
        if attempt_dir is not None:
            atomic_write_text(attempt_dir / "salvaged_payload.json", payload)
        notify(
            on_event,
            f"extractor: {stage_name} salvaged structured payload from "
            f"text stream ({why})",
        )
        try:
            parsed = output_type.model_validate_json(payload)
        except ValidationError as exc:
            if attempt_dir is not None:
                atomic_write_text(
                    attempt_dir / "salvage_validation_error.txt", str(exc)
                )
            return ClaudeStageResult(
                parsed=None,
                validation_error=exc,
                result_event=result_event,
                raw_payload=payload,
                telemetry=telem,
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
            telemetry=telem,
        )

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
        # Plan mode exposes ExitPlanMode; chatty local models (e.g. Qwen/Qwen3.8-27B)
        # call it to "finish" instead of the required StructuredOutput tool, then
        # loop against the Stop hook until max_turns. Deny it so the only terminal
        # tool is StructuredOutput. Read-only guarantee of plan mode is unaffected.
        "--disallowed-tools",
        "ExitPlanMode",
        "--max-turns",
        str(max_turns),
    ]
    if claude_model:
        argv += ["--model", claude_model]

    try:
        result = run_streaming_claude(
            argv=argv,
            prompt=prompt,
            env=clean_env(),
            cwd=repo_path,
            timeout_s=timeout_s,
            on_event=on_event,
            # When salvaging, kill the child the moment the full payload has
            # streamed as text — otherwise the CLI keeps re-nudging a
            # non-tool-calling local model to max_turns for no gain.
            early_stop=_SalvageWatcher(output_type) if salvage else None,
        )
    except StreamingTimeout as exc:
        if attempt_dir is not None:
            atomic_write_bytes(attempt_dir / "stream.jsonl", exc.stdout or b"")
            atomic_write_bytes(attempt_dir / "stderr.log", exc.stderr or b"")
        salvaged = _salvage_result(
            exc.stdout or b"",
            fallback_duration_s=exc.duration_s,
            why=f"timeout after {exc.duration_s:.1f}s",
        )
        if salvaged is not None:
            return salvaged
        context: dict[str, object] = {
            "timeout_s": timeout_s,
            "stage": stage_name,
            "telemetry": _stage_telemetry_from_stream(
                exc.stdout or b"", fallback_duration_s=exc.duration_s
            ),
        }
        api_reason = api_failure_reason(exc.stdout or b"")
        suffix = ""
        if api_reason is not None:
            context["api_failure"] = api_reason
            suffix = f" [api_failure: {api_reason}]"
        raise ExtractorAgentError(
            f"claude ({stage_name}) timed out after {exc.duration_s:.1f}s{suffix}",
            **context,
        ) from exc

    if attempt_dir is not None:
        atomic_write_bytes(attempt_dir / "stream.jsonl", result.stdout or b"")
        atomic_write_bytes(attempt_dir / "stderr.log", result.stderr or b"")

    stage_telemetry = _stage_telemetry_from_stream(
        result.stdout, fallback_duration_s=result.duration_s
    )
    if getattr(result, "early_stopped", False):
        # The watcher saw the full text-embedded payload and killed the child.
        # Salvage it from the captured stream; only if that unexpectedly finds
        # nothing do we fall through to the normal (now nonzero) exit handling.
        salvaged = _salvage_result(
            result.stdout,
            fallback_duration_s=result.duration_s,
            why="early-stop: required fields emitted as text",
        )
        if salvaged is not None:
            return salvaged
    if result.returncode != 0:
        salvaged = _salvage_result(
            result.stdout,
            fallback_duration_s=result.duration_s,
            why=f"exit={result.returncode}",
        )
        if salvaged is not None:
            return salvaged
        stderr_tail = result.stderr[-500:].decode("utf-8", "replace")
        stdout_tail = result.stdout[-500:].decode("utf-8", "replace")
        context = {
            "returncode": result.returncode,
            "stderr_tail": stderr_tail,
            "stdout_tail": stdout_tail,
            "stage": stage_name,
            "telemetry": stage_telemetry,
        }
        api_reason = api_failure_reason(result.stdout)
        suffix = ""
        if api_reason is not None:
            context["api_failure"] = api_reason
            suffix = f" [api_failure: {api_reason}]"
        raise ExtractorAgentError(
            f"claude ({stage_name}) exit={result.returncode}{suffix}",
            **context,
        )

    result_event = extract_result_event(result.stdout)
    if result_event is None:
        salvaged = _salvage_result(
            result.stdout,
            fallback_duration_s=result.duration_s,
            why="no terminal result event",
        )
        if salvaged is not None:
            return salvaged
        raise ExtractorAgentError(
            f"claude ({stage_name}) stream-json had no terminal result event",
            stage=stage_name,
            telemetry=stage_telemetry,
        )
    if result_event.get("is_error"):
        api_reason = api_failure_reason(result.stdout)
        # Only salvage a non-API terminal error: an API-classified failure
        # (rate limit, timeout, overload) must stay retryable upstream rather
        # than be resolved from a partial stream. The loop-to-max_turns case
        # this salvage targets is not API-classified.
        if api_reason is None:
            salvaged = _salvage_result(
                result.stdout,
                fallback_duration_s=result.duration_s,
                why="terminal is_error",
            )
            if salvaged is not None:
                return salvaged
        terminal_context: dict[str, object] = {
            "stage": stage_name,
            "telemetry": stage_telemetry,
        }
        suffix = ""
        if api_reason is not None:
            terminal_context["api_failure"] = api_reason
            suffix = f" [api_failure: {api_reason}]"
        raise ExtractorAgentError(
            f"claude ({stage_name}) returned a terminal error{suffix}",
            **terminal_context,
        )
    if attempt_dir is not None:
        atomic_write_json(attempt_dir / "result_event.json", result_event)

    payload = final_message_text(result_event)
    if not payload.strip():
        # An empty final message is a truncated stream, not a prompt/repo
        # fault: the turn ended (often after burning output tokens on thinking)
        # before any answer text arrived — the signature of a backend/tunnel
        # drop mid-generation on a long call. Tag it `api_failure` so the retry
        # loop restarts with a fresh session and backoff instead of hard-failing
        # a multi-hour run on one dropped call. Exhaustion still re-raises.
        raise ExtractorAgentError(
            f"claude ({stage_name}) returned an empty final message",
            stage=stage_name,
            telemetry=stage_telemetry,
            api_failure="empty_final_message",
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
            telemetry=stage_telemetry,
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
        telemetry=stage_telemetry,
    )


__all__ = [
    "MAX_SCHEMA_BYTES",
    "ClaudeStageResult",
    "StageTelemetry",
    "add_optional_float",
    "add_optional_int",
    "api_failure_reason",
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
    "salvage_structured_payload",
]
