"""Agent subprocess runners for candidate discovery.

The runners are thin adapters over the centralized `llm_session` package:
`ClaudeSession`/`CodexSession` own argv building, env scrubbing, spawn, and
stream/last-message parsing, and `run_with_retry` wraps each spawn in the
process-wide concurrency limiter and the transport-level backoff-retry. What
stays here is discovery-specific: the `AgentInvocation` telemetry shape, the
`raw_stdout.log`/`raw_stderr.log` retry-separator artifact convention, and the
`_SchemaParseError` schema-retry signal.

`_SchemaParseError` is raised by `invoke` (timeout / nonzero exit / no terminal
result event) and by `parse_last_message` (bad `last_message.json`); the
orchestrator is the only consumer and translates each occurrence into either a
retry or a `DiscoveryValidationError`.
"""

from __future__ import annotations

import json
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from spotlights_engine.candidate_discovery.errors import DiscoverySetupError
from spotlights_engine.costing.usage import codex_usage_from_stream
from spotlights_engine.llm_session import (
    ClaudeSession,
    ClaudeSessionOptions,
    CodexSession,
    CodexSessionOptions,
    SessionResult,
    run_with_retry,
)
from spotlights_engine.llm_session.transport import (
    ResultEventError,
    extract_result_event,
)

if TYPE_CHECKING:  # pragma: no cover
    from spotlights_engine.candidate_discovery.api import DiscoveryConfig


class _SchemaParseError(Exception):
    """Package-internal signal that the iteration should be retried (§6.2).

    Raised by `AgentRunner.invoke` (timeout / nonzero exit / no terminal result
    event) and by `AgentRunner.parse_last_message` (missing / empty / non-JSON
    output).
    """


_RETRY_SEPARATOR = b"\n--- retry separator ---\n"


@dataclass
class AgentInvocation:
    session_id: str | None
    duration_s: float
    # CLI-reported cost — audit only, never used for billing (see costing).
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None = None
    cache_create_tokens: int | None = None
    # Resolved model id from the CLI output (falls back to the configured
    # model for codex); None means unresolved.
    model: str | None = None
    # Model-reported API duration when available (Claude `duration_api_ms`).
    api_time_s: float | None = None


class AgentRunner(ABC):
    name: str
    _executable: str

    def __init__(self, config: DiscoveryConfig) -> None:
        self._config = config
        if shutil.which(self._executable) is None:
            raise DiscoverySetupError(
                f"required CLI not on PATH: {self._executable}",
                executable=self._executable,
            )

    @abstractmethod
    def _build_argv(self, schema_path: Path, iter_dir: Path) -> list[str]: ...

    @abstractmethod
    def parse_last_message(self, iter_dir: Path) -> str: ...

    @abstractmethod
    def _session(self, schema_path: Path, iter_dir: Path):
        """Build the centralized session for this iteration."""

    @abstractmethod
    def _parse_invocation_metadata(
        self, result: SessionResult, iter_dir: Path
    ) -> AgentInvocation: ...

    def _label(self) -> str:
        return f"candidate_discovery.{self.name}"

    def invoke(
        self,
        prompt: str,
        iter_dir: Path,
        schema_path: Path,
    ) -> AgentInvocation:
        session = self._session(schema_path=schema_path, iter_dir=iter_dir)

        stdout_path = iter_dir / "raw_stdout.log"
        stderr_path = iter_dir / "raw_stderr.log"
        is_retry = stdout_path.exists()

        # `run_with_retry` acquires the process-wide limiter slot and applies
        # transport-level backoff-retry underneath the orchestrator's (0, 1)
        # schema-retry loop. We do NOT hand the session a `log_dir`: discovery
        # owns its raw-stream artifact convention (append-with-separator across
        # schema retries), so we persist the returned streams ourselves.
        result = run_with_retry(
            session,
            prompt,
            cwd=self._config.repo_path,
            on_event=None,
            label=self._label(),
        )

        _append_streams(stdout_path, stderr_path, result.stdout, result.stderr, is_retry)

        if result.timed_out:
            self._on_failure(result, iter_dir)
            raise _SchemaParseError(
                f"agent {self.name} timed out after {result.duration_s:.1f}s"
            )
        if result.error is not None:
            self._on_failure(result, iter_dir)
            raise _SchemaParseError(f"agent {self.name}: {result.error}")

        return self._parse_invocation_metadata(result, iter_dir)

    def _on_failure(self, result: SessionResult, iter_dir: Path) -> None:  # noqa: B027
        """Hook invoked before raising on a failed run (default no-op).

        `ClaudeRunner` overrides this to blank `last_message.json`; `CodexRunner`
        relies on the default (codex owns its own output file).
        """


def _append_streams(
    stdout_path: Path,
    stderr_path: Path,
    stdout: bytes,
    stderr: bytes,
    is_retry: bool,
) -> None:
    mode = "ab" if is_retry else "wb"
    with stdout_path.open(mode) as fh:
        if is_retry:
            fh.write(_RETRY_SEPARATOR)
        fh.write(stdout or b"")
    with stderr_path.open(mode) as fh:
        if is_retry:
            fh.write(_RETRY_SEPARATOR)
        fh.write(stderr or b"")


class ClaudeRunner(AgentRunner):
    name = "claude_code"
    _executable = "claude"

    def _session(self, schema_path: Path, iter_dir: Path) -> ClaudeSession:
        schema_text = schema_path.read_text(encoding="utf-8")
        return ClaudeSession(
            ClaudeSessionOptions(
                json_schema=schema_text,
                permission_mode="plan",
                max_turns=self._config.claude_max_turns,
                timeout_s=self._config.per_iteration_wallclock_s,
                claude_bin=self._executable,
                # Blocking capture (no live event stream): discovery has never
                # rendered per-event progress, and the terminal result event is
                # parsed from the captured stdout.
                output_format="stream-json",
                stream_events=False,
            )
        )

    def _build_argv(self, schema_path: Path, iter_dir: Path) -> list[str]:
        return self._session(schema_path=schema_path, iter_dir=iter_dir).build_argv()

    def _on_failure(self, result: SessionResult, iter_dir: Path) -> None:
        # Preserve the pre-migration contract: a failed Claude run leaves an
        # empty last_message.json so the orchestrator's parse_last_message sees
        # "empty" rather than a stale success payload.
        (iter_dir / "last_message.json").write_text("", encoding="utf-8")

    def _parse_invocation_metadata(
        self, result: SessionResult, iter_dir: Path
    ) -> AgentInvocation:
        last_message_path = iter_dir / "last_message.json"
        payload = result.final_message or ""
        last_message_path.write_text(payload, encoding="utf-8")

        # Re-derive the CLI-reported per-iteration duration from the terminal
        # event (the session keeps the raw stdout); fall back to wall-clock.
        reported_duration: float | None = None
        try:
            event = extract_result_event(result.stdout)
        except ResultEventError:
            event = None
        if event is not None:
            reported_duration = _duration_seconds_from_event(event)

        usage = result.usage
        return AgentInvocation(
            session_id=result.session_id,
            duration_s=reported_duration if reported_duration is not None else result.duration_s,
            cost_usd=result.cost_usd,
            input_tokens=usage.input if usage else None,
            output_tokens=usage.output if usage else None,
            cache_read_tokens=usage.cache_read if usage else None,
            cache_create_tokens=usage.cache_create if usage else None,
            model=usage.model if usage else None,
            api_time_s=usage.api_time_s if usage else None,
        )

    def parse_last_message(self, iter_dir: Path) -> str:
        path = iter_dir / "last_message.json"
        if not path.exists():
            raise _SchemaParseError("claude last_message.json missing")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise _SchemaParseError("claude last_message.json empty")
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            raise _SchemaParseError(f"claude last_message.json not JSON: {e}") from e
        return text


class CodexRunner(AgentRunner):
    name = "codex"
    _executable = "codex"

    def _session(self, schema_path: Path, iter_dir: Path) -> CodexSession:
        repo_path = self._config.repo_path
        assert repo_path is not None, "DiscoveryConfig.repo_path is required at step entry"
        return CodexSession(
            CodexSessionOptions(
                repo_path=repo_path,
                output_last_message=iter_dir / "last_message.json",
                output_schema=schema_path,
                model=self._config.codex_model,
                reasoning_effort=self._config.codex_reasoning_effort,
                sandbox="read-only",
                timeout_s=self._config.per_iteration_wallclock_s,
                codex_bin=self._executable,
            )
        )

    def _build_argv(self, schema_path: Path, iter_dir: Path) -> list[str]:
        return self._session(schema_path=schema_path, iter_dir=iter_dir).build_argv()

    def _parse_invocation_metadata(
        self, result: SessionResult, iter_dir: Path
    ) -> AgentInvocation:
        # Codex writes last_message.json itself; the session validated it.
        duration = result.duration_s
        session_id: str | None = None
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            reported_duration = _duration_seconds_from_event(obj)
            if reported_duration is not None:
                duration = reported_duration
            session_id = obj.get("session_id") or session_id

        usage = codex_usage_from_stream(result.stdout)
        return AgentInvocation(
            session_id=session_id,
            duration_s=duration,
            cost_usd=usage.cli_reported_cost_usd if usage else None,
            input_tokens=usage.input if usage else None,
            output_tokens=usage.output if usage else None,
            cache_read_tokens=usage.cache_read if usage else None,
            cache_create_tokens=usage.cache_create if usage else None,
            model=(usage.model if usage and usage.model else None)
            or self._config.codex_model,
            api_time_s=None,
        )

    def parse_last_message(self, iter_dir: Path) -> str:
        path = iter_dir / "last_message.json"
        if not path.exists():
            raise _SchemaParseError("codex last_message.json missing")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise _SchemaParseError("codex last_message.json empty")
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            raise _SchemaParseError(f"codex last_message.json not JSON: {e}") from e
        return text


def _as_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _duration_seconds_from_event(event: dict) -> float | None:
    for key in ("duration_s", "elapsed_s"):
        value = _as_float(event.get(key))
        if value is not None:
            return value
    for key in ("duration_ms", "elapsed_ms"):
        value = _as_float(event.get(key))
        if value is not None:
            return value / 1000.0
    return None


# `time` is imported for parity with the previous module surface; no longer
# used directly here (the session owns wall-clock timing) but kept so callers
# monkeypatching `agents.time` in tests keep a valid target.
_ = time


__all__ = [
    "AgentInvocation",
    "AgentRunner",
    "ClaudeRunner",
    "CodexRunner",
    "_SchemaParseError",
]
