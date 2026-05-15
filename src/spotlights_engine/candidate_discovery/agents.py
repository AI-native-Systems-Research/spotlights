"""Agent subprocess runners and env scrubbing.

Each runner owns its own `parse_last_message` so CLI-specific missing/empty
/output-shape checks stay out of the orchestrator. The package-internal
`_SchemaParseError` is raised by `invoke` (timeout / nonzero exit) and by
`parse_last_message` (bad `last_message.json`); the orchestrator is the only
consumer and translates each occurrence into either a retry or a
`DiscoveryValidationError`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from spotlights_engine.candidate_discovery.errors import DiscoverySetupError

if TYPE_CHECKING:  # pragma: no cover
    from spotlights_engine.candidate_discovery.api import DiscoveryConfig


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


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in _DROP_EXACT or key.startswith(_DROP_PREFIX):
            env.pop(key)
    return env


class _SchemaParseError(Exception):
    """Package-internal signal that the iteration should be retried (§6.2).

    Raised by `AgentRunner.invoke` (timeout / nonzero exit) and by
    `AgentRunner.parse_last_message` (missing / empty / non-JSON output).
    """


_RETRY_SEPARATOR = b"\n--- retry separator ---\n"


@dataclass
class AgentInvocation:
    session_id: str | None
    duration_s: float
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None


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

    def invoke(
        self,
        prompt: str,
        iter_dir: Path,
        schema_path: Path,
    ) -> AgentInvocation:
        argv = self._build_argv(schema_path=schema_path, iter_dir=iter_dir)
        env = _clean_env()
        cwd = self._config.repo_path

        stdout_path = iter_dir / "raw_stdout.log"
        stderr_path = iter_dir / "raw_stderr.log"
        is_retry = stdout_path.exists()

        start = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                input=prompt.encode("utf-8"),
                capture_output=True,
                env=env,
                cwd=str(cwd),
                timeout=self._config.per_iteration_wallclock_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            _append_streams(stdout_path, stderr_path, stdout, stderr, is_retry)
            raise _SchemaParseError(
                f"agent {self.name} timed out after {duration:.1f}s"
            ) from exc

        _append_streams(stdout_path, stderr_path, completed.stdout, completed.stderr, is_retry)

        if completed.returncode != 0:
            stdout_tail = (completed.stdout or b"")[-500:].decode("utf-8", "replace")
            stderr_tail = (completed.stderr or b"")[-500:].decode("utf-8", "replace")
            raise _SchemaParseError(
                f"agent {self.name} exit={completed.returncode}: "
                f"stderr={stderr_tail!r} stdout={stdout_tail!r}"
            )

        return self._parse_invocation_metadata(completed.stdout, iter_dir, start)

    @abstractmethod
    def _parse_invocation_metadata(
        self, stdout: bytes, iter_dir: Path, start: float
    ) -> AgentInvocation: ...


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

    def _build_argv(self, schema_path: Path, iter_dir: Path) -> list[str]:
        schema_text = schema_path.read_text(encoding="utf-8")
        return [
            self._executable,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--json-schema",
            schema_text,
            "--permission-mode",
            "plan",
            "--max-turns",
            str(self._config.claude_max_turns),
        ]

    def _parse_invocation_metadata(
        self, stdout: bytes, iter_dir: Path, start: float
    ) -> AgentInvocation:
        duration = time.monotonic() - start
        last_message_path = iter_dir / "last_message.json"

        result_event = self._extract_result_event(stdout)
        if result_event is None:
            last_message_path.write_text("", encoding="utf-8")
            raise _SchemaParseError("claude stream-json had no terminal result event")

        message_text = self._final_message_text(result_event)
        last_message_path.write_text(message_text, encoding="utf-8")

        usage = result_event.get("usage") or {}
        reported_duration = _duration_seconds_from_event(result_event)
        return AgentInvocation(
            session_id=result_event.get("session_id"),
            duration_s=reported_duration if reported_duration is not None else duration,
            cost_usd=_as_float(result_event.get("total_cost_usd")),
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
        )

    @staticmethod
    def _extract_result_event(stdout: bytes) -> dict | None:
        last_event: dict | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise _SchemaParseError(f"claude stream-json line not JSON: {e}") from e
            if not isinstance(obj, dict):
                raise _SchemaParseError("claude stream-json line was not an object")
            last_event = obj
        if last_event is None or last_event.get("type") != "result":
            return None
        return last_event

    @staticmethod
    def _final_message_text(result_event: dict) -> str:
        # With --json-schema, the CLI parses the model output and places the
        # validated object on `structured_output`; the `result` string is empty
        # in that mode. Prefer structured_output when present.
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

    def _build_argv(self, schema_path: Path, iter_dir: Path) -> list[str]:
        return [
            self._executable,
            "exec",
            "-",
            "--json",
            "--output-last-message",
            str(iter_dir / "last_message.json"),
            "--output-schema",
            str(schema_path),
            "--sandbox",
            "read-only",
            "-C",
            str(self._config.repo_path),
            "-c",
            f'model="{self._config.codex_model}"',
            "-c",
            f'model_reasoning_effort="{self._config.codex_reasoning_effort}"',
        ]

    def _parse_invocation_metadata(
        self, stdout: bytes, iter_dir: Path, start: float
    ) -> AgentInvocation:
        duration = time.monotonic() - start
        session_id: str | None = None
        cost_usd: float | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

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
            reported_duration = _duration_seconds_from_event(obj)
            if reported_duration is not None:
                duration = reported_duration
            session_id = obj.get("session_id") or session_id
            usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
            if usage is not None:
                if input_tokens is None:
                    input_tokens = _as_int(usage.get("input_tokens"))
                if output_tokens is None:
                    output_tokens = _as_int(usage.get("output_tokens"))
            if cost_usd is None:
                cost_usd = _as_float(obj.get("total_cost_usd") or obj.get("cost_usd"))

        return AgentInvocation(
            session_id=session_id,
            duration_s=duration,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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


__all__ = [
    "AgentInvocation",
    "AgentRunner",
    "ClaudeRunner",
    "CodexRunner",
    "_SchemaParseError",
    "_clean_env",
]
