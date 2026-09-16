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
from spotlights_engine.costing.usage import (
    claude_usage_from_payload,
    codex_usage_from_stream,
)

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

    `contract` separates the two reasons an attempt can fail, because they
    have opposite consequences once both attempts are spent: an agent that
    breaks the output contract is a deterministic bug that must stay fatal,
    while an agent the environment cut off (rate limit, timeout, context
    exhaustion) leaves the earlier iterations perfectly valid and so is
    salvageable. Everything raised from this module is environmental, hence
    the default.
    """

    def __init__(self, message: str = "", *, contract: bool = False) -> None:
        super().__init__(message)
        self.contract = contract


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
                _explain(f"agent {self.name} timed out after {duration:.1f}s", iter_dir)
            ) from exc

        _append_streams(stdout_path, stderr_path, completed.stdout, completed.stderr, is_retry)

        if completed.returncode != 0:
            stdout_tail = (completed.stdout or b"")[-500:].decode("utf-8", "replace")
            stderr_tail = (completed.stderr or b"")[-500:].decode("utf-8", "replace")
            diagnosis = _diagnose(completed.stdout or b"", completed.stderr or b"")
            raise _SchemaParseError(
                f"agent {self.name} exit={completed.returncode}: "
                + (f"{diagnosis}; " if diagnosis else "")
                + f"stderr={stderr_tail!r} stdout={stdout_tail!r}"
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


# Terminal-failure signatures we know how to name. `parse_last_message` can only
# report that `last_message.json` is missing, which is what a 429 storm, a
# context-window compaction, and a genuinely malformed reply all look like from
# the outside — the run that produced "schema parse failed twice" for all eight
# failed modules was in fact eight rate limits. The real terminal event is in the
# raw stream, so scan it and say so.
def _scan_stream_events(stdout: bytes) -> dict[str, object]:
    found: dict[str, object] = {}
    api_retries = 0
    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw or not raw.startswith(b"{"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        # codex --json nests some events under "msg"; claude does not.
        for event in (obj, obj.get("msg")):
            if not isinstance(event, dict):
                continue
            etype = event.get("type")
            if etype == "error":
                message = event.get("message")
                if isinstance(message, str) and message:
                    found["hard_error"] = message
            elif etype == "turn.failed":
                err = event.get("error")
                message = err.get("message") if isinstance(err, dict) else err
                if isinstance(message, str) and message:
                    found["hard_error"] = message
            elif etype == "result" and event.get("is_error"):
                found["result_error"] = str(
                    event.get("result") or event.get("subtype") or "unspecified"
                )
            elif etype == "system":
                if event.get("subtype") == "api_retry":
                    api_retries += 1
                    reason = event.get("error")
                    if isinstance(reason, str) and reason:
                        found["retry_reason"] = reason
                if event.get("status") == "compacting":
                    found["compacting"] = True
    if api_retries:
        found["api_retries"] = api_retries
    return found


def _diagnose(stdout: bytes, stderr: bytes = b"") -> str | None:
    """Best-effort one-line explanation of why an agent produced no output."""
    found = _scan_stream_events(stdout)
    parts: list[str] = []
    if isinstance(found.get("hard_error"), str):
        parts.append(f"agent reported: {found['hard_error']}")
    elif isinstance(found.get("result_error"), str):
        parts.append(f"agent result was an error: {found['result_error']}")
    retries = found.get("api_retries")
    if isinstance(retries, int):
        reason = found.get("retry_reason")
        suffix = f" (last: {reason})" if isinstance(reason, str) else ""
        parts.append(f"{retries} API retries in stream{suffix}")
    if found.get("compacting"):
        parts.append("stream hit context compaction before finishing")
    if not parts:
        tail = stderr[-300:].decode("utf-8", "replace").strip()
        if tail:
            parts.append(f"stderr tail: {tail!r}")
    return "; ".join(parts) or None


def _diagnose_iter_dir(iter_dir: Path) -> str | None:
    """`_diagnose` over the persisted raw streams for an iteration directory."""
    def _read(name: str) -> bytes:
        path = iter_dir / name
        try:
            return path.read_bytes()
        except OSError:
            return b""

    return _diagnose(_read("raw_stdout.log"), _read("raw_stderr.log"))


def _explain(message: str, iter_dir: Path) -> str:
    diagnosis = _diagnose_iter_dir(iter_dir)
    return f"{message}; {diagnosis}" if diagnosis else message


class ClaudeRunner(AgentRunner):
    name = "claude_code"
    _executable = "claude"

    def _build_argv(self, schema_path: Path, iter_dir: Path) -> list[str]:
        schema_text = schema_path.read_text(encoding="utf-8")
        # Resolve to claude.exe (not claude.CMD). The .CMD shim buffers
        # stdout and deadlocks subprocess.run(capture_output=True) on
        # large outputs; the shared helper finds the npm-installed
        # claude.exe and refuses the .CMD fallback.
        from spotlights_engine.signal_pipeline._subprocess_util import (
            resolve_claude_argv0,
        )
        argv0 = resolve_claude_argv0(self._executable)
        return [
            *argv0,
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
            *(
                ["--model", self._config.claude_model]
                if self._config.claude_model
                else []
            ),
        ]

    def _parse_invocation_metadata(
        self, stdout: bytes, iter_dir: Path, start: float
    ) -> AgentInvocation:
        duration = time.monotonic() - start
        last_message_path = iter_dir / "last_message.json"

        result_event = self._extract_result_event(stdout)
        if result_event is None:
            last_message_path.write_text("", encoding="utf-8")
            raise _SchemaParseError(
                _explain("claude stream-json had no terminal result event", iter_dir)
            )

        message_text = self._final_message_text(result_event)
        last_message_path.write_text(message_text, encoding="utf-8")

        usage = claude_usage_from_payload(result_event)
        reported_duration = _duration_seconds_from_event(result_event)
        return AgentInvocation(
            session_id=result_event.get("session_id"),
            duration_s=reported_duration if reported_duration is not None else duration,
            cost_usd=_as_float(result_event.get("total_cost_usd")),
            input_tokens=usage.input if usage else None,
            output_tokens=usage.output if usage else None,
            cache_read_tokens=usage.cache_read if usage else None,
            cache_create_tokens=usage.cache_create if usage else None,
            model=usage.model if usage else None,
            api_time_s=usage.api_time_s if usage else None,
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
            raise _SchemaParseError(_explain("claude last_message.json missing", iter_dir))
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise _SchemaParseError(_explain("claude last_message.json empty", iter_dir))
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            raise _SchemaParseError(f"claude last_message.json not JSON: {e}") from e
        return text


class CodexRunner(AgentRunner):
    name = "codex"
    _executable = "codex"

    def _build_argv(self, schema_path: Path, iter_dir: Path) -> list[str]:
        # Codex runs with `-C <repo_path>`, so any relative path here would
        # resolve under the target repo. Pass absolutes for both schema and
        # last_message outputs.
        # Resolve via shutil.which so Windows finds the .CMD/.ps1 shim;
        # bare "codex" → FileNotFoundError because subprocess on Windows
        # doesn't follow PATHEXT for unqualified argv[0].
        resolved = shutil.which(self._executable) or self._executable
        return [
            resolved,
            "exec",
            "-",
            "--json",
            "--output-last-message",
            str((iter_dir / "last_message.json").resolve()),
            "--output-schema",
            str(schema_path.resolve()),
            "--sandbox",
            "read-only",
            "-C",
            str(self._config.repo_path),
            *(
                ["-c", f'model="{self._config.codex_model}"']
                if self._config.codex_model
                else []
            ),
            "-c",
            f'model_reasoning_effort="{self._config.codex_reasoning_effort}"',
        ]

    def _parse_invocation_metadata(
        self, stdout: bytes, iter_dir: Path, start: float
    ) -> AgentInvocation:
        duration = time.monotonic() - start
        session_id: str | None = None

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

        # Shared parser: keeps the latest cumulative usage payload and
        # normalizes codex's cached-subset convention into disjoint buckets.
        usage = codex_usage_from_stream(stdout)
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
            raise _SchemaParseError(_explain("codex last_message.json missing", iter_dir))
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise _SchemaParseError(_explain("codex last_message.json empty", iter_dir))
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
