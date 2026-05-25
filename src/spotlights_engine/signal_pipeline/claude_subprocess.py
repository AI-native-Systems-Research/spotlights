"""Generic `claude -p` subprocess wrapper for stages 03/04/05.

Modeled on `agent_proposals.claude_exec` but parameterised: stages
choose `permission_mode`, `cwd`, `json_schema`, and `allowed_tools`.

Persists the assembled prompt + raw stdout/stderr under `log_dir` for
debugging, regardless of success / failure. The runner's per-stage
`log_dir` is `_logs/<NN>_<name>/<timestamp>/`, so each invocation gets
its own directory and old runs aren't clobbered.

**Auth-leak fix:** main's `_clean_env` (in `agent_proposals.claude_exec`
and `modules_extractor.agent`) drops `ANTHROPIC_BASE_URL` but not
`ANTHROPIC_AUTH_TOKEN`. When this pipeline is invoked from inside a
Claude Code session, that token leaks into the spawned subprocess and
overrides keychain credentials → `401 Invalid bearer token`. We drop
both. (The same fix on main is a separate cleanup; reproducing here
keeps the signal pipeline self-sufficient.)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


_DROP_EXACT = frozenset(
    {
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
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
    """

    structured_output: dict | list | None
    duration_s: float
    error: str | None = None


def ensure_claude_available(claude_bin: str = "claude") -> None:
    if shutil.which(claude_bin) is None:
        raise ClaudeNotAvailableError(
            f"required CLI not on PATH: {claude_bin}"
        )


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
) -> ClaudeRunResult:
    """Spawn one `claude -p` session and return its parsed structured_output.

    The prompt is fed on stdin; output is `stream-json`. Streams + the
    prompt itself are persisted to `log_dir` for post-mortem.
    """
    ensure_claude_available(claude_bin)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    if json_schema is not None:
        (log_dir / "schema.json").write_text(json_schema, encoding="utf-8")

    argv: list[str] = [
        claude_bin,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        permission_mode,
        "--max-turns",
        str(max_turns),
    ]
    if json_schema is not None:
        argv += ["--json-schema", json_schema]
    if allowed_tools is not None:
        argv += ["--allowed-tools", ",".join(allowed_tools)]

    env = _clean_env()
    start = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            input=prompt.encode("utf-8"),
            capture_output=True,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        _persist_streams(log_dir, exc.stdout or b"", exc.stderr or b"")
        return ClaudeRunResult(
            structured_output=None,
            duration_s=duration,
            error=f"claude timed out after {duration:.1f}s",
        )

    duration = time.monotonic() - start
    _persist_streams(log_dir, completed.stdout or b"", completed.stderr or b"")

    if completed.returncode != 0:
        stderr_tail = (completed.stderr or b"")[-500:].decode("utf-8", "replace")
        return ClaudeRunResult(
            structured_output=None,
            duration_s=duration,
            error=(
                f"claude exit={completed.returncode}: stderr={stderr_tail!r} "
                f"(see {log_dir / 'raw_stderr.log'})"
            ),
        )

    try:
        result_event = _extract_result_event(completed.stdout)
    except _ResultEventError as e:
        return ClaudeRunResult(
            structured_output=None,
            duration_s=duration,
            error=str(e),
        )
    if result_event is None:
        return ClaudeRunResult(
            structured_output=None,
            duration_s=duration,
            error="claude stream-json had no terminal result event",
        )

    structured = result_event.get("structured_output")
    if isinstance(structured, (dict, list)):
        return ClaudeRunResult(
            structured_output=structured, duration_s=duration
        )

    # Older CLI versions: the JSON payload may live in `result` as text.
    fallback = result_event.get("result")
    if isinstance(fallback, str) and fallback.strip():
        try:
            parsed = json.loads(fallback)
        except json.JSONDecodeError as e:
            return ClaudeRunResult(
                structured_output=None,
                duration_s=duration,
                error=f"claude result text not JSON: {e}",
            )
        if isinstance(parsed, (dict, list)):
            return ClaudeRunResult(
                structured_output=parsed, duration_s=duration
            )

    return ClaudeRunResult(
        structured_output=None,
        duration_s=duration,
        error="claude produced no structured_output and no usable fallback",
    )


def _persist_streams(log_dir: Path, stdout: bytes, stderr: bytes) -> None:
    (log_dir / "raw_stdout.log").write_bytes(stdout)
    (log_dir / "raw_stderr.log").write_bytes(stderr)


class _ResultEventError(Exception):
    pass


def _extract_result_event(stdout: bytes) -> dict | None:
    """Parse stream-json stdout; return the final event iff it's a `result`.

    Same behavior as `agent_proposals.claude_exec._extract_result_event`,
    duplicated here to keep the signal pipeline self-contained.
    """
    last_event: dict | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise _ResultEventError(
                f"claude stream-json line not JSON: {e}"
            ) from e
        if not isinstance(obj, dict):
            raise _ResultEventError("claude stream-json line was not an object")
        last_event = obj
    if last_event is None or last_event.get("type") != "result":
        return None
    return last_event


__all__ = [
    "ClaudeNotAvailableError",
    "ClaudeRunResult",
    "ClaudeSubprocessError",
    "ensure_claude_available",
    "run_claude",
]
