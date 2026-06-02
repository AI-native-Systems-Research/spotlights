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

**Windows shim bypass:** `claude` on Windows is `claude.CMD`, a cmd.exe
shim that buffers the child's stdout — live readers see no events until
the child exits. We resolve `claude.exe` directly under
`%APPDATA%\\npm\\node_modules\\@anthropic-ai\\claude-code\\`.

Output is streamed live via a reader thread; each stream-json event is
forwarded to the caller-supplied `on_event` callback (defaults to
printing). On timeout the entire process tree is killed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal


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


def _resolve_argv0(claude_bin: str) -> list[str]:
    """argv prefix that bypasses the Windows .CMD shim.

    The `claude.CMD` shim on PATH buffers child stdout and breaks live
    streaming. Resolve `claude.exe` directly on Windows; pass through on
    POSIX.
    """
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            base = Path(appdata) / "npm" / "node_modules" / "@anthropic-ai" / "claude-code"
            for candidate in (base / "claude.exe", base / "bin" / "claude.exe"):
                if candidate.exists():
                    return [str(candidate)]

    resolved = shutil.which(claude_bin) or claude_bin
    return [resolved]


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, check=False,
        )
        return
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except (OSError, ProcessLookupError):
            pass


def _summarize_event(ev: dict, t0: float) -> str | None:
    et = ev.get("type")
    elapsed = f"+{time.monotonic() - t0:6.1f}s"
    if et == "system" and ev.get("subtype") == "init":
        sid = (ev.get("session_id") or "?")[:8]
        return f"{elapsed}  init session={sid}"
    if et == "assistant":
        msg = ev.get("message") or {}
        for c in msg.get("content") or []:
            if c.get("type") == "tool_use":
                name = c.get("name", "?")
                inp = c.get("input") or {}
                if name == "Bash":
                    desc = inp.get("description") or (inp.get("command") or "")[:80]
                elif name in ("Glob", "Grep"):
                    desc = inp.get("pattern", "")
                elif name == "Read":
                    desc = inp.get("file_path", "")
                elif name in ("Edit", "Write"):
                    desc = inp.get("file_path", "")
                else:
                    desc = json.dumps(inp)[:80]
                return f"{elapsed}  {name}: {desc}"
            if c.get("type") == "text":
                t = (c.get("text") or "").strip()
                if t:
                    return f"{elapsed}  text: {t[:120]}"
        return None
    if et == "user":
        msg = ev.get("message") or {}
        for c in msg.get("content") or []:
            if c.get("type") == "tool_result":
                content = c.get("content")
                if isinstance(content, str):
                    body = content
                elif isinstance(content, list):
                    body = json.dumps(content)
                else:
                    body = ""
                err = " ERR" if c.get("is_error") else ""
                if not body:
                    snippet = "(empty)"
                elif len(body) > 1024:
                    snippet = f"<{len(body)} chars>"
                else:
                    snippet = body.replace("\n", " | ")[:80]
                return f"{elapsed}  result{err}: {snippet}"
        return None
    if et == "result":
        sub = ev.get("subtype", "?")
        turns = ev.get("num_turns")
        cost = ev.get("total_cost_usd")
        if cost is not None:
            return f"{elapsed}  RESULT subtype={sub} turns={turns} cost=${cost:.3f}"
        return f"{elapsed}  RESULT subtype={sub} turns={turns}"
    return None


def _default_on_event(line: str) -> None:
    print(line, flush=True)


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
    on_event: Callable[[str], None] | None = _default_on_event,
) -> ClaudeRunResult:
    """Spawn one `claude -p` session and return its parsed structured_output.

    The prompt is fed on stdin; output is `stream-json`. Streams + the
    prompt itself are persisted to `log_dir` for post-mortem. `on_event` is
    invoked with a one-line summary of each stream-json event as it arrives;
    pass `None` to silence.
    """
    ensure_claude_available(claude_bin)
    argv0 = _resolve_argv0(claude_bin)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    if json_schema is not None:
        (log_dir / "schema.json").write_text(json_schema, encoding="utf-8")

    argv: list[str] = argv0 + [
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

    popen_kwargs: dict = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True

    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **popen_kwargs,
    )
    proc.stdin.write(prompt)
    proc.stdin.close()

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _drain(stream, sink: list[str], parse: bool) -> None:
        for line in iter(stream.readline, ""):
            sink.append(line)
            if not parse or on_event is None:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            summary = _summarize_event(ev, start)
            if summary:
                try:
                    on_event(summary)
                except Exception:  # noqa: BLE001
                    pass
        stream.close()

    t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_chunks, True), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_chunks, False), daemon=True)
    t_out.start(); t_err.start()

    timed_out = False
    deadline = start + timeout_s
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if time.monotonic() > deadline:
            _kill_tree(proc)
            timed_out = True
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            break
        time.sleep(0.5)

    t_out.join(timeout=10)
    t_err.join(timeout=10)
    duration = time.monotonic() - start
    stdout_bytes = "".join(stdout_chunks).encode("utf-8")
    stderr_bytes = "".join(stderr_chunks).encode("utf-8")

    _persist_streams(log_dir, stdout_bytes, stderr_bytes)

    if timed_out:
        return ClaudeRunResult(
            structured_output=None,
            duration_s=duration,
            error=f"claude timed out after {duration:.1f}s",
        )

    rc = proc.returncode if proc.returncode is not None else -1
    if rc != 0:
        stderr_tail = stderr_bytes[-500:].decode("utf-8", "replace")
        return ClaudeRunResult(
            structured_output=None,
            duration_s=duration,
            error=(
                f"claude exit={rc}: stderr={stderr_tail!r} "
                f"(see {log_dir / 'raw_stderr.log'})"
            ),
        )

    try:
        result_event = _extract_result_event(stdout_bytes)
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
