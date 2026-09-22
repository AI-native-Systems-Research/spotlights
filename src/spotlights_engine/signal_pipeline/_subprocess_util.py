"""Shared helpers for invoking `claude -p` with live streaming.

Both `modules_extractor/agent.py` (stage 02) and
`signal_pipeline/claude_subprocess.py` (stages 01/03/04/05) spawn
`claude -p` and parse its `stream-json` stdout. The mechanics are
identical; only the prompt, schema, and post-processing differ.
This module is the single source of truth for:

- `resolve_claude_argv0` — argv prefix that bypasses the Windows
  `claude.CMD` shim (which buffers child stdout and breaks live
  streaming). Falls through to `shutil.which` on POSIX.
- `kill_tree` — terminate the spawned child + its descendants.
  Windows uses `taskkill /F /T`; POSIX uses `killpg` (with the
  caller's `start_new_session=True`).
- `summarize_event` — convert one stream-json dict to a one-line
  string suitable for the `on_event` callback.
- `default_on_event` — the default sink (line-flushing print).
- `run_streaming_claude` — the full Popen + reader-threads +
  deadline loop. Returns captured stdout/stderr as bytes plus the
  process return code; raises on hard errors (timeout, .CMD shim
  fallback). Callers parse the terminal `result` event themselves.

Anything platform-specific or repeated in both call sites lives
here so a future tweak to event summarization, kill semantics, or
shim resolution lands in exactly one place.
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
from typing import Callable


# ── Resolution ───────────────────────────────────────────────────────────


class ClaudeResolutionError(RuntimeError):
    """`claude` couldn't be resolved to a non-shim binary on this platform."""


def resolve_claude_argv0(claude_bin: str = "claude") -> list[str]:
    """argv prefix that invokes `claude` without going through the cmd shim.

    On Windows: prefer `claude.exe` shipped under the npm install root.
    Refuses to fall back to `claude.CMD` because the shim buffers child
    stdout and silently breaks live streaming — the very bug this whole
    machinery exists to avoid. On POSIX: `shutil.which` is fine.

    `SPOTLIGHTS_CLAUDE_BIN` overrides the search entirely if set.
    """
    override = os.environ.get("SPOTLIGHTS_CLAUDE_BIN")
    if override:
        if not Path(override).exists():
            raise ClaudeResolutionError(
                f"SPOTLIGHTS_CLAUDE_BIN={override!r} does not exist"
            )
        return [override]

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            base = Path(appdata) / "npm" / "node_modules" / "@anthropic-ai" / "claude-code"
            for candidate in (base / "claude.exe", base / "bin" / "claude.exe"):
                if candidate.exists():
                    return [str(candidate)]

        resolved = shutil.which(claude_bin)
        if resolved is None:
            raise ClaudeResolutionError(
                f"required CLI not on PATH: {claude_bin}"
            )
        # Refuse the .CMD shim — it would silently re-introduce the streaming hang.
        if resolved.lower().endswith((".cmd", ".bat")):
            raise ClaudeResolutionError(
                f"resolved {claude_bin} to a cmd shim ({resolved}); the shim "
                f"buffers child stdout and breaks live streaming. Install "
                f"claude.exe under %APPDATA%\\npm\\node_modules\\@anthropic-ai"
                f"\\claude-code\\ or set SPOTLIGHTS_CLAUDE_BIN to a real .exe"
            )
        return [resolved]

    resolved = shutil.which(claude_bin)
    if resolved is None:
        raise ClaudeResolutionError(f"required CLI not on PATH: {claude_bin}")
    return [resolved]


# ── Process tree kill ────────────────────────────────────────────────────


def kill_tree(proc: subprocess.Popen) -> None:
    """Terminate proc and its descendants.

    Windows: `taskkill /F /T` walks descendants so no orphan node child
    survives. POSIX: `killpg` (caller must Popen with
    `start_new_session=True` for this to be safe — otherwise we'd kill
    ourselves).
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except (OSError, ProcessLookupError):
            pass


# ── Event summarization ──────────────────────────────────────────────────


def summarize_event(ev: dict, t0: float) -> str | None:
    """One-line summary of a stream-json event, or None to skip.

    The format is consumed by `StageEventFormatter` (which strips the
    elapsed prefix to do burst dedup), so the prefix shape is part of
    the contract: `+{seconds:6.1f}s  <body>`.
    """
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


def default_on_event(line: str) -> None:
    print(line, flush=True)


# ── Streaming runner ─────────────────────────────────────────────────────


@dataclass
class StreamingResult:
    """Outcome of one `run_streaming_claude` call.

    `stdout`/`stderr` are the captured bytes (utf-8 decoded then re-encoded);
    callers parse the terminal `result` event from `stdout` themselves.
    `returncode` is `proc.returncode`. `duration_s` is wall-clock from
    Popen until process exit / kill. `early_stopped` is True when the
    `early_stop` predicate fired and the child was killed before it exited
    on its own — the caller should salvage the captured stream rather than
    treat the nonzero returncode as a failure.
    """

    stdout: bytes
    stderr: bytes
    returncode: int
    duration_s: float
    early_stopped: bool = False


class StreamingTimeout(RuntimeError):
    """Raised when the deadline elapsed and the child was killed."""

    def __init__(
        self, *, timeout_s: float, duration_s: float,
        stdout: bytes, stderr: bytes,
    ) -> None:
        super().__init__(f"claude timed out after {duration_s:.1f}s")
        self.timeout_s = timeout_s
        self.duration_s = duration_s
        self.stdout = stdout
        self.stderr = stderr


def run_streaming_claude(
    *,
    argv: list[str],
    prompt: str,
    env: dict[str, str],
    cwd: Path | str | None,
    timeout_s: float,
    on_event: Callable[[str], None] | None = default_on_event,
    early_stop: Callable[[dict], bool] | None = None,
) -> StreamingResult:
    """Spawn `argv`, feed `prompt` on stdin, stream stdout events to `on_event`.

    Returns the captured streams + returncode on normal exit. Raises
    `StreamingTimeout` if the deadline elapses (process tree is killed
    before the exception is raised).

    `early_stop`, when given, is called with every parsed stdout event dict.
    The first time it returns True the child tree is killed and the call
    returns with `early_stopped=True`. This exists for structured stages run
    against a model that types the answer as assistant text instead of calling
    the tool: once the full payload has streamed there is no reason to let the
    CLI keep re-nudging the model to max_turns. The predicate is inert for a
    model that emits a real tool_use (nothing to detect in text), so callers
    can pass it unconditionally.
    """
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
        # Forward slashes on Windows: the agent reads its cwd from the
        # claude system-init event and copies it into Bash commands; bash
        # treats `\p`, `\v`, etc. as escape sequences and the listing
        # fails. POSIX form is accepted by both Windows native APIs and
        # bash, so normalize.
        cwd=Path(cwd).as_posix() if cwd is not None else None,
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
    early_stop_event = threading.Event()

    def _drain(stream, sink: list[str], parse: bool) -> None:
        for line in iter(stream.readline, ""):
            sink.append(line)
            if not parse:
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
            if early_stop is not None and not early_stop_event.is_set():
                try:
                    if early_stop(ev):
                        early_stop_event.set()
                except Exception:  # noqa: BLE001 - detector must not abort the run
                    pass
            if on_event is None:
                continue
            summary = summarize_event(ev, start)
            if summary:
                try:
                    on_event(summary)
                except Exception:  # noqa: BLE001 - UI errors must not abort the run
                    pass
        stream.close()

    t_out = threading.Thread(
        target=_drain, args=(proc.stdout, stdout_chunks, True), daemon=True
    )
    t_err = threading.Thread(
        target=_drain, args=(proc.stderr, stderr_chunks, False), daemon=True
    )
    t_out.start()
    t_err.start()

    deadline = start + timeout_s
    timed_out = False
    early_stopped = False
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if early_stop_event.is_set():
            kill_tree(proc)
            early_stopped = True
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            break
        if time.monotonic() > deadline:
            kill_tree(proc)
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

    if timed_out:
        raise StreamingTimeout(
            timeout_s=timeout_s,
            duration_s=duration,
            stdout=stdout_bytes,
            stderr=stderr_bytes,
        )

    return StreamingResult(
        stdout=stdout_bytes,
        stderr=stderr_bytes,
        returncode=proc.returncode if proc.returncode is not None else -1,
        duration_s=duration,
        early_stopped=early_stopped,
    )


__all__ = [
    "ClaudeResolutionError",
    "StreamingResult",
    "StreamingTimeout",
    "default_on_event",
    "kill_tree",
    "resolve_claude_argv0",
    "run_streaming_claude",
    "summarize_event",
]
