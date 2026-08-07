"""Low-level subprocess transport shared by every Claude/Codex session.

This is the single home for the machinery that used to be copy-pasted across
the spawners:

- `run_streaming_claude` / `kill_tree` / `summarize_event` / `default_on_event`
  / `resolve_claude_argv0` — moved verbatim from
  `signal_pipeline/_subprocess_util.py` (which is now a re-export shim).
- `run_blocking` — the `subprocess.run(capture_output=True)` path used by the
  blocking spawners (`agent_proposals`, `proposal_from_finding_creator`,
  `candidate_discovery`, `module_deep_research`).
- `extract_result_event` — the stream-json terminal-event parser (5 copies).
- `final_message_text` — structured_output → result → message fallback
  (3 named defs + inlined variants).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Re-exported for callers migrating off resolve_claude_argv0's old home.
from spotlights_engine.llm_session.resolve import (  # noqa: F401
    ClaudeResolutionError,
    CLIResolutionError,
    resolve_claude_argv0,
)

# ── Errors ───────────────────────────────────────────────────────────────


class ResultEventError(Exception):
    """A stream-json line was not JSON / not an object."""


# ── Process tree kill ────────────────────────────────────────────────────


def kill_tree(proc: subprocess.Popen) -> None:
    """Terminate proc and its descendants.

    Windows: `taskkill /F /T` walks descendants. POSIX: `killpg` (caller must
    Popen with `start_new_session=True`).
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        return
    import os

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

    The format is consumed by `StageEventFormatter` (which strips the elapsed
    prefix to do burst dedup), so the prefix shape is part of the contract:
    `+{seconds:6.1f}s  <body>`.
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
    """Outcome of one `run_streaming_claude` call."""

    stdout: bytes
    stderr: bytes
    returncode: int
    duration_s: float


class StreamingTimeout(RuntimeError):
    """Raised when the deadline elapsed and the child was killed."""

    def __init__(
        self,
        *,
        timeout_s: float,
        duration_s: float,
        stdout: bytes,
        stderr: bytes,
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
) -> StreamingResult:
    """Spawn `argv`, feed `prompt` on stdin, stream stdout events to `on_event`.

    Returns the captured streams + returncode on normal exit. Raises
    `StreamingTimeout` if the deadline elapses (process tree killed first).
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
        # Forward slashes on Windows: the agent copies its cwd into Bash
        # commands and bash treats `\p`, `\v`, etc. as escapes. POSIX form is
        # accepted by both Windows native APIs and bash.
        cwd=Path(cwd).as_posix() if cwd is not None else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **popen_kwargs,
    )
    assert proc.stdin is not None
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
    while True:
        rc = proc.poll()
        if rc is not None:
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
    )


# ── Blocking runner ──────────────────────────────────────────────────────


@dataclass
class BlockingResult:
    """Outcome of one blocking `subprocess.run` invocation.

    `timed_out` distinguishes a wall-clock kill (its own retry budget) from a
    normal nonzero exit.
    """

    stdout: bytes
    stderr: bytes
    returncode: int
    duration_s: float
    timed_out: bool = False


def run_blocking(
    *,
    argv: list[str],
    prompt: str,
    env: dict[str, str],
    cwd: Path | str | None,
    timeout_s: float | None,
) -> BlockingResult:
    """Spawn `argv`, feed `prompt` on stdin (bytes), block until exit/timeout.

    On timeout the partial streams are captured and `timed_out=True` is set
    (the process is reaped by `subprocess.run`); no exception is raised so the
    session can shape a `SessionResult` uniformly.
    """
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
        return BlockingResult(
            stdout=exc.stdout or b"",
            stderr=exc.stderr or b"",
            returncode=-1,
            duration_s=duration,
            timed_out=True,
        )
    duration = time.monotonic() - start
    return BlockingResult(
        stdout=completed.stdout or b"",
        stderr=completed.stderr or b"",
        returncode=completed.returncode,
        duration_s=duration,
        timed_out=False,
    )


# ── stream-json parsing (shared) ─────────────────────────────────────────


def extract_result_event(stdout: bytes | str) -> dict | None:
    """Return the final stream-json event iff it is a `type == "result"`.

    Raises `ResultEventError` if any non-empty line is not a JSON object —
    matching the strict behaviour of the copies it replaces.
    """
    text = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else stdout
    last_event: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResultEventError(f"claude stream-json line not JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ResultEventError("claude stream-json line was not an object")
        last_event = obj
    if last_event is None or last_event.get("type") != "result":
        return None
    return last_event


def final_message_text(result_event: dict) -> str:
    """structured_output → result → message.content fallback, as a string.

    With `--json-schema`, the CLI parses the model output and places the
    validated object on `structured_output`; the `result` string is empty in
    that mode, so prefer `structured_output`. Returns "" when nothing usable.
    """
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


def structured_output_from_result(result_event: dict) -> dict | list | None:
    """Return `structured_output`, or a JSON-parsed `result` fallback.

    Raises `ResultEventError` when a non-empty `result` string is not JSON —
    matching the "claude result text not JSON" error the copies produced.
    """
    structured = result_event.get("structured_output")
    if isinstance(structured, (dict, list)):
        return structured
    fallback = result_event.get("result")
    if isinstance(fallback, str) and fallback.strip():
        try:
            parsed = json.loads(fallback)
        except json.JSONDecodeError as exc:
            raise ResultEventError(f"claude result text not JSON: {exc}") from exc
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def model_from_init(stdout: bytes | str) -> str | None:
    """Read the first stream-json line; return its `model` if it's `system/init`."""
    text = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else stdout
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if (
            isinstance(obj, dict)
            and obj.get("type") == "system"
            and obj.get("subtype") == "init"
        ):
            model = obj.get("model")
            return model if isinstance(model, str) else None
        return None
    return None


__all__ = [
    "BlockingResult",
    "CLIResolutionError",
    "ClaudeResolutionError",
    "ResultEventError",
    "StreamingResult",
    "StreamingTimeout",
    "default_on_event",
    "extract_result_event",
    "final_message_text",
    "kill_tree",
    "model_from_init",
    "resolve_claude_argv0",
    "run_blocking",
    "run_streaming_claude",
    "structured_output_from_result",
    "summarize_event",
]
