"""Small subprocess wrapper around non-interactive `codex exec`."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import codex_usage_from_stream
from spotlights_engine.module_deep_research.agent_exec import (
    AgentExecResult,
    resolve_cli_executable,
)

_log = logging.getLogger(__name__)


_RATE_LIMIT_PATTERNS = (
    "exceeded rate limit",
    "rate limit exceeded",
    "stream disconnected before completion",
    "rate-limited",
    "ratelimitexceeded",
    "http 429",
    "(429)",
)


def _looks_rate_limited(stdout: str, stderr: str) -> bool:
    """Heuristic: does the codex CLI output look like a gateway rate-limit eviction?

    The Azure LiteLLM gateway can terminate the SSE stream mid-response with
    strings like "Your requests to <model> have exceeded rate limit." or
    "stream disconnected before completion". The codex CLI exits non-zero
    with no clean error path, so the only way to detect this is to match the
    rate-limit substring in its captured output.
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    haystack = combined.lower()
    return any(pattern in haystack for pattern in _RATE_LIMIT_PATTERNS)


class CodexExecOptions(BaseModel):
    """Options for running Codex in non-interactive mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: Path | str = Field(default_factory=Path.cwd)
    codex_bin: str = "codex"
    model: str | None = None
    profile: str | None = None
    sandbox: str = "workspace-write"
    approval: str = "never"
    search: bool = True
    skip_git_repo_check: bool = True
    ephemeral: bool = False
    json_events: bool = False
    timeout_seconds: int | None = None
    output_last_message: Path | str | None = None
    extra_args: Sequence[str] = Field(default_factory=tuple)
    env: Mapping[str, str] | None = None
    stream_logs: bool = False
    rate_limit_retries: int = Field(default=3, ge=0)
    rate_limit_backoff_seconds: float = Field(default=120.0, ge=0.0)


class CodexExecResult(AgentExecResult):
    """Captured result from one `codex exec` invocation."""

    output_last_message: Path | None

    def raise_for_status(self) -> None:
        if self.returncode != 0:
            raise RuntimeError(
                "codex exec failed with exit code "
                f"{self.returncode}\nCOMMAND: {' '.join(self.command)}\n"
                f"STDOUT:\n{self.stdout}\nSTDERR:\n{self.stderr}"
            )


class CodexExecClient:
    """Run Codex in non-interactive mode from Python."""

    name = "codex"

    def __init__(self, options: CodexExecOptions | None = None) -> None:
        self.options = options or CodexExecOptions()

    def build_command(self, prompt_file: str = "-") -> tuple[list[str], Path | None]:
        opt = self.options
        cwd = Path(opt.cwd).expanduser().resolve()

        if opt.output_last_message is None:
            handle = tempfile.NamedTemporaryFile(
                prefix="spotlights-module-research-", suffix=".md", delete=False
            )
            output_last_message = Path(handle.name)
            handle.close()
        else:
            output_last_message = Path(opt.output_last_message).expanduser().resolve()
            output_last_message.parent.mkdir(parents=True, exist_ok=True)

        cmd = [resolve_cli_executable(opt.codex_bin)]
        if opt.model:
            cmd += ["--model", opt.model]
        if opt.profile:
            cmd += ["--profile", opt.profile]
        if opt.approval:
            cmd += ["--ask-for-approval", opt.approval]
        if opt.search:
            cmd += ["--search"]

        cmd += ["exec", "--cd", str(cwd)]
        if opt.sandbox:
            cmd += ["--sandbox", opt.sandbox]
        if opt.skip_git_repo_check:
            cmd += ["--skip-git-repo-check"]
        if opt.ephemeral:
            cmd += ["--ephemeral"]
        if opt.json_events:
            cmd += ["--json"]
        if output_last_message:
            cmd += ["--output-last-message", str(output_last_message)]
        cmd += list(opt.extra_args)
        cmd.append(prompt_file)
        return cmd, output_last_message

    def run(self, prompt: str, *, check: bool = True) -> CodexExecResult:
        """Run `codex exec` with `prompt` on stdin, streaming output live.

        Retries on rate-limit eviction (LiteLLM/Azure terminating the SSE
        stream when the per-minute / per-hour quota is hit). See
        ``CodexExecOptions.rate_limit_retries`` / ``rate_limit_backoff_seconds``
        for the policy; retries use exponential backoff. Non-rate-limit
        non-zero exits are returned without retry.
        """
        result = self._run_once(prompt)
        attempts_remaining = max(0, self.options.rate_limit_retries)
        attempt = 0
        while (
            attempts_remaining > 0
            and result.returncode != 0
            and _looks_rate_limited(result.stdout, result.stderr)
        ):
            attempt += 1
            wait = self.options.rate_limit_backoff_seconds * (2 ** (attempt - 1))
            _log.warning(
                "codex hit a rate limit (attempt %d); sleeping %.0fs before retry",
                attempt,
                wait,
            )
            time.sleep(wait)
            result = self._run_once(prompt)
            attempts_remaining -= 1

        if check:
            result.raise_for_status()
        return result

    def _run_once(self, prompt: str) -> CodexExecResult:
        """One non-retried codex invocation. Factored out of ``run`` so the
        retry wrapper above can call it without recursion."""
        cmd, last_path = self.build_command("-")
        env = os.environ.copy()
        if self.options.env:
            env.update(dict(self.options.env))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Pin UTF-8 for stdin/stdout/stderr. Without this, Python's
            # text mode uses locale.getpreferredencoding() — cp1252 on
            # Windows — and any non-cp1252 char in codex's response
            # (arrows, quotes, em-dashes, accented chars, …) crashes
            # the reader thread with a 'charmap' codec error.
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(self.options.cwd).expanduser().resolve()),
            env=env,
            bufsize=1,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_sink = sys.stdout if self.options.stream_logs else None
        stderr_sink = sys.stderr if self.options.stream_logs else None

        stdout_thread = threading.Thread(
            target=_tee_stream,
            args=(proc.stdout, stdout_sink, stdout_chunks),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_tee_stream,
            args=(proc.stderr, stderr_sink, stderr_chunks),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            proc.stdin.write(prompt)
        finally:
            proc.stdin.close()

        try:
            returncode = proc.wait(timeout=self.options.timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stdout_thread.join()
            stderr_thread.join()
            raise

        stdout_thread.join()
        stderr_thread.join()

        final_message = None
        if last_path and last_path.exists():
            final_message = last_path.read_text(encoding="utf-8", errors="replace")

        result = CodexExecResult(
            command=cmd,
            returncode=returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            final_message=final_message,
            usage=codex_usage_from_stream("".join(stdout_chunks)),
            output_last_message=last_path,
        )
        if result.usage is not None and result.usage.model is None and self.options.model:
            result.usage = result.usage.model_copy(update={"model": self.options.model})
        return result


def _tee_stream(source: IO[str], sink: IO[str] | None, buffer: list[str]) -> None:
    for line in source:
        buffer.append(line)
        if sink is None:
            continue
        try:
            sink.write(line)
            sink.flush()
        except Exception:
            pass


__all__ = ["CodexExecClient", "CodexExecOptions", "CodexExecResult"]
