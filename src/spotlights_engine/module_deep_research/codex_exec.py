"""Small subprocess wrapper around non-interactive `codex exec`."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import codex_usage_from_stream
from spotlights_engine.local_agent.base import (
    AgentExecResult,
    resolve_cli_executable,
)


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
        """Run `codex exec` with `prompt` on stdin, streaming output live."""
        cmd, last_path = self.build_command("-")
        env = os.environ.copy()
        if self.options.env:
            env.update(dict(self.options.env))

        # Local-model dispatch: a Qwen model routes to pi/vLLM instead of the
        # codex CLI. Deep research is free-text, so pi's final message becomes
        # `final_message` and is mirrored into the `--output-last-message` file.
        from spotlights_engine.costing.usage import AgentUsage
        from spotlights_engine.local_agent.dispatch import is_local_model, run_pi

        if is_local_model(self.options.model):
            pi = run_pi(
                prompt=prompt,
                cwd=Path(self.options.cwd).expanduser().resolve(),
                env=env,
                timeout_s=self.options.timeout_seconds or 3600,
                model=self.options.model,
                read_only=self.options.sandbox in (None, "read-only"),
            )
            if last_path is not None:
                last_path.write_text(pi.final_text, encoding="utf-8")
            result = CodexExecResult(
                command=cmd,
                returncode=pi.returncode,
                stdout=pi.stdout,
                stderr=pi.stderr,
                final_message=pi.final_text,
                usage=AgentUsage(
                    input=pi.input_tokens, output=pi.output_tokens, model=pi.model
                ),
                output_last_message=last_path,
            )
            if check:
                result.raise_for_status()
            return result

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
            # Own process group so we can kill codex AND its grandchildren.
            # codex spawns a node child that inherits the stdout/stderr pipe
            # write-ends; killing only codex leaves the grandchild holding the
            # pipe open, so the reader threads never see EOF and their join()
            # blocks forever (whole run wedges at 0% CPU).
            start_new_session=True,
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

        def _kill_group() -> None:
            # SIGKILL the whole session (codex + node grandchildren) so the
            # pipe write-ends close and the reader threads reach EOF.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()

        try:
            returncode = proc.wait(timeout=self.options.timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill_group()
            proc.wait()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            raise

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            # codex exited but a grandchild still holds the pipe open. Kill the
            # group so the threads see EOF, then reap; never join unbounded.
            _kill_group()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

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
        if check:
            result.raise_for_status()
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
