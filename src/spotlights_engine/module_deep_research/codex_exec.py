"""Small subprocess wrapper around non-interactive `codex exec`."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult


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

        # Resolve via shutil.which so Windows finds the .CMD/.ps1 shim.
        # Python's subprocess on Windows doesn't follow PATHEXT for an
        # unqualified argv[0], so a bare "codex" → FileNotFoundError
        # even when the CLI is on PATH.
        codex_resolved = shutil.which(opt.codex_bin) or opt.codex_bin
        cmd = [codex_resolved]
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
            output_last_message=last_path,
        )
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
