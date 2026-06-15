"""Small subprocess wrapper around non-interactive `claude -p`."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.module_deep_research.agent_exec import (
    AgentExecResult,
    resolve_cli_executable,
)

CLAUDE_RESEARCH_TOOLS = (
    "Read",
    "Grep",
    "Glob",
    "LS",
    "Bash",
    "WebSearch",
    "WebFetch",
)


class ClaudeExecOptions(BaseModel):
    """Options for running Claude in non-interactive mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: Path | str = Field(default_factory=Path.cwd)
    claude_bin: str = "claude"
    model: str | None = None
    permission_mode: str = "acceptEdits"
    allowed_tools: Sequence[str] = CLAUDE_RESEARCH_TOOLS
    max_turns: int = 8
    timeout_seconds: int | None = None
    extra_args: Sequence[str] = Field(default_factory=tuple)
    env: Mapping[str, str] | None = None


class ClaudeExecClient:
    """Run Claude Code in non-interactive print mode from Python."""

    name = "claude"

    def __init__(self, options: ClaudeExecOptions | None = None) -> None:
        self.options = options or ClaudeExecOptions()

    def build_command(self) -> list[str]:
        opt = self.options
        cmd = [
            resolve_cli_executable(opt.claude_bin),
            "-p",
            "--output-format",
            "json",
            "--permission-mode",
            opt.permission_mode,
            "--max-turns",
            str(opt.max_turns),
        ]
        if opt.model:
            cmd += ["--model", opt.model]
        if opt.allowed_tools:
            cmd += ["--tools", ",".join(opt.allowed_tools)]
            cmd += ["--allowedTools", ",".join(opt.allowed_tools)]
        cmd += list(opt.extra_args)
        return cmd

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult:
        cmd = self.build_command()
        env = os.environ.copy()
        if self.options.env:
            env.update(dict(self.options.env))

        completed = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=str(Path(self.options.cwd).expanduser().resolve()),
            env=env,
            timeout=self.options.timeout_seconds,
            check=False,
        )
        result = AgentExecResult(
            command=cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            final_message=_final_message(completed.stdout),
        )
        if check:
            result.raise_for_status()
        return result


def _final_message(stdout: str) -> str | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    if not isinstance(payload, dict):
        return text

    structured = payload.get("structured_output")
    if isinstance(structured, (dict, list)):
        return json.dumps(structured)

    result = payload.get("result")
    if isinstance(result, str) and result.strip():
        return result

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = [item.get("text", "") for item in content if isinstance(item, dict)]
            joined = "".join(chunks)
            if joined.strip():
                return joined

    return None


__all__ = ["ClaudeExecClient", "ClaudeExecOptions"]
