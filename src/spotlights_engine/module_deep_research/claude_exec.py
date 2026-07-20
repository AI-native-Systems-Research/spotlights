"""Small subprocess wrapper around non-interactive `claude -p`."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import AgentUsage, claude_usage_from_payload
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
    "WebFetch",
    "WebSearch",
)


class ClaudeExecOptions(BaseModel):
    """Options for running Claude in non-interactive mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: Path | str = Field(default_factory=Path.cwd)
    claude_bin: str = "claude"
    model: str | None = None
    permission_mode: str = "acceptEdits"
    allowed_tools: Sequence[str] = CLAUDE_RESEARCH_TOOLS
    max_turns: int = 80
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
        from spotlights_engine._backlog import enabled as _bl_enabled
        stream = _bl_enabled()
        cmd = [
            resolve_cli_executable(opt.claude_bin),
            "-p",
            "--output-format",
            "stream-json" if stream else "json",
            "--permission-mode",
            opt.permission_mode,
            "--max-turns",
            str(opt.max_turns),
        ]
        if stream:
            cmd += ["--verbose", "--include-partial-messages"]
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
        # `--output-format json` puts text and usage on the same payload;
        # parse it once so usage is captured before the stream is discarded.
        payload = _parse_payload(completed.stdout)
        result = AgentExecResult(
            command=cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            final_message=_final_message(completed.stdout, payload=payload),
            usage=_usage(payload, fallback_model=self.options.model),
        )
        if check:
            result.raise_for_status()
        return result


def _parse_payload(stdout: str) -> dict | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    # stream-json: JSONL — pick the terminal event with type=result.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(p, dict) and p.get("type") == "result":
            return p
    return None


def _usage(payload: dict | None, *, fallback_model: str | None) -> AgentUsage | None:
    if payload is None:
        return None
    usage = claude_usage_from_payload(payload)
    if usage is not None and usage.model is None and fallback_model:
        usage = usage.model_copy(update={"model": fallback_model})
    return usage


def _final_message(stdout: str, payload: dict | None = None) -> str | None:
    text = stdout.strip()
    if not text:
        return None
    if payload is None:
        payload = _parse_payload(text)
        if payload is None:
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
