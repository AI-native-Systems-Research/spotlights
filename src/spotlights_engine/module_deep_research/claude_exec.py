"""Small subprocess wrapper around non-interactive `claude -p`."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import AgentUsage, claude_usage_from_payload
from spotlights_engine.llm_session.env import clean_env
from spotlights_engine.llm_session.transport import final_message_text
from spotlights_engine.module_deep_research.agent_exec import (
    AgentExecResult,
    guarded_run,
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
        # Centralized env scrubbing (drops ANTHROPIC_AUTH_TOKEN — the auth-leak
        # fix — plus the proxy/base-url vars that used to bypass keychain auth).
        env = clean_env()
        if self.options.env:
            env.update(dict(self.options.env))

        def _inner() -> AgentExecResult:
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
            return AgentExecResult(
                command=cmd,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                final_message=_json_output_text(completed.stdout, payload),
                usage=_usage(payload, fallback_model=self.options.model),
            )

        result = guarded_run(_inner, cli="claude", label="module_deep_research.claude")
        if check:
            result.raise_for_status()
        return result


def _parse_payload(stdout: str) -> dict | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _usage(payload: dict | None, *, fallback_model: str | None) -> AgentUsage | None:
    if payload is None:
        return None
    usage = claude_usage_from_payload(payload)
    if usage is not None and usage.model is None and fallback_model:
        usage = usage.model_copy(update={"model": fallback_model})
    return usage


def _json_output_text(stdout: str, payload: dict | None) -> str | None:
    """Extract the assistant's final text/structured output from `--output-format json`.

    Delegates the structured_output → result → message.content fallback to the
    centralized `llm_session.transport.final_message_text`, keeping the
    module-deep-research nuance that a non-JSON stdout is returned verbatim
    (this client tolerates a bare text response, unlike the strict stream-json
    parsers).
    """
    text = stdout.strip()
    if not text:
        return None
    if payload is None:
        # stdout was not a JSON object — return the raw text as-is.
        return text
    return final_message_text(payload) or None


__all__ = ["ClaudeExecClient", "ClaudeExecOptions"]
