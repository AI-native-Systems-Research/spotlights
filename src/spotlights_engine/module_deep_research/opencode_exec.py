"""Small subprocess wrapper around non-interactive `opencode run`."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import AgentUsage, opencode_usage_from_stream
from spotlights_engine.module_deep_research.agent_exec import (
    AgentExecResult,
    resolve_cli_executable,
)

# Explicit so telemetry/rate keys are deterministic even when OpenCode's event
# stream does not report a model id (see opencode_usage_from_stream).
DEFAULT_OPENCODE_MODEL = "litellm/gemini-2.5-pro"
# Custom read-only research agent (webfetch + websearch, no bash/edit/write).
# See the README's "Web research tooling" section for how to create it.
DEFAULT_OPENCODE_AGENT = "research"


class OpenCodeExecOptions(BaseModel):
    """Options for running OpenCode in non-interactive mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: Path | str = Field(default_factory=Path.cwd)
    opencode_bin: str = "opencode"
    model: str | None = DEFAULT_OPENCODE_MODEL
    agent: str = DEFAULT_OPENCODE_AGENT
    variant: str | None = None
    pure: bool = False
    timeout_seconds: int | None = None
    extra_args: Sequence[str] = Field(default_factory=tuple)
    env: Mapping[str, str] | None = None


class OpenCodeExecClient:
    """Run OpenCode in headless mode from Python."""

    name = "opencode"

    def __init__(self, options: OpenCodeExecOptions | None = None) -> None:
        self.options = options or OpenCodeExecOptions()

    def build_command(self) -> list[str]:
        opt = self.options
        cmd = [
            resolve_cli_executable(opt.opencode_bin),
            "run",
            "--format",
            "json",
            "--agent",
            opt.agent,
        ]
        if opt.model:
            cmd += ["--model", opt.model]
        if opt.variant:
            cmd += ["--variant", opt.variant]
        if opt.pure:
            cmd += ["--pure"]
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
            usage=_usage(completed.stdout, fallback_model=self.options.model),
        )
        if check:
            result.raise_for_status()
        return result


def _usage(stdout: str, *, fallback_model: str | None) -> AgentUsage | None:
    usage = opencode_usage_from_stream(stdout)
    if usage is not None and usage.model is None and fallback_model:
        usage = usage.model_copy(update={"model": fallback_model})
    return usage


def _final_message(stdout: str) -> str | None:
    """Concatenate the `text` parts from OpenCode's NDJSON event stream.

    Falls back to the raw stdout when the stream is unparseable so the block's
    existing `parse_agent_output` still sees the `find-NNNN` payload.
    """
    text = stdout.strip()
    if not text:
        return None

    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "text":
            continue
        part = obj.get("part")
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])

    joined = "".join(chunks)
    if joined.strip():
        return joined
    return text


__all__ = [
    "DEFAULT_OPENCODE_AGENT",
    "DEFAULT_OPENCODE_MODEL",
    "OpenCodeExecClient",
    "OpenCodeExecOptions",
]
