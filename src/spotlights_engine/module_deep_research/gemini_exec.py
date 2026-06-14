"""Small subprocess wrapper around Gemini CLI headless mode."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult


class GeminiExecOptions(BaseModel):
    """Options for running Gemini CLI in non-interactive mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: Path | str = Field(default_factory=Path.cwd)
    gemini_bin: str = "gemini"
    model: str | None = None
    approval_mode: str = "plan"
    timeout_seconds: int | None = None
    extra_args: Sequence[str] = Field(default_factory=tuple)
    env: Mapping[str, str] | None = None


class GeminiExecClient:
    """Run Gemini CLI in headless mode from Python."""

    name = "gemini"

    def __init__(self, options: GeminiExecOptions | None = None) -> None:
        self.options = options or GeminiExecOptions()

    def build_command(self) -> list[str]:
        opt = self.options
        cmd = [
            opt.gemini_bin,
            "--prompt",
            "",
            "--output-format",
            "json",
            "--approval-mode",
            opt.approval_mode,
            "--skip-trust",
        ]
        if opt.model:
            cmd += ["--model", opt.model]
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

    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, str) and response.strip():
            return response
    return text


__all__ = ["GeminiExecClient", "GeminiExecOptions"]
