"""Small subprocess wrapper around Gemini CLI headless mode."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult

IBM_LITELLM_PROXY_URL = "https://ete-litellm.ai-models.vpc-int.res.ibm.com"
IBM_GEMINI_MODEL = "gcp/gemini-3.1-pro-preview"


class GeminiExecOptions(BaseModel):
    """Options for running Gemini CLI in non-interactive mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: Path | str = Field(default_factory=Path.cwd)
    gemini_bin: str = "gemini"
    model: str | None = None
    approval_mode: str = "plan"
    gemini_base_url: str | None = None
    gemini_api_key_env: str | None = None
    api_key_auth_mechanism: Literal["x-goog-api-key", "bearer"] | None = None
    settings_path: Path | str | None = None
    skip_trust: bool = False
    timeout_seconds: int | None = None
    extra_args: Sequence[str] = Field(default_factory=tuple)
    env: Mapping[str, str] | None = None

    @classmethod
    def ibm_litellm(
        cls,
        *,
        cwd: Path | str | None = None,
        model: str = IBM_GEMINI_MODEL,
        settings_path: Path | str | None = None,
        timeout_seconds: int | None = None,
        approval_mode: str = "yolo",
        env: Mapping[str, str] | None = None,
    ) -> GeminiExecOptions:
        """Return the explicit IBM LiteLLM configuration used by live research runs."""
        return cls(
            cwd=Path.cwd() if cwd is None else cwd,
            model=model,
            approval_mode=approval_mode,
            gemini_base_url=IBM_LITELLM_PROXY_URL,
            gemini_api_key_env="LITELLM_API_KEY",
            api_key_auth_mechanism="bearer",
            settings_path=settings_path,
            skip_trust=True,
            timeout_seconds=timeout_seconds,
            env=env,
        )


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
        ]
        if opt.model:
            cmd += ["--model", opt.model]
        if opt.skip_trust:
            cmd += ["--skip-trust"]
        cmd += list(opt.extra_args)
        return cmd

    def build_env(self) -> dict[str, str]:
        opt = self.options
        env = os.environ.copy()
        if opt.env:
            env.update(dict(opt.env))

        if opt.gemini_base_url:
            env["GOOGLE_GEMINI_BASE_URL"] = opt.gemini_base_url
        if opt.gemini_api_key_env:
            key = env.get(opt.gemini_api_key_env)
            if key:
                env["GEMINI_API_KEY"] = key
        if opt.api_key_auth_mechanism:
            env["GEMINI_API_KEY_AUTH_MECHANISM"] = opt.api_key_auth_mechanism
        if opt.settings_path:
            env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(opt.settings_path)
        return env

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult:
        cmd = self.build_command()

        completed = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=str(Path(self.options.cwd).expanduser().resolve()),
            env=self.build_env(),
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
