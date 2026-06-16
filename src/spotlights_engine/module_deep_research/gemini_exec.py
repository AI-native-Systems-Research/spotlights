"""Small subprocess wrapper around Gemini CLI headless mode."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.module_deep_research.agent_exec import (
    AgentExecResult,
    resolve_cli_executable,
)

DEFAULT_LITELLM_GEMINI_MODEL = "gcp/gemini-3.1-pro-preview"


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
    managed_settings: Mapping[str, Any] | None = None
    skip_trust: bool = False
    timeout_seconds: int | None = None
    extra_args: Sequence[str] = Field(default_factory=tuple)
    env: Mapping[str, str] | None = None

    @classmethod
    def litellm_proxy(
        cls,
        *,
        base_url: str,
        cwd: Path | str | None = None,
        model: str = DEFAULT_LITELLM_GEMINI_MODEL,
        api_key_env: str = "LITELLM_API_KEY",
        settings_path: Path | str | None = None,
        timeout_seconds: int | None = None,
        approval_mode: str = "yolo",
        env: Mapping[str, str] | None = None,
        configure_web_tools: bool = True,
        web_utility_model: str | None = None,
    ) -> GeminiExecOptions:
        """Return explicit Gemini CLI settings for a LiteLLM gateway."""
        return cls(
            cwd=Path.cwd() if cwd is None else cwd,
            model=model,
            approval_mode=approval_mode,
            gemini_base_url=base_url.rstrip("/"),
            gemini_api_key_env=api_key_env,
            api_key_auth_mechanism="bearer",
            settings_path=settings_path,
            managed_settings=(
                litellm_settings_payload(
                    model=model,
                    web_utility_model=web_utility_model or model,
                )
                if configure_web_tools and settings_path is None
                else None
            ),
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
            resolve_cli_executable(opt.gemini_bin),
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

    def build_env(self, *, settings_path: Path | str | None = None) -> dict[str, str]:
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
        effective_settings_path = settings_path or opt.settings_path
        if effective_settings_path:
            env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(effective_settings_path)
        return env

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult:
        with _managed_settings_file(self.options.managed_settings) as settings_path:
            completed = subprocess.run(
                self.build_command(),
                input=prompt,
                capture_output=True,
                text=True,
                cwd=str(Path(self.options.cwd).expanduser().resolve()),
                env=self.build_env(settings_path=settings_path),
                timeout=self.options.timeout_seconds,
                check=False,
            )
        result = AgentExecResult(
            command=self.build_command(),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            final_message=_final_message(completed.stdout),
        )
        if check:
            result.raise_for_status()
        return result


def litellm_settings_payload(
    *,
    model: str,
    web_utility_model: str | None = None,
) -> dict[str, Any]:
    """Return Gemini CLI settings that route internal web-tool aliases via LiteLLM.

    Gemini CLI implements `google_web_search` and `web_fetch` with internal model
    aliases (`web-search`, `web-fetch`, and `web-fetch-fallback`). LiteLLM
    deployments often expose provider-qualified public names, so remap those
    aliases directly instead of relying on Gemini CLI's unqualified defaults.
    """
    utility_model = web_utility_model or model
    return {
        "model": {"name": model},
        "modelConfigs": {
            "customAliases": {
                "web-search": {
                    "extends": "base",
                    "modelConfig": {
                        "model": utility_model,
                        "generateContentConfig": {"tools": [{"googleSearch": {}}]},
                    },
                },
                "web-fetch": {
                    "extends": "base",
                    "modelConfig": {
                        "model": utility_model,
                        "generateContentConfig": {"tools": [{"urlContext": {}}]},
                    },
                },
                "web-fetch-fallback": {
                    "extends": "base",
                    "modelConfig": {"model": utility_model},
                },
            }
        },
        "advanced": {"ignoreLocalEnv": True},
        "security": {"auth": {"selectedType": "gemini-api-key"}},
    }


def write_litellm_settings(
    path: Path | str,
    *,
    model: str,
    web_utility_model: str | None = None,
) -> Path:
    """Write a Gemini CLI settings file for LiteLLM-backed web tools."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            litellm_settings_payload(model=model, web_utility_model=web_utility_model),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


class _managed_settings_file:
    def __init__(self, payload: Mapping[str, Any] | None) -> None:
        self.payload = payload
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path | None:
        if self.payload is None:
            return None
        self._tmpdir = tempfile.TemporaryDirectory(prefix="spotlights-gemini-")
        self.path = Path(self._tmpdir.name) / "settings.json"
        self.path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.path

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()


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


__all__ = [
    "GeminiExecClient",
    "GeminiExecOptions",
    "litellm_settings_payload",
    "write_litellm_settings",
]
