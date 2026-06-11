"""Gemini CLI implementation for cli_agents."""

from __future__ import annotations

import os

from spotlights_engine.cli_agents.config import GeminiCliConfig, base_env
from spotlights_engine.cli_agents.interface import (
    CliAgentRequest,
    CliAgentResult,
    effective_output_format,
    run_subprocess_agent,
)
from spotlights_engine.cli_agents.structured import apply_structured_prompt


class GeminiCliAgent:
    runtime = "gemini"

    def __init__(self, config: GeminiCliConfig | None = None) -> None:
        self.config = config or GeminiCliConfig()

    def build_command(self, request: CliAgentRequest) -> list[str]:
        model = request.model if request.model is not None else self.config.model
        if request.allow_write:
            approval = self.config.approval_write
        elif request.allow_network:
            approval = self.config.approval_network
        else:
            approval = self.config.approval_read
        cmd = [
            self.config.bin,
            "--prompt",
            apply_structured_prompt(request.prompt, request.structured_output),
            "--output-format",
            effective_output_format(request),
        ]
        if model:
            cmd += ["--model", model]
        if approval:
            cmd += ["--approval-mode", approval]
        if self.config.skip_trust:
            cmd += ["--skip-trust"]
        cmd += [str(arg) for arg in request.extra_args]
        return cmd

    def build_env(self, request: CliAgentRequest) -> dict[str, str]:
        inject: dict[str, str] = dict(request.env)
        if self.config.gemini_base_url:
            inject["GOOGLE_GEMINI_BASE_URL"] = self.config.gemini_base_url
        if self.config.gemini_api_key_env:
            key = request.env.get(self.config.gemini_api_key_env) or os.getenv(
                self.config.gemini_api_key_env
            )
            if key:
                inject["GEMINI_API_KEY"] = key
        if self.config.api_key_auth_mechanism:
            inject["GEMINI_API_KEY_AUTH_MECHANISM"] = self.config.api_key_auth_mechanism
        if self.config.settings_path:
            inject["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(self.config.settings_path)
        return base_env(keep=self.config.keep_env, inject=inject)

    def run(self, request: CliAgentRequest) -> CliAgentResult:
        return run_subprocess_agent(
            runtime="gemini",
            command=self.build_command(request),
            env=self.build_env(request),
            cwd=request.cwd,
            timeout_seconds=request.timeout_seconds,
            stdin=None,
            structured_output=request.structured_output,
        )
