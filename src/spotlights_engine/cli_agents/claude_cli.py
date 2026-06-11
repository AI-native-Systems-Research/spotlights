"""Claude Code CLI implementation for cli_agents."""

from __future__ import annotations

import json

from spotlights_engine.cli_agents.config import ClaudeCliConfig, base_env
from spotlights_engine.cli_agents.interface import (
    CliAgentRequest,
    CliAgentResult,
    effective_output_format,
    run_subprocess_agent,
)
from spotlights_engine.cli_agents.structured import apply_structured_prompt


class ClaudeCliAgent:
    runtime = "claude"

    def __init__(self, config: ClaudeCliConfig | None = None) -> None:
        self.config = config or ClaudeCliConfig()

    def build_command(self, request: CliAgentRequest) -> list[str]:
        model = request.model if request.model is not None else self.config.model
        if request.allow_write:
            permission = self.config.permission_write
        elif request.allow_network:
            permission = self.config.permission_network
        else:
            permission = self.config.permission_read
        cmd = [self.config.bin, "--print", "--output-format", effective_output_format(request)]
        if model:
            cmd += ["--model", model]
        if permission:
            cmd += ["--permission-mode", permission]
        if request.allow_network and self.config.network_allowed_tools:
            cmd += ["--allowedTools", ",".join(self.config.network_allowed_tools)]
        if request.structured_output is not None:
            cmd += [
                "--json-schema",
                json.dumps(request.structured_output.native_json_schema(), ensure_ascii=False),
            ]
        if request.system_prompt:
            cmd += ["--system-prompt", request.system_prompt]
        cmd += [str(arg) for arg in request.extra_args]
        return cmd

    def build_env(self, request: CliAgentRequest) -> dict[str, str]:
        inject: dict[str, str] = dict(request.env)
        if self.config.anthropic_base_url:
            inject["ANTHROPIC_BASE_URL"] = self.config.anthropic_base_url
        if self.config.anthropic_auth_token_env:
            token = request.env.get(self.config.anthropic_auth_token_env) or __import__(
                "os"
            ).getenv(self.config.anthropic_auth_token_env)
            if token:
                inject["ANTHROPIC_AUTH_TOKEN"] = token
        if self.config.anthropic_api_key_env:
            key = request.env.get(self.config.anthropic_api_key_env) or __import__("os").getenv(
                self.config.anthropic_api_key_env
            )
            if key:
                inject["ANTHROPIC_API_KEY"] = key
        if self.config.disable_experimental_betas:
            inject["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
        env = base_env(keep=self.config.keep_env, inject=inject)
        for key in self.config.unset_env:
            env.pop(key, None)
        return env

    def run(self, request: CliAgentRequest) -> CliAgentResult:
        return run_subprocess_agent(
            runtime="claude",
            command=self.build_command(request),
            env=self.build_env(request),
            cwd=request.cwd,
            timeout_seconds=request.timeout_seconds,
            stdin=apply_structured_prompt(
                request.prompt,
                request.structured_output,
                include_schema=False,
            ),
            structured_output=request.structured_output,
        )
