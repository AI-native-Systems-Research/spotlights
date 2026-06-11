"""Codex CLI implementation for cli_agents."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from spotlights_engine.cli_agents.config import CodexCliConfig, base_env
from spotlights_engine.cli_agents.interface import (
    CliAgentRequest,
    CliAgentResult,
    effective_output_format,
    run_subprocess_agent,
)
from spotlights_engine.cli_agents.structured import apply_structured_prompt


class CodexCliAgent:
    runtime = "codex"

    def __init__(self, config: CodexCliConfig | None = None) -> None:
        self.config = config or CodexCliConfig()

    def build_command(self, request: CliAgentRequest) -> list[str]:
        cwd = Path(request.cwd or ".").expanduser().resolve()
        profile = request.profile if request.profile is not None else self.config.profile
        model = request.model if request.model is not None else self.config.model
        sandbox = self.config.sandbox_write if request.allow_write else self.config.sandbox_read
        output_path = self.config.output_last_message or Path(
            tempfile.NamedTemporaryFile(
                prefix="cli-agents-codex-last-", suffix=".md", delete=False
            ).name
        )

        cmd = [self.config.bin]
        if model:
            cmd += ["--model", model]
        if profile:
            cmd += ["--profile", profile]
        if self.config.approval:
            cmd += ["--ask-for-approval", self.config.approval]
        if request.allow_network:
            cmd += ["--search"]
        cmd += ["exec", "--cd", str(cwd), "--sandbox", sandbox, "--skip-git-repo-check"]
        if self.config.json_events or effective_output_format(request) in {"json", "stream-json"}:
            cmd += ["--json"]
        if request.structured_output is not None:
            cmd += [
                "--output-schema",
                str(_write_temp_schema_file(request.structured_output.native_json_schema())),
            ]
        cmd += ["--output-last-message", str(output_path)]
        cmd += [str(arg) for arg in request.extra_args]
        cmd.append("-")
        return cmd

    def build_env(self, request: CliAgentRequest) -> dict[str, str]:
        return base_env(keep=self.config.keep_env, inject=request.env)

    def run(self, request: CliAgentRequest) -> CliAgentResult:
        cmd = self.build_command(request)
        output_path = _output_last_message_path(cmd)
        result = run_subprocess_agent(
            runtime="codex",
            command=cmd,
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
        try:
            if output_path and output_path.exists():
                text = output_path.read_text(encoding="utf-8", errors="replace")
                if text:
                    updates = {**result.__dict__, "text": text}
                    if request.structured_output is not None:
                        try:
                            updates["structured_data"] = request.structured_output.parse(text=text)
                            updates["structured_error"] = None
                        except (TypeError, ValueError) as exc:
                            updates["structured_data"] = None
                            updates["structured_error"] = str(exc)
                    return CliAgentResult(**updates)
            return result
        finally:
            schema_path = _output_schema_path(cmd)
            if schema_path is not None:
                schema_path.unlink(missing_ok=True)


def _output_last_message_path(cmd: list[str]) -> Path | None:
    if "--output-last-message" not in cmd:
        return None
    idx = cmd.index("--output-last-message")
    if idx + 1 >= len(cmd):
        return None
    return Path(cmd[idx + 1])


def _output_schema_path(cmd: list[str]) -> Path | None:
    if "--output-schema" not in cmd:
        return None
    idx = cmd.index("--output-schema")
    if idx + 1 >= len(cmd):
        return None
    return Path(cmd[idx + 1])


def _write_temp_schema_file(schema: object) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix="cli-agents-codex-schema-",
        suffix=".json",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as file:
        json.dump(schema, file, ensure_ascii=False, sort_keys=True)
        file.write("\n")
        return Path(file.name)
