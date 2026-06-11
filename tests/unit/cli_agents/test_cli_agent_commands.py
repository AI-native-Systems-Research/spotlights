"""Command-shape tests for local CLI agent adapters."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from spotlights_engine.cli_agents import CliAgentRequest, StructuredOutputSpec
from spotlights_engine.cli_agents.claude_cli import ClaudeCliAgent
from spotlights_engine.cli_agents.codex_cli import CodexCliAgent
from spotlights_engine.cli_agents.config import ClaudeCliConfig, CodexCliConfig, GeminiCliConfig
from spotlights_engine.cli_agents.gemini_cli import GeminiCliAgent


class TinyPayload(BaseModel):
    label: str = Field(description="Smoke-test label.")


def _request(tmp_path: Path) -> CliAgentRequest:
    return CliAgentRequest(
        prompt="return a label",
        cwd=tmp_path,
        allow_network=True,
        structured_output=StructuredOutputSpec.from_pydantic(TinyPayload),
    )


def test_codex_uses_profile_and_native_output_schema(tmp_path: Path) -> None:
    cmd = CodexCliAgent(CodexCliConfig.ibm_litellm(profile="gpt55")).build_command(
        _request(tmp_path)
    )

    assert cmd[:3] == ["codex", "--profile", "gpt55"]
    assert "--output-schema" in cmd
    assert "--search" in cmd
    assert cmd[-1] == "-"


def test_claude_uses_anthropic_schema_flag_and_model(tmp_path: Path) -> None:
    cmd = ClaudeCliAgent(ClaudeCliConfig.ibm_litellm(model="claude-opus-4-7")).build_command(
        _request(tmp_path)
    )

    assert cmd[:3] == ["claude", "--print", "--output-format"]
    assert "claude-opus-4-7" in cmd
    assert "--json-schema" in cmd
    assert "--allowedTools" in cmd


def test_gemini_uses_json_output_without_fake_schema_flag(tmp_path: Path) -> None:
    cmd = GeminiCliAgent(
        GeminiCliConfig.ibm_litellm(model="gcp/gemini-3.1-pro-preview")
    ).build_command(_request(tmp_path))

    assert cmd[0] == "gemini"
    assert "--model" in cmd
    assert "gcp/gemini-3.1-pro-preview" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--json-schema" not in cmd
