"""Opt-in live smoke tests for expanded deep-research CLI agents."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from spotlights_engine.cli_agents import CliAgentRequest, StructuredOutputSpec
from spotlights_engine.cli_agents.claude_cli import ClaudeCliAgent
from spotlights_engine.cli_agents.codex_cli import CodexCliAgent
from spotlights_engine.cli_agents.config import ClaudeCliConfig, CodexCliConfig, GeminiCliConfig
from spotlights_engine.cli_agents.gemini_cli import GeminiCliAgent


class LiveSmokePayload(BaseModel):
    ok: bool = Field(description="True when the live agent completed the smoke task.")
    label: str = Field(description="Agent-specific smoke label.")


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_CLI_AGENT_TESTS") != "1" or not os.getenv("LITELLM_API_KEY"),
    reason="set RUN_LIVE_CLI_AGENT_TESTS=1 and LITELLM_API_KEY to run live CLI smoke tests",
)


@pytest.mark.parametrize(
    ("name", "agent"),
    [
        ("codex-gpt55", CodexCliAgent(CodexCliConfig.ibm_litellm(profile="gpt55"))),
        ("claude-opus-4-7", ClaudeCliAgent(ClaudeCliConfig.ibm_litellm())),
        (
            "gemini-3.1-pro",
            GeminiCliAgent(
                GeminiCliConfig.ibm_litellm(
                    model="gcp/gemini-3.1-pro-preview"
                )
            ),
        ),
    ],
)
def test_live_cli_agent_structured_smoke(name: str, agent: object, tmp_path: Path) -> None:
    settings = tmp_path / "gemini-settings.json"
    settings.write_text(
        '{"security":{"auth":{"selectedType":"gemini-api-key"}}}',
        encoding="utf-8",
    )
    if name.startswith("gemini"):
        agent.config = agent.config.with_settings_path(settings)  # type: ignore[attr-defined]

    result = agent.run(  # type: ignore[attr-defined]
        CliAgentRequest(
            prompt=f"Return a JSON object with ok=true and label={name!r}.",
            cwd=tmp_path,
            allow_network=True,
            timeout_seconds=180,
            structured_output=StructuredOutputSpec.from_pydantic(LiveSmokePayload),
        )
    )

    result.raise_for_status()
    payload = result.require_structured()
    assert payload.ok is True
    assert payload.label == name
