"""Subprocess adapters for local agent CLIs."""

from spotlights_engine.cli_agents.interface import CliAgentRequest, CliAgentResult
from spotlights_engine.cli_agents.structured import StructuredOutputSpec

__all__ = [
    "CliAgentRequest",
    "CliAgentResult",
    "StructuredOutputSpec",
]
