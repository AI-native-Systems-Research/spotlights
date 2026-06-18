"""Module deep-research step for Spotlights."""

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult, ModuleResearchRunner
from spotlights_engine.module_deep_research.api import research_module
from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient, ClaudeExecOptions
from spotlights_engine.module_deep_research.codex_exec import (
    CodexExecClient,
    CodexExecOptions,
    CodexExecResult,
)
from spotlights_engine.module_deep_research.gemini_exec import GeminiExecClient, GeminiExecOptions
from spotlights_engine.module_deep_research.prompts import render_module_deep_research_prompt
from spotlights_engine.module_deep_research.validation import parse_module_deep_research_output

__all__ = [
    "AgentExecResult",
    "ClaudeExecClient",
    "ClaudeExecOptions",
    "CodexExecClient",
    "CodexExecOptions",
    "CodexExecResult",
    "GeminiExecClient",
    "GeminiExecOptions",
    "ModuleResearchRunner",
    "parse_module_deep_research_output",
    "render_module_deep_research_prompt",
    "research_module",
]
