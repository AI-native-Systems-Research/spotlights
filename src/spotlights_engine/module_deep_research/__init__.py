"""Module deep-research step for Spotlights."""

from spotlights_engine.module_deep_research.api import research_module
from spotlights_engine.module_deep_research.codex_exec import (
    CodexExecClient,
    CodexExecOptions,
    CodexExecResult,
)
from spotlights_engine.module_deep_research.expanded import ExpandedResearchConfig
from spotlights_engine.module_deep_research.prompts import render_module_deep_research_prompt
from spotlights_engine.module_deep_research.validation import parse_module_deep_research_output

__all__ = [
    "CodexExecClient",
    "CodexExecOptions",
    "CodexExecResult",
    "ExpandedResearchConfig",
    "parse_module_deep_research_output",
    "render_module_deep_research_prompt",
    "research_module",
]
