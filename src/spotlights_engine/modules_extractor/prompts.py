"""Prompt loading for the modules extractor.

The prompt template lives in `prompts_data/extraction.md` as package data and
is loaded once via `importlib.resources` so behavior does not depend on cwd.
The template is parameter-free — module discovery is structurally identical
for every repo, and `repo_path` is delivered to the agent via the subprocess
`cwd`, not the prompt.
"""

from __future__ import annotations

from importlib import resources


def _load(name: str) -> str:
    return (
        resources.files("spotlights_engine.modules_extractor.prompts_data")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


EXTRACTION_PROMPT: str = _load("extraction.md")


__all__ = ["EXTRACTION_PROMPT"]
