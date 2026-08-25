"""one-shot-apply: turn one Spotlights candidate into one reviewable patch.

The cheap arm next to `prep-evolve`: instead of generating a bundle for an
evolutionary search, run a single `claude -p` session in a throwaway git
worktree and hand back `apply.patch` + `APPLY-NOTES.md`. Nothing is executed,
tested, or benchmarked here; the candidate's oracles travel with the patch as
the verification recipe for whoever has the hardware.

See `docs/superpowers/specs/2026-08-20-one-shot-claude-code-fix-design.md`.
"""

from __future__ import annotations

from spotlights_engine.one_shot_apply.api import (
    ApplyArtifact,
    OneShotApplyConfig,
    OneShotApplyInput,
    OneShotApplyResult,
    PromptPreview,
    SkippedApply,
    one_shot_apply,
    render_prompt_block,
)
from spotlights_engine.one_shot_apply.errors import (
    ArtifactWriteError,
    ClaudeUnavailableError,
    NotAGitRepoError,
    OneShotApplyError,
    WorktreeError,
)
from spotlights_engine.one_shot_apply.prompts import build_apply_prompt

__all__ = [
    "ArtifactWriteError",
    "ClaudeUnavailableError",
    "ApplyArtifact",
    "NotAGitRepoError",
    "OneShotApplyConfig",
    "OneShotApplyError",
    "OneShotApplyInput",
    "OneShotApplyResult",
    "PromptPreview",
    "SkippedApply",
    "WorktreeError",
    "build_apply_prompt",
    "one_shot_apply",
    "render_prompt_block",
]
