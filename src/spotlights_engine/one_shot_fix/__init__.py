"""one-shot-fix: turn one Spotlights candidate into one reviewable patch.

The cheap arm next to `prep-evolve`: instead of generating a bundle for an
evolutionary search, run a single `claude -p` session in a throwaway git
worktree and hand back `fix.patch` + `FIX-NOTES.md`. Nothing is executed,
tested, or benchmarked here; the candidate's oracles travel with the patch as
the verification recipe for whoever has the hardware.

See `docs/superpowers/specs/2026-08-20-one-shot-claude-code-fix-design.md`.
"""

from __future__ import annotations

from spotlights_engine.one_shot_fix.errors import (
    NotAGitRepoError,
    OneShotFixError,
    WorktreeError,
)

__all__ = [
    "NotAGitRepoError",
    "OneShotFixError",
    "WorktreeError",
]
