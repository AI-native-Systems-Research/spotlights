"""Error hierarchy for the one-shot fix stage.

Resolution, repo-path, and staleness failures reuse
`spotlights_engine.prep_evolve.errors` — the fix stage runs the same resolver
and the same staleness gate, so it raises the same types. What is new here is
git-worktree failure: `fix` requires a real git checkout, which `prep-evolve`
does not.

The CLI catches `(PrepEvolveError, OneShotFixError)` once and maps both to a
clean stderr message plus a nonzero exit.
"""

from __future__ import annotations


class OneShotFixError(Exception):
    """Base class for one-shot-fix-specific failures."""


class NotAGitRepoError(OneShotFixError):
    """The target repo is not a git checkout, so no worktree can be made.

    Unlike `prep-evolve`, which tolerates a non-git target and records a `None`
    commit, `fix` fails loudly: the base commit is what makes the emitted patch
    applicable, and a worktree cannot exist without one.
    """


class WorktreeError(OneShotFixError):
    """A `git worktree` / `git diff` invocation failed."""


__all__ = [
    "NotAGitRepoError",
    "OneShotFixError",
    "WorktreeError",
]
