"""The one-shot fix prompt — the single source of truth.

`spotlights-engine fix` builds the prompt with `build_fix_prompt` and feeds it
to `claude -p`; `fix --print-prompt` prints the same string for
`/spotlights-fix-candidate` to work from in-session. The prompt is shared so
the two paths cannot drift on the part that matters.

The findings/proposals/target body is `prep_evolve.digest.render_digest`, the
same block every evolver bundle embeds. What this module adds around it is the
fix-specific contract: the worktree to edit, the hard scope boundary, the base
commit, the oracles as a recipe rather than a task, and the instruction to
write a rationale file the collector folds into FIX-NOTES.md.
"""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.one_shot_fix.scope import candidate_target, scope_lines
from spotlights_engine.prep_evolve.digest import render_digest
from spotlights_engine.prep_evolve.spec import EvolveSpec, Target

# The agent writes its rationale here, in the worktree root. `collect_patch`
# reads it and removes it *before* `git add -N`, so it never lands in the
# patch; `render_fix_notes` folds the text into FIX-NOTES.md.
CHANGE_SUMMARY_NAME = "CHANGE-SUMMARY.md"


def _oracle_block(target: Target) -> str:
    """The oracles verbatim. Recorded, never enforced here."""
    lines: list[str] = []
    if target.oracles.correctness:
        lines.append("Correctness oracle (recorded, NOT run here):")
        lines.extend(f"  {cmd}" for cmd in target.oracles.correctness)
    else:
        lines.append("Correctness oracle: (none recorded)")
    if target.oracles.performance:
        lines.append(
            "Performance oracle (recorded, NOT measured here): "
            f"{target.oracles.performance}"
        )
    else:
        lines.append("Performance oracle: (none recorded — see the objective)")
    return "\n".join(lines)


def build_fix_prompt(*, spec: EvolveSpec, worktree: Path) -> str:
    """Render the fix prompt for one candidate.

    `spec.run.repo_path` is the real target repo (identity); `worktree` is the
    detached, throwaway checkout the agent actually edits.
    """
    target = candidate_target(spec)
    base_sha = spec.source_revision.git_commit or "(unknown)"
    return f"""You are implementing ONE proposed optimization in an isolated git worktree.

Working directory: {worktree}
This is a detached worktree of {spec.run.repo_path} at commit {base_sha}.
It is throwaway. It has no build artifacts, no virtualenv, and no compiled
extensions — nothing in it is runnable.

{render_digest(spec)}

## In-scope files — do not edit anything else
{scope_lines(spec)}

## Oracles
{_oracle_block(target)}

## What to do
1. Read the in-scope code, and any other file you need for context.
2. Implement the change described above, editing ONLY the in-scope files.
   Prefer the smallest faithful implementation of the proposal over a rewrite.
3. Write `{CHANGE_SUMMARY_NAME}` in the working-directory root: what you
   changed, why, which findings/proposals you drew on, and anything a reviewer
   should check by hand.

## Hard rules
- Do not run tests. Do not run benchmarks. Do not try to build or install
  anything. This worktree cannot execute them, and a fabricated result is
  worse than none. The oracles above are recorded for whoever has the
  hardware — they are not your task.
- Do not commit, branch, stash, or otherwise run git write commands. The patch
  is collected from your uncommitted working-tree changes.
- Do not edit files outside the in-scope list.
- If the change cannot be made faithfully within scope, make no edit and say
  so in `{CHANGE_SUMMARY_NAME}`, with the reason. A missing patch is a fine
  outcome; a patch that cannot be trusted is not.
""".strip()


__all__ = ["CHANGE_SUMMARY_NAME", "build_fix_prompt"]
