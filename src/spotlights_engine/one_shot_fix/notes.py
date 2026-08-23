"""`FIX-NOTES.md` — the patch's travelling documentation.

The notes make the patch self-contained for a reviewer on another machine:
what was proposed, what changed and why, which research backed it, **which
commit the patch applies to**, and the recorded oracles as a verification
recipe. Recording the base commit is load-bearing — `git diff` embeds no base,
and a patch applied to the wrong commit either fails or, worse, misapplies.

The notes never claim a result. Nothing was executed here (see the design's
"No verification here"), and saying so plainly is part of the deliverable.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from spotlights_engine.one_shot_fix.scope import (
    candidate_target,
    declared_scope,
    out_of_scope_paths,
    scope_lines,
)
from spotlights_engine.one_shot_fix.worktree import FileChange
from spotlights_engine.prep_evolve.spec import EvolveSpec, Target


def _findings_lines(spec: EvolveSpec) -> str:
    if not spec.findings:
        return "_(none linked to this candidate)_"
    return "\n".join(
        f"- {f.title} ({f.source_type}) — {f.url}\n  {f.technique_summary}"
        for f in spec.findings
    )


def _oracle_lines(target: Target) -> str:
    lines: list[str] = []
    if target.oracles.correctness:
        lines.append("**Correctness:**")
        lines.extend(f"- `{cmd}`" for cmd in target.oracles.correctness)
    else:
        lines.append("**Correctness:** _(none recorded)_")
    perf = target.oracles.performance or "_(none recorded — see the objective)_"
    lines.append(f"\n**Performance:** {perf}")
    return "\n".join(lines)


def _escape_pipe(text: str) -> str:
    """Escape `|` so a path containing one can't break a GFM table row.

    Backticks (used to set the path in code font) do not escape pipes in GFM
    tables; an un-escaped `|` renders as extra columns. GFM resolves `\\|`
    to a literal pipe *before* inline parsing, so this is still correct for
    text that then lands inside a code span.
    """
    return text.replace("|", "\\|")


def _code_span(text: str) -> str:
    """Wrap `text` in a code span that survives backticks inside it.

    Paths in the manifest come from the diff, so an agent-created filename
    containing a backtick (legal on POSIX) reaches this function. A plain
    `` `...` `` wrapper would end the span at that backtick and spill the
    rest of the path into the row as literal text — a broken table exactly
    where a reviewer is looking for an out-of-scope edit.

    Escaping the backtick is not the fix: CommonMark (and so GFM) does not
    honour backslash escapes inside a code span, so `` \\` `` would render
    the backslash. The spec's own mechanism is a delimiter run longer than
    any run in the content, plus one space of padding — stripped on render
    when the content both starts and ends with a space — so a leading or
    trailing backtick cannot merge with the fence.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    if longest == 0:
        return f"`{text}`"
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _scope_cell(change: FileChange, declared: set[str]) -> str:
    """The Scope column, naming *which* end of a rename strayed.

    A rename touches two paths and only one of them is `change.path`. Keyed on
    the destination alone, a rename that drags an undeclared file *into* the
    declared scope reads as plainly "in scope" — while what the patch actually
    does is delete a file the agent was never permitted to touch. Both ends are
    checked, and the label says which, because "the source" and "the
    destination" call for completely different scrutiny from a reviewer.
    """
    offending = out_of_scope_paths(change, declared)
    if not offending:
        return "in scope"
    if change.old_path is None:
        return "**OUT OF SCOPE**"
    source_strayed = change.old_path in offending
    dest_strayed = change.path in offending
    if source_strayed and dest_strayed:
        return "**OUT OF SCOPE** (both ends)"
    return f"**OUT OF SCOPE** ({'source' if source_strayed else 'destination'})"


def _change_row(change: FileChange, declared: set[str]) -> str:
    path = f"{change.old_path} → {change.path}" if change.old_path else change.path
    if change.binary:
        lines = "binary"
    elif not change.counts_known:
        lines = "?"
    else:
        lines = f"+{change.insertions}/-{change.deletions}"
    return (
        f"| {_code_span(_escape_pipe(path))} | {change.change_kind} | {lines} | "
        f"{_scope_cell(change, declared)} |"
    )


def _manifest_section(
    spec: EvolveSpec,
    manifest: Sequence[FileChange],
    patch_produced: bool,
    out_of_scope: Sequence[str],
    collect_error: str | None = None,
) -> str:
    """The patch's real blast radius: every file it touched, kind, size, and scope.

    This is deliberately separate from "In-scope files" above: that section
    documents what was *declared*; this one documents what was *done*. The
    difference between them is exactly what a reviewer needs to catch before
    applying a patch they cannot otherwise see without applying it.

    `out_of_scope` is passed in, never recomputed here: `FixArtifact` (and so
    the CLI warning) carries the same list, and a reader who sees the CLI warn
    about two files must not open the notes and find three. One computation,
    in `api._write_artifacts`, feeds both.
    """
    if collect_error:
        # "The diff failed" and "the agent changed nothing" are different
        # facts, and only one of them is knowable here.
        return (
            "## Files changed\n\n"
            "_(unknown — the diff could not be taken, so there is no per-file "
            "breakdown; see \"Outcome\" below)_\n"
        )

    if not patch_produced:
        return (
            "## Files changed\n\n"
            "_(no patch was produced — there is no diff for this section to describe)_\n"
        )

    if not manifest:
        # A non-empty patch with an empty manifest means manifest collection
        # itself degraded (see `worktree.collect_patch`'s exception handling)
        # — not that nothing changed. Saying "no patch was produced" here
        # would be a second false claim layered on the first; say plainly
        # that the breakdown is unavailable and point at the real diff.
        return (
            "## Files changed\n\n"
            "_(a patch was produced, but the per-file breakdown could not be "
            "collected — see `fix.patch` directly for what changed)_\n"
        )

    declared = declared_scope(spec)
    rows = "\n".join(_change_row(c, declared) for c in manifest)
    total_files = len(manifest)
    unknown = [c for c in manifest if not c.counts_known]
    # No `or 0` fallback: a missing count summed as zero silently understates
    # the change, which is the false-total bug `counts_known` exists to prevent.
    # `FileChange.line_counts()` returns `None` for exactly the records that
    # have nothing to add (binary, or degraded), and the `is not None` filter
    # narrows the pairs to `int` — so the sum needs neither a fallback nor a
    # `type: ignore`.
    counted = [
        p
        for p in (c.line_counts() for c in manifest if c.counts_known and not c.binary)
        if p is not None
    ]
    total_ins = sum(ins for ins, _ in counted)
    total_del = sum(dels for _, dels in counted)
    plural = "s" if total_files != 1 else ""

    if unknown:
        # Never print a totals number that silently excludes files whose
        # counts are unknown — that understates the real change. Say so
        # explicitly instead of a clean-looking (but false) "+X/-Y".
        unknown_plural = "s" if len(unknown) != 1 else ""
        totals_line = (
            f"**Totals:** {total_files} file{plural} changed. Line counts "
            f"unavailable for {len(unknown)} file{unknown_plural}; known "
            f"subset: +{total_ins}/-{total_del}."
        )
    else:
        totals_line = f"**Totals:** {total_files} file{plural} changed, +{total_ins}/-{total_del}."

    section = f"""## Files changed

What the diff actually touched, derived from the patch itself — not from the
declared scope above. Compare it against "In-scope files": any disagreement
is exactly what a reviewer needs to see before applying this patch.

| File | Change | Lines | Scope |
| --- | --- | --- | --- |
{rows}

{totals_line}
"""
    if out_of_scope:
        listed = ", ".join(_code_span(p) for p in out_of_scope)
        section += f"""
> **This patch touches files outside the declared scope.** The agent was
> instructed to edit only the files under "In-scope files" above, but the
> diff also touches: {listed}. That was not declared and not permitted —
> review those hunks specifically, on their own merits, before applying
> anything in this patch. A path listed here as a rename's **source** is one
> the patch *removes* from the repo, which the Change column's "renamed" can
> otherwise make look routine.
"""
    return section


def _outcome_section(
    *,
    repo: Path,
    base_sha: str,
    change_summary: str | None,
    patch_produced: bool,
    agent_error: str | None,
    target: Target,
    collect_error: str | None,
) -> str:
    summary = change_summary.strip() if change_summary else "_(the agent left no summary)_"
    if agent_error:
        # `_code_span`, not a bare `` `...` ``: `agent_error` carries the tail of
        # the agent's own stderr (see `claude_exec.run_fix_claude`), so its
        # content is arbitrary CLI output. One backtick in there closes a plain
        # span early and spills the rest of the error into the document as
        # prose — precisely the line a reader turns to this section for.
        summary = f"{summary}\n\n**Agent session error:** {_code_span(agent_error)}"

    if collect_error:
        # The one outcome this document must not describe as "no patch was
        # produced": the session ran, may well have edited files, and the diff
        # that would have captured them failed. Those edits are gone with the
        # worktree, and saying so is the only honest report available.
        return (
            "## Outcome\n\n"
            "**No patch could be collected — this is a failure, not a decision.** "
            "The agent session finished, but taking the diff of its worktree "
            "failed, so what it changed is unknown. The throwaway worktree has "
            "since been removed, so those edits are not recoverable: re-run "
            "this candidate.\n\n"
            f"**Patch collection error:** {_code_span(collect_error)}\n\n"
            "The agent's own summary, if it left one, follows — treat it as a "
            "claim about work that was never captured.\n\n"
            f"{summary}\n"
        )

    if not patch_produced:
        return (
            "## Outcome\n\n"
            "**No patch was produced.** The agent made no in-scope edit. Its "
            "reasoning follows; a missing patch is a legitimate outcome and is "
            "preferable to one that cannot be trusted.\n\n"
            f"{summary}\n"
        )

    # *Every* recorded correctness oracle, not just the first. This block is
    # what a reviewer copy-pastes, so a command omitted here is a suite that
    # never runs against the patched tree — while "Recorded oracles" above
    # still lists it, leaving the patch looking checked against tests nobody
    # ran. The oracles are recorded as a list precisely because a candidate can
    # name more than one.
    oracle_cmds = target.oracles.correctness
    oracle_lines = "\n".join(
        oracle_cmds or ["# (no correctness oracle was recorded for this candidate)"]
    )
    oracle_intro = (
        "# the recorded correctness oracles — run them all on a machine that can:"
        if len(oracle_cmds) > 1
        else "# the recorded correctness oracle — run it on a machine that can:"
    )
    # These lines are copy-pasted into a shell, so the repo path has to be
    # shell-quoted: an unquoted `/home/alice/my projects/vllm` makes the shell
    # split the argument at the space, and `git -C /home/alice/my` fails (or,
    # with an unlucky directory layout, targets the wrong repo). `shlex.quote`
    # leaves an ordinary path untouched, so the common case reads the same.
    repo_arg = shlex.quote(str(repo))
    return f"""## What changed and why

{summary}

## Applying and verifying this patch

Run this from the directory containing this file — the same directory
`fix.patch` sits in:

```bash
git -C {repo_arg} checkout {base_sha}
git -C {repo_arg} apply --check "$PWD/fix.patch" && git -C {repo_arg} apply "$PWD/fix.patch"

{oracle_intro}
{oracle_lines}
```

If the patch does not apply cleanly, `git -C {repo_arg} apply -3 "$PWD/fix.patch"`
falls back to a three-way merge. Without git, `patch -p1 < fix.patch` works
from the repo root.
"""


def render_fix_notes(
    *,
    spec: EvolveSpec,
    candidate_id: str,
    module_qn: str,
    base_sha: str,
    repo: Path,
    change_summary: str | None,
    patch_produced: bool,
    agent_error: str | None,
    manifest: Sequence[FileChange],
    out_of_scope: Sequence[str],
    collect_error: str | None = None,
) -> str:
    """Render `FIX-NOTES.md` for one candidate.

    `manifest` is the per-file breakdown of what the patch actually touched
    (see `worktree.collect_patch`) — always derived from the diff, never from
    `spec.targets`, so it can surface a patch that strayed outside scope.

    `out_of_scope` is that comparison's verdict, computed once by the caller
    (`api._write_artifacts`) and shared with `FixArtifact`, so the notes and
    the CLI warning cannot disagree.

    `collect_error` outranks `patch_produced`: when the diff itself could not
    be taken, an empty patch says nothing about what the agent did, so both
    the manifest and the outcome sections report the collection failure
    instead of the (unknowable) claim that no edit was made.
    """
    target = candidate_target(spec)
    return f"""# Fix notes — {candidate_id}

- **Candidate:** `{candidate_id}`
- **Module:** `{module_qn}`
- **Repo:** `{repo}`
- **Base commit:** `{base_sha}`
- **Objective:** {spec.objective.goal} (direction: {spec.objective.direction})

## In-scope files

{scope_lines(spec)}

{_manifest_section(spec, manifest, patch_produced, out_of_scope, collect_error)}

## The proposal

{target.description}

**Current approach:** {target.current_approach}

**Why it was worth changing:** {target.evolve_rationale}

## Research findings used

{_findings_lines(spec)}

## Recorded oracles

These are carried forward **verbatim** from the candidate. They are the
verification recipe for a machine that can run them.

{_oracle_lines(target)}

> **Nothing in this directory was verified.** No test was run, no benchmark was
> measured, no build was attempted. The patch was produced in a fresh detached
> worktree with no virtualenv and no compiled extensions, on a machine that may
> lack the hardware the performance oracle needs. Treat this as a proposal
> faithfully implemented — not as a measured win.

{_outcome_section(
    repo=repo,
    base_sha=base_sha,
    change_summary=change_summary,
    patch_produced=patch_produced,
    agent_error=agent_error,
    collect_error=collect_error,
    target=target,
)}"""


__all__ = ["render_fix_notes"]
