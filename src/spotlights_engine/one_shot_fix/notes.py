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

import posixpath
from collections.abc import Sequence
from pathlib import Path

from spotlights_engine.one_shot_fix.worktree import FileChange
from spotlights_engine.prep_evolve.spec import EvolveSpec, Target


def _candidate_target(spec: EvolveSpec) -> Target:
    for t in spec.targets:
        if t.scope_kind == "candidate":
            return t
    return spec.targets[0]


def _scope_lines(spec: EvolveSpec) -> str:
    """Same no-backtick rule as `prompts._scope_block` — see the note there."""
    lines: list[str] = []
    for t in spec.targets:
        if t.scope_kind == "candidate" and t.line_start is not None:
            sym = f" — {t.symbol}" if t.symbol else ""
            lines.append(f"- {t.file}:{t.line_start}-{t.line_end}{sym}")
        else:
            role = f" — {t.role}" if t.role else ""
            lines.append(f"- {t.file} (whole file{role})")
    return "\n".join(lines)


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


def _normalize_scope_path(path: str) -> str:
    """Canonicalize a repo-relative path for scope comparison.

    `validate_target.py` only checks containment and existence, so a declared
    target spelled `./pkg/attn/tile.py` or `pkg/attn/../attn/tile.py`
    validates and is stored verbatim — while git always emits the canonical
    POSIX relative path in the diff. Without normalizing both sides, such a
    target falsely reports **OUT OF SCOPE** against the candidate's own file
    (over-reporting only, never under — but a callout that cries wolf is one
    reviewers learn to skip). Backslashes are normalized too, in case a
    target was recorded with Windows-style separators.
    """
    return posixpath.normpath(path.replace("\\", "/"))


def _escape_pipe(text: str) -> str:
    """Escape `|` so a path containing one can't break a GFM table row.

    Backticks (used to set the path in code font) do not escape pipes in GFM
    tables; an un-escaped `|` renders as extra columns.
    """
    return text.replace("|", "\\|")


def _declared_scope(spec: EvolveSpec) -> set[str]:
    return {_normalize_scope_path(t.file) for t in spec.targets}


def out_of_scope_files(spec: EvolveSpec, manifest: Sequence[FileChange]) -> list[str]:
    """Files the patch actually touched that are not in `spec.targets`' declared scope.

    This is the whole point of the manifest: derived from the diff, so it can
    (and does, when the agent strays) disagree with what was declared. Shared
    between the notes renderer and the CLI so the two never compute it twice.
    """
    declared = _declared_scope(spec)
    return [c.path for c in manifest if _normalize_scope_path(c.path) not in declared]


def _change_row(change: FileChange, declared: set[str]) -> str:
    path = f"{change.old_path} → {change.path}" if change.old_path else change.path
    path = _escape_pipe(path)
    if change.binary:
        lines = "binary"
    elif not change.counts_known:
        lines = "?"
    else:
        lines = f"+{change.insertions}/-{change.deletions}"
    scope = "in scope" if _normalize_scope_path(change.path) in declared else "**OUT OF SCOPE**"
    return f"| `{path}` | {change.change_kind} | {lines} | {scope} |"


def _manifest_section(
    spec: EvolveSpec, manifest: Sequence[FileChange], patch_produced: bool
) -> str:
    """The patch's real blast radius: every file it touched, kind, size, and scope.

    This is deliberately separate from "In-scope files" above: that section
    documents what was *declared*; this one documents what was *done*. The
    difference between them is exactly what a reviewer needs to catch before
    applying a patch they cannot otherwise see without applying it.
    """
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

    declared = _declared_scope(spec)
    out_of_scope = out_of_scope_files(spec, manifest)
    rows = "\n".join(_change_row(c, declared) for c in manifest)
    total_files = len(manifest)
    unknown = [c for c in manifest if not c.counts_known]
    known_text = [c for c in manifest if c.counts_known and not c.binary]
    total_ins = sum(c.insertions or 0 for c in known_text)
    total_del = sum(c.deletions or 0 for c in known_text)
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
        listed = ", ".join(f"`{p}`" for p in out_of_scope)
        section += f"""
> **This patch touches files outside the declared scope.** The agent was
> instructed to edit only the files under "In-scope files" above, but the
> diff also changes: {listed}. That was not declared and not permitted —
> review those hunks specifically, on their own merits, before applying
> anything in this patch.
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
) -> str:
    summary = change_summary.strip() if change_summary else "_(the agent left no summary)_"
    if agent_error:
        summary = f"{summary}\n\n**Agent session error:** `{agent_error}`"

    if not patch_produced:
        return (
            "## Outcome\n\n"
            "**No patch was produced.** The agent made no in-scope edit. Its "
            "reasoning follows; a missing patch is a legitimate outcome and is "
            "preferable to one that cannot be trusted.\n\n"
            f"{summary}\n"
        )

    oracle_cmd = (
        target.oracles.correctness[0]
        if target.oracles.correctness
        else "# (no correctness oracle was recorded for this candidate)"
    )
    return f"""## What changed and why

{summary}

## Applying and verifying this patch

Run this from the directory containing this file — the same directory
`fix.patch` sits in:

```bash
git -C {repo} checkout {base_sha}
git -C {repo} apply --check "$PWD/fix.patch" && git -C {repo} apply "$PWD/fix.patch"

# the recorded correctness oracle — run it on a machine that can:
{oracle_cmd}
```

If the patch does not apply cleanly, `git -C {repo} apply -3 "$PWD/fix.patch"`
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
) -> str:
    """Render `FIX-NOTES.md` for one candidate.

    `manifest` is the per-file breakdown of what the patch actually touched
    (see `worktree.collect_patch`) — always derived from the diff, never from
    `spec.targets`, so it can surface a patch that strayed outside scope.
    """
    target = _candidate_target(spec)
    return f"""# Fix notes — {candidate_id}

- **Candidate:** `{candidate_id}`
- **Module:** `{module_qn}`
- **Repo:** `{repo}`
- **Base commit:** `{base_sha}`
- **Objective:** {spec.objective.goal} (direction: {spec.objective.direction})

## In-scope files

{_scope_lines(spec)}

{_manifest_section(spec, manifest, patch_produced)}

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
    target=target,
)}"""


__all__ = ["out_of_scope_files", "render_fix_notes"]
