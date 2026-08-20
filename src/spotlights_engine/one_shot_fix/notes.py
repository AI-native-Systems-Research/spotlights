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

from pathlib import Path

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

```bash
git -C {repo} checkout {base_sha}
git -C {repo} apply --check fix.patch && git -C {repo} apply fix.patch

# the recorded correctness oracle — run it on a machine that can:
{oracle_cmd}
```

If the patch does not apply cleanly, `git -C {repo} apply -3 fix.patch` falls
back to a three-way merge. Without git, `patch -p1 < fix.patch` works from the
repo root.
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
) -> str:
    """Render `FIX-NOTES.md` for one candidate."""
    target = _candidate_target(spec)
    return f"""# Fix notes — {candidate_id}

- **Candidate:** `{candidate_id}`
- **Module:** `{module_qn}`
- **Repo:** `{repo}`
- **Base commit:** `{base_sha}`
- **Objective:** {spec.objective.goal} (direction: {spec.objective.direction})

## In-scope files

{_scope_lines(spec)}

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


__all__ = ["render_fix_notes"]
