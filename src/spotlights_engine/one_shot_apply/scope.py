"""Reading a candidate's declared scope out of an `EvolveSpec`.

`prompts.py` writes the scope into the agent's prompt; `notes.py` writes it
into `APPLY-NOTES.md` and compares it against what the diff actually touched.
Both had their own copy of `candidate_target` and of the scope-line
formatter, kept in sync by a docstring cross-reference. A single home removes
the drift risk: the prompt's "do not edit anything else" list and the notes'
"In-scope files" list are the same claim, so they must be the same string.

The comparison helpers live here too, for the same reason: `_write_artifacts`
and `_manifest_section` both need "is this diffed path in scope?", and a path
must not be judged in scope by one and out of scope by the other.
"""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from typing import TYPE_CHECKING

from spotlights_engine.prep_evolve.spec import EvolveSpec, Target

if TYPE_CHECKING:  # `worktree` imports `prompts`, which imports this module.
    from spotlights_engine.one_shot_apply.worktree import FileChange


def candidate_target(spec: EvolveSpec) -> Target:
    """The candidate's own target, or the first one if none is marked."""
    for t in spec.targets:
        if t.scope_kind == "candidate":
            return t
    return spec.targets[0]


def scope_lines(spec: EvolveSpec) -> str:
    """The in-scope files, with validated line ranges. This is the hard bound.

    Written WITHOUT backticks around `file:start-end` on purpose: the range is
    a copy-pasteable location, and a trailing backtick between the path and
    the colon breaks both grep-ability and the tests that assert on it.
    """
    lines: list[str] = []
    for t in spec.targets:
        if t.scope_kind == "candidate" and t.line_start is not None:
            sym = f" — {t.symbol}" if t.symbol else ""
            lines.append(f"- {t.file}:{t.line_start}-{t.line_end}{sym}")
        else:
            role = f" — {t.role}" if t.role else ""
            lines.append(f"- {t.file} (whole file{role})")
    return "\n".join(lines)


def normalize_scope_path(path: str) -> str:
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


def declared_scope(spec: EvolveSpec) -> set[str]:
    """The normalized set of files `spec.targets` declares as editable."""
    return {normalize_scope_path(t.file) for t in spec.targets}


def out_of_scope_paths(change: FileChange, declared: set[str]) -> list[str]:
    """The paths `change` touches that `declared` does not permit — both ends of a rename.

    A rename touches two paths, and `change.path` is only the destination. An
    agent that renames an undeclared file *into* the declared scope has
    deleted a file it was never permitted to touch, yet every check keyed on
    the destination alone reports the row as in scope. The source is the more
    consequential half there: the destination is a file the reviewer expected
    to change anyway, while the source vanishing from the repo is entirely
    unannounced.

    Returns the offending paths in diff order (source before destination) so a
    row can be labelled by *which* end strayed, not merely that one did.
    """
    offending: list[str] = []
    if change.old_path is not None and normalize_scope_path(change.old_path) not in declared:
        offending.append(change.old_path)
    if normalize_scope_path(change.path) not in declared:
        offending.append(change.path)
    return offending


def out_of_scope_files(spec: EvolveSpec, manifest: Sequence[FileChange]) -> list[str]:
    """Files the patch actually touched that are not in `spec.targets`' declared scope.

    This is the whole point of the manifest: derived from the diff, so it can
    (and does, when the agent strays) disagree with what was declared.
    Computed exactly once per candidate, in `api._write_artifacts`, and
    threaded from there into both `ApplyArtifact` and the notes renderer — so
    the artifact and the notes can never disagree about what strayed.

    Counts *both* ends of a rename (see `out_of_scope_paths`), so a rename out
    of the declared scope and a rename into it are each reported once for the
    path that was not permitted.
    """
    declared = declared_scope(spec)
    return [p for c in manifest for p in out_of_scope_paths(c, declared)]


__all__ = [
    "candidate_target",
    "declared_scope",
    "normalize_scope_path",
    "out_of_scope_files",
    "out_of_scope_paths",
    "scope_lines",
]
