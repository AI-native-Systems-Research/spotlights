"""Deterministic post-parse normalization of an `EnrichedTree`.

Runs between parsing and validation in every Stage-3 path (single, per-shard,
and merged). It applies mechanical fixes that would otherwise burn the bounded
LLM repair on problems no model judgment is needed for:

1. **Single-child collapse (Rule 4).** A parent with exactly one emitted child
   absorbs it: the child's submodules are re-parented, its description is
   appended to the parent's, and a synthesized fold record covers the child's
   path. Batch evidence showed this was the single most common repair trigger,
   and the LLM repair for it twice produced a tree that traded the shape
   violation for a coverage regression.

2. **Fold-evidence sync (Rule 7).** A fold whose `evidence_files` contain a
   valid evidence file that the model merely forgot to duplicate into the
   target's `main_files` gets that file inserted (when a slot is free, or by
   replacing a non-evidence entry when not). The fold decision itself is the
   model's; only the cross-reference bookkeeping is repaired.

Both transforms are conservative: they never invent paths, never drop coverage
(a collapsed child moves from `emitted` to `folded`), and leave anything they
cannot fix untouched for `validate_enriched_tree` to report precisely.

The pass is stateless and idempotent, so re-validation at merge and Stage-5
assembly sees a tree that passes the byte-for-byte unchanged validators — no
"trusted fold" state needs to survive fragment persistence or the merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.coverage import (
    _dir_has_direct_source_file,
    _is_real_source_file,
    _nearest_emitted_owner,
)
from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedFile,
    EnrichedSubmodule,
    EnrichedTopModule,
    EnrichedTree,
    FoldRecord,
    Skeleton,
)

# Mirrors the bounds in `stage_schemas` (kept private there by design).
_MAX_DESCRIPTION = 2000
_MAX_MAIN_FILES = 5


class NormalizationAction(BaseModel):
    """One mechanical fix applied to the tree, for the artifact trail."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["collapse_single_child", "sync_fold_evidence"]
    path: str
    """The collapsed module's path, or the fold path whose evidence was synced."""
    into: str
    """The absorbing parent, or the fold target whose `main_files` changed."""
    detail: str


class NormalizationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[NormalizationAction] = Field(default_factory=list)


def _norm(path: str) -> str:
    return path.strip().strip("/")


def normalize_enriched_tree(
    enriched: EnrichedTree,
    repo_path: Path,
    *,
    skeleton: Skeleton,
    protected_paths: set[str] | None = None,
) -> tuple[EnrichedTree, NormalizationReport]:
    """Return a normalized deep copy of `enriched` plus the action ledger.

    `protected_paths` are module paths the collapse must not touch — neither as
    the single-child parent (a spine's promotion parent legitimately shows one
    local child; the rest return at merge) nor as the absorbed child (the merge
    grafts promoted children into it by path, so it must stay emitted).
    """
    tree = enriched.model_copy(deep=True)
    protected = {_norm(p) for p in (protected_paths or set())}
    actions: list[NormalizationAction] = []

    _collapse_single_children(tree, protected, actions)
    _sync_fold_evidence(tree, repo_path, skeleton, actions)

    return tree, NormalizationReport(actions=actions)


# ── Single-child collapse ─────────────────────────────────────────────────


def _collapse_single_children(
    tree: EnrichedTree, protected: set[str], actions: list[NormalizationAction]
) -> None:
    # Restart the traversal after each absorb: the structure just changed, and
    # an absorb can expose a new single-child parent higher up (a chain
    # collapses one link per pass). Each absorb removes one module, so the
    # outer loop is bounded by the module count.
    while True:
        absorbed = False
        for parent in tree.iter_all_modules():
            if len(parent.submodules) != 1:
                continue
            ppath = _norm(parent.path)
            if ppath in protected:
                continue  # its missing siblings return at merge
            child = parent.submodules[0]
            cpath = _norm(child.path)
            if cpath in protected:
                continue  # merge grafts into this path; it must stay emitted
            _absorb(tree, parent, child)
            actions.append(
                NormalizationAction(
                    kind="collapse_single_child",
                    path=cpath,
                    into=ppath,
                    detail=(
                        f"absorbed single child {cpath!r} into {ppath!r}; "
                        f"{len(child.submodules)} grandchild(ren) re-parented"
                    ),
                )
            )
            absorbed = True
            break
        if not absorbed:
            return


def _absorb(
    tree: EnrichedTree,
    parent: EnrichedTopModule | EnrichedSubmodule,
    child: EnrichedSubmodule,
) -> None:
    ppath = _norm(parent.path)
    cpath = _norm(child.path)

    parent.submodules = list(child.submodules)
    merged = f"{parent.description.rstrip()} Includes {child.name}: {child.description}"
    parent.description = merged[:_MAX_DESCRIPTION]

    # Cover the child's path with a synthesized fold so the coverage equation
    # (`missing = required - emitted - folded`) still accounts for it. Evidence
    # is the child's own citations; `_sync_fold_evidence` moves one into the
    # parent's main_files afterwards. A model fold for this path cannot already
    # exist in a valid tree (the child was emitted), but guard anyway.
    if all(_norm(f.path) != cpath for f in tree.folds):
        evidence = [f.path for f in child.main_files] or [cpath]
        tree.folds.append(
            FoldRecord(
                path=cpath,
                into=ppath,
                reason="single-child parent collapsed deterministically (Rule 4)",
                evidence_files=evidence[:_MAX_MAIN_FILES],
            )
        )

    # Folds that targeted the absorbed child now target the parent — their
    # nearest emitted ancestor after the collapse.
    for fold in tree.folds:
        if _norm(fold.into) == cpath:
            fold.into = ppath


# ── Fold-evidence sync ────────────────────────────────────────────────────


def _is_valid_evidence(
    ev: str, fpath: str, into: str, repo_path: Path, emitted: set[str]
) -> bool:
    """Mirror `coverage._validate_folds`' per-file evidence test exactly."""
    if not (ev == fpath or ev.startswith(fpath + "/")):
        return False
    if not _is_real_source_file(repo_path, ev):
        return False
    owner = _nearest_emitted_owner(ev, emitted)
    return owner is None or owner == into


def _sync_fold_evidence(
    tree: EnrichedTree,
    repo_path: Path,
    skeleton: Skeleton,
    actions: list[NormalizationAction],
) -> None:
    modules_by_path = {_norm(m.path): m for m in tree.iter_all_modules()}
    emitted = set(modules_by_path)
    organizational = set(skeleton.organizational_only)

    for fold in tree.folds:
        fpath = _norm(fold.path)
        into = _norm(fold.into)
        target = modules_by_path.get(into)
        if target is None:
            continue  # malformed fold; validation reports it precisely
        # Folds the validator waives need no main_files cross-reference: an
        # organizational passthrough, or a directory with no direct source file
        # of its own (nothing substantive to cite).
        if fpath in organizational:
            continue
        if not _dir_has_direct_source_file(repo_path, fpath):
            continue

        valid_evidence = [
            ev
            for ev in fold.evidence_files
            if _is_valid_evidence(_norm(ev), fpath, into, repo_path, emitted)
        ]
        if not valid_evidence:
            continue  # no mechanical fix exists; leave for validation/repair
        target_paths = {f.path for f in target.main_files}
        if any(ev in target_paths for ev in valid_evidence):
            continue  # Rule 7 already satisfied

        ev = valid_evidence[0]
        entry = EnrichedFile(path=ev, role=f"Key file of folded {fpath}")
        if len(target.main_files) < _MAX_MAIN_FILES:
            target.main_files = [*target.main_files, entry]
            detail = f"appended evidence {ev!r} to {into!r} main_files"
        else:
            # Full: replace the last main_file no *other* fold into this target
            # depends on. If every slot is another fold's evidence, there is no
            # safe mechanical fix — leave it for the repair.
            locked = {
                _norm(e)
                for other in tree.folds
                if other is not fold and _norm(other.into) == into
                for e in other.evidence_files
            }
            droppable = [f for f in target.main_files if _norm(f.path) not in locked]
            if not droppable:
                continue
            victim = droppable[-1]
            target.main_files = [
                f for f in target.main_files if f.path != victim.path
            ] + [entry]
            detail = (
                f"replaced {victim.path!r} with evidence {ev!r} in "
                f"{into!r} main_files"
            )
        actions.append(
            NormalizationAction(
                kind="sync_fold_evidence", path=fpath, into=into, detail=detail
            )
        )


__all__ = [
    "NormalizationAction",
    "NormalizationReport",
    "normalize_enriched_tree",
]
