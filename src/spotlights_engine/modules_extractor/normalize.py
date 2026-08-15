"""Deterministic post-parse normalization of an `EnrichedTree`.

Runs between parsing and validation in every Stage-3 path (single, per-shard,
and merged). It applies one mechanical fix that would otherwise burn the
bounded LLM repair on a problem no model judgment is needed for:

**Single-child collapse (Rule 4).** A parent with exactly one emitted child
absorbs it: the child's submodules are re-parented, its description is
appended to the parent's, and a synthesized fold record covers the child's
path. Batch evidence showed this was the single most common repair trigger,
and the LLM repair for it twice produced a tree that traded the shape
violation for a coverage regression.

The transform is conservative: it never invents paths, never drops coverage
(a collapsed child moves from `emitted` to `folded`), and leaves anything it
cannot fix untouched for `validate_enriched_tree` to report precisely. Fold
evidence needs no synchronization into the target's `main_files` — Rule 8
validates `evidence_files` directly against the filesystem.

The pass is stateless and idempotent, so re-validation at merge and Stage-5
assembly sees a tree that passes the byte-for-byte unchanged validators — no
"trusted fold" state needs to survive fragment persistence or the merge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.constants import MAX_MAIN_FILES
from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedSubmodule,
    EnrichedTopModule,
    EnrichedTree,
    FoldRecord,
)

# Mirrors the description bound in `stage_schemas` (kept private there).
_MAX_DESCRIPTION = 2000


class NormalizationAction(BaseModel):
    """One mechanical fix applied to the tree, for the artifact trail."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["collapse_single_child"]
    path: str
    """The collapsed module's path."""
    into: str
    """The absorbing parent."""
    detail: str


class NormalizationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[NormalizationAction] = Field(default_factory=list)


def _norm(path: str) -> str:
    return path.strip().strip("/")


def normalize_enriched_tree(
    enriched: EnrichedTree,
    *,
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
    # is the child's own citations, which Rule 8 validates directly against the
    # filesystem. A model fold for this path cannot already exist in a valid
    # tree (the child was emitted), but guard anyway.
    if all(_norm(f.path) != cpath for f in tree.folds):
        evidence = [f.path for f in child.main_files] or [cpath]
        tree.folds.append(
            FoldRecord(
                path=cpath,
                into=ppath,
                reason="single-child parent collapsed deterministically (Rule 4)",
                evidence_files=evidence[:MAX_MAIN_FILES],
            )
        )

    # Folds that targeted the absorbed child now target the parent — their
    # nearest emitted ancestor after the collapse.
    for fold in tree.folds:
        if _norm(fold.into) == cpath:
            fold.into = ppath


__all__ = [
    "NormalizationAction",
    "NormalizationReport",
    "normalize_enriched_tree",
]
