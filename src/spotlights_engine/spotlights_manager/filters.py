"""Subset selection for the manager's per-module pipeline.

`ModuleFilter` is a testing aid (and a hand convenience for one-off runs)
that limits which modules the manager pipelines. See plan §6.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.project import Module

_log = logging.getLogger(__name__)


class ModuleFilter(BaseModel):
    """Limit which modules the manager pipelines.

    Empty `include` = run every module. Each non-empty entry that names a real
    module selects that module and all of its submodules (e.g. `v1/worker`
    selects both `v1/worker` and `v1/worker/gpu`). When an entry
    does not resolve to any module it is treated as a virtual prefix and
    expands to every module nested beneath it (e.g. a source-root package
    segment like `spotlights_engine`). User-supplied order is preserved across
    entries so smoke runs are reproducible. A name finer-grained than any
    module falls back to its nearest existing ancestor module; a name matching
    neither a module, a prefix, nor an ancestor is warned and skipped.
    """

    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=list)


def _nearest_ancestor(name: str, available: set[str]) -> str | None:
    """Return the longest existing module that is a strict ancestor of `name`.

    `name` is slash-form (e.g. `v1/attention/backends`); its ancestors are the
    successively shorter prefixes (`v1/attention`, `v1`). The nearest — longest —
    ancestor present in `available` wins. Returns None when no ancestor exists
    (e.g. a top-level `utils` whose only ancestor would be the source root).
    """
    parts = name.split("/")
    for cut in range(len(parts) - 1, 0, -1):
        candidate = "/".join(parts[:cut])
        if candidate in available:
            return candidate
    return None


def apply_filter(qualified_names: Iterable[str], filt: ModuleFilter | None) -> list[str]:
    """Return the subset of `qualified_names` selected by `filt`.

    A name that resolves to a real module selects that module and all of its
    submodules (e.g. `v1/worker` matches both `v1/worker` and
    `v1/worker/gpu`). A name that matches no module is treated as a virtual
    prefix and expands to every module nested beneath it (e.g. a source-root
    package segment like `spotlights_engine`). A name that is finer-grained than
    any module the extractor emitted (matches nothing above) falls back to its
    nearest existing ancestor module (e.g. `v1/attention/backends` selects
    `v1/attention` when the extractor kept attention as one module). Names that
    resolve to none of the above (a typo, or a leaf with no ancestor module) are
    logged as a warning and ignored, so a coarse extractor granularity does not
    abort a run after the expensive extraction step.
    """
    qns = list(qualified_names)
    if filt is None or not filt.include:
        return qns

    available = set(qns)
    selected: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for name in filt.include:
        if name in available:
            # Exact module: the CLI contract treats it as a subtree root.
            prefix = name + "/"
            matches = [qn for qn in qns if qn == name or qn.startswith(prefix)]
        else:
            # Virtual prefix (no module resolves to `name`): expand descendants.
            prefix = name + "/"
            matches = [qn for qn in qns if qn.startswith(prefix)]
        if not matches:
            # Finer-grained than any module: fall back to nearest ancestor.
            ancestor = _nearest_ancestor(name, available)
            if ancestor is not None:
                _log.warning(
                    "ModuleFilter.include: %r matches no module; "
                    "using nearest ancestor module %r instead",
                    name,
                    ancestor,
                )
                matches = [ancestor]
            else:
                unknown.append(name)
                continue
        for qn in matches:
            if qn not in seen:
                seen.add(qn)
                selected.append(qn)
    if unknown:
        _log.warning(
            "ModuleFilter.include: ignoring %d entr%s that match no module: %r; "
            "available modules: %r",
            len(unknown),
            "y" if len(unknown) == 1 else "ies",
            unknown,
            sorted(available),
        )
    return selected


# A module is pipelined on its own merits alongside its submodules, so a pure
# routing node is a target too: `hrl_ocr/models/detection`, whose entire own
# content is a 9-line `__init__.py` re-exporting three submodules, pays a full
# discovery pass to analyze nothing. Every candidate it could produce has to
# live in one of its *own* files -- `candidate_discovery.Validator` drops
# anything inside a submodule -- and those files are import plumbing.
_PLUMBING_FILENAMES = frozenset({"__init__.py"})

# A guard, not the discriminator. Containers are selected on the shape of their
# own files (all `__init__.py`); this bound exists only to rescue the rare
# `__init__.py` that carries real implementation instead of re-exports. Measured
# on the IOCR tree (30 modules): the four containers hold 9, 9, 9 and 18
# non-blank lines, the smallest non-container module holds 96. Any value in that
# gap selects the same four, so the number is deliberately generous rather than
# tuned.
_PLUMBING_MAX_NONBLANK_LINES = 50


def container_module_reason(module: Module, repo_path: Path) -> str | None:
    """Why `module` cannot produce a candidate of its own, or None if it can.

    Returns a human reason suitable for a log line and a `SKIPPED` status, so a
    run says *why* a module was never discovered rather than silently omitting
    it.

    Submodules are required deliberately. The skipped content is not lost: it is
    analyzed as those submodules' own targets. A *leaf* module with one small
    file is the opposite case -- small, but possibly the whole point of the repo
    -- and is never reported here.

    Returns None whenever the answer is not certain: an unreadable own file, or
    own files that are not all package plumbing. Discovery is the expensive but
    correct fallback, so ambiguity resolves toward spending the money.
    """
    if not module.submodules:
        return None

    sub_prefixes = [s.path.rstrip("/") + "/" for s in module.submodules]
    own = [
        f.path
        for f in module.main_files
        if not any(f.path.startswith(prefix) for prefix in sub_prefixes)
    ]
    if not own:
        return (
            f"no files of its own -- every main_file belongs to one of its "
            f"{len(module.submodules)} submodules, which are audited separately"
        )

    if {PurePosixPath(p).name for p in own} - _PLUMBING_FILENAMES:
        return None

    nonblank = 0
    for rel in own:
        try:
            text = (repo_path / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Cannot judge what we cannot read; let discovery decide.
            return None
        nonblank += sum(1 for line in text.splitlines() if line.strip())
    if nonblank >= _PLUMBING_MAX_NONBLANK_LINES:
        return None

    return (
        f"own content is package plumbing only ({', '.join(sorted(own))}, "
        f"{nonblank} non-blank lines); the code lives in "
        f"{len(module.submodules)} submodules, which are audited separately"
    )


__all__ = ["ModuleFilter", "apply_filter", "container_module_reason"]
