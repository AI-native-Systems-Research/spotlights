"""Subset selection for the manager's per-module pipeline.

`ModuleFilter` is a testing aid (and a hand convenience for one-off runs)
that limits which modules the manager pipelines. See plan §6.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

_log = logging.getLogger(__name__)


class ModuleFilter(BaseModel):
    """Limit which modules the manager pipelines.

    Empty `include` = run every module. Each non-empty entry that names a real
    module selects exactly that module and none of its submodules (e.g.
    `v1/worker` selects `v1/worker` alone, not `v1/worker/gpu`). When an entry
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


def apply_filter(
    qualified_names: Iterable[str], filt: ModuleFilter | None
) -> list[str]:
    """Return the subset of `qualified_names` selected by `filt`.

    A name that resolves to a real module selects exactly that module and none
    of its submodules (e.g. `v1/worker` matches `v1/worker` but not
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
            # Exact module: select just this module, not its submodules.
            matches = [name]
        else:
            # Virtual prefix (no module resolves to `name`): expand descendants.
            prefix = name + "/"
            matches = [qn for qn in qns if qn.startswith(prefix)]
        if not matches:
            # Finer-grained than any module: fall back to nearest ancestor.
            ancestor = _nearest_ancestor(name, available)
            if ancestor is not None:
                matches = [ancestor]
            else:
                unknown.append(name)
                continue
        for qn in matches:
            if qn not in seen:
                seen.add(qn)
                selected.append(qn)
    if unknown:
        _log.debug(
            "ModuleFilter.include: ignoring %d entr%s that match no module: %r; "
            "available modules: %r",
            len(unknown),
            "y" if len(unknown) == 1 else "ies",
            unknown,
            sorted(available),
        )
    return selected


__all__ = ["ModuleFilter", "apply_filter"]
