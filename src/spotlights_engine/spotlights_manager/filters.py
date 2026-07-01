"""Subset selection for the manager's per-module pipeline.

`ModuleFilter` is a testing aid (and a hand convenience for one-off runs)
that limits which modules the manager pipelines. See plan §6.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field


class ModuleFilter(BaseModel):
    """Limit which modules the manager pipelines.

    Empty `include` = run every module. Each non-empty entry that names a real
    module selects exactly that module and none of its submodules (e.g.
    `v1/worker` selects `v1/worker` alone, not `v1/worker/gpu`). When an entry
    does not resolve to any module it is treated as a virtual prefix and
    expands to every module nested beneath it (e.g. a source-root package
    segment like `spotlights_engine`). User-supplied order is preserved across
    entries so smoke runs are reproducible.
    """

    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=list)


def apply_filter(
    qualified_names: Iterable[str], filt: ModuleFilter | None
) -> list[str]:
    """Return the subset of `qualified_names` selected by `filt`.

    A name that resolves to a real module selects exactly that module and none
    of its submodules (e.g. `v1/worker` matches `v1/worker` but not
    `v1/worker/gpu`). A name that matches no module is treated as a virtual
    prefix and expands to every module nested beneath it (e.g. a source-root
    package segment like `spotlights_engine`). Names matching neither raise
    `ValueError` so a typo in the CLI fails fast before any per-module step
    runs.
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
            unknown.append(name)
            continue
        for qn in matches:
            if qn not in seen:
                seen.add(qn)
                selected.append(qn)
    if unknown:
        raise ValueError(
            f"ModuleFilter.include contains unknown qualified names: {unknown!r}; "
            f"available modules: {sorted(available)!r}"
        )
    return selected


__all__ = ["ModuleFilter", "apply_filter"]
