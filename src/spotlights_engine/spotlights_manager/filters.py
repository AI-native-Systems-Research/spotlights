"""Subset selection for the manager's per-module pipeline.

`ModuleFilter` is a testing aid (and a hand convenience for one-off runs)
that limits which leaves the manager pipelines. See plan §6.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field


class ModuleFilter(BaseModel):
    """Limit which leaf modules the manager pipelines.

    Empty `include` = run every leaf. Non-empty entries match either an
    exact leaf or any leaf nested beneath them (e.g. `v1.worker` expands
    to `v1.worker.gpu`). User-supplied order is preserved across entries
    so smoke runs are reproducible.
    """

    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=list)


def apply_filter(
    qualified_names: Iterable[str], filt: ModuleFilter | None
) -> list[str]:
    """Return the subset of `qualified_names` selected by `filt`.

    Each name in `filt.include` matches either an exact leaf or any leaf
    nested beneath it (e.g. `v1.worker` matches `v1.worker.gpu`). Unknown
    names raise `ValueError` so a typo in the CLI fails fast before any
    per-module step runs.
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
            matches = [name]
        else:
            prefix = name + "."
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
            f"available leaves: {sorted(available)!r}"
        )
    return selected


__all__ = ["ModuleFilter", "apply_filter"]
