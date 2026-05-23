"""Subset selection for the manager's per-module pipeline.

`ModuleFilter` is a testing aid (and a hand convenience for one-off runs)
that limits which leaves the manager pipelines. See plan §6.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field


class ModuleFilter(BaseModel):
    """Limit which leaf modules the manager pipelines.

    Empty `include` = run every leaf. Non-empty = exact-match keep-list in
    the order given (later resolution preserves the user-supplied order so
    smoke runs are reproducible).
    """

    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=list)


def apply_filter(
    qualified_names: Iterable[str], filt: ModuleFilter | None
) -> list[str]:
    """Return the subset of `qualified_names` selected by `filt`.

    Unknown names in `filt.include` raise `ValueError` so a typo in the CLI
    fails fast before any per-module step runs.
    """
    qns = list(qualified_names)
    if filt is None or not filt.include:
        return qns

    available = set(qns)
    unknown = [name for name in filt.include if name not in available]
    if unknown:
        raise ValueError(
            f"ModuleFilter.include contains unknown qualified names: {unknown!r}; "
            f"available leaves: {sorted(available)!r}"
        )
    return list(filt.include)


__all__ = ["ModuleFilter", "apply_filter"]
