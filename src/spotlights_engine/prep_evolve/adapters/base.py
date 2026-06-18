"""Adapter protocol and the `GeneratedFile` unit of materialization.

Each adapter consumes an `EvolveSpec` and returns the bundle-relative files for
its evolver. `supports()` is where the single-file vs multi-file routing rule
lives (plan §3 / §5).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from spotlights_engine.prep_evolve.spec import EvolveSpec


class GeneratedFile(BaseModel):
    """One file an adapter wants written into the bundle.

    Bundles are fully generator-owned: every file is overwritten on a `--force`
    re-run.
    """

    model_config = ConfigDict(extra="forbid")

    path: str  # bundle-relative
    text: str


@runtime_checkable
class Adapter(Protocol):
    name: str

    def supports(self, spec: EvolveSpec) -> tuple[bool, str]:
        """Return `(ok, reason-if-not)` for whether this evolver can run `spec`."""
        ...

    def render(self, spec: EvolveSpec) -> list[GeneratedFile]:
        """Render the evolver-native files (excludes the shared README, which
        `api.py` adds)."""
        ...


__all__ = ["Adapter", "GeneratedFile"]
