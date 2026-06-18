"""Adapter protocol and the `GeneratedFile` unit of materialization.

Each adapter consumes an `EvolveSpec` and returns the bundle-relative files for
its evolver. `supports()` is where the single-file vs multi-file routing rule
lives (plan §3 / §5).
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from spotlights_engine.prep_evolve.spec import EvolveSpec

OverwritePolicy = Literal["always", "preserve_if_modified"]


class GeneratedFile(BaseModel):
    """One file an adapter wants written into the bundle.

    `overwrite="preserve_if_modified"` marks user-editable scaffolds
    (evaluator/grader) the writer must not clobber once hand-edited.
    """

    model_config = ConfigDict(extra="forbid")

    path: str  # bundle-relative
    text: str
    overwrite: OverwritePolicy = "always"


@runtime_checkable
class Adapter(Protocol):
    name: str

    def supports(self, spec: EvolveSpec) -> tuple[bool, str]:
        """Return `(ok, reason-if-not)` for whether this evolver can run `spec`."""
        ...

    def render(self, spec: EvolveSpec) -> list[GeneratedFile]:
        """Render the evolver-native files (excludes the always-emitted
        metadata files, which `api.py` adds)."""
        ...


__all__ = ["Adapter", "GeneratedFile", "OverwritePolicy"]
