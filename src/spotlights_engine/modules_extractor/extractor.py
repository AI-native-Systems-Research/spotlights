"""Top-level `extract` orchestration.

Stub. The real implementation will compose the sub-steps in this package
(repository metadata, module discovery, dependency resolution, describer)
to build the `ProjectTree`.
"""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.schemas.modules import ProjectTree, Repository


def extract(root: Path) -> ProjectTree:
    """Return a `ProjectTree` describing the project at `root`.

    Stub: returns a tree with `repository.name` set to the directory basename
    and no modules.
    """
    return ProjectTree(
        repository=Repository(name=root.name, summary=""),
        modules=[],
    )
