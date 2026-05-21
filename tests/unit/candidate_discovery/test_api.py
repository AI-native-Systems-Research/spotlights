"""Unit tests for the candidate_discovery public API surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.candidate_discovery.api import (
    DiscoveryConfig,
    resolve_target_module,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import CandidateDiscoveryInput
from spotlights_engine.schemas.project import File, Module, ProjectTree, Repository


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="demo", summary="d"),
        modules=[
            Module(
                name="foo",
                path="src/foo",
                main_files=[File(path="src/foo/x.py", role="entry")],
            )
        ],
    )


def _ctx() -> SpotlightContext:
    return SpotlightContext(objective="reduce latency")


def test_discovery_config_holds_only_infra(tmp_path: Path) -> None:
    """`DiscoveryConfig` is the infra-only sibling of `CandidateDiscoveryInput`."""
    cfg = DiscoveryConfig(
        repo_path=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert cfg.repo_path == tmp_path
    assert cfg.artifacts_dir == tmp_path / "artifacts"
    assert cfg.num_review_iterations == 3


def test_resolve_target_module_slash_form() -> None:
    inp = CandidateDiscoveryInput(
        project_tree=_tree(),
        module_qualified_name="foo",
        context=_ctx(),
    )
    m = resolve_target_module(inp)
    assert m.name == "foo"
    assert m.path == "src/foo"


def test_resolve_target_module_dot_form() -> None:
    """Architecture spec uses dot-joined qualified names; ProjectTree.walk
    yields slash-joined names. The resolver must accept both."""
    tree = ProjectTree(
        repository=Repository(name="demo", summary="d"),
        modules=[
            Module(
                name="inference",
                path="src/inference",
                submodules=[
                    Module(name="attention", path="src/inference/attention"),
                ],
            )
        ],
    )
    inp = CandidateDiscoveryInput(
        project_tree=tree,
        module_qualified_name="inference.attention",
        context=_ctx(),
    )
    m = resolve_target_module(inp)
    assert m.name == "attention"


def test_resolve_target_module_unknown_raises() -> None:
    inp = CandidateDiscoveryInput(
        project_tree=_tree(),
        module_qualified_name="nope",
        context=_ctx(),
    )
    with pytest.raises(ValueError, match="not found"):
        resolve_target_module(inp)
