"""Unit tests for the candidate_discovery public API surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spotlights_engine.candidate_discovery.api import DiscoveryConfig
from spotlights_engine.schemas.common import SpotlightContext
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


def test_discovery_config_resolves_module_from_project_tree(tmp_path: Path):
    cfg = DiscoveryConfig(
        repo_path=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        project_tree=_tree(),
        module_qualified_name="foo",
        context=_ctx(),
    )
    assert cfg.module is not None
    assert cfg.module.name == "foo"
    assert cfg.module.path == "src/foo"


def test_discovery_config_accepts_explicit_module_legacy_shape(tmp_path: Path):
    """Pre-existing call sites pass `module=Module(...)` directly without a
    `project_tree`. That path must keep working."""
    m = Module(name="foo", path="src/foo")
    cfg = DiscoveryConfig(
        repo_path=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        module_qualified_name="v1/foo",
        module=m,
    )
    assert cfg.module is m
    assert cfg.context is None


def test_discovery_config_requires_module_or_project_tree(tmp_path: Path):
    with pytest.raises(ValidationError, match="module"):
        DiscoveryConfig(
            repo_path=tmp_path,
            artifacts_dir=tmp_path / "artifacts",
            module_qualified_name="v1/foo",
        )


def test_discovery_config_resolves_dot_qualified_name(tmp_path: Path):
    """The architecture spec uses dot-joined qualified names; ProjectTree.walk
    yields slash-joined names. The config must accept both."""
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
    cfg = DiscoveryConfig(
        repo_path=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        project_tree=tree,
        module_qualified_name="inference.attention",
        context=_ctx(),
    )
    assert cfg.module.name == "attention"


def test_discovery_config_rejects_unknown_qualified_name(tmp_path: Path):
    with pytest.raises(ValidationError, match="not found"):
        DiscoveryConfig(
            repo_path=tmp_path,
            artifacts_dir=tmp_path / "artifacts",
            project_tree=_tree(),
            module_qualified_name="nope",
            context=_ctx(),
        )
