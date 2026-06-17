"""Unit tests for the candidate_discovery public API surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.candidate_discovery import DiscoverySetupError
from spotlights_engine.candidate_discovery.api import (
    DiscoveryConfig,
    discover,
    resolve_target_module,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import CandidateDiscoveryInput
from spotlights_engine.schemas.project import File, Module, ProjectTree, Repository


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="demo", summary="d", source_root="src"),
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


def test_resolve_target_module_nested_slash_form() -> None:
    """Qualified names are the slash-joined names `ProjectTree.walk` yields —
    the single canonical form the resolver keys on."""
    tree = ProjectTree(
        repository=Repository(name="demo", summary="d", source_root="src"),
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
        module_qualified_name="inference/attention",
        context=_ctx(),
    )
    m = resolve_target_module(inp)
    assert m.name == "attention"


def test_discovery_config_paths_optional_at_construction() -> None:
    """Plan §3.1: relaxed to `Path | None` so callers (SpotlightsManager) can
    build a config once and fill the per-module paths via `model_copy`."""
    cfg = DiscoveryConfig()
    assert cfg.repo_path is None
    assert cfg.artifacts_dir is None


def test_discover_raises_when_repo_path_none() -> None:
    inp = CandidateDiscoveryInput(
        project_tree=_tree(),
        module_qualified_name="foo",
        context=_ctx(),
    )
    cfg = DiscoveryConfig(artifacts_dir=Path("/tmp/will-not-be-used"))
    with pytest.raises(DiscoverySetupError, match="repo_path is required"):
        discover(inp, config=cfg)


def test_discover_raises_when_artifacts_dir_none(tmp_path: Path) -> None:
    inp = CandidateDiscoveryInput(
        project_tree=_tree(),
        module_qualified_name="foo",
        context=_ctx(),
    )
    cfg = DiscoveryConfig(repo_path=tmp_path)
    with pytest.raises(DiscoverySetupError, match="artifacts_dir is required"):
        discover(inp, config=cfg)


def test_resolve_target_module_unknown_raises() -> None:
    inp = CandidateDiscoveryInput(
        project_tree=_tree(),
        module_qualified_name="nope",
        context=_ctx(),
    )
    with pytest.raises(ValueError, match="not found"):
        resolve_target_module(inp)
