"""Loader tests: disk -> in-memory model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.results_renderer.errors import (
    RendererLoadError,
    RendererSetupError,
)
from spotlights_engine.results_renderer.loader import load_run
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.spotlights_manager.persistence import ManagerPaths

from tests.unit.results_renderer._fixtures import (
    make_candidate,
    make_full_run,
    write_extractor_outputs,
    write_manifest_with_modules,
    write_module,
    make_tree,
)


def test_load_full_run(tmp_path: Path) -> None:
    paths, _ = make_full_run(tmp_path)

    loaded = load_run(tmp_path)

    assert loaded.context is not None
    assert loaded.context.objective == "reduce latency"
    assert set(loaded.modules.keys()) == {"v1.kv_offload", "kernels"}
    kv = loaded.modules["v1.kv_offload"]
    assert kv.candidates is not None
    assert len(kv.candidates.candidates) == 3
    assert kv.deep_research is not None
    assert kv.agent_proposals is not None


def test_load_run_missing_root(tmp_path: Path) -> None:
    with pytest.raises(RendererSetupError):
        load_run(tmp_path / "does_not_exist")


def test_load_run_missing_extractor_outputs(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    paths.root.mkdir(parents=True)
    paths.modules_root.mkdir()
    write_manifest_with_modules(
        paths,
        context=SpotlightContext(objective="x"),
        repo_path="/repo",
        module_statuses={},
    )
    with pytest.raises(RendererLoadError):
        load_run(tmp_path)


def test_load_run_missing_context_emits_warning(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    write_extractor_outputs(paths, make_tree())
    write_manifest_with_modules(
        paths, context=None, repo_path="/repo", module_statuses={}
    )
    loaded = load_run(tmp_path)
    assert loaded.context is None
    assert any("context" in w for w in loaded.warnings)


def test_load_run_invalid_manifest_json(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    write_extractor_outputs(paths, make_tree())
    paths.manifest_path.write_text("not-json")
    with pytest.raises(RendererLoadError):
        load_run(tmp_path)


def test_load_run_falls_back_to_module_dir_walk(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    write_extractor_outputs(paths, make_tree())
    # No manifest at all -> walk modules_root.
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.modules_root.mkdir(parents=True, exist_ok=True)
    cands = Candidates(
        module_qualified_name="kernels",
        candidates=[make_candidate(1, impact="high", file="src/kernels/main.py")],
    )
    write_module(paths, "kernels", candidates=cands, status="SUCCEEDED")

    loaded = load_run(tmp_path)
    # The on-disk slug for "kernels" is "kernels"; checkpoint qn restores it.
    assert "kernels" in loaded.modules
