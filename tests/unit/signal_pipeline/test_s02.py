"""Stage 02 — ProjectTree extraction wiring.

State-machine concerns (resume, inject, etc.) are covered in
`test_runner.py`. This file verifies the *delegation* to
`modules_extractor.extract`: that the right `ModulesExtractorInput` /
`ExtractorConfig` are constructed, that the artifacts subdir is
timestamp-named under `log_dir`, and that the returned `ProjectTree`
round-trips into `02_projecttree.json` cleanly.

The `--only-stage 02 --repo <real_repo>` smoke test that fires
the real claude subprocess is a manual verification step (per the plan)
— don't add it here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.signal_pipeline import (
    SignalPipelineInput,
    StageSelection,
    run_pipeline,
)
from spotlights_engine.signal_pipeline.stages import s02_projecttree


def _custom_tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="custom-repo", summary="from-test"),
        modules=[
            Module(name="alpha", path="alpha", description="alpha module"),
            Module(name="beta", path="beta", description="beta module"),
        ],
    )


def test_run_calls_extract_with_subject_root_and_log_subdir(monkeypatch, tmp_path):
    """The wrapper must hand `subject_root` to `extract` verbatim and place
    the modules-extractor working dir under a fresh subdir of the stage's
    log_dir (so re-runs don't trip main's pre-existing-dir guard)."""
    captured: dict = {}

    def fake_extract(subject_root: Path, log_dir: Path, on_event=None) -> ProjectTree:
        captured["subject_root"] = subject_root
        captured["log_dir"] = log_dir
        return _custom_tree()

    monkeypatch.setattr(s02_projecttree, "_extract_project_tree", fake_extract)

    subject = tmp_path / "subject"
    subject.mkdir()
    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=subject),
        run_dir=run_dir,
        stages=StageSelection.only("02"),
    )

    assert captured["subject_root"] == subject
    # log_dir is the stage-level dir; the wrapper makes its own per-call
    # subdir inside it (we test that subdir-naming is fresh per-call below).
    assert captured["log_dir"] == run_dir.resolve() / "_logs" / "02_projecttree"


def test_extract_output_is_persisted_as_canonical_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(
        s02_projecttree, "_extract_project_tree", lambda *_a, **_k: _custom_tree()
    )
    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path),
        run_dir=run_dir,
        stages=StageSelection.only("02"),
    )

    written = ProjectTree.model_validate(
        json.loads((run_dir / "02_projecttree.json").read_text())
    )
    assert written.repository.name == "custom-repo"
    assert [m.name for m in written.modules] == ["alpha", "beta"]


def test_ts_subdir_is_filesystem_safe():
    """No colons or pluses — those break Windows; some macOS tools also dislike them."""
    s = s02_projecttree._ts_subdir()
    assert ":" not in s and "+" not in s
    assert s.endswith("Z") or s.endswith("-00")  # UTC offset normalized to dashes
    # Looks like 2026-05-25T21-45-00-00-00 or 2026-05-25T21-45-00Z depending
    # on platform; both shapes are filesystem-safe.


def test_safety_order_no_top_level_extractor_import():
    """`s02_projecttree` must not bind `extract` (or any main-only symbol)
    at module top — the lazy-import inside `_extract_project_tree` is what
    makes the runner's `_check_layout()` precondition the user-visible
    error site instead of an opaque ImportError. Defends the "Safety order"
    note in the approved plan."""
    forbidden = {"extract", "ExtractorConfig", "ModulesExtractorInput"}
    bound = set(dir(s02_projecttree)) & forbidden
    assert bound == set(), (
        f"safety-order regression: {bound} bound at module top of s02_projecttree"
    )


@pytest.mark.no_stub_stages
def test_artifacts_dir_is_a_fresh_subdir_of_log_dir(monkeypatch, tmp_path):
    """The path passed to `ExtractorConfig.artifacts_dir` must be a fresh
    subdir of `log_dir`, so concurrent / repeated invocations don't trip
    main's "pre-existing run dir is unsupported" guard. Exercises the
    *real* `_extract_project_tree` (opted out of the conftest stub) with
    the inner `extract` mocked so we can observe the artifacts_dir
    argument, and `_ts_subdir` mocked so the freshness assertion is
    deterministic instead of wall-clock-dependent."""
    seen: list[Path] = []

    def fake_inner_extract(input, *, config, on_event=None):
        seen.append(config.artifacts_dir)
        return _custom_tree()

    counter = iter(["A", "B", "C"])
    monkeypatch.setattr(s02_projecttree, "_ts_subdir", lambda: next(counter))
    # `_extract_project_tree` lazy-imports `extract` from
    # `spotlights_engine.modules_extractor` each call, so patching the
    # module attribute *before* invocation reaches the patched value.
    import spotlights_engine.modules_extractor as me

    monkeypatch.setattr(me, "extract", fake_inner_extract)

    subject = tmp_path / "subject"
    subject.mkdir()
    log_dir = tmp_path / "log"

    s02_projecttree._extract_project_tree(subject, log_dir)
    s02_projecttree._extract_project_tree(subject, log_dir)

    assert seen == [log_dir / "A", log_dir / "B"]
    assert seen[0] != seen[1]
