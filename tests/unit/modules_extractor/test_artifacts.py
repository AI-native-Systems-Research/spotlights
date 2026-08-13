"""Artifact-persistence and config-dispatch tests for the two-phase extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.modules_extractor import errors as me_errors
from spotlights_engine.modules_extractor.extractor import (
    ExtractorConfig,
    extract_with_telemetry,
)
from spotlights_engine.modules_extractor.two_phase import (
    _required_write_json,
    run_two_phase_extraction,
)
from spotlights_engine.schemas.pipeline import ModulesExtractorInput
from tests.unit.modules_extractor.test_two_phase import (  # reuse fakes
    _codex_result,
    _enriched_tree,
    _FakeClaude,
    _FakeCodex,
    _patch,
    _repo,
    _source_root_decision,
    _stream_result,
)


def _run(tmp_path, monkeypatch, *, artifacts: Path | None, extra_enrich=None, config=None):
    repo = _repo(tmp_path)
    results = [
        _stream_result(_source_root_decision()),
    ]
    if extra_enrich is not None:
        results.append(_stream_result(extra_enrich))
    results.append(_stream_result(_enriched_tree()))
    claude = _FakeClaude(results)
    codex = _FakeCodex(lambda p, o: _codex_result({"ok": True, "issues": []}))
    _patch(monkeypatch, claude, codex)
    run = run_two_phase_extraction(
        repo,
        config=config or ExtractorConfig(),
        on_event=None,
        artifacts_dir=artifacts,
    )
    return run, claude


# The synthetic repo has one top-level source-bearing branch (`pkg/core` under
# `source_root="pkg"`), so derivation yields exactly one shard keyed `pkg__core`.
_SHARD = "pkg__core"


def test_per_stage_artifacts_written(tmp_path, monkeypatch) -> None:
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    _run(tmp_path, monkeypatch, artifacts=artifacts)

    # Stage 1
    assert (artifacts / "01_source_root" / "prompt.md").exists()
    assert (artifacts / "01_source_root" / "attempt_01" / "schema.json").exists()
    assert (artifacts / "01_source_root" / "attempt_01" / "stream.jsonl").exists()
    assert (artifacts / "01_source_root" / "attempt_01" / "last_message.json").exists()
    assert (artifacts / "source_root_decision.json").exists()
    # Stage 2
    assert (artifacts / "02_skeleton" / "skeleton.json").exists()
    assert (artifacts / "skeleton.json").exists()
    # Stage 3 — per-shard layout.
    enrich = artifacts / "03_enrich"
    assert (enrich / "shards.json").exists()
    shard = enrich / _SHARD
    for name in ("scope.json", "subtree_skeleton.json", "data_blocks.json",
                 "prompt.md", "schema.json", "fragment.json"):
        assert (shard / name).exists(), name
    for name in ("prompt.md", "schema.json", "stream.jsonl", "last_message.json",
                 "enriched_tree.json", "coverage.json"):
        assert (shard / "attempt_01" / name).exists(), name
    for name in ("enriched_tree.json", "coverage.json", "validation.json"):
        assert (enrich / "merged" / name).exists(), name
    assert (enrich / "branches" / _SHARD / "coverage.json").exists()
    # Stage 4 — per-branch review + merged ledger.
    assert (artifacts / "04_review" / "shards.json").exists()
    assert (artifacts / "04_review" / _SHARD / "attempt_01" / "status.json").exists()
    assert (artifacts / "04_review" / "merged_review.json").exists()
    # Top-level
    assert (artifacts / "enriched_tree.json").exists()
    assert (artifacts / "coverage.json").exists()
    assert (artifacts / "review.json").exists()
    assert (artifacts / "project_tree.json").exists()
    assert (artifacts / "sessions.json").exists()


def test_single_mode_keeps_the_monolithic_artifact_layout(tmp_path, monkeypatch) -> None:
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    _run(
        tmp_path,
        monkeypatch,
        artifacts=artifacts,
        config=ExtractorConfig(enrich_sharding="single"),
    )
    assert (artifacts / "03_enrich" / "prompt.md").exists()
    assert (artifacts / "03_enrich" / "attempt_01" / "stream.jsonl").exists()
    assert (artifacts / "03_enrich" / "attempt_01" / "coverage.json").exists()
    assert not (artifacts / "03_enrich" / "shards.json").exists()


def test_repair_writes_second_attempt_dir_without_clobbering(tmp_path, monkeypatch) -> None:
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    # First enrichment is a coverage miss (empty submodules), triggering a repair.
    incomplete = _enriched_tree()
    incomplete["modules"][0]["submodules"] = []
    _run(tmp_path, monkeypatch, artifacts=artifacts, extra_enrich=incomplete)

    a1 = artifacts / "03_enrich" / _SHARD / "attempt_01"
    a2 = artifacts / "03_enrich" / _SHARD / "attempt_02"
    assert a1.exists() and a2.exists()
    # Attempt 1 recorded a coverage miss.
    cov1 = json.loads((a1 / "coverage.json").read_text())
    assert "pkg/core/kv_offload" in cov1["missing"]
    # Attempt 2 (accepted) has no missing.
    cov2 = json.loads((a2 / "coverage.json").read_text())
    assert cov2["missing"] == []
    # The rejected attempt never became the accepted fragment.
    fragment = json.loads(
        (artifacts / "03_enrich" / _SHARD / "fragment.json").read_text()
    )
    assert [s["name"] for s in fragment["modules"][0]["submodules"]] == [
        "kv_offload",
        "scheduler",
    ]


def test_artifacts_none_disables_writing(tmp_path, monkeypatch) -> None:
    run, _ = _run(tmp_path, monkeypatch, artifacts=None)
    assert run.project_tree.modules[0].name == "core"


def test_required_write_failure_raises_artifact_error(tmp_path) -> None:
    # Point the base at a path whose parent is a file → mkdir/write fails.
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(me_errors.ExtractorArtifactError):
        _required_write_json(blocker, "sub/thing.json", {"a": 1})


def test_required_write_noop_when_base_none() -> None:
    # No exception, no write.
    _required_write_json(None, "whatever.json", {"a": 1})


# ── Config dispatch / CLI flag ────────────────────────────────────────────


def test_default_config_is_two_phase() -> None:
    assert ExtractorConfig().two_phase is True


def test_extract_with_telemetry_dispatches_to_two_phase(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(_enriched_tree()),
    ])
    codex = _FakeCodex(lambda p, o: _codex_result({"ok": True, "issues": []}))
    _patch(monkeypatch, claude, codex)

    result = extract_with_telemetry(
        ModulesExtractorInput(repo_path=repo),
        config=ExtractorConfig(two_phase=True),
    )
    assert result.project_tree.modules[0].name == "core"
    # Two-phase used two Claude sessions (source-root + enrichment).
    assert len(claude.prompts) == 2


def test_extract_with_telemetry_legacy_path(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    # Legacy single-shot path returns a full ProjectTree in one Claude call.
    payload = {
        "repository": {
            "name": "demo",
            "summary": "A demo package.",
            "source_root": "pkg",
        },
        "modules": [
            {
                "name": "core",
                "path": "pkg/core",
                "description": "Core.",
                "main_files": [{"path": "pkg/core/engine.py", "role": "Engine."}],
                "submodules": [],
            }
        ],
    }
    calls: list[str] = []

    def fake_stream(**kwargs):
        calls.append(kwargs["prompt"])
        return _stream_result(payload)

    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.resolve_claude_argv0",
        lambda _: ["claude"],
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.run_streaming_claude",
        fake_stream,
    )
    result = extract_with_telemetry(
        ModulesExtractorInput(repo_path=repo),
        config=ExtractorConfig(two_phase=False),
    )
    assert result.project_tree.modules[0].name == "core"
    # Legacy path is a single Claude session.
    assert len(calls) == 1
