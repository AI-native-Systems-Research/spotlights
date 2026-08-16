"""Artifact failure handling and top-level extractor dispatch."""

from __future__ import annotations

import pytest

from spotlights_engine.modules_extractor import errors as me_errors
from spotlights_engine.modules_extractor.extractor import (
    ExtractorConfig,
    extract_with_telemetry,
)
from spotlights_engine.modules_extractor.two_phase import _required_write_json
from spotlights_engine.schemas.pipeline import ModulesExtractorInput
from tests.unit.modules_extractor.test_two_phase_assignments import (
    _result,
    _single_root_repo,
)


def test_required_write_failure_raises_artifact_error(tmp_path) -> None:
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(me_errors.ExtractorArtifactError):
        _required_write_json(blocker, "sub/thing.json", {"a": 1})


def test_required_write_noop_when_base_none() -> None:
    _required_write_json(None, "whatever.json", {"a": 1})


def test_default_config_is_two_phase() -> None:
    assert ExtractorConfig().two_phase is True


def test_extract_with_telemetry_direct_single_shot_path(tmp_path, monkeypatch) -> None:
    repo = _single_root_repo(tmp_path)
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
                "main_files": [
                    {"path": "pkg/core/engine.py", "role": "Engine."}
                ],
                "submodules": [],
            }
        ],
    }
    calls: list[str] = []

    def fake_stream(**kwargs):
        calls.append(kwargs["prompt"])
        return _result(payload, "single-shot")

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
    assert len(calls) == 1
