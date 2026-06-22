from __future__ import annotations

import json

import pytest

from spotlights_engine.modules_extractor.agent import run_extraction
from spotlights_engine.modules_extractor.errors import ExtractorValidationError
from spotlights_engine.signal_pipeline._subprocess_util import StreamingResult


def _stream_result(payload: dict) -> StreamingResult:
    event = {
        "type": "result",
        "subtype": "success",
        "structured_output": payload,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    return StreamingResult(
        stdout=(json.dumps(event) + "\n").encode("utf-8"),
        stderr=b"",
        returncode=0,
        duration_s=1.0,
    )


def _invalid_conceptual_split_payload() -> dict:
    return {
        "repository": {
            "name": "inference-sim",
            "summary": "A Go simulator.",
            "source_root": "",
            "external_dependencies": [],
        },
        "modules": [
            {
                "name": "sim",
                "path": "sim",
                "description": "Simulation runtime.",
                "depends_on": [],
                "main_files": [
                    {"path": "sim/simulator.go", "role": "Drives simulation."}
                ],
                "submodules": [
                    {
                        "name": "routing",
                        "path": "sim",
                        "description": "Routes requests.",
                        "main_files": [
                            {"path": "sim/simulator.go", "role": "Routing logic."}
                        ],
                    }
                ],
            }
        ],
    }


def _valid_leaf_payload() -> dict:
    return {
        "repository": {
            "name": "inference-sim",
            "summary": "A Go simulator.",
            "source_root": "",
            "external_dependencies": [],
        },
        "modules": [
            {
                "name": "sim",
                "path": "sim",
                "description": "Simulation runtime.",
                "depends_on": [],
                "main_files": [
                    {"path": "sim/simulator.go", "role": "Drives simulation."}
                ],
                "submodules": [],
            }
        ],
    }


def test_run_extraction_retries_once_after_project_tree_validation_error(
    tmp_path, monkeypatch
) -> None:
    prompts: list[str] = []
    results = [
        _stream_result(_invalid_conceptual_split_payload()),
        _stream_result(_valid_leaf_payload()),
    ]

    def fake_run_streaming_claude(**kwargs) -> StreamingResult:
        prompts.append(kwargs["prompt"])
        return results.pop(0)

    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.resolve_claude_argv0",
        lambda _: ["claude"],
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.run_streaming_claude",
        fake_run_streaming_claude,
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    events: list[str] = []

    result = run_extraction(
        repo_path=tmp_path,
        prompt="base prompt",
        artifacts_dir=artifacts,
        on_event=events.append,
    )

    assert result.project_tree.modules[0].name == "sim"
    assert result.invocation.duration_s == 1.0
    assert result.invocation.input_tokens == 10
    assert result.invocation.output_tokens == 20
    assert len(prompts) == 2
    assert "failed Spotlights' stricter semantic validation" in prompts[1]
    assert "Do not create logical submodules" in prompts[1]
    assert any("ProjectTree semantic validation failed" in event for event in events)
    assert (artifacts / "validation_error.txt").exists()
    assert (artifacts / "last_message.json").exists()
    assert (artifacts / "repair_1_prompt.md").exists()
    assert (artifacts / "repair_1_last_message.json").exists()


def test_run_extraction_raises_after_validation_retry_fails(tmp_path, monkeypatch) -> None:
    prompts: list[str] = []
    results = [
        _stream_result(_invalid_conceptual_split_payload()),
        _stream_result(_invalid_conceptual_split_payload()),
        _stream_result(_invalid_conceptual_split_payload()),
    ]

    def fake_run_streaming_claude(**kwargs) -> StreamingResult:
        prompts.append(kwargs["prompt"])
        return results.pop(0)

    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.resolve_claude_argv0",
        lambda _: ["claude"],
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.run_streaming_claude",
        fake_run_streaming_claude,
    )

    with pytest.raises(ExtractorValidationError, match="after 2 repair attempts") as exc:
        run_extraction(repo_path=tmp_path, prompt="base prompt", on_event=None)

    assert len(exc.value.context["repair_errors"]) == 3
    assert len(prompts) == 3
