from __future__ import annotations

import json

import pytest

from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.llm_session import SessionResult
from spotlights_engine.modules_extractor.agent import run_extraction
from spotlights_engine.modules_extractor.errors import ExtractorValidationError


def _session_result(payload: dict) -> SessionResult:
    """A successful stream-json run whose terminal event carried `payload`.

    The extractor now consumes a centralized `SessionResult`; the session has
    already parsed `structured_output` and usage out of the raw stream, so the
    test constructs the parsed result directly.
    """
    return SessionResult(
        cli="claude",
        returncode=0,
        duration_s=1.0,
        structured_output=payload,
        final_message=json.dumps(payload),
        usage=AgentUsage(input=10, output=20),
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
        _session_result(_invalid_conceptual_split_payload()),
        _session_result(_valid_leaf_payload()),
    ]

    def fake_run_with_retry(session, prompt, **kwargs) -> SessionResult:
        prompts.append(prompt)
        return results.pop(0)

    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.resolve_cli",
        lambda _cli, _bin: "/usr/local/bin/claude",
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.run_with_retry",
        fake_run_with_retry,
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
    assert result.invocation.duration_s == 2.0
    assert result.invocation.input_tokens == 20
    assert result.invocation.output_tokens == 40
    assert len(prompts) == 2
    assert "Retry after local ProjectTree validation failure" in prompts[1]
    assert "Do not split one directory into conceptual children" in prompts[1]
    assert any("retrying once" in event for event in events)
    assert (artifacts / "validation_error_attempt_0.txt").exists()
    assert (artifacts / "last_message_attempt_0.json").exists()
    assert (artifacts / "last_message_attempt_1.json").exists()


def test_run_extraction_raises_after_validation_retry_fails(tmp_path, monkeypatch) -> None:
    prompts: list[str] = []
    results = [
        _session_result(_invalid_conceptual_split_payload()),
        _session_result(_invalid_conceptual_split_payload()),
    ]

    def fake_run_with_retry(session, prompt, **kwargs) -> SessionResult:
        prompts.append(prompt)
        return results.pop(0)

    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.resolve_cli",
        lambda _cli, _bin: "/usr/local/bin/claude",
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.agent.run_with_retry",
        fake_run_with_retry,
    )

    with pytest.raises(ExtractorValidationError, match="after 2 attempts") as exc:
        run_extraction(repo_path=tmp_path, prompt="base prompt", on_event=None)

    assert exc.value.context["attempts"] == 2
    assert len(prompts) == 2
