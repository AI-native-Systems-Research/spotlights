from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine.modules_extractor import agent
from spotlights_engine.signal_pipeline._subprocess_util import StreamingResult


def _result_event(payload: dict) -> bytes:
    event = {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "total_cost_usd": 0.01,
        "duration_ms": 1234,
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "structured_output": payload,
    }
    return (json.dumps(event) + "\n").encode()


def test_run_extraction_retries_semantic_project_tree_validation(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    invalid_tree = {
        "repository": {"name": "demo", "summary": "demo repo", "source_root": ""},
        "modules": [
            {
                "name": "pkg",
                "path": "pkg",
                "description": "package",
                "depends_on": [],
                "main_files": [{"path": "pkg/__init__.py", "role": "entry"}],
                "submodules": [
                    {
                        "name": "logical_unit",
                        "path": "pkg",
                        "description": "invalid same-path logical unit",
                        "main_files": [{"path": "pkg/core.py", "role": "core"}],
                    }
                ],
            }
        ],
    }
    valid_tree = {
        "repository": {"name": "demo", "summary": "demo repo", "source_root": ""},
        "modules": [
            {
                "name": "pkg",
                "path": "pkg",
                "description": "package",
                "depends_on": [],
                "main_files": [
                    {"path": "pkg/__init__.py", "role": "entry"},
                    {"path": "pkg/core.py", "role": "core"},
                ],
            }
        ],
    }
    payloads = [invalid_tree, valid_tree]
    prompts: list[str] = []

    def fake_run_streaming_claude(**kwargs):
        prompts.append(kwargs["prompt"])
        return StreamingResult(
            stdout=_result_event(payloads.pop(0)),
            stderr=b"",
            returncode=0,
            duration_s=1.0,
        )

    monkeypatch.setattr(agent, "resolve_claude_argv0", lambda _bin: ["claude"])
    monkeypatch.setattr(agent, "run_streaming_claude", fake_run_streaming_claude)

    result = agent.run_extraction(
        repo_path=repo,
        prompt="Map the repository.",
        artifacts_dir=artifacts,
        on_event=None,
    )

    assert result.project_tree.resolve("pkg") is not None
    assert len(prompts) == 2
    assert "failed Spotlights' stricter semantic validation" in prompts[1]
    assert (artifacts / "validation_error.txt").exists()
    assert (artifacts / "repair_1_prompt.md").exists()
    assert (artifacts / "repair_1_last_message.json").exists()
    assert (artifacts / "project_tree.json").exists()
