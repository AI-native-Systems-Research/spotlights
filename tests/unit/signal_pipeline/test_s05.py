"""Stage 05 — execution backend wiring.

Per-change fan-out: tests cover the prompt assembly, the runner-owned
fields (`result_id`, `change_ref`, `backend_id`), the file-edit
post-processing (read after, truncation, path-traversal guard), and
the elevated `bypassPermissions` permission mode that distinguishes
this stage from 03/04.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.signal_pipeline import (
    SignalPipelineInput,
    StageSelection,
    run_pipeline,
)
from spotlights_engine.signal_pipeline.schemas import (
    Change,
    ExecutionResult,
    FileEdit,
)
from spotlights_engine.signal_pipeline.stages import s05_execution
from spotlights_engine.signal_pipeline.stages.s05_execution import (
    ExecutionError,
    _ExecutionSummary,
    _MAX_FILE_BYTES,
    _read_file_edit,
)


def _change(candidate_ref: str = "cand-0001") -> Change:
    return Change(
        change_id=f"chg-{candidate_ref}",
        candidate_ref=candidate_ref,
        change_type="tune",
        mechanism="lower watermark",
        expected_effect="latency ↓",
        required_changes="scheduler.py",
        evaluation_metric="latency-vs-baseline",
    )


# ── Prompt + schema ─────────────────────────────────────────────────────


def test_build_prompt_substitutes_inputs(tmp_path):
    prompt = s05_execution._build_prompt(_change("cand-0042"), tmp_path)
    assert "cand-0042" in prompt
    assert str(tmp_path) in prompt
    assert "{change_json}" not in prompt
    assert "{subject_root}" not in prompt


def test_summary_schema_is_closed():
    schema = _ExecutionSummary.model_json_schema()
    assert schema["additionalProperties"] is False
    # Status enum is the closed set we expect.
    status_def = schema["properties"]["status"]
    # pydantic emits either `enum` or a `$ref` to a Literal-derived type;
    # accept both shapes.
    assert "enum" in status_def or "$ref" in status_def


# ── _read_file_edit ─────────────────────────────────────────────────────


def test_read_file_edit_returns_after_content(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("hello world")
    edit = _read_file_edit(tmp_path, "a.py")
    assert edit.path == "a.py"
    assert edit.format == "after_content"
    assert edit.payload == "hello world"


def test_read_file_edit_rejects_path_traversal(tmp_path):
    # Create the target outside subject_root.
    outside = tmp_path.parent / "leak.txt"
    outside.write_text("secret")
    with pytest.raises(ExecutionError, match="escapes subject_root"):
        _read_file_edit(tmp_path, "../leak.txt")


def test_read_file_edit_marks_missing_file(tmp_path):
    edit = _read_file_edit(tmp_path, "ghost.py")
    assert edit.format == "after_content"
    assert "not found" in edit.payload


def test_read_file_edit_truncates_huge_files(tmp_path):
    f = tmp_path / "big.py"
    big = b"a" * (_MAX_FILE_BYTES + 1024)
    f.write_bytes(big)
    edit = _read_file_edit(tmp_path, "big.py")
    assert "truncated" in edit.payload
    assert edit.payload.startswith("a" * 100)  # spot-check the head survived


# ── _execute_change wiring ──────────────────────────────────────────────


@pytest.mark.no_stub_stages
def test_execute_change_uses_bypass_permissions_and_edit_tools(
    monkeypatch, tmp_path
):
    """Stage 05 is the only stage that gets edit perms — verify it
    actually requests them."""
    captured: dict = {}

    def fake_run_claude(*, prompt, log_dir, json_schema, cwd, max_turns,
                       timeout_s, permission_mode, allowed_tools, **kwargs):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        captured.update(
            permission_mode=permission_mode,
            allowed_tools=tuple(allowed_tools),
            cwd=cwd,
        )
        return ClaudeRunResult(
            structured_output={
                "status": "applied",
                "files_touched": [],
                "rationale": "stubbed",
            },
            duration_s=0.1,
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    subject = tmp_path / "subject"
    subject.mkdir()
    s05_execution._execute_change(
        change=_change(),
        subject_root=subject,
        log_dir=tmp_path / "log",
        backend_id="claude_code",
    )
    assert captured["permission_mode"] == "bypassPermissions"
    # Edit + Write must be present so claude can actually modify code.
    assert "Edit" in captured["allowed_tools"]
    assert "Write" in captured["allowed_tools"]
    assert captured["cwd"] == subject


@pytest.mark.no_stub_stages
def test_execute_change_assembles_result_from_summary_and_disk(
    monkeypatch, tmp_path
):
    """Runner constructs the ExecutionResult — including reading touched
    files back from disk — from the LLM's summary, not from anything
    the LLM dumps in the response. Defends the schema invariant that
    `file_edits` content always comes from the filesystem post-edit."""
    subject = tmp_path / "subject"
    subject.mkdir()
    (subject / "scheduler.py").write_text("# tuned content\n")
    (subject / "watermark.py").write_text("WATERMARK = 0.85\n")

    def fake_run_claude(**_kw):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        return ClaudeRunResult(
            structured_output={
                "status": "applied",
                "files_touched": ["scheduler.py", "watermark.py"],
                "rationale": "tuned the watermark",
            },
            duration_s=0.1,
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    result = s05_execution._execute_change(
        change=_change("cand-0007"),
        subject_root=subject,
        log_dir=tmp_path / "log",
        backend_id="claude_code",
    )
    assert isinstance(result, ExecutionResult)
    assert result.result_id == "res-cand-0007"
    assert result.change_ref == "chg-cand-0007"
    assert result.backend_id == "claude_code"
    assert result.status == "applied"
    assert result.rationale == "tuned the watermark"
    paths = sorted(e.path for e in result.file_edits)
    assert paths == ["scheduler.py", "watermark.py"]
    by_path = {e.path: e for e in result.file_edits}
    assert "tuned content" in by_path["scheduler.py"].payload
    assert "WATERMARK" in by_path["watermark.py"].payload


@pytest.mark.no_stub_stages
def test_execute_change_propagates_claude_error(monkeypatch, tmp_path):
    def fake_run_claude(**_kw):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        return ClaudeRunResult(
            structured_output=None, duration_s=2.0, error="claude exit=2"
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    with pytest.raises(ExecutionError, match="cand-0001.*exit=2"):
        s05_execution._execute_change(
            change=_change(),
            subject_root=tmp_path,
            log_dir=tmp_path / "log",
            backend_id="claude_code",
        )


# ── End-to-end via the runner ───────────────────────────────────────────


def test_run_pipeline_writes_per_change_result_files(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path),
        run_dir=run_dir,
        stages=StageSelection.all(),
    )
    fanout = run_dir / "05_results"
    assert (fanout / "cand-0001.json").exists()
    assert (fanout / "cand-0002.json").exists()
    one = json.loads((fanout / "cand-0001.json").read_text())
    assert one["change_ref"].startswith("chg-")
    assert one["result_id"] == "res-cand-0001"


# ── Safety order ────────────────────────────────────────────────────────


def test_safety_order_no_top_level_main_imports():
    forbidden = {"extract", "ExtractorConfig", "ModulesExtractorInput", "run_claude"}
    bound = set(dir(s05_execution)) & forbidden
    assert bound == set(), (
        f"safety-order regression: {bound} bound at module top of s05_execution"
    )
