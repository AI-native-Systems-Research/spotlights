"""Step 5 must report the real cause of an agent failure, not just its symptom.

On the IOCR run all 44 step-5 failures were labelled either `timed out after
600.0s` (30) or `exit=1: stderr=''` (14). Replaying the persisted streams shows
~33 of them were rate limits or server errors -- the 600s went to the CLI's own
retry loop. The evidence was on disk the whole time; the failure path just never
read it. These tests pin that it does now, and that the old text survives as the
prefix so anything grepping for it still matches.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from spotlights_engine.agent_proposals import claude_exec, codex_exec

# Real event shapes, as in tests/unit/utils/test_agent_stream.py.
CODEX_429 = (
    b'{"type":"error","message":"exceeded retry limit, last status: '
    b'429 Too Many Requests"}\n'
)
CLAUDE_RATE_LIMIT_RETRY = (
    b'{"type":"system","subtype":"api_retry","attempt":1,"max_retries":10,'
    b'"retry_delay_ms":562,"error_status":429,"error":"rate_limit"}\n'
)


def _fake_run(*, stdout=b"", stderr=b"", returncode=0):
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, returncode, stdout=stdout, stderr=stderr
        )

    return run


def _fake_timeout(*, stdout=b"", stderr=b""):
    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(
            argv, kwargs.get("timeout", 600), output=stdout, stderr=stderr
        )

    return run


def _claude(**kwargs):
    return claude_exec.run_candidate_claude(
        candidate_id="cand-0001",
        prompt="p",
        schema_text="{}",
        repo_path=Path("."),
        max_turns=1,
        wallclock_s=600,
        **kwargs,
    )


def _codex(tmp_path, **kwargs):
    return codex_exec.run_candidate_codex(
        candidate_id="cand-0001",
        prompt="p",
        schema_text="{}",
        repo_path=Path("."),
        wallclock_s=600,
        last_message_path=tmp_path / "last.json",
        schema_path=tmp_path / "schema.json",
        **kwargs,
    )


# --- the 30 mislabelled timeouts --------------------------------------------


def test_claude_timeout_names_the_rate_limit_behind_it(monkeypatch):
    monkeypatch.setattr(
        claude_exec.subprocess, "run", _fake_timeout(stdout=CLAUDE_RATE_LIMIT_RETRY * 12)
    )
    result = _claude()
    assert result.error is not None
    # Old text stays the prefix; the cause is appended.
    assert result.error.startswith("claude timed out after ")
    assert "12 API retries in stream (last: rate_limit)" in result.error


def test_claude_timeout_with_a_clean_stream_still_reports_only_the_timeout(
    monkeypatch,
):
    """A genuinely slow call must not be dressed up as a rate limit."""
    monkeypatch.setattr(
        claude_exec.subprocess,
        "run",
        _fake_timeout(stdout=b'{"type":"assistant","message":{"content":"work"}}\n'),
    )
    result = _claude()
    assert result.error is not None
    assert result.error.startswith("claude timed out after ")
    assert ";" not in result.error


def test_codex_timeout_names_the_rate_limit_behind_it(monkeypatch, tmp_path):
    monkeypatch.setattr(
        codex_exec.subprocess, "run", _fake_timeout(stdout=CODEX_429)
    )
    result = _codex(tmp_path)
    assert result.error is not None
    assert result.error.startswith("codex timed out after ")
    assert "exceeded retry limit, last status: 429" in result.error


# --- the 14 silent `exit=1, stderr=''` failures ------------------------------


def test_codex_nonzero_exit_names_the_429(monkeypatch, tmp_path):
    monkeypatch.setattr(
        codex_exec.subprocess, "run", _fake_run(stdout=CODEX_429, returncode=1)
    )
    result = _codex(tmp_path)
    assert result.error is not None
    assert result.error.startswith("codex exit=1: stderr=''")
    assert "exceeded retry limit, last status: 429" in result.error


def test_claude_nonzero_exit_names_the_rate_limit(monkeypatch):
    monkeypatch.setattr(
        claude_exec.subprocess,
        "run",
        _fake_run(stdout=CLAUDE_RATE_LIMIT_RETRY * 3, returncode=1),
    )
    result = _claude()
    assert result.error is not None
    assert result.error.startswith("claude exit=1: stderr=''")
    assert "3 API retries in stream" in result.error


# --- streams that end without producing output -------------------------------


def test_claude_missing_result_event_names_the_compaction(monkeypatch):
    """"no terminal result event" is what a compacted stream looks like."""
    monkeypatch.setattr(
        claude_exec.subprocess,
        "run",
        _fake_run(stdout=b'{"type":"system","status":"compacting"}\n'),
    )
    result = _claude()
    assert result.error is not None
    assert result.error.startswith("claude stream-json had no terminal result event")
    assert "context compaction" in result.error


def test_codex_missing_last_message_names_the_429(monkeypatch, tmp_path):
    monkeypatch.setattr(
        codex_exec.subprocess, "run", _fake_run(stdout=CODEX_429, returncode=0)
    )
    result = _codex(tmp_path)
    assert result.error is not None
    assert result.error.startswith("codex output_last_message file missing")
    assert "exceeded retry limit" in result.error


def test_unexplainable_failure_is_unchanged(monkeypatch, tmp_path):
    """No evidence, no invention -- the message stays exactly as it was."""
    monkeypatch.setattr(
        codex_exec.subprocess, "run", _fake_run(stdout=b"", returncode=1)
    )
    result = _codex(tmp_path)
    assert result.error == "codex exit=1: stderr=''"
