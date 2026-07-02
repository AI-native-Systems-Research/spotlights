"""Unit tests for the step-3 debug artifacts (`debug_dir`)."""

from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult
from spotlights_engine.module_deep_research.orchestration import (
    RunnerOutcome,
    merge_outcomes,
)
from spotlights_engine.schemas.pipeline import PaperFilter


def _wire(finding_id: str, title: str, url: str) -> dict:
    return {
        "finding_id": finding_id,
        "title": title,
        "url": url,
        "source_type": "paper",
        "technique_summary": "summary",
    }


def _outcome(agent_name: str, findings: list[dict]) -> RunnerOutcome:
    payload = json.dumps({"findings": findings, "issues": []})
    return RunnerOutcome(
        agent_name=agent_name,
        result=AgentExecResult(
            command=[agent_name],
            returncode=0,
            stdout="",
            stderr="",
            final_message=payload,
        ),
    )


def test_writes_per_runner_output_and_merged_set(tmp_path: Path) -> None:
    debug_dir = tmp_path / "debug"
    outcomes = [
        _outcome("codex", [_wire("find-0001", "Flash", "https://example.com/flash")]),
        _outcome("claude", [_wire("find-0002", "Preble", "https://example.com/preble")]),
        RunnerOutcome(agent_name="gemini", error="gemini exploded"),
    ]

    merge_outcomes(
        outcomes,
        max_findings_per_module=30,
        segment="mod",
        debug_dir=debug_dir,
    )

    # One raw-output file per runner, including the failed one.
    codex_out = (debug_dir / "codex.output.md").read_text(encoding="utf-8")
    assert "Flash" in codex_out
    assert (debug_dir / "claude.output.md").read_text(encoding="utf-8")
    gemini_out = (debug_dir / "gemini.output.md").read_text(encoding="utf-8")
    assert "FAILED" in gemini_out
    assert "gemini exploded" in gemini_out

    # Merged (deduped) set before any filter — both findings present.
    merged = json.loads((debug_dir / "merged_before_filter.json").read_text("utf-8"))
    assert {f["title"] for f in merged} == {"Flash", "Preble"}


def test_merged_before_filter_captures_prefilter_set(tmp_path: Path) -> None:
    debug_dir = tmp_path / "debug"
    outcome = _outcome(
        "codex",
        [
            _wire("find-0001", "Flash", "https://example.com/flash"),
            _wire("find-0002", "Preble", "https://arxiv.org/abs/2504.19874v2"),
        ],
    )

    output = merge_outcomes(
        [outcome],
        max_findings_per_module=30,
        segment="mod",
        paper_filter=PaperFilter(url="https://arxiv.org/pdf/2504.19874"),
        debug_dir=debug_dir,
    )

    # The filter collapsed `findings` to one, but the debug set keeps both.
    assert len(output.findings) == 1
    merged = json.loads((debug_dir / "merged_before_filter.json").read_text("utf-8"))
    assert {f["title"] for f in merged} == {"Flash", "Preble"}


def test_no_debug_dir_writes_nothing(tmp_path: Path) -> None:
    debug_dir = tmp_path / "debug"
    outcome = _outcome("codex", [_wire("find-0001", "Flash", "https://example.com/flash")])

    merge_outcomes([outcome], max_findings_per_module=30, segment="mod")

    assert not debug_dir.exists()
