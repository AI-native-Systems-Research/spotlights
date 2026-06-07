"""Stage 01 — telemetry-loader wiring.

The synthetic-fallback branch is exercised by every test in
`test_runner.py` (none of them pass `telemetry_from`); this file covers
the loader path: file vs directory targets, alternative filenames inside
a directory, and the failure modes a real user will hit.

Once Bundle A lands and the synthetic branch goes away, the
`test_synthetic_fallback_when_telemetry_from_is_none` test below should
flip to asserting that omitting `--telemetry-from` raises a clear error.
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
from spotlights_engine.signal_pipeline.schemas import Signals
from spotlights_engine.signal_pipeline.stages import s01_signal_extraction


def _signals_payload(workload_id: str = "from-fixture") -> dict:
    return {
        "workload": {"workload_id": workload_id, "description": "from-fixture"},
        "traces": [
            {
                "trace_id": "t1",
                "summary": "fixture trace",
                "raw_trace_pointer": None,
            }
        ],
        "anomalies": [
            {
                "anomaly_id": "a1",
                "type": "fixture",
                "description": "fixture anomaly",
            }
        ],
    }


def test_synthetic_fallback_when_telemetry_from_is_none(tmp_path):
    """Backwards-compatible bridge: no telemetry_from → synthetic Signals."""
    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path, telemetry_from=None),
        run_dir=run_dir,
        stages=StageSelection.only("01"),
    )
    written = Signals.model_validate_json(
        (run_dir / "01_signals.json").read_text()
    )
    assert "Synthetic" in written.workload.description or "stub" in written.workload.description.lower()


def test_loader_reads_file_directly(tmp_path):
    src = tmp_path / "my_signals.json"
    src.write_text(json.dumps(_signals_payload(workload_id="W-FILE")))

    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path, telemetry_from=src),
        run_dir=run_dir,
        stages=StageSelection.only("01"),
    )
    written = Signals.model_validate_json(
        (run_dir / "01_signals.json").read_text()
    )
    assert written.workload.workload_id == "W-FILE"


def test_loader_reads_canonical_filename_in_dir(tmp_path):
    """Directory target with `01_signals.json` inside — the documented happy path."""
    fixture_dir = tmp_path / "telemetry"
    fixture_dir.mkdir()
    (fixture_dir / "01_signals.json").write_text(
        json.dumps(_signals_payload(workload_id="W-CANON"))
    )
    # Ignored: anything else in the directory shouldn't matter.
    (fixture_dir / "noise.txt").write_text("noise")

    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path, telemetry_from=fixture_dir),
        run_dir=run_dir,
        stages=StageSelection.only("01"),
    )
    written = Signals.model_validate_json(
        (run_dir / "01_signals.json").read_text()
    )
    assert written.workload.workload_id == "W-CANON"


@pytest.mark.parametrize("name", ["signals.json", "signal.json"])
def test_loader_accepts_alternative_filenames_in_dir(tmp_path, name):
    fixture_dir = tmp_path / "telemetry"
    fixture_dir.mkdir()
    (fixture_dir / name).write_text(
        json.dumps(_signals_payload(workload_id=f"W-{name}"))
    )

    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path, telemetry_from=fixture_dir),
        run_dir=run_dir,
        stages=StageSelection.only("01"),
    )
    written = Signals.model_validate_json(
        (run_dir / "01_signals.json").read_text()
    )
    assert written.workload.workload_id == f"W-{name}"


def test_loader_dir_without_known_filename_raises(tmp_path):
    fixture_dir = tmp_path / "empty"
    fixture_dir.mkdir()
    (fixture_dir / "unrelated.json").write_text("{}")

    run_dir = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="contains neither.*pre-cooked"):
        run_pipeline(
            SignalPipelineInput(subject_root=tmp_path, telemetry_from=fixture_dir),
            run_dir=run_dir,
            stages=StageSelection.only("01"),
        )


def test_loader_missing_path_raises(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_pipeline(
            SignalPipelineInput(
                subject_root=tmp_path,
                telemetry_from=tmp_path / "nope",
            ),
            run_dir=run_dir,
            stages=StageSelection.only("01"),
        )


def test_loader_invalid_payload_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"this_is_not": "Signals"}))

    run_dir = tmp_path / "run"
    with pytest.raises(Exception):  # pydantic ValidationError; runner re-raises.
        run_pipeline(
            SignalPipelineInput(subject_root=tmp_path, telemetry_from=bad),
            run_dir=run_dir,
            stages=StageSelection.only("01"),
        )


def test_safety_order_no_top_level_main_imports():
    """Stage 01 must not bind any main-only symbol at module top — same
    invariant as stage 02. Defends the "Safety order" note in the
    approved plan."""
    forbidden = {"extract", "ExtractorConfig", "ModulesExtractorInput", "run_claude"}
    bound = set(dir(s01_signal_extraction)) & forbidden
    assert bound == set(), (
        f"safety-order regression: {bound} bound at module top of s01_signal_extraction"
    )


# ── Agent-extraction branch (raw OTel dir) ──────────────────────────────


def _otel_dir(parent: Path, with_traces: bool = True) -> Path:
    """Build a minimal raw-telemetry dir for branch-dispatch tests."""
    d = parent / "otel-capture"
    d.mkdir()
    if with_traces:
        # Single OTel resourceSpans line — enough to trigger the
        # discriminator. Tests that exercise the real agent path mock
        # `_extract_signals_via_claude` so this content is never parsed
        # by python; it just needs to be non-empty.
        (d / "traces.jsonl").write_text(
            json.dumps({"resourceSpans": []}) + "\n"
        )
    (d / "vllm_server.log").write_text("INFO model=test\n")
    return d


def test_dir_with_traces_jsonl_dispatches_to_agent_path(tmp_path):
    """Dir without pre-cooked Signals JSON but with `traces.jsonl` →
    agent extraction. The conftest fixture stubs the agent call, so
    this test runs claude-free but exercises the dispatch logic."""
    target = _otel_dir(tmp_path)
    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path, telemetry_from=target),
        run_dir=run_dir,
        stages=StageSelection.only("01"),
    )
    written = Signals.model_validate_json(
        (run_dir / "01_signals.json").read_text()
    )
    # The conftest stub identifies itself; this confirms dispatch hit
    # the agent branch, not the pre-cooked-JSON loader.
    assert written.workload.workload_id == "conftest-stub-workload"


def test_dir_with_both_signals_json_and_traces_prefers_pre_cooked(tmp_path):
    """When both branches' triggers are present, the pre-cooked JSON
    wins (cheaper, deterministic). Documented behavior — `rm` the JSON
    to force re-extraction."""
    target = _otel_dir(tmp_path)
    (target / "01_signals.json").write_text(
        json.dumps(_signals_payload(workload_id="W-PRECOOKED"))
    )
    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path, telemetry_from=target),
        run_dir=run_dir,
        stages=StageSelection.only("01"),
    )
    written = Signals.model_validate_json(
        (run_dir / "01_signals.json").read_text()
    )
    assert written.workload.workload_id == "W-PRECOOKED"


def test_dir_with_neither_branch_trigger_raises_clear_error(tmp_path):
    """No `01_signals.json`, no `traces.jsonl` → clear error message
    that names both branches' triggers so the user can fix it."""
    target = tmp_path / "empty"
    target.mkdir()
    (target / "unrelated.txt").write_text("noise")

    run_dir = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="01_signals.json.*traces.jsonl"):
        run_pipeline(
            SignalPipelineInput(subject_root=tmp_path, telemetry_from=target),
            run_dir=run_dir,
            stages=StageSelection.only("01"),
        )


@pytest.mark.no_stub_stages
def test_extract_signals_via_claude_call_args(monkeypatch, tmp_path):
    """The real agent path must (a) cwd into the telemetry dir, (b)
    grant Bash + Read so the agent can grep/jq/head, (c) use plan
    permission mode (read-only), (d) hand a closed `Signals`-shape
    schema."""
    captured: dict = {}

    def fake_run_claude(*, prompt, log_dir, json_schema, cwd, max_turns,
                       timeout_s, permission_mode, allowed_tools, **kw):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        captured.update(
            prompt=prompt,
            cwd=cwd,
            permission_mode=permission_mode,
            allowed_tools=tuple(allowed_tools),
            json_schema=json_schema,
        )
        return ClaudeRunResult(
            structured_output={
                "workload": {"workload_id": "agent-w"},
                "traces": [],
                "anomalies": [],
            },
            duration_s=0.05,
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    from spotlights_engine.signal_pipeline.stages.s01_signal_extraction import (
        _extract_signals_via_claude,
    )

    target = _otel_dir(tmp_path)
    log_dir = tmp_path / "log"
    out = _extract_signals_via_claude(target, log_dir)
    assert isinstance(out, Signals)
    assert out.workload.workload_id == "agent-w"

    assert captured["cwd"] == target
    assert captured["permission_mode"] == "plan"
    assert "Bash" in captured["allowed_tools"]
    assert "Read" in captured["allowed_tools"]
    # Schema enforces top-level required keys without forbidding extras
    # (telemetry-format flexibility — see stage 01 docstring).
    schema = json.loads(captured["json_schema"])
    assert schema["required"] == ["workload", "traces", "anomalies"]
    assert "additionalProperties" not in schema  # extras allowed


@pytest.mark.no_stub_stages
def test_extract_signals_via_claude_preserves_extra_fields(monkeypatch, tmp_path):
    """The agent is encouraged to surface richer fields than the lite
    schemas' minimum. `extra="allow"` on the Pydantic types must let
    those round-trip through `model_validate` so Bundle C sees them."""

    def fake_run_claude(**_kw):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        return ClaudeRunResult(
            structured_output={
                "workload": {
                    "workload_id": "rich-w",
                    "model": "Hermes-3-8B",
                    "kv_config": {"eviction": "lru", "size_gb": 16},
                },
                "traces": [
                    {
                        "trace_id": "t1",
                        "summary": "rich",
                        "count": 80,
                        "percentiles_ms": {"p50": 12, "p99": 230},
                    }
                ],
                "anomalies": [
                    {
                        "anomaly_id": "a1",
                        "type": "tail_latency",
                        "description": "p99 19x p50 on llm_request",
                        "magnitude": 19.2,
                        "confidence": 0.6,
                    }
                ],
            },
            duration_s=0.05,
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    from spotlights_engine.signal_pipeline.stages.s01_signal_extraction import (
        _extract_signals_via_claude,
    )

    target = _otel_dir(tmp_path)
    out = _extract_signals_via_claude(target, tmp_path / "log")
    # Pydantic with `extra="allow"` exposes extras via `model_dump`.
    dumped = out.model_dump()
    assert dumped["workload"]["model"] == "Hermes-3-8B"
    assert dumped["workload"]["kv_config"] == {"eviction": "lru", "size_gb": 16}
    assert dumped["traces"][0]["percentiles_ms"]["p99"] == 230
    assert dumped["anomalies"][0]["magnitude"] == 19.2


@pytest.mark.no_stub_stages
def test_extract_signals_via_claude_propagates_errors(monkeypatch, tmp_path):
    from spotlights_engine.signal_pipeline.stages.s01_signal_extraction import (
        SignalExtractionError,
        _extract_signals_via_claude,
    )

    def fake_run_claude(**_kw):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        return ClaudeRunResult(
            structured_output=None, duration_s=2.0, error="claude exit=1"
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    target = _otel_dir(tmp_path)
    with pytest.raises(SignalExtractionError, match="exit=1"):
        _extract_signals_via_claude(target, tmp_path / "log")
