"""Stage 01 — SigNoz/SQL source (`s01_signal_extraction_signoz`).

Run resolution + auto-select warning, the agent call-args, credential/network
error wrapping, and end-to-end dispatch from `run()`. No live dependency:
`SignozClient` and `run_claude` are monkeypatched.
"""

from __future__ import annotations

import json

import pytest

from spotlights_engine.signal_pipeline import (
    SignalPipelineInput,
    StageSelection,
    run_pipeline,
)
from spotlights_engine.signal_pipeline.schemas import Signals
from spotlights_engine.signal_pipeline.stages import (
    s01_signal_extraction,
)
from spotlights_engine.signal_pipeline.stages import (
    s01_signal_extraction_signoz as signoz_stage,
)
from spotlights_engine.signal_pipeline.stages._s01_signal_common import (
    SignalExtractionError,
)


class _FakeSignozClient:
    """Stand-in for `SignozClient` whose `list_runs` returns a fixed list."""

    def __init__(self, runs):
        self._runs = runs

    def list_runs(self):
        return self._runs


def _patch_list_runs(monkeypatch, runs):
    import spotlights_engine.signal_pipeline.signoz_tool as st

    monkeypatch.setattr(
        st.SignozClient, "from_env", classmethod(lambda cls: _FakeSignozClient(runs))
    )


# ── run resolution ──────────────────────────────────────────────────────


def test_resolve_run_explicit_id_skips_list_runs(monkeypatch):
    """An explicit --run-id is used verbatim and never touches SigNoz."""
    import spotlights_engine.signal_pipeline.signoz_tool as st

    def _boom(cls):
        raise AssertionError("from_env must not be called for an explicit run_id")

    monkeypatch.setattr(st.SignozClient, "from_env", classmethod(_boom))
    assert signoz_stage._resolve_run("20260615T073440Z", None) == "20260615T073440Z"


def test_resolve_run_single_run_no_warning(monkeypatch):
    _patch_list_runs(monkeypatch, ["20260615T073440Z"])
    events: list[str] = []
    assert signoz_stage._resolve_run(None, events.append) == "20260615T073440Z"
    assert events == []  # single run → silent


def test_resolve_run_multiple_picks_latest_and_warns(monkeypatch):
    runs = ["20260614T120000Z", "20260615T073440Z", "20260613T010101Z"]
    _patch_list_runs(monkeypatch, runs)
    events: list[str] = []
    chosen = signoz_stage._resolve_run(None, events.append)

    assert chosen == "20260615T073440Z"  # max() == latest
    assert len(events) == 1
    banner = events[0]
    assert "3 runs found" in banner
    assert chosen in banner
    assert "--run-id" in banner  # tells the user how to override
    # the non-chosen runs are named
    assert "20260614T120000Z" in banner and "20260613T010101Z" in banner


def test_resolve_run_no_runs_raises(monkeypatch):
    _patch_list_runs(monkeypatch, [])
    with pytest.raises(SignalExtractionError, match="no runs found"):
        signoz_stage._resolve_run(None, None)


def test_resolve_run_wraps_signoz_error_as_extraction_error(monkeypatch):
    """Credential/network SignozError (e.g. missing .env) must surface as
    SignalExtractionError, honouring the stage's error-type contract."""
    import spotlights_engine.signal_pipeline.signoz_tool as st

    def _raise(cls):
        raise st.SignozError("SIGNOZ_API_KEY missing from .env")

    monkeypatch.setattr(st.SignozClient, "from_env", classmethod(_raise))
    with pytest.raises(SignalExtractionError, match="--signoz"):
        signoz_stage._resolve_run(None, None)


# ── agent extraction ─────────────────────────────────────────────────────


@pytest.mark.no_stub_stages
def test_extract_via_signoz_call_args(monkeypatch, tmp_path):
    """The SigNoz agent path must (a) cwd into the project root so
    `claude -p` finds the project's `.claude/` auth (no telemetry dir on
    this path), (b) grant Bash + Read, (c) use plan permission mode,
    (d) hand the closed `Signals`-shape schema, and (e) embed the run_id
    in the prompt."""
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
                "workload": {"workload_id": "20260615T073440Z"},
                "traces": [],
                "anomalies": [],
            },
            duration_s=0.05,
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    log_dir = tmp_path / "log"
    out = signoz_stage._extract_via_signoz("20260615T073440Z", log_dir)
    assert isinstance(out, Signals)

    assert captured["cwd"] == signoz_stage._PROJECT_ROOT  # project root → .claude/ auth
    assert (signoz_stage._PROJECT_ROOT / "pyproject.toml").is_file()  # really the root
    assert captured["permission_mode"] == "plan"
    assert "Bash" in captured["allowed_tools"]
    assert "Read" in captured["allowed_tools"]
    assert "20260615T073440Z" in captured["prompt"]  # run_id reached the agent
    schema = json.loads(captured["json_schema"])
    assert schema["required"] == ["workload", "traces", "anomalies"]


@pytest.mark.no_stub_stages
def test_run_pipeline_signoz_branch_dispatches_with_run_id(monkeypatch, tmp_path):
    """End-to-end through the runner: `--signoz --run-id X` dispatches to the
    SigNoz module and reaches the extractor with the explicit id (no network)."""
    captured: dict = {}

    def fake_extract_via(run_id, log_dir, on_event=None, model=None):
        captured["run_id"] = run_id
        return s01_signal_extraction._synthetic_signals()

    monkeypatch.setattr(signoz_stage, "_extract_via_signoz", fake_extract_via)

    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(
            subject_root=tmp_path, signoz=True, run_id="20260615T073440Z"
        ),
        run_dir=run_dir,
        stages=StageSelection.only("01"),
    )
    assert captured["run_id"] == "20260615T073440Z"
