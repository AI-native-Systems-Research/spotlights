"""CLI wiring for the signal pipeline — focused on the SigNoz source flags.

Covers the parse-time validation added with `--signoz` / `--run-id`:
mutual exclusion with `--telemetry-from`, and `--run-id` requiring `--signoz`.
`run_pipeline` is patched out so these tests exercise only arg parsing and the
`SignalPipelineInput` mapping, never a real run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.signal_pipeline import cli
from spotlights_engine.signal_pipeline.schemas import SignalPipelineResult


@pytest.fixture
def captured(monkeypatch):
    """Patch `run_pipeline` to capture the `SignalPipelineInput` main builds."""
    box: dict = {}

    def fake_run_pipeline(sp_input, **kwargs):
        box["input"] = sp_input
        return SignalPipelineResult(run_dir=Path("rd"), output_folder=Path("of"))

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    return box


# ── valid source selections ─────────────────────────────────────────────


def test_signoz_alone_sets_flag_and_auto_selects(captured):
    rc = cli.main(["--signoz"])
    assert rc == 0
    sp = captured["input"]
    assert sp.signoz is True
    assert sp.run_id is None  # auto-select latest
    assert sp.telemetry_from is None


def test_signoz_with_run_id_passes_id_through(captured):
    rc = cli.main(["--signoz", "--run-id", "20260615T073440Z"])
    assert rc == 0
    sp = captured["input"]
    assert sp.signoz is True
    assert sp.run_id == "20260615T073440Z"


def test_telemetry_from_alone_still_works(captured):
    rc = cli.main(["--telemetry-from", "some/path"])
    assert rc == 0
    sp = captured["input"]
    assert sp.telemetry_from == Path("some/path")
    assert sp.signoz is False
    assert sp.run_id is None


def test_no_source_flags_defaults_to_synthetic(captured):
    rc = cli.main([])
    assert rc == 0
    sp = captured["input"]
    assert sp.signoz is False
    assert sp.telemetry_from is None
    assert sp.run_id is None


# ── rejected combinations (parse-time) ───────────────────────────────────


def test_signoz_and_telemetry_from_are_mutually_exclusive(captured, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--signoz", "--telemetry-from", "some/path"])
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err  # argparse group message
    assert "input" not in captured  # never reached run_pipeline


def test_run_id_without_signoz_is_rejected(captured, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--run-id", "20260615T073440Z"])
    assert exc.value.code == 2
    assert "--run-id requires --signoz" in capsys.readouterr().err
    assert "input" not in captured


def test_run_id_with_telemetry_from_is_rejected(captured, capsys):
    # --run-id without --signoz fails first, regardless of --telemetry-from.
    with pytest.raises(SystemExit) as exc:
        cli.main(["--run-id", "20260615T073440Z", "--telemetry-from", "some/path"])
    assert exc.value.code == 2
    assert "--run-id requires --signoz" in capsys.readouterr().err
    assert "input" not in captured
