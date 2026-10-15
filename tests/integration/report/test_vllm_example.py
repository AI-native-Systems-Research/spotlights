"""Render the tracked example run end to end.

The unit tests use a two-candidate fixture; this one exercises the paths only a
real run reaches — the research layer, the ranking overlay, and the evolve/ and
apply/ artifact trees. `examples/vllm_subset/result.json` is 4.3 MB, so this
lives in the integration suite rather than the unit one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.report.cli import main
from spotlights_engine.report.render import load

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "vllm_subset"

pytestmark = pytest.mark.skipif(
    not (EXAMPLE / "result.json").is_file(),
    reason="examples/vllm_subset/result.json is not present in this checkout",
)


def test_load_reads_the_full_example_run():
    d = load(EXAMPLE)

    assert d["run_id"] == "run-4b5859bcc6cc3ad1"
    assert len(d["rows"]) == 173
    assert sum(1 for r in d["rows"] if r["impact"] == "high") == 60
    assert len(d["modules"]) == 8
    # unlike the unit fixture, this run has a research layer and a ranking
    assert d["research"] is not None
    assert d["ranked"] is True


def test_cli_renders_the_example_to_an_explicit_path(tmp_path, capsys):
    dest = tmp_path / "experiment.html"

    assert main([str(EXAMPLE), "-o", str(dest)]) == 0

    assert dest.is_file()
    # the real page is ~2.8 MB; a page that collapsed to a stub would not be
    assert dest.stat().st_size > 2_000_000

    printed = capsys.readouterr().out
    assert "candidates 173 (60 high impact) over 8 modules" in printed
    assert "research " in printed
