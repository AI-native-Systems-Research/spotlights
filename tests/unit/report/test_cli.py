from __future__ import annotations

from spotlights_engine.report.cli import main

from ._fixtures import write_run


def test_writes_experiment_html_beside_the_run_by_default(tmp_path, capsys):
    run = write_run(tmp_path)

    assert main([str(run)]) == 0

    out = run / "experiment.html"
    assert out.is_file()
    assert out.stat().st_size > 10_000

    printed = capsys.readouterr().out
    assert "run-min-0001" in printed
    assert "candidates 2 (1 high impact) over 1 modules" in printed
    # no findings in the fixture, so the research line is omitted entirely
    assert "research " not in printed


def test_honours_an_explicit_out_path_and_creates_parents(tmp_path):
    run = write_run(tmp_path)
    dest = tmp_path / "nested" / "deeper" / "page.html"

    assert main([str(run), "-o", str(dest)]) == 0

    assert dest.is_file()
    assert not (run / "experiment.html").exists()


def test_report_routed_via_main(tmp_path):
    """The engine's argv dispatch reaches this subcommand.

    The tests above call `report.cli.main` directly, which bypasses the branch
    in `spotlights_engine.cli` that routes the word `report` — so nothing else
    guards that wiring against a rename. Mirrors
    `test_init_skills.TestCliEntryPoint.test_init_routed_via_main`.
    """
    from spotlights_engine.cli import main as engine_main

    run = write_run(tmp_path)
    dest = tmp_path / "routed.html"

    assert engine_main(["report", str(run), "-o", str(dest)]) == 0
    assert dest.is_file()
