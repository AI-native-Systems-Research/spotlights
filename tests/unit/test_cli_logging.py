"""CLI logging setup and stdout-contract coverage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from spotlights_engine import cli
from spotlights_engine.candidate_discovery import DiscoveryConfig


def _args(
    *,
    quiet: bool = False,
    verbose: bool = False,
    log_file: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(quiet=quiet, verbose=verbose, log_file=log_file)


def _cli_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [
        h
        for h in logger.handlers
        if getattr(h, cli._HANDLER_TAG, False)
    ]


def _clear_cli_handlers(logger: logging.Logger) -> None:
    for h in _cli_handlers(logger):
        logger.removeHandler(h)
        h.close()


@pytest.fixture(autouse=True)
def _restore_spotlights_logger():
    logger = logging.getLogger("spotlights_engine")
    old_level = logger.level
    old_propagate = logger.propagate
    _clear_cli_handlers(logger)
    try:
        yield
    finally:
        _clear_cli_handlers(logger)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def test_configure_logging_is_idempotent() -> None:
    logger = logging.getLogger("spotlights_engine")

    cli._configure_logging(_args())
    cli._configure_logging(_args())

    handlers = _cli_handlers(logger)
    assert len(handlers) == 1
    assert handlers[0].level == logging.INFO


def test_configure_logging_replaces_and_closes_old_file_handler(
    tmp_path: Path,
) -> None:
    logger = logging.getLogger("spotlights_engine")
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"

    cli._configure_logging(_args(log_file=first))
    logger.info("first message")
    cli._configure_logging(_args(log_file=second))
    logger.info("second message")

    for handler in _cli_handlers(logger):
        handler.flush()

    assert "first message" in first.read_text(encoding="utf-8")
    assert "second message" not in first.read_text(encoding="utf-8")
    assert "second message" in second.read_text(encoding="utf-8")
    assert len(_cli_handlers(logger)) == 2


def test_quiet_and_verbose_are_mutually_exclusive() -> None:
    parser = cli._build_argparser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--quiet", "--verbose"])


def test_review_iterations_defaults_to_manager_default() -> None:
    args = cli._build_argparser().parse_args(["--repo", "."])

    # No flag => the DiscoveryConfig default (3) holds.
    #
    # This used to assert `discovery is None`. It no longer can: the bundled
    # models.yaml pins the Codex model, so `_build_config` always constructs a
    # DiscoveryConfig to carry it. Assert the effective value instead of the
    # object's absence — that is what the flag actually promises.
    assert args.review_iterations is None
    discovery = cli._build_config(args).discovery
    assert discovery is not None
    assert discovery.num_review_iterations == DiscoveryConfig().num_review_iterations


def test_no_review_disables_review_session() -> None:
    args = cli._build_argparser().parse_args(["--repo", ".", "--no-review"])

    assert cli._build_config(args).discovery.num_review_iterations == 0


def test_review_iterations_sets_count() -> None:
    args = cli._build_argparser().parse_args(["--repo", ".", "--review-iterations", "5"])

    assert cli._build_config(args).discovery.num_review_iterations == 5


def test_review_iterations_and_no_review_are_mutually_exclusive() -> None:
    parser = cli._build_argparser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--no-review", "--review-iterations", "3"])


def test_negative_review_iterations_rejected() -> None:
    args = cli._build_argparser().parse_args(["--review-iterations", "-1"])

    with pytest.raises(ValidationError):
        cli._build_config(args)


def test_enable_claude_search_defaults_off() -> None:
    args = cli._build_argparser().parse_args(["--repo", "."])

    assert args.enable_claude_search is False
    assert cli._build_input(args).enable_claude_search is False


def test_enable_claude_search_flag_turns_on() -> None:
    args = cli._build_argparser().parse_args(["--repo", ".", "--enable-claude-search"])

    assert args.enable_claude_search is True
    assert cli._build_input(args).enable_claude_search is True


def test_deep_research_enabled_by_default() -> None:
    args = cli._build_argparser().parse_args(["--repo", "."])

    assert args.enable_deep_research is True
    assert cli._build_input(args).enable_deep_research is True


def test_no_deep_research_flag_disables() -> None:
    args = cli._build_argparser().parse_args(["--repo", ".", "--no-deep-research"])

    assert args.enable_deep_research is False
    assert cli._build_input(args).enable_deep_research is False


def test_dry_run_reports_disabled_deep_research(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(
        [
            "--repo",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-folder",
            str(tmp_path / "output"),
            "--no-deep-research",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert "deep-research: DISABLED" in capsys.readouterr().out


def test_dry_run_omits_deep_research_line_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(
        [
            "--repo",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-folder",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert "deep-research" not in capsys.readouterr().out


def test_no_deep_research_warns_about_ignored_step3_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Warning, not `parser.error`: wrapper scripts that always pass
    `--enable-claude-search` must keep working. It goes to stderr rather than
    through `logging` because the package logger carries a NullHandler and
    `_configure_logging` has not run yet at this point in `main()`."""
    rc = cli.main(
        [
            "--repo",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-folder",
            str(tmp_path / "output"),
            "--no-deep-research",
            "--enable-claude-search",
            "--no-candidate-hotspots",
            "--max-findings-per-module",
            "5",
            "--dry-run",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "--no-deep-research" in err
    assert "--enable-claude-search" in err
    assert "--no-candidate-hotspots" in err
    assert "--max-findings-per-module" in err


def test_no_warning_when_deep_research_enabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(
        [
            "--repo",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-folder",
            str(tmp_path / "output"),
            "--enable-claude-search",
            "--dry-run",
        ]
    )

    assert capsys.readouterr().err == ""


def test_no_deep_research_alone_does_not_warn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(
        [
            "--repo",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-folder",
            str(tmp_path / "output"),
            "--no-deep-research",
            "--dry-run",
        ]
    )

    assert capsys.readouterr().err == ""


def test_main_preserves_summary_stdout_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"

    class _FakeResult:
        module_runs: dict = {}

    def _fake_run(input, *, config):
        assert input.repo_path == repo
        assert config.artifacts_dir == artifacts
        assert config.output_folder == output
        return _FakeResult()

    monkeypatch.setattr(cli, "run_with_telemetry", _fake_run)
    monkeypatch.setattr(cli, "_print_summary", lambda result, **_kw: print("summary"))

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "--artifacts-dir",
            str(artifacts),
            "--output-folder",
            str(output),
            "--quiet",
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out == "summary\n"
    assert not (output / "result.json").exists()
