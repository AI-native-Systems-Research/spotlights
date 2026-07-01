"""CLI logging setup and stdout-contract coverage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from spotlights_engine import cli


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

    # No flag => leave DiscoveryConfig unset so the manager default (3) holds.
    assert args.review_iterations is None
    assert cli._build_config(args).discovery is None


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
    monkeypatch.setattr(cli, "_print_summary", lambda result: print("summary"))

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
