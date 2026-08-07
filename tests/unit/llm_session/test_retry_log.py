"""Dedicated retry-log writer: file format, greppability, graceful fallback."""

from __future__ import annotations

import logging

import pytest

from spotlights_engine.llm_session.retry_log import (
    RetryLogger,
    configure_retry_log,
    get_retry_logger,
    reset_retry_log_for_tests,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_retry_log_for_tests()
    yield
    reset_retry_log_for_tests()


def test_writes_one_greppable_line_per_record(tmp_path) -> None:
    path = tmp_path / "run" / "retries.log"  # parent does not exist yet
    logger = configure_retry_log(path)
    assert get_retry_logger() is logger

    logger.record(cli="claude", label="modules_extractor", attempt=1, budget=5,
                  reason="transient", delay_s=2.5, error="429 too many concurrent requests")
    logger.record(cli="codex", label="module_deep_research.codex", attempt=2, budget=2,
                  reason="timeout", delay_s=4.0, error=None)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "retry cli=claude label=modules_extractor attempt=1/5 reason=transient" in lines[0]
    assert "delay=2.50s" in lines[0]
    assert "429 too many concurrent requests" in lines[0]
    assert "cli=codex label=module_deep_research.codex attempt=2/2 reason=timeout" in lines[1]
    # A greppable fixed prefix on every line.
    assert all(" retry cli=" in ln for ln in lines)


def test_appends_across_records(tmp_path) -> None:
    path = tmp_path / "retries.log"
    logger = configure_retry_log(path)
    for i in range(3):
        logger.record(cli="claude", label="x", attempt=i + 1, budget=5,
                      reason="transient", delay_s=1.0, error=None)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_falls_back_to_logging_when_no_path(caplog) -> None:
    logger = RetryLogger(None)  # no artifacts folder configured
    with caplog.at_level(logging.WARNING, logger="spotlights_engine.llm_session.retry"):
        logger.record(cli="claude", label="x", attempt=1, budget=5,
                      reason="transient", delay_s=1.0, error="boom")
    assert any("retry cli=claude" in rec.message for rec in caplog.records)


def test_write_failure_does_not_raise(tmp_path, caplog) -> None:
    # Point at a path whose parent is a file, so open() fails; record() must
    # swallow it (a retry log write must never crash the retry loop).
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    logger = RetryLogger.__new__(RetryLogger)
    logger._path = blocker / "retries.log"  # type: ignore[attr-defined]
    import threading

    logger._lock = threading.Lock()  # type: ignore[attr-defined]
    with caplog.at_level(logging.WARNING, logger="spotlights_engine.llm_session.retry"):
        logger.record(cli="claude", label="x", attempt=1, budget=5,
                      reason="transient", delay_s=1.0, error=None)  # must not raise


def test_default_logger_before_configure_has_no_path() -> None:
    assert get_retry_logger().path is None
