"""Manager startup wires the retry log + optional CLI concurrency budget."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.llm_session.config import (
    RetryPolicy,
    get_default_retry_policy,
    reset_default_retry_policy_for_tests,
)
from spotlights_engine.llm_session.limiter import (
    get_global_limiter,
    reset_global_limiter_for_tests,
)
from spotlights_engine.llm_session.retry_log import (
    get_retry_logger,
    reset_retry_log_for_tests,
)
from spotlights_engine.spotlights_manager.api import SpotlightsManagerConfig
from spotlights_engine.spotlights_manager.orchestrator import _configure_llm_session
from spotlights_engine.spotlights_manager.persistence import ManagerPaths


def _config(
    tmp_path: Path,
    *,
    max_cli_concurrency: int | None,
    retry_policy: RetryPolicy | None = None,
) -> SpotlightsManagerConfig:
    return SpotlightsManagerConfig(
        artifacts_dir=tmp_path / "artifacts",
        output_folder=tmp_path / "out",
        max_cli_concurrency=max_cli_concurrency,
        retry_policy=retry_policy,
    )


def _run(monkeypatch, tmp_path, *, max_cli_concurrency, retry_policy=None):
    monkeypatch.delenv("SPOTLIGHTS_MAX_CLI_CONCURRENCY", raising=False)
    reset_global_limiter_for_tests()
    reset_retry_log_for_tests()
    reset_default_retry_policy_for_tests()
    cfg = _config(tmp_path, max_cli_concurrency=max_cli_concurrency, retry_policy=retry_policy)
    paths = ManagerPaths(cfg.artifacts_dir)
    _configure_llm_session(cfg, paths)
    return cfg, paths


def test_retry_log_lands_in_artifacts_folder(monkeypatch, tmp_path) -> None:
    _, paths = _run(monkeypatch, tmp_path, max_cli_concurrency=None)
    logger = get_retry_logger()
    assert logger.path == paths.root / "retries.log"
    # Records now land in the run's artifacts folder.
    logger.record(cli="claude", label="x", attempt=1, budget=5,
                  reason="transient", delay_s=1.0, error=None)
    assert (paths.root / "retries.log").exists()
    reset_global_limiter_for_tests()
    reset_retry_log_for_tests()


def test_pins_cli_budget_when_set(monkeypatch, tmp_path) -> None:
    _run(monkeypatch, tmp_path, max_cli_concurrency=3)
    assert get_global_limiter().max_concurrency == 3
    reset_global_limiter_for_tests()
    reset_retry_log_for_tests()


def test_unset_budget_leaves_default(monkeypatch, tmp_path) -> None:
    _run(monkeypatch, tmp_path, max_cli_concurrency=None)
    # No pin → lazy default budget (6) when first resolved.
    assert get_global_limiter().max_concurrency == 6
    reset_global_limiter_for_tests()
    reset_retry_log_for_tests()


def test_pins_retry_policy_when_set(monkeypatch, tmp_path) -> None:
    policy = RetryPolicy(max_attempts=9, base_delay_s=1.0)
    _run(monkeypatch, tmp_path, max_cli_concurrency=None, retry_policy=policy)
    assert get_default_retry_policy() is policy
    reset_global_limiter_for_tests()
    reset_retry_log_for_tests()
    reset_default_retry_policy_for_tests()


def test_unset_retry_policy_leaves_default(monkeypatch, tmp_path) -> None:
    _run(monkeypatch, tmp_path, max_cli_concurrency=None)
    # No pin → built-in RetryPolicy defaults (max_attempts=5).
    assert get_default_retry_policy() == RetryPolicy()
    reset_global_limiter_for_tests()
    reset_retry_log_for_tests()
    reset_default_retry_policy_for_tests()


def test_late_reconfigure_downgraded_to_warning(monkeypatch, tmp_path, caplog) -> None:
    # A slot already taken (limiter lazily created) must not crash startup.
    monkeypatch.delenv("SPOTLIGHTS_MAX_CLI_CONCURRENCY", raising=False)
    reset_global_limiter_for_tests()
    reset_retry_log_for_tests()
    get_global_limiter()  # force lazy creation
    cfg = _config(tmp_path, max_cli_concurrency=3)
    paths = ManagerPaths(cfg.artifacts_dir)
    _configure_llm_session(cfg, paths)  # must not raise
    reset_global_limiter_for_tests()
    reset_retry_log_for_tests()
