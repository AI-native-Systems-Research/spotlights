"""Global CLI limiter: true simultaneous cap, no deadlock, singleton wiring."""

from __future__ import annotations

import threading
import time

import pytest

from spotlights_engine.llm_session.limiter import (
    GlobalCLILimiter,
    configure_global_limiter,
    get_global_limiter,
    reset_global_limiter_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_limiter():
    reset_global_limiter_for_tests()
    yield
    reset_global_limiter_for_tests()


def test_caps_true_simultaneous_workers() -> None:
    limiter = GlobalCLILimiter(3)
    peak = 0
    current = 0
    lock = threading.Lock()
    start = threading.Barrier(10)

    def worker() -> None:
        nonlocal peak, current
        start.wait()
        with limiter.sync_slot():
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.02)
            with lock:
                current -= 1

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert all(not t.is_alive() for t in threads)  # no deadlock
    assert peak <= 3
    assert peak == 3  # cap actually reached under contention


def test_slot_releases_on_exception() -> None:
    limiter = GlobalCLILimiter(1)
    with pytest.raises(ValueError):
        with limiter.sync_slot():
            raise ValueError("boom")
    # If the slot leaked, this second acquire would block forever.
    acquired = threading.Event()

    def take() -> None:
        with limiter.sync_slot():
            acquired.set()

    t = threading.Thread(target=take)
    t.start()
    t.join(timeout=2)
    assert acquired.is_set()


def test_rejects_zero_budget() -> None:
    with pytest.raises(ValueError):
        GlobalCLILimiter(0)


def test_configure_before_first_use_succeeds() -> None:
    # Importing the package must not take a slot / create the singleton, so a
    # startup configure still wins.
    limiter = configure_global_limiter(4)
    assert limiter.max_concurrency == 4
    assert get_global_limiter() is limiter


def test_configure_twice_raises() -> None:
    configure_global_limiter(2)
    with pytest.raises(RuntimeError):
        configure_global_limiter(3)


def test_get_global_lazily_creates_from_default(monkeypatch) -> None:
    monkeypatch.delenv("SPOTLIGHTS_MAX_CLI_CONCURRENCY", raising=False)
    reset_global_limiter_for_tests()
    limiter = get_global_limiter()
    assert limiter.max_concurrency == 6


def test_env_var_sets_default_budget(monkeypatch) -> None:
    monkeypatch.setenv("SPOTLIGHTS_MAX_CLI_CONCURRENCY", "2")
    reset_global_limiter_for_tests()
    assert get_global_limiter().max_concurrency == 2
