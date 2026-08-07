"""run_with_retry: transient retry, write-mode single-attempt, budget honoring."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from spotlights_engine.llm_session.config import RetryPolicy
from spotlights_engine.llm_session.limiter import (
    GlobalCLILimiter,
    configure_global_limiter,
    reset_global_limiter_for_tests,
)
from spotlights_engine.llm_session.result import SessionResult
from spotlights_engine.llm_session.runner import arun_with_retry, run_with_retry


@dataclass
class _Options:
    permission_mode: str = "plan"


class _FakeSession:
    """Records call count and returns a scripted sequence of results."""

    name = "claude"
    cli = "claude"

    def __init__(self, results: list[SessionResult], permission_mode: str = "plan") -> None:
        self._results = results
        self.options = _Options(permission_mode=permission_mode)
        self.calls = 0

    def run(self, prompt, *, cwd=None, on_event=None, log_dir=None) -> SessionResult:
        self.calls += 1
        idx = min(self.calls - 1, len(self._results) - 1)
        return self._results[idx]


def _transient() -> SessionResult:
    return SessionResult(cli="claude", returncode=1, duration_s=0.1,
                         error="429 too many concurrent requests")


def _ok() -> SessionResult:
    return SessionResult(cli="claude", returncode=0, duration_s=0.1,
                         structured_output={"ok": True})


def _fatal() -> SessionResult:
    return SessionResult(cli="claude", returncode=2, duration_s=0.1, error="bad schema")


@pytest.fixture(autouse=True)
def _fast_and_isolated(monkeypatch):
    # No real sleeping between attempts.
    monkeypatch.setattr("spotlights_engine.llm_session.runner.time.sleep", lambda _s: None)
    reset_global_limiter_for_tests()
    configure_global_limiter(4)
    yield
    reset_global_limiter_for_tests()


def test_retries_transient_then_succeeds() -> None:
    session = _FakeSession([_transient(), _transient(), _ok()])
    res = run_with_retry(session, "p", cwd=None,
                         policy=RetryPolicy(base_delay_s=0, jitter="none"))
    assert res.ok
    assert session.calls == 3


def test_gives_up_after_transient_budget() -> None:
    session = _FakeSession([_transient()])
    res = run_with_retry(session, "p", cwd=None,
                         policy=RetryPolicy(max_attempts=3, base_delay_s=0, jitter="none"))
    assert not res.ok
    assert session.calls == 3  # exactly max_attempts


def test_non_transient_fails_fast_no_retry() -> None:
    session = _FakeSession([_fatal()])
    res = run_with_retry(session, "p", cwd=None,
                         policy=RetryPolicy(max_attempts=5, base_delay_s=0, jitter="none"))
    assert res.returncode == 2
    assert session.calls == 1


def test_write_mode_single_attempt_bypass_permissions() -> None:
    # A partially-applied edit must never be re-applied: bypassPermissions →
    # exactly one call even on a transient-looking failure, with no explicit
    # policy (the runner derives max_attempts=1).
    session = _FakeSession([_transient()], permission_mode="bypassPermissions")
    res = run_with_retry(session, "p", cwd=None)
    assert not res.ok
    assert session.calls == 1


def test_write_mode_single_attempt_accept_edits() -> None:
    session = _FakeSession([_transient()], permission_mode="acceptEdits")
    run_with_retry(session, "p", cwd=None)
    assert session.calls == 1


def test_plan_mode_retries_by_default() -> None:
    # Default (read) mode with the default policy retries transient failures.
    session = _FakeSession([_transient(), _ok()], permission_mode="plan")
    res = run_with_retry(session, "p", cwd=None,
                         policy=RetryPolicy(base_delay_s=0, jitter="none"))
    assert res.ok
    assert session.calls == 2


def test_slot_is_released_before_backoff_sleep(monkeypatch) -> None:
    # The slot must NOT be held across the backoff sleep, or waiting callers are
    # serialized behind sleepers (throughput throttle / deadlock under load).
    # Pin a budget-1 limiter and capture the number of free permits observed at
    # sleep time: if the slot was released before sleeping, the (only) permit is
    # free while we sleep.
    from spotlights_engine.llm_session import limiter as limiter_mod

    reset_global_limiter_for_tests()
    configure_global_limiter(1)
    limiter = limiter_mod.get_global_limiter()

    free_during_sleep: list[int] = []

    def _record_sleep(_delay: float) -> None:
        # BoundedSemaphore._value is the count of currently-available permits.
        free_during_sleep.append(limiter._sem._value)  # type: ignore[attr-defined]

    monkeypatch.setattr("spotlights_engine.llm_session.runner.time.sleep", _record_sleep)

    session = _FakeSession([_transient(), _transient(), _ok()])
    res = run_with_retry(session, "p", cwd=None,
                         policy=RetryPolicy(base_delay_s=1.0, jitter="none"))
    assert res.ok
    # Two sleeps (before attempts 2 and 3); the sole permit was free each time.
    assert free_during_sleep == [1, 1]


# ── async wrapper (steps 3/4/5 run through this in production) ──────────────


def test_arun_retries_transient_then_succeeds() -> None:
    session = _FakeSession([_transient(), _transient(), _ok()])
    res = asyncio.run(
        arun_with_retry(session, "p", cwd=None,
                        policy=RetryPolicy(base_delay_s=0, jitter="none"))
    )
    assert res.ok
    assert session.calls == 3


def test_arun_write_mode_single_attempt() -> None:
    session = _FakeSession([_transient()], permission_mode="bypassPermissions")
    res = asyncio.run(arun_with_retry(session, "p", cwd=None))
    assert not res.ok
    assert session.calls == 1


def test_arun_slot_released_across_await_no_deadlock() -> None:
    # The async path acquires the slot INSIDE the worker thread. With a budget
    # of 1, two concurrent async callers must still both complete (the first
    # releases before the second acquires) — a held slot across the await would
    # deadlock this.
    limiter = GlobalCLILimiter(1)

    async def _main() -> tuple[SessionResult, SessionResult]:
        s1 = _FakeSession([_ok()])
        s2 = _FakeSession([_ok()])
        return await asyncio.gather(
            arun_with_retry(s1, "p", cwd=None, limiter=limiter),
            arun_with_retry(s2, "p", cwd=None, limiter=limiter),
        )

    r1, r2 = asyncio.run(asyncio.wait_for(_main(), timeout=5))
    assert r1.ok and r2.ok
