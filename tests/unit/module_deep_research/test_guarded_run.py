"""guarded_run: limiter + transient/timeout retry over AgentExecResult."""

from __future__ import annotations

import subprocess

import pytest

from spotlights_engine.llm_session.config import RetryPolicy
from spotlights_engine.llm_session.limiter import (
    configure_global_limiter,
    reset_global_limiter_for_tests,
)
from spotlights_engine.module_deep_research.agent_exec import AgentExecResult, guarded_run


def _res(returncode: int, *, stderr: str = "", stdout: str = "") -> AgentExecResult:
    return AgentExecResult(
        command=["claude"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture(autouse=True)
def _fast_and_isolated(monkeypatch):
    monkeypatch.setattr(
        "spotlights_engine.module_deep_research.agent_exec.time.sleep", lambda _s: None
    )
    reset_global_limiter_for_tests()
    configure_global_limiter(4)
    yield
    reset_global_limiter_for_tests()


def test_success_returns_immediately() -> None:
    calls = 0

    def inner() -> AgentExecResult:
        nonlocal calls
        calls += 1
        return _res(0)

    result = guarded_run(inner, cli="claude", label="t")
    assert result.ok
    assert calls == 1


def test_transient_exit_is_retried_then_succeeds() -> None:
    seq = [_res(1, stderr="429 too many concurrent requests"), _res(0)]

    def inner() -> AgentExecResult:
        return seq.pop(0)

    result = guarded_run(
        inner, cli="claude", label="t",
        policy=RetryPolicy(base_delay_s=0, jitter="none"),
    )
    assert result.ok
    assert seq == []


def test_non_transient_exit_returned_as_is_no_retry() -> None:
    calls = 0

    def inner() -> AgentExecResult:
        nonlocal calls
        calls += 1
        return _res(3, stderr="ValidationError: bad output")

    result = guarded_run(
        inner, cli="claude", label="t",
        policy=RetryPolicy(max_attempts=5, base_delay_s=0, jitter="none"),
    )
    assert result.returncode == 3  # surfaced to the orchestrator, not retried
    assert calls == 1


def test_timeout_is_retried_within_smaller_budget() -> None:
    calls = 0

    def inner() -> AgentExecResult:
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    with pytest.raises(subprocess.TimeoutExpired):
        guarded_run(
            inner, cli="claude", label="t",
            policy=RetryPolicy(max_attempts=5, timeout_max_attempts=2,
                               base_delay_s=0, jitter="none"),
        )
    assert calls == 2  # timeout budget, not max_attempts


def test_gives_up_after_transient_budget_returns_last() -> None:
    calls = 0

    def inner() -> AgentExecResult:
        nonlocal calls
        calls += 1
        return _res(1, stderr="503 service unavailable")

    result = guarded_run(
        inner, cli="claude", label="t",
        policy=RetryPolicy(max_attempts=3, base_delay_s=0, jitter="none"),
    )
    assert result.returncode == 1
    assert calls == 3
