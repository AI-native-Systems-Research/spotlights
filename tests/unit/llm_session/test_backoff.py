"""Retry classifier truth table + backoff/jitter bounds."""

from __future__ import annotations

import random

import pytest

from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.llm_session.backoff import (
    RetryReason,
    budget_for,
    classify,
    classify_transport,
    is_retryable,
    next_delay,
)
from spotlights_engine.llm_session.config import RetryPolicy
from spotlights_engine.llm_session.result import (
    LLMSessionTimeout,
    LLMTransientError,
    SessionResult,
)


def _result(*, returncode: int = 0, error: str | None = None, timed_out: bool = False,
            stderr: bytes = b"") -> SessionResult:
    return SessionResult(
        cli="claude",
        returncode=returncode,
        duration_s=1.0,
        error=error,
        timed_out=timed_out,
        stderr=stderr,
        usage=AgentUsage(input=1, output=1),
    )


class TestClassifyTransport:
    def test_ok_is_never_retryable(self) -> None:
        assert classify_transport(ok=True, timed_out=False, text="rate limit") is RetryReason.NONE

    def test_timeout_wins_over_text(self) -> None:
        assert classify_transport(ok=False, timed_out=True, text="") is RetryReason.TIMEOUT

    @pytest.mark.parametrize(
        "text",
        [
            "Error: too many concurrent requests",
            "429 Too Many Requests",
            "rate_limit_error",
            "the service is Overloaded",
            "529 overloaded_error",
            "503 Service Unavailable",
            "temporarily unavailable, retry",
        ],
    )
    def test_transient_substrings(self, text: str) -> None:
        assert classify_transport(ok=False, timed_out=False, text=text) is RetryReason.TRANSIENT

    @pytest.mark.parametrize(
        "text",
        [
            "SyntaxError: invalid schema",
            "the tool reported a timeout in its output",  # 'timeout' must NOT match
            "no such file",
            "",
        ],
    )
    def test_non_transient_text_fails_fast(self, text: str) -> None:
        assert classify_transport(ok=False, timed_out=False, text=text) is RetryReason.NONE


class TestClassifySessionResult:
    def test_success_none(self) -> None:
        assert classify(_result(returncode=0)) is RetryReason.NONE

    def test_transient_from_stderr(self) -> None:
        res = _result(returncode=1, error="failed", stderr=b"429 rate limit")
        assert classify(res) is RetryReason.TRANSIENT
        assert is_retryable(res)

    def test_timeout_signal(self) -> None:
        assert classify(_result(returncode=1, timed_out=True)) is RetryReason.TIMEOUT

    def test_nonzero_unknown_text_is_none(self) -> None:
        assert classify(_result(returncode=2, error="boom")) is RetryReason.NONE


class TestClassifyException:
    def test_timeout_exc(self) -> None:
        assert classify(LLMSessionTimeout("hang")) is RetryReason.TIMEOUT

    def test_transient_exc(self) -> None:
        assert classify(LLMTransientError("overloaded")) is RetryReason.TRANSIENT

    def test_generic_exc_matches_text(self) -> None:
        assert classify(RuntimeError("429 too many")) is RetryReason.TRANSIENT

    def test_generic_exc_non_transient(self) -> None:
        assert classify(RuntimeError("bad input")) is RetryReason.NONE


class TestBudgetFor:
    def test_timeout_budget_is_smaller(self) -> None:
        p = RetryPolicy(max_attempts=5, timeout_max_attempts=2)
        assert budget_for(RetryReason.TIMEOUT, p) == 2
        assert budget_for(RetryReason.TRANSIENT, p) == 5

    def test_timeout_budget_never_exceeds_max_attempts(self) -> None:
        p = RetryPolicy(max_attempts=1, timeout_max_attempts=9)
        assert budget_for(RetryReason.TIMEOUT, p) == 1


class TestNextDelay:
    def test_full_jitter_within_cap(self) -> None:
        p = RetryPolicy(base_delay_s=2.0, max_delay_s=60.0, multiplier=2.0, jitter="full")
        rng = random.Random(1234)
        for attempt in range(8):
            raw = min(p.max_delay_s, p.base_delay_s * (p.multiplier**attempt))
            for _ in range(50):
                d = next_delay(attempt, p, rand=rng)
                assert 0.0 <= d <= raw

    def test_delay_capped_at_max(self) -> None:
        p = RetryPolicy(base_delay_s=2.0, max_delay_s=10.0, multiplier=2.0, jitter="none")
        # attempt 10 would be 2*2^10=2048 without the cap
        assert next_delay(10, p) == 10.0

    def test_equal_jitter_lower_half(self) -> None:
        p = RetryPolicy(base_delay_s=4.0, max_delay_s=60.0, multiplier=2.0, jitter="equal")
        rng = random.Random(7)
        for attempt in range(5):
            raw = min(p.max_delay_s, p.base_delay_s * (p.multiplier**attempt))
            for _ in range(50):
                d = next_delay(attempt, p, rand=rng)
                assert raw / 2 <= d <= raw

    def test_no_jitter_is_deterministic(self) -> None:
        p = RetryPolicy(base_delay_s=1.0, max_delay_s=100.0, multiplier=3.0, jitter="none")
        assert next_delay(0, p) == 1.0
        assert next_delay(1, p) == 3.0
        assert next_delay(2, p) == 9.0
