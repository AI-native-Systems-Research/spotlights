"""Backoff policy for relaunching a rate-limited agent call.

The stream fragments are imported from `test_agent_stream` rather than restated,
so both suites are pinned to the same real captures from the IOCR run.
"""

from __future__ import annotations

import random

import pytest

from spotlights_engine.utils.agent_retry import (
    DEFAULT_ATTEMPTS,
    RetryPolicy,
    should_retry,
)
from tests.unit.utils.test_agent_stream import (
    CLAUDE_CLEAN_RESULT,
    CLAUDE_RATE_LIMIT_RETRY,
    CLAUDE_SERVER_ERROR_RETRY,
    CODEX_429,
    NOISE,
    QUOTA_EXHAUSTED,
)

RETRYING = RetryPolicy(attempts=3, base_s=30.0, max_s=300.0)


# --- the default must be a no-op --------------------------------------------


def test_default_policy_is_off() -> None:
    """Shipping opt-in is the whole safety argument: backoff makes a bad run
    longer, so a run that did not ask for it must take today's path exactly."""
    assert DEFAULT_ATTEMPTS == 1
    assert RetryPolicy().enabled is False
    assert should_retry(CODEX_429, policy=RetryPolicy(), attempt=1) is False


def test_enabled_only_above_one_attempt() -> None:
    assert RetryPolicy(attempts=2).enabled is True


# --- what earns a relaunch ---------------------------------------------------


@pytest.mark.parametrize(
    "stream",
    [CODEX_429, CLAUDE_RATE_LIMIT_RETRY * 4, CLAUDE_SERVER_ERROR_RETRY],
)
def test_rate_limits_and_transient_errors_retry(stream: bytes) -> None:
    assert should_retry(stream, policy=RETRYING, attempt=1) is True


def test_quota_exhaustion_never_retries() -> None:
    """No wait inside a run fixes an account out of budget until next month."""
    assert should_retry(QUOTA_EXHAUSTED, policy=RETRYING, attempt=1) is False


@pytest.mark.parametrize("stream", [CLAUDE_CLEAN_RESULT, NOISE, b""])
def test_a_slow_call_is_not_a_rate_limited_one(stream: bytes) -> None:
    """30 of the run's 44 step-5 failures were reported as timeouts, and the
    ~33 that were really rate limits say so in their stream. A call that was
    merely slow carries no such evidence and must not be relaunched -- otherwise
    the retry doubles the cost of every genuinely long call."""
    assert should_retry(stream, policy=RETRYING, attempt=1) is False


def test_budget_is_exhausted_by_attempt_count() -> None:
    """A rate limit that keeps recurring must still terminate."""
    assert should_retry(CODEX_429, policy=RETRYING, attempt=2) is True
    assert should_retry(CODEX_429, policy=RETRYING, attempt=3) is False


# --- how long it waits -------------------------------------------------------


def test_first_retry_waits_far_longer_than_the_cli_did() -> None:
    """The CLIs' own median backoff was 601 ms. A retry that waits the same
    order of magnitude adds cost and changes nothing, so the floor here is the
    thing under test, not the ceiling."""
    rand = random.Random(0)
    waits = [RETRYING.delay_s(2, rand=rand) for _ in range(200)]
    assert max(waits) <= 30.0
    assert sum(waits) / len(waits) > 5.0


def test_backoff_grows_then_stops_at_the_cap() -> None:
    policy = RetryPolicy(attempts=10, base_s=30.0, max_s=300.0)
    # Ceilings double per attempt: 30, 60, 120, 240, then clamp at 300.
    for attempt, ceiling in ((2, 30.0), (3, 60.0), (4, 120.0), (5, 240.0), (6, 300.0)):
        rand = random.Random(attempt)
        observed = [policy.delay_s(attempt, rand=rand) for _ in range(500)]
        assert max(observed) <= ceiling
        assert max(observed) > ceiling * 0.9, "full jitter should reach near its ceiling"
    assert policy.delay_s(9, rand=random.Random(1)) <= 300.0


def test_jitter_is_full_not_fixed() -> None:
    """Several candidates run in parallel and are throttled by the same event.
    A fixed delay would march them back in lockstep and rebuild the burst."""
    rand = random.Random(7)
    waits = {round(RETRYING.delay_s(3, rand=rand), 4) for _ in range(50)}
    assert len(waits) > 40
    assert min(waits) < 6.0, "full jitter must be able to return a short wait"


def test_no_delay_is_defined_for_the_first_launch() -> None:
    with pytest.raises(ValueError, match="retries start at 2"):
        RETRYING.delay_s(1)


# --- a policy has to describe what the run actually did ---------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attempts": 0},
        {"base_s": 0.0},
        {"max_s": -1.0},
        {"base_s": 300.0, "max_s": 30.0},
    ],
)
def test_incoherent_policies_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]
