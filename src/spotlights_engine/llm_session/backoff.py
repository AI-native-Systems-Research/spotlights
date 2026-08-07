"""Transient-error classification and exponential-backoff-with-jitter delays."""

from __future__ import annotations

import enum
import random

from spotlights_engine.llm_session.config import RetryPolicy
from spotlights_engine.llm_session.result import (
    LLMSessionTimeout,
    LLMTransientError,
    SessionResult,
)

# Case-insensitive substrings that mark a transport error as retryable. NOTE:
# the word "timeout" is deliberately ABSENT — it shows up in unrelated
# model/tool output and matching it caused false retries. Timeouts are
# classified from the structured signal (`SessionResult.timed_out` /
# `LLMSessionTimeout`), not from text.
_RETRYABLE_SUBSTRINGS = (
    "too many concurrent requests",
    "rate limit",
    "rate_limit",
    "429",
    "overloaded",
    "overloaded_error",
    "529",
    "503",
    "service unavailable",
    "temporarily unavailable",
)


class RetryReason(enum.Enum):
    NONE = "none"  # not retryable
    TRANSIENT = "transient"  # rate-limit / overloaded / 429 / 529 / 503 …
    TIMEOUT = "timeout"  # wall-clock timeout (its own, smaller budget)


def _text_is_transient(text: str) -> bool:
    low = text.lower()
    return any(sub in low for sub in _RETRYABLE_SUBSTRINGS)


def classify_transport(*, ok: bool, timed_out: bool, text: str) -> RetryReason:
    """Core classifier over structured signals + error text.

    Shared by `classify` (SessionResult path) and the module_deep_research
    clients (AgentExecResult path) so the retryable-substring rules live in
    exactly one place.
    """
    if ok:
        return RetryReason.NONE
    if timed_out:
        return RetryReason.TIMEOUT
    return RetryReason.TRANSIENT if _text_is_transient(text) else RetryReason.NONE


def classify(result_or_exc: SessionResult | BaseException) -> RetryReason:
    """Classify a session outcome (or raised exception) for retry.

    Inspects the structured timeout signal first, then the error/stderr text
    against `_RETRYABLE_SUBSTRINGS`. A nonzero exit whose text doesn't match →
    `NONE` (fail fast, as today).
    """
    if isinstance(result_or_exc, SessionResult):
        res = result_or_exc
        haystack = (res.error or "") + "\n" + res.stderr.decode("utf-8", "replace")
        return classify_transport(ok=res.ok, timed_out=res.timed_out, text=haystack)

    exc = result_or_exc
    if isinstance(exc, LLMSessionTimeout):
        return RetryReason.TIMEOUT
    if isinstance(exc, LLMTransientError):
        return RetryReason.TRANSIENT
    if _text_is_transient(str(exc)):
        return RetryReason.TRANSIENT
    return RetryReason.NONE


def is_retryable(result_or_exc: SessionResult | BaseException) -> bool:
    return classify(result_or_exc) is not RetryReason.NONE


def budget_for(reason: RetryReason, p: RetryPolicy) -> int:
    """Attempt budget for the given reason under policy `p`."""
    if reason is RetryReason.TIMEOUT:
        return min(p.max_attempts, p.timeout_max_attempts)
    return p.max_attempts


def next_delay(attempt: int, p: RetryPolicy, *, rand: random.Random | None = None) -> float:
    """Backoff delay before retry `attempt` (0-indexed).

    `raw = min(cap, base * multiplier ** attempt)`, then apply jitter. Full
    jitter (AWS-recommended) decorrelates a thundering herd. `rand` is
    injectable for deterministic tests.
    """
    rng = rand or random
    raw = min(p.max_delay_s, p.base_delay_s * (p.multiplier**attempt))
    if p.jitter == "full":
        return rng.uniform(0, raw)
    if p.jitter == "equal":
        return raw / 2 + rng.uniform(0, raw / 2)
    return raw


__all__ = [
    "RetryReason",
    "budget_for",
    "classify",
    "classify_transport",
    "is_retryable",
    "next_delay",
]
