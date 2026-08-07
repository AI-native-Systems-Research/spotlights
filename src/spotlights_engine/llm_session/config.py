"""Policy dataclasses for retry and concurrency."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff-plus-jitter retry policy.

    `max_attempts` bounds transient (rate-limit/overloaded) retries;
    `timeout_max_attempts` is the smaller budget for wall-clock timeouts, since
    a hang retried the full N× multiplies real elapsed time.
    """

    max_attempts: int = 5
    timeout_max_attempts: int = 2
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    multiplier: float = 2.0
    jitter: Literal["full", "equal", "none"] = "full"


@dataclass(frozen=True)
class ConcurrencyPolicy:
    """Process-wide simultaneous-CLI-process cap."""

    max_concurrency: int = 6


# ── process-wide default RetryPolicy ──────────────────────────────────────
# Mirrors the limiter (`limiter.py`) and retry-log (`retry_log.py`) singletons:
# a process-wide default that the manager pins at startup from its
# `retry_policy` config arg (`SpotlightsManagerConfig.retry_policy`). When left
# unset, the built-in `RetryPolicy()` field defaults apply. Every
# `run_with_retry` / `guarded_run` caller passes `policy=None` and picks this
# up, so retry/backoff is tunable via the config arg with no per-call-site
# changes.

_DEFAULT_POLICY: RetryPolicy | None = None
_DEFAULT_POLICY_LOCK = threading.Lock()


def configure_default_retry_policy(policy: RetryPolicy) -> RetryPolicy:
    """Pin the process-wide default RetryPolicy (idempotent, last-wins).

    Unlike the limiter (which guards a live semaphore and fails loud on a late
    reconfigure), RetryPolicy is immutable and stateless, so re-pinning is safe.
    """
    global _DEFAULT_POLICY
    with _DEFAULT_POLICY_LOCK:
        _DEFAULT_POLICY = policy
        return policy


def get_default_retry_policy() -> RetryPolicy:
    """Return the process-wide default, or the built-in defaults if unset."""
    global _DEFAULT_POLICY
    with _DEFAULT_POLICY_LOCK:
        if _DEFAULT_POLICY is None:
            _DEFAULT_POLICY = RetryPolicy()
        return _DEFAULT_POLICY


def reset_default_retry_policy_for_tests() -> None:
    """Test-only: drop the pinned default so the built-in defaults apply."""
    global _DEFAULT_POLICY
    with _DEFAULT_POLICY_LOCK:
        _DEFAULT_POLICY = None


def write_mode_policy(base: RetryPolicy | None = None) -> RetryPolicy:
    """Single-attempt variant of `base` (default: the process-wide default).

    Preserves the configured delays/jitter while forcing one attempt, so a
    partially-applied edit is never re-applied (see `runner._effective_policy`).
    """
    base = base if base is not None else get_default_retry_policy()
    return replace(base, max_attempts=1, timeout_max_attempts=1)


__all__ = [
    "ConcurrencyPolicy",
    "RetryPolicy",
    "configure_default_retry_policy",
    "get_default_retry_policy",
    "reset_default_retry_policy_for_tests",
    "write_mode_policy",
]
