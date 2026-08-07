"""Process-wide default RetryPolicy: config-arg pinning and effective policy."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from spotlights_engine.llm_session.config import (
    RetryPolicy,
    configure_default_retry_policy,
    get_default_retry_policy,
    reset_default_retry_policy_for_tests,
    write_mode_policy,
)
from spotlights_engine.llm_session.runner import _effective_policy


@pytest.fixture(autouse=True)
def _isolated():
    reset_default_retry_policy_for_tests()
    yield
    reset_default_retry_policy_for_tests()


@dataclass
class _Options:
    permission_mode: str = "plan"


class _Session:
    def __init__(self, permission_mode: str = "plan") -> None:
        self.options = _Options(permission_mode=permission_mode)


# ── default / configure / get ───────────────────────────────────────────────


def test_unset_gives_builtin_defaults():
    assert get_default_retry_policy() == RetryPolicy()


def test_configure_is_last_wins():
    configure_default_retry_policy(RetryPolicy(max_attempts=7))
    assert get_default_retry_policy().max_attempts == 7
    configure_default_retry_policy(RetryPolicy(max_attempts=2))
    assert get_default_retry_policy().max_attempts == 2


def test_configure_returns_the_policy():
    p = RetryPolicy(max_attempts=4, base_delay_s=1.0)
    assert configure_default_retry_policy(p) is p
    assert get_default_retry_policy() is p


# ── write_mode_policy ───────────────────────────────────────────────────────


def test_write_mode_preserves_delays_forces_single_attempt():
    configure_default_retry_policy(
        RetryPolicy(max_attempts=5, base_delay_s=3.0, multiplier=4.0, jitter="none")
    )
    wm = write_mode_policy()
    assert wm.max_attempts == 1
    assert wm.timeout_max_attempts == 1
    # Delays/jitter preserved from the configured default.
    assert wm.base_delay_s == 3.0
    assert wm.multiplier == 4.0
    assert wm.jitter == "none"


def test_write_mode_accepts_explicit_base():
    wm = write_mode_policy(RetryPolicy(max_attempts=9, base_delay_s=7.0))
    assert wm.max_attempts == 1
    assert wm.base_delay_s == 7.0


# ── _effective_policy ───────────────────────────────────────────────────────


def test_effective_policy_plan_mode_uses_default():
    configure_default_retry_policy(RetryPolicy(max_attempts=8))
    assert _effective_policy(_Session("plan"), None).max_attempts == 8


@pytest.mark.parametrize("mode", ["acceptEdits", "bypassPermissions"])
def test_effective_policy_write_mode_single_attempt(mode):
    configure_default_retry_policy(RetryPolicy(max_attempts=8, base_delay_s=5.0))
    p = _effective_policy(_Session(mode), None)
    assert p.max_attempts == 1
    assert p.base_delay_s == 5.0  # configured delay preserved


def test_effective_policy_explicit_policy_wins():
    configure_default_retry_policy(RetryPolicy(max_attempts=8))
    explicit = RetryPolicy(max_attempts=3)
    assert _effective_policy(_Session("bypassPermissions"), explicit) is explicit
