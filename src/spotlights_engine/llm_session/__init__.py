"""Centralized Claude & Codex session management.

One place that owns session creation, invocation, transport-level retry with
exponential backoff + jitter, and a process-wide concurrency limiter. See
`design/claude_codex_in_one_place.md`.
"""

from __future__ import annotations

from spotlights_engine.llm_session.backoff import (
    RetryReason,
    budget_for,
    classify,
    classify_transport,
    is_retryable,
    next_delay,
)
from spotlights_engine.llm_session.claude import ClaudeSession, ClaudeSessionOptions
from spotlights_engine.llm_session.codex import CodexSession, CodexSessionOptions
from spotlights_engine.llm_session.config import (
    ConcurrencyPolicy,
    RetryPolicy,
    configure_default_retry_policy,
    get_default_retry_policy,
    reset_default_retry_policy_for_tests,
    write_mode_policy,
)
from spotlights_engine.llm_session.env import clean_env
from spotlights_engine.llm_session.limiter import (
    GlobalCLILimiter,
    configure_global_limiter,
    get_global_limiter,
)
from spotlights_engine.llm_session.resolve import (
    CLIResolutionError,
    resolve_claude_argv0,
    resolve_cli,
)
from spotlights_engine.llm_session.result import (
    LLMSessionTimeout,
    LLMTransientError,
    SessionResult,
)
from spotlights_engine.llm_session.retry_log import (
    RetryLogger,
    configure_retry_log,
    get_retry_logger,
)
from spotlights_engine.llm_session.runner import arun_with_retry, run_with_retry

__all__ = [
    "CLIResolutionError",
    "ClaudeSession",
    "ClaudeSessionOptions",
    "CodexSession",
    "CodexSessionOptions",
    "ConcurrencyPolicy",
    "GlobalCLILimiter",
    "LLMSessionTimeout",
    "LLMTransientError",
    "RetryLogger",
    "RetryPolicy",
    "RetryReason",
    "SessionResult",
    "arun_with_retry",
    "budget_for",
    "classify",
    "classify_transport",
    "clean_env",
    "configure_default_retry_policy",
    "configure_global_limiter",
    "configure_retry_log",
    "get_default_retry_policy",
    "get_global_limiter",
    "get_retry_logger",
    "is_retryable",
    "next_delay",
    "reset_default_retry_policy_for_tests",
    "resolve_claude_argv0",
    "resolve_cli",
    "run_with_retry",
    "write_mode_policy",
]
