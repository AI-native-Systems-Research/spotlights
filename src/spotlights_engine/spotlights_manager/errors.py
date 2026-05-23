"""SpotlightsManager exception hierarchy.

Mirrors the per-step `*SetupError` convention so callers can branch on cause
without parsing the message string.
"""

from __future__ import annotations

from typing import Any


class _ManagerError(RuntimeError):
    """Base for manager-level exceptions; carries a structured `context`."""

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context)


class ManagerSetupError(_ManagerError):
    """Raised before any step runs (bad repo_path / artifacts_dir)."""


class ResumeMismatchError(_ManagerError):
    """Raised on resume when the existing run dir's frozen inputs differ
    from the current invocation."""


__all__ = ["ManagerSetupError", "ResumeMismatchError"]
