"""Stage 1 candidate discovery exception hierarchy.

All three classes carry a structured `context: dict` so callers (Stage 3,
tests) can branch on cause without parsing the message string.
"""

from __future__ import annotations

from typing import Any


class _DiscoveryError(Exception):
    """Base for the three Stage-1 discovery exceptions."""

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context)


class DiscoverySetupError(_DiscoveryError):
    """Raised before any iteration starts (bad config, missing CLI, pre-existing run dir)."""


class DiscoveryValidationError(_DiscoveryError):
    """Raised by §6.2 second-failure, §6.3, §6.7, and §6.8-empty checks."""


class DiscoveryMutationError(_DiscoveryError):
    """Raised by §6.1 when the target repo is mutated during a subprocess call."""


__all__ = [
    "DiscoveryMutationError",
    "DiscoverySetupError",
    "DiscoveryValidationError",
]
