"""Errors raised by the modules extractor."""

from __future__ import annotations


class ModulesExtractorError(RuntimeError):
    """Base class for failures in the modules extractor."""

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.context: dict[str, object] = dict(context)


class ExtractorSetupError(ModulesExtractorError):
    """Pre-flight failure: missing CLI, bad `repo_path`, or oversized schema."""


class ExtractorAgentError(ModulesExtractorError):
    """The Claude subprocess failed (timeout, nonzero exit, malformed output)."""


class ExtractorValidationError(ModulesExtractorError):
    """The agent's response did not validate against `ProjectTree`."""


__all__ = [
    "ExtractorAgentError",
    "ExtractorSetupError",
    "ExtractorValidationError",
    "ModulesExtractorError",
]
