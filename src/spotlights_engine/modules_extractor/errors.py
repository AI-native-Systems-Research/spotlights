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


class ExtractorCoverageError(ModulesExtractorError):
    """The enriched tree is structurally valid but omits required directories.

    Carries the sorted `missing` paths — deterministically-required,
    non-symlink source directories that were neither emitted nor validly
    folded — in `context["missing"]`.
    """


class ExtractorReviewError(ModulesExtractorError):
    """Strict-mode review found remaining semantic issues, or a review-driven
    revision failed validation while `fail_on_review_issues=True`."""


class ExtractorArtifactError(ModulesExtractorError):
    """A required artifact write failed on an otherwise successful stage.

    Only raised when artifact persistence is enabled (`artifacts_dir` set):
    silently succeeding would contradict the enabled artifact contract. An
    artifact-write failure *during* error handling never masks the original
    stage error (it is attached/logged instead).
    """


__all__ = [
    "ExtractorAgentError",
    "ExtractorArtifactError",
    "ExtractorCoverageError",
    "ExtractorReviewError",
    "ExtractorSetupError",
    "ExtractorValidationError",
    "ModulesExtractorError",
]
