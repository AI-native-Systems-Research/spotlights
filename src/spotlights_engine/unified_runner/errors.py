"""Errors raised by the unified runner."""

from __future__ import annotations


class UnifiedSetupError(RuntimeError):
    """Setup-time failure (paths missing, fingerprint mismatch with no resume, etc.)."""


class UnifiedResumeMismatchError(UnifiedSetupError):
    """Existing unified manifest's fingerprint differs from this invocation's.

    Mirrors `spotlights_manager.errors.ResumeMismatchError` for the unified
    layer. Raised when a previous unified run dir exists but the new
    invocation's `mode` / `repo_path` / `context` / sub-pipeline configs
    don't match — pass `--no-resume` to clear the dir and start fresh.
    """


class MergeIdCollisionError(RuntimeError):
    """Two reports share an id (candidate / proposal / finding / anomaly).

    The merger raises this when the concatenated id list contains duplicates.
    The schema's segmented id pattern (`<type>-<segment>-NNNN`) does not by
    itself guarantee disjointness — DR uses the module slug as its segment,
    so a module literally named `signal` would collide with signal pipeline's
    `signal` segment. Surface as a hard error so silent merging never produces
    a report with non-unique ids.
    """
