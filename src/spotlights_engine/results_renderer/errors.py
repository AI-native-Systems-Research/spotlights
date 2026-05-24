"""Error types for the results renderer."""

from __future__ import annotations


class RendererSetupError(Exception):
    """Output folder invalid, or `artifacts_dir` missing the
    `spotlights_manager/` subtree."""


class RendererLoadError(Exception):
    """Manifest invalid or required extractor outputs missing on disk."""


__all__ = ["RendererLoadError", "RendererSetupError"]
