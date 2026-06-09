"""Shared CLI defaults for spotlights-engine, signal-pipeline, and
spotlights-objectives so the three entry points stay in lock-step.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_REPO = Path("../vllm")
DEFAULT_OUTPUT = Path("./spotlights-out")
DEFAULT_ARTIFACTS = Path("./artifacts")
