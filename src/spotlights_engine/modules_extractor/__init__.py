"""Modules extractor — produce a `ProjectTree` from a project directory.

This is step 1 of the deep-research pipeline described in
`docs/architecture/spotlights_deep_research_path_architecture.md`. Public
entrypoints:

- `extract(input)` — architecture-shaped contract, takes a
  `ModulesExtractorInput` and returns a `ProjectTree`.
- `extract_with_telemetry(input, *, config)` — runtime-rich form with
  optional artifacts persistence and tunable agent budgets.

See [docs/modules_extractor.md](../../../docs/modules_extractor.md) for the
narrative spec.
"""

from spotlights_engine.modules_extractor.errors import (
    ExtractorAgentError,
    ExtractorArtifactError,
    ExtractorCoverageError,
    ExtractorSetupError,
    ExtractorValidationError,
    ModulesExtractorError,
)
from spotlights_engine.modules_extractor.extractor import (
    ExtractorConfig,
    ExtractorResult,
    extract,
    extract_with_telemetry,
)

__all__ = [
    "ExtractorAgentError",
    "ExtractorArtifactError",
    "ExtractorConfig",
    "ExtractorCoverageError",
    "ExtractorResult",
    "ExtractorSetupError",
    "ExtractorValidationError",
    "ModulesExtractorError",
    "extract",
    "extract_with_telemetry",
]
