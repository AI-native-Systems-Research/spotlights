"""Top-level schemas package.

This package is being restructured to host the new SpotlightReport
unified-output types (Stage B). For now, this module is intentionally
empty — all current schemas have been moved under
``spotlights_engine.schemas.legacy``. Importers should use the
``legacy`` subpackage explicitly:

    from spotlights_engine.schemas.legacy import Candidate, Finding, ...

The new SpotlightReport types will land in this namespace in a
follow-up PR (see
``docs/_review-notes/2026-06-15_spotlight_report_schema.md``).
"""
