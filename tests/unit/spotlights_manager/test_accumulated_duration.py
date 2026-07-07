"""Unit tests for `_accumulated_duration_s` (design/accumulated_duration.md).

The helper sums durably-persisted per-step durations so the reported duration
is reconstructed from disk and stays stable across a crash + resume, rather
than reflecting only the last (resuming) process leg's wall clock.
"""

from __future__ import annotations

from spotlights_engine.spotlights_manager import orchestrator as orch
from spotlights_engine.spotlights_manager.api import ModuleTelemetry


def _tel(**kwargs: object) -> ModuleTelemetry:
    return ModuleTelemetry(**kwargs)  # type: ignore[arg-type]


def test_sums_extractor_and_all_module_steps() -> None:
    manifest = {"extractor": {"completed": True, "duration_s": 2.0}}
    per_module = {
        "pkg/a": _tel(
            discovery_total_duration_s=3.0,
            deep_research_duration_s=4.0,
            proposal_from_finding_duration_s=5.0,
            agent_proposals_duration_s=6.0,
        ),
        "pkg/b": _tel(
            discovery_total_duration_s=1.5,
            deep_research_duration_s=2.5,
            proposal_from_finding_duration_s=0.0,
            agent_proposals_duration_s=1.0,
        ),
    }

    total = orch._accumulated_duration_s(manifest, per_module)

    # 2 + (3+4+5+6) + (1.5+2.5+0+1) = 2 + 18 + 5 = 25
    assert total == 25.0


def test_none_terms_count_as_zero() -> None:
    manifest = {"extractor": {"duration_s": 1.0}}
    per_module = {
        "pkg/a": _tel(
            discovery_total_duration_s=None,
            deep_research_duration_s=None,
            proposal_from_finding_duration_s=None,
            agent_proposals_duration_s=None,
        ),
    }

    assert orch._accumulated_duration_s(manifest, per_module) == 1.0


def test_missing_or_malformed_extractor_contributes_zero() -> None:
    per_module = {"pkg/a": _tel(discovery_total_duration_s=7.0)}

    assert orch._accumulated_duration_s({}, per_module) == 7.0
    assert orch._accumulated_duration_s({"extractor": None}, per_module) == 7.0
    assert (
        orch._accumulated_duration_s(
            {"extractor": {"duration_s": None}}, per_module
        )
        == 7.0
    )


def test_empty_run_is_zero() -> None:
    assert orch._accumulated_duration_s({}, {}) == 0.0
