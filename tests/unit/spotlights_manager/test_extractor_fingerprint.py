"""Semantic extractor fingerprint after the assignments-only cutover."""

from __future__ import annotations

from spotlights_engine.modules_extractor.extractor import ExtractorConfig
from spotlights_engine.spotlights_manager.persistence import (
    build_config_fingerprint,
    extractor_semantic_hash,
    extractor_semantic_payload,
)


def test_two_phase_payload_has_the_single_stage3_contract() -> None:
    payload = extractor_semantic_payload(ExtractorConfig())
    assert payload["contract"] == "assignments"
    assert payload["merge_threshold"] == 15


def test_two_phase_hash_includes_the_merge_threshold() -> None:
    assert extractor_semantic_hash(
        ExtractorConfig(merge_threshold=12)
    ) != extractor_semantic_hash(ExtractorConfig(merge_threshold=15))


def test_single_shot_hash_ignores_assignment_policy() -> None:
    payload = extractor_semantic_payload(
        ExtractorConfig(two_phase=False, merge_threshold=12)
    )
    assert "contract" not in payload
    assert "merge_threshold" not in payload
    assert extractor_semantic_hash(
        ExtractorConfig(two_phase=False, merge_threshold=12)
    ) == extractor_semantic_hash(
        ExtractorConfig(two_phase=False, merge_threshold=15)
    )


def test_config_fingerprint_uses_the_semantic_extractor_hash() -> None:
    fp = build_config_fingerprint(
        module_filter=None,
        extractor_cfg=ExtractorConfig(),
        discovery_cfg=None,
        deep_research_cfg=None,
        proposal_from_finding_cfg=None,
        agent_proposals_cfg=None,
    )
    assert fp["extractor_hash"] == extractor_semantic_hash(ExtractorConfig())
