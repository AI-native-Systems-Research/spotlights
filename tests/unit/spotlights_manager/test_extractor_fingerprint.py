"""Semantic extractor fingerprint: tree-mode hashes stay pre-migration
compatible; assignment-mode hashes carry the contract and threshold."""

from __future__ import annotations

from spotlights_engine.modules_extractor.extractor import ExtractorConfig
from spotlights_engine.spotlights_manager.persistence import (
    _stable_hash,
    build_config_fingerprint,
    extractor_semantic_hash,
    extractor_semantic_payload,
)


def _pre_migration_hash(cfg: ExtractorConfig) -> str:
    """What `hash_pydantic_excluding(cfg, exclude={'artifacts_dir'})` produced
    before the `contract`/`merge_threshold` fields existed."""
    payload = cfg.model_dump(mode="json", exclude={"artifacts_dir"})
    payload.pop("contract", None)
    payload.pop("merge_threshold", None)
    return _stable_hash(payload)


def test_tree_mode_hash_is_byte_compatible_with_pre_migration() -> None:
    cfg = ExtractorConfig()
    assert cfg.contract == "tree"
    assert extractor_semantic_hash(cfg) == _pre_migration_hash(cfg)


def test_tree_mode_ignores_the_semantically_inert_threshold() -> None:
    # merge_threshold is ignored under the tree contract, so changing it must
    # not invalidate a resume.
    assert extractor_semantic_hash(
        ExtractorConfig(merge_threshold=12)
    ) == extractor_semantic_hash(ExtractorConfig(merge_threshold=15))


def test_assignment_mode_hash_includes_contract_and_threshold() -> None:
    tree_hash = extractor_semantic_hash(ExtractorConfig())
    a15 = extractor_semantic_hash(ExtractorConfig(contract="assignments"))
    a12 = extractor_semantic_hash(
        ExtractorConfig(contract="assignments", merge_threshold=12)
    )
    assert a15 != tree_hash
    assert a12 != a15
    payload = extractor_semantic_payload(ExtractorConfig(contract="assignments"))
    assert payload["contract"] == "assignments"
    assert payload["merge_threshold"] == 15


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
