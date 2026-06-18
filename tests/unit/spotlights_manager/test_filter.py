"""Filter resolution: include keep-list, ordering, unknown-name error."""

from __future__ import annotations

import pytest

from spotlights_engine.spotlights_manager.filters import ModuleFilter, apply_filter


def test_empty_filter_keeps_all() -> None:
    leaves = ["v1/kv_offload", "v1/attention/paged_kv"]
    assert apply_filter(leaves, None) == leaves
    assert apply_filter(leaves, ModuleFilter()) == leaves


def test_include_preserves_user_supplied_order() -> None:
    leaves = ["v1/kv_offload", "v1/attention/paged_kv", "v1/scheduler"]
    out = apply_filter(
        leaves, ModuleFilter(include=["v1/scheduler", "v1/kv_offload"])
    )
    assert out == ["v1/scheduler", "v1/kv_offload"]


def test_unknown_name_raises() -> None:
    leaves = ["v1/kv_offload"]
    with pytest.raises(ValueError, match="unknown qualified names"):
        apply_filter(leaves, ModuleFilter(include=["does/not/exist"]))


def test_parent_prefix_expands_to_descendants() -> None:
    leaves = ["v1/worker/gpu", "v1/worker/tpu", "v1/kv_offload"]
    out = apply_filter(leaves, ModuleFilter(include=["v1/worker"]))
    assert out == ["v1/worker/gpu", "v1/worker/tpu"]


def test_parent_prefix_dedupes_with_explicit_leaves() -> None:
    leaves = ["v1/worker/gpu", "v1/worker/tpu"]
    out = apply_filter(
        leaves, ModuleFilter(include=["v1/worker/gpu", "v1/worker"])
    )
    assert out == ["v1/worker/gpu", "v1/worker/tpu"]


def test_source_root_prefixed_names() -> None:
    """With source-root-relative qns, slash-form leaves carry the package prefix
    (`spotlights_engine/modules_extractor`). Exact and virtual-prefix matches
    both work; the package segment matches as a prefix even though no tree node
    resolves to it on its own."""
    leaves = [
        "spotlights_engine/modules_extractor/agent",
        "spotlights_engine/modules_extractor/errors",
        "spotlights_engine/schemas",
    ]
    # Exact leaf match.
    assert apply_filter(
        leaves, ModuleFilter(include=["spotlights_engine/schemas"])
    ) == ["spotlights_engine/schemas"]
    # Virtual prefix (no node resolves to `spotlights_engine`) expands all.
    assert apply_filter(
        leaves, ModuleFilter(include=["spotlights_engine"])
    ) == leaves
    # Mid prefix expands its descendants.
    assert apply_filter(
        leaves, ModuleFilter(include=["spotlights_engine/modules_extractor"])
    ) == [
        "spotlights_engine/modules_extractor/agent",
        "spotlights_engine/modules_extractor/errors",
    ]
