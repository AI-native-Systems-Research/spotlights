"""Filter resolution: include keep-list, ordering, unknown-name warn-and-skip."""

from __future__ import annotations

import logging

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


def test_unknown_name_is_warned_and_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaves = ["v1/kv_offload"]
    with caplog.at_level(logging.WARNING):
        out = apply_filter(
            leaves, ModuleFilter(include=["v1/kv_offload", "does/not/exist"])
        )
    assert out == ["v1/kv_offload"]
    assert "does/not/exist" in caplog.text


def test_all_unknown_names_yield_empty_selection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaves = ["v1/kv_offload"]
    with caplog.at_level(logging.WARNING):
        out = apply_filter(leaves, ModuleFilter(include=["does/not/exist"]))
    assert out == []
    assert "does/not/exist" in caplog.text


def test_finer_grained_name_falls_back_to_nearest_ancestor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scope entry finer-grained than any module selects its nearest existing
    ancestor module (e.g. the extractor kept `v1/attention` as one module)."""
    modules = ["v1/attention", "config"]
    with caplog.at_level(logging.WARNING):
        out = apply_filter(
            modules,
            ModuleFilter(include=["v1/attention/backends", "v1/attention/ops"]),
        )
    # Both fine-grained entries collapse onto the single ancestor, deduped.
    assert out == ["v1/attention"]
    assert "v1/attention/backends" in caplog.text


def test_ancestor_fallback_picks_longest_ancestor() -> None:
    """When several ancestors exist, the nearest (longest) one wins."""
    modules = ["v1", "v1/attention"]
    out = apply_filter(modules, ModuleFilter(include=["v1/attention/backends"]))
    assert out == ["v1/attention"]


def test_leaf_without_ancestor_module_is_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A top-level name with no ancestor module (no bare source-root module) is
    warned and ignored rather than falling back."""
    modules = ["v1/attention", "config"]
    with caplog.at_level(logging.WARNING):
        out = apply_filter(modules, ModuleFilter(include=["utils"]))
    assert out == []
    assert "utils" in caplog.text


def test_virtual_prefix_expands_to_descendants() -> None:
    """A name that resolves to no module is a virtual prefix and expands."""
    leaves = ["v1/worker/gpu", "v1/worker/tpu", "v1/kv_offload"]
    out = apply_filter(leaves, ModuleFilter(include=["v1/worker"]))
    assert out == ["v1/worker/gpu", "v1/worker/tpu"]


def test_real_module_selects_only_itself() -> None:
    """When the named module exists, it selects exactly that module and none
    of its submodules."""
    modules = ["v1/worker", "v1/worker/gpu", "v1/worker/tpu", "v1/kv_offload"]
    out = apply_filter(modules, ModuleFilter(include=["v1/worker"]))
    assert out == ["v1/worker"]


def test_explicit_module_and_submodule_both_selected() -> None:
    """Naming a module and one of its submodules selects exactly those two,
    in user-supplied order."""
    modules = ["v1/worker", "v1/worker/gpu", "v1/worker/tpu"]
    out = apply_filter(
        modules, ModuleFilter(include=["v1/worker/gpu", "v1/worker"])
    )
    assert out == ["v1/worker/gpu", "v1/worker"]


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
