"""Filter resolution: include keep-list, ordering, unknown-name error."""

from __future__ import annotations

import pytest

from spotlights_engine.spotlights_manager.filters import ModuleFilter, apply_filter


def test_empty_filter_keeps_all() -> None:
    leaves = ["v1.kv_offload", "v1.attention.paged_kv"]
    assert apply_filter(leaves, None) == leaves
    assert apply_filter(leaves, ModuleFilter()) == leaves


def test_include_preserves_user_supplied_order() -> None:
    leaves = ["v1.kv_offload", "v1.attention.paged_kv", "v1.scheduler"]
    out = apply_filter(
        leaves, ModuleFilter(include=["v1.scheduler", "v1.kv_offload"])
    )
    assert out == ["v1.scheduler", "v1.kv_offload"]


def test_unknown_name_raises() -> None:
    leaves = ["v1.kv_offload"]
    with pytest.raises(ValueError, match="unknown qualified names"):
        apply_filter(leaves, ModuleFilter(include=["does.not.exist"]))
