"""Unit tests for `schemas/project.py` — source-root-relative qualified names.

Covers the qn rule (qualified name = module path relative to `source_root`,
normalized per segment), the `Module.name`/basename invariant, the
`ProjectTree` path/qn validators, per-segment normalization of awkward
directory names, and the omitted-vs-explicit `source_root` fallback.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.project import Module, ProjectTree, Repository


def _tree(modules: list[dict], source_root: str | None = None) -> ProjectTree:
    repo: dict = {"name": "demo", "summary": "s"}
    if source_root is not None:
        repo["source_root"] = source_root
    return ProjectTree.model_validate({"repository": repo, "modules": modules})


# --------------------------------------------------------------------------
# Qualified-name examples (the plan's table)
# --------------------------------------------------------------------------


def test_src_layout_strips_source_root_prefix() -> None:
    tree = _tree(
        [{"path": "src/spotlights_engine/modules_extractor"}],
        source_root="src",
    )
    qns = [qn for qn, _ in tree.walk()]
    assert qns == ["spotlights_engine/modules_extractor"]


def test_src_layout_nested_submodule() -> None:
    tree = _tree(
        [
            {
                "path": "src/spotlights_engine/modules_extractor",
                "submodules": [
                    {"path": "src/spotlights_engine/modules_extractor/agent"}
                ],
            }
        ],
        source_root="src",
    )
    qns = [qn for qn, _ in tree.walk()]
    assert qns == [
        "spotlights_engine/modules_extractor",
        "spotlights_engine/modules_extractor/agent",
    ]


def test_root_layout_keeps_package_prefix() -> None:
    tree = _tree([{"path": "vllm/attention"}], source_root="")
    assert [qn for qn, _ in tree.walk()] == ["vllm/attention"]


def test_root_layout_flat() -> None:
    tree = _tree([{"path": "cli"}], source_root="")
    assert [qn for qn, _ in tree.walk()] == ["cli"]


# --------------------------------------------------------------------------
# round-trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_root", ["src", ""])
def test_resolve_round_trips_for_every_walk_qn(source_root: str) -> None:
    prefix = "src/" if source_root == "src" else ""
    tree = _tree(
        [
            {
                "path": f"{prefix}pkg",
                "submodules": [
                    {"path": f"{prefix}pkg/a"},
                    {"path": f"{prefix}pkg/b"},
                ],
            }
        ],
        source_root=source_root,
    )
    for qn, module in tree.walk():
        assert tree.resolve(qn) is module


def test_leaves_yields_only_leaf_modules() -> None:
    tree = _tree(
        [
            {
                "path": "src/pkg",
                "submodules": [{"path": "src/pkg/leaf"}],
            }
        ],
        source_root="src",
    )
    assert [qn for qn, _ in tree.leaves()] == ["pkg/leaf"]


# --------------------------------------------------------------------------
# validators
# --------------------------------------------------------------------------


def test_rejects_non_prefix_source_root() -> None:
    with pytest.raises(ValidationError, match="not under source_root"):
        _tree([{"path": "lib/pkg"}], source_root="src")


def test_rejects_path_equal_to_source_root() -> None:
    with pytest.raises(ValidationError, match="equals source_root"):
        _tree([{"path": "src"}], source_root="src")


def test_rejects_child_outside_parent() -> None:
    with pytest.raises(ValidationError, match="not nested under parent"):
        _tree(
            [
                {
                    "path": "src/v1",
                    "submodules": [{"path": "src/other"}],
                }
            ],
            source_root="src",
        )


def test_rejects_duplicate_qn_after_normalization() -> None:
    # `foo-bar` and `foo_bar` both normalize to `foo_bar`.
    with pytest.raises(ValidationError, match="duplicate qualified name"):
        _tree(
            [{"path": "src/foo-bar"}, {"path": "src/foo_bar"}],
            source_root="src",
        )


def test_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError, match="repo-relative"):
        _tree([{"path": "/abs/pkg"}], source_root="")


def test_rejects_parent_traversal_in_path() -> None:
    with pytest.raises(ValidationError, match=r"\.\."):
        _tree([{"path": "../escape"}], source_root="")


def test_rejects_absolute_source_root() -> None:
    with pytest.raises(ValidationError, match="repo-relative"):
        _tree([{"path": "pkg"}], source_root="/abs")


# --------------------------------------------------------------------------
# segment normalization + name/basename invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basename, expected_name",
    [
        ("my-module", "my_module"),
        ("v1.2", "v1_2"),
        ("2d", "m_2d"),
        ("Foo", "foo"),
    ],
)
def test_default_name_normalizes_basename(basename: str, expected_name: str) -> None:
    m = Module.model_validate({"path": f"src/{basename}"})
    assert m.name == expected_name


def test_qn_segments_are_normalized() -> None:
    tree = _tree([{"path": "src/2d-kernels"}], source_root="src")
    assert [qn for qn, _ in tree.walk()] == ["m_2d_kernels"]


def test_name_disagreeing_with_basename_is_rejected() -> None:
    with pytest.raises(ValidationError, match="disagrees with"):
        Module.model_validate({"name": "attention", "path": "pkg/attn"})


# --------------------------------------------------------------------------
# source_root inference fallback
# --------------------------------------------------------------------------


def test_omitted_source_root_infers_src_when_all_under_src() -> None:
    tree = _tree([{"path": "src/pkg/a"}, {"path": "src/pkg/b"}], source_root=None)
    assert tree.repository.source_root == "src"
    assert sorted(qn for qn, _ in tree.walk()) == ["pkg/a", "pkg/b"]


def test_omitted_source_root_infers_empty_for_root_layout() -> None:
    tree = _tree([{"path": "vllm/attention"}], source_root=None)
    assert tree.repository.source_root == ""
    assert [qn for qn, _ in tree.walk()] == ["vllm/attention"]


def test_bare_src_module_with_omitted_source_root_is_root_layout() -> None:
    # A module whose path is literally "src" must not infer source_root="src"
    # (which the path/qn validator would then reject as path==source_root).
    tree = _tree([{"path": "src"}], source_root=None)
    assert tree.repository.source_root == ""
    assert [qn for qn, _ in tree.walk()] == ["src"]


def test_to_json_from_json_annotations_resolve() -> None:
    # Regression: the module imports PurePosixPath; to_json/from_json still
    # annotate `Path`, so `Path` must remain importable for get_type_hints.
    import typing

    typing.get_type_hints(ProjectTree.to_json)
    typing.get_type_hints(ProjectTree.from_json)


def test_explicit_empty_source_root_is_respected_not_inferred() -> None:
    # All paths under src/, but the caller explicitly set "" — keep the prefix.
    tree = _tree([{"path": "src/pkg"}], source_root="")
    assert tree.repository.source_root == ""
    assert [qn for qn, _ in tree.walk()] == ["src/pkg"]


def test_source_root_normalization_strips_trailing_slash_and_dot() -> None:
    repo = Repository.model_validate(
        {"name": "r", "summary": "s", "source_root": "./src/"}
    )
    assert repo.source_root == "src"


# --------------------------------------------------------------------------
# extractor schema emission
# --------------------------------------------------------------------------


def test_json_schema_includes_source_root_under_byte_ceiling() -> None:
    import json

    from spotlights_engine.modules_extractor.agent import _MAX_SCHEMA_BYTES

    schema = ProjectTree.model_json_schema()
    repo_props = schema["$defs"]["Repository"]["properties"]
    assert "source_root" in repo_props

    schema_bytes = len(json.dumps(schema, indent=2).encode("utf-8"))
    assert schema_bytes < _MAX_SCHEMA_BYTES
