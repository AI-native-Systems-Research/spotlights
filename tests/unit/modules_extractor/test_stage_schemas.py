"""Strict inter-stage model tests.

The stage models are deliberately stricter than the public `ProjectTree`:
they forbid extra fields (including extras nested in the raw `Repository`),
drop `depends_on` from submodules, bound descriptions/roles, require 1-5 unique
`main_files`, and enforce `review ok == (issues == [])`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedTree,
    FoldRecord,
    ReviewReport,
    SourceRootDecision,
)


def _min_top_module(**over) -> dict:
    d = {
        "name": "core",
        "path": "pkg/core",
        "description": "Core.",
        "depends_on": [],
        "main_files": [{"path": "pkg/core/a.py", "role": "A."}],
    }
    d.update(over)
    return d


def test_source_root_decision_rejects_unknown_repository_key() -> None:
    with pytest.raises(ValidationError, match="unknown repository field"):
        SourceRootDecision.model_validate(
            {
                "repository": {
                    "name": "r",
                    "summary": "s",
                    "source_root": "src",
                    "surprise": 1,
                },
                "excluded_source_paths": [],
            }
        )


def test_source_root_decision_rejects_extra_top_level_field() -> None:
    with pytest.raises(ValidationError):
        SourceRootDecision.model_validate(
            {
                "repository": {"name": "r", "summary": "s"},
                "excluded_source_paths": [],
                "extra": True,
            }
        )


def test_enriched_tree_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        EnrichedTree.model_validate(
            {"modules": [_min_top_module(bogus=1)], "folds": []}
        )


def test_submodule_rejects_depends_on() -> None:
    with pytest.raises(ValidationError):
        EnrichedTree.model_validate(
            {
                "modules": [
                    _min_top_module(
                        submodules=[
                            {
                                "name": "sub",
                                "path": "pkg/core/sub",
                                "description": "Sub.",
                                "depends_on": ["x"],
                                "main_files": [
                                    {"path": "pkg/core/sub/x.py", "role": "X."}
                                ],
                            }
                        ]
                    )
                ],
                "folds": [],
            }
        )


def test_main_files_bounds_enforced() -> None:
    with pytest.raises(ValidationError, match="1.5"):
        EnrichedTree.model_validate(
            {"modules": [_min_top_module(main_files=[])], "folds": []}
        )
    dup = [
        {"path": "pkg/core/a.py", "role": "A."},
        {"path": "pkg/core/a.py", "role": "A again."},
    ]
    with pytest.raises(ValidationError, match="unique"):
        EnrichedTree.model_validate(
            {"modules": [_min_top_module(main_files=dup)], "folds": []}
        )


def test_fold_record_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        FoldRecord.model_validate(
            {"path": "a", "into": "b", "reason": "r", "evidence_files": []}
        )


def test_review_report_ok_must_agree_with_issues() -> None:
    with pytest.raises(ValidationError, match="ok"):
        ReviewReport.model_validate(
            {
                "ok": True,
                "issues": [
                    {"kind": "bad_fold", "path": "p", "detail": "d"}
                ],
            }
        )
    with pytest.raises(ValidationError, match="ok"):
        ReviewReport.model_validate({"ok": False, "issues": []})
    # Valid both ways.
    assert ReviewReport.model_validate({"ok": True, "issues": []}).ok
    r = ReviewReport.model_validate(
        {"ok": False, "issues": [{"kind": "missing_dir", "path": "p", "detail": "d"}]}
    )
    assert not r.ok


def test_review_issue_unknown_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewReport.model_validate(
            {
                "ok": False,
                "issues": [{"kind": "not_a_kind", "path": "p", "detail": "d"}],
            }
        )


def test_name_defaults_from_path() -> None:
    tree = EnrichedTree.model_validate(
        {
            "modules": [
                {
                    "path": "pkg/core",
                    "description": "Core.",
                    "main_files": [{"path": "pkg/core/a.py", "role": "A."}],
                }
            ],
            "folds": [],
        }
    )
    assert tree.modules[0].name == "core"
