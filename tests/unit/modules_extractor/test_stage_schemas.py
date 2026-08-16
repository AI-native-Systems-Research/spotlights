"""Strict inter-stage schema tests for the assignments pipeline."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentTree,
    ModuleInfo,
    SourceRootDecision,
)


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


def test_assignment_tree_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AssignmentTree.model_validate(
            {
                "assignments": {"pkg/core": "MODULE"},
                "module_decisions": {"pkg/core": {"keep_reason": None}},
                "folds": [],
            }
        )


def test_module_info_main_file_bounds_and_uniqueness() -> None:
    six = [{"path": f"pkg/core/f{i}.py", "role": "F."} for i in range(6)]
    with pytest.raises(ValidationError, match="at most 5"):
        ModuleInfo.model_validate({"description": "Core.", "main_files": six})

    ModuleInfo.model_validate({"description": "Container.", "main_files": []})
    duplicate = [
        {"path": "pkg/core/a.py", "role": "A."},
        {"path": "pkg/core/a.py", "role": "Again."},
    ]
    with pytest.raises(ValidationError, match="unique"):
        ModuleInfo.model_validate(
            {"description": "Core.", "main_files": duplicate}
        )
